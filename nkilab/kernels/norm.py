"""Row-wise softmax and RMSNorm.

Rows map to the 128 SBUF partitions and the reduced dimension lives on the
free axis, so every reduction is a per-partition free-axis reduction that
the Vector/Scalar engines do in one instruction. Row counts that are not a
multiple of 128 are handled with a short final tile (no host padding).
"""
from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


def _row_tiles(rows: int):
    for r0 in range(0, rows, P):
        yield r0, min(P, rows - r0)


@nki.jit
def softmax_unfused(x):
    """Framework-style softmax: three passes, each round-tripping through HBM.

    pass 1: m = max(x)            -> HBM
    pass 2: e = exp(x - m), s = sum(e) -> HBM
    pass 3: y = e / s             -> HBM
    """
    R, D = x.shape
    out = nl.ndarray((R, D), dtype=x.dtype, buffer=nl.shared_hbm)
    m_hbm = nl.ndarray((R, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    e_hbm = nl.ndarray((R, D), dtype=nl.float32, buffer=nl.shared_hbm)
    s_hbm = nl.ndarray((R, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    for r0, rp in _row_tiles(R):
        xt = nl.ndarray((rp, D), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x[r0:r0 + rp, :])
        mt = nl.ndarray((rp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mt, op=nl.max, data=xt, axis=1)
        nisa.dma_copy(dst=m_hbm[r0:r0 + rp, :], src=mt)
    for r0, rp in _row_tiles(R):
        xt = nl.ndarray((rp, D), dtype=x.dtype, buffer=nl.sbuf)
        mt = nl.ndarray((rp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x[r0:r0 + rp, :])
        nisa.dma_copy(dst=mt, src=m_hbm[r0:r0 + rp, :])
        sh = nl.ndarray((rp, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sh, data=xt, op0=nl.subtract, operand0=mt)
        et = nl.ndarray((rp, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=et, op=nl.exp, data=sh)
        st = nl.ndarray((rp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=st, op=nl.add, data=et, axis=1)
        nisa.dma_copy(dst=e_hbm[r0:r0 + rp, :], src=et)
        nisa.dma_copy(dst=s_hbm[r0:r0 + rp, :], src=st)
    for r0, rp in _row_tiles(R):
        et = nl.ndarray((rp, D), dtype=nl.float32, buffer=nl.sbuf)
        st = nl.ndarray((rp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=et, src=e_hbm[r0:r0 + rp, :])
        nisa.dma_copy(dst=st, src=s_hbm[r0:r0 + rp, :])
        yt = nl.ndarray((rp, D), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=yt, data=et, op0=nl.divide, operand0=st)
        nisa.dma_copy(dst=out[r0:r0 + rp, :], src=yt)
    return out


@nki.jit
def softmax_fused(x):
    """One HBM read and one write per row tile.

    Vector Engine: -max(x) (negate is free). Scalar Engine: exp(x + (-max))
    with the row sum accumulated in the same instruction. Vector Engine:
    reciprocal, then one broadcast multiply.
    """
    R, D = x.shape
    out = nl.ndarray((R, D), dtype=x.dtype, buffer=nl.shared_hbm)
    for r0, rp in _row_tiles(R):
        xt = nl.ndarray((rp, D), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x[r0:r0 + rp, :])
        neg_max = nl.ndarray((rp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=neg_max, op=nl.max, data=xt, axis=1, negate=True)
        et = nl.ndarray((rp, D), dtype=nl.float32, buffer=nl.sbuf)
        ssum = nl.ndarray((rp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=et, op=nl.exp, data=xt, bias=neg_max,
                        reduce_op=nl.add, reduce_res=ssum, reduce_cmd=nisa.reduce_cmd.reset_reduce)
        rinv = nl.ndarray((rp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rinv, data=ssum)
        yt = nl.ndarray((rp, D), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=yt, data=et, op0=nl.multiply, operand0=rinv)
        nisa.dma_copy(dst=out[r0:r0 + rp, :], src=yt)
    return out


@nki.jit
def rmsnorm_fused(x, weight, eps: float = 1e-6):
    """y = x * rsqrt(mean(x^2) + eps) * weight, one HBM pass per row tile.

    The weight row is DMA-broadcast across partitions once and reused.
    """
    R, D = x.shape
    out = nl.ndarray((R, D), dtype=x.dtype, buffer=nl.shared_hbm)
    w = nl.ndarray((P, D), dtype=weight.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=weight.reshape((1, D)).broadcast(0, P))
    for r0, rp in _row_tiles(R):
        xt = nl.ndarray((rp, D), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x[r0:r0 + rp, :])
        sq = nl.ndarray((rp, D), dtype=nl.float32, buffer=nl.sbuf)
        ssq = nl.ndarray((rp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=xt,
                        reduce_op=nl.add, reduce_res=ssq, reduce_cmd=nisa.reduce_cmd.reset_reduce)
        ms = nl.ndarray((rp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ms, data=ssq, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
        rstd = nl.ndarray((rp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=ms)
        xn = nl.ndarray((rp, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=xn, data=xt, op0=nl.multiply, operand0=rstd)
        yt = nl.ndarray((rp, D), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=yt, data1=xn, data2=w[0:rp, :], op=nl.multiply)
        nisa.dma_copy(dst=out[r0:r0 + rp, :], src=yt)
    return out


@nki.jit
def swiglu_fused(gate, up):
    """y = silu(gate) * up in one pass: Scalar Engine silu, Vector Engine multiply."""
    R, D = gate.shape
    out = nl.ndarray((R, D), dtype=gate.dtype, buffer=nl.shared_hbm)
    for r0, rp in _row_tiles(R):
        g = nl.ndarray((rp, D), dtype=gate.dtype, buffer=nl.sbuf)
        u = nl.ndarray((rp, D), dtype=up.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=g, src=gate[r0:r0 + rp, :])
        nisa.dma_copy(dst=u, src=up[r0:r0 + rp, :])
        a = nl.ndarray((rp, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=a, op=nl.silu, data=g)
        y = nl.ndarray((rp, D), dtype=gate.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y, data1=a, data2=u, op=nl.multiply)
        nisa.dma_copy(dst=out[r0:r0 + rp, :], src=y)
    return out
