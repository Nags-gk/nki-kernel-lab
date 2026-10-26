"""Causal scaled-dot-product attention: O = softmax(Q K^T * scale + mask) V.

Host layout (prepared by ``nkilab.ops.attention``):
    qT, kT : [H, d, S]   head_dim on partitions, so Q K^T is one nc_matmul
    v      : [H, S, d]   keys on partitions, so P V is one nc_matmul
    out    : [H, S, d]

Three versions, same numerics:
  * attention_naive     materializes the S x S score and probability matrices
                        in HBM (what an unfused framework graph does).
  * flash_attention     FlashAttention-style online softmax: per 128-query
                        tile, stream 128-key tiles, keep running max / sum /
                        output in SBUF, never write scores to HBM, and skip
                        key tiles that are fully masked by causality.
  * flash_attention_kv_resident  same, but each head's K and V are loaded
                        into SBUF once and reused by every query tile.
"""
from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

T = 128  # query / key tile
NEG_INF = -3.0e38


def _causal_mask_tile(dst, src, q0, k0):
    """dst = src where (q0 + p) - (k0 + c) >= 0 else -inf  (GpSimd affine_select)."""
    nisa.affine_select(dst=dst, pattern=[[-1, src.shape[1]]], channel_multiplier=1,
                       on_true_tile=src, on_false_value=NEG_INF,
                       cmp_op=nl.greater_equal, offset=q0 - k0)


@nki.jit
def attention_naive(qT, kT, v, scale: float):
    H, d, S = qT.shape
    assert S % T == 0 and d <= 128
    nt = S // T
    out = nl.ndarray((H, S, d), dtype=v.dtype, buffer=nl.shared_hbm)
    scores = nl.ndarray((H, S, S), dtype=nl.float32, buffer=nl.shared_hbm)
    probs = nl.ndarray((H, S, S), dtype=v.dtype, buffer=nl.shared_hbm)
    for h in range(H):
        # pass 1: S = Q K^T * scale, masked, written to HBM
        for i in range(nt):
            q = nl.ndarray((d, T), dtype=qT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q, src=qT[h, :, i * T:(i + 1) * T])
            for j in range(nt):
                k = nl.ndarray((d, T), dtype=kT.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=k, src=kT[h, :, j * T:(j + 1) * T])
                s_ps = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=s_ps, stationary=q, moving=k)
                s_sb = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=s_sb, data=s_ps, op0=nl.multiply, operand0=scale)
                s_m = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.sbuf)
                _causal_mask_tile(s_m, s_sb, i * T, j * T)
                nisa.dma_copy(dst=scores[h, i * T:(i + 1) * T, j * T:(j + 1) * T], src=s_m)
        # pass 2: row softmax over the full S-wide score rows
        for i in range(nt):
            row = nl.ndarray((T, S), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=row, src=scores[h, i * T:(i + 1) * T, :])
            nmax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=nmax, op=nl.max, data=row, axis=1, negate=True)
            e = nl.ndarray((T, S), dtype=nl.float32, buffer=nl.sbuf)
            ssum = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=e, op=nl.exp, data=row, bias=nmax, reduce_op=nl.add,
                            reduce_res=ssum, reduce_cmd=nisa.reduce_cmd.reset_reduce)
            rinv = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rinv, data=ssum)
            p = nl.ndarray((T, S), dtype=v.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=p, data=e, op0=nl.multiply, operand0=rinv)
            nisa.dma_copy(dst=probs[h, i * T:(i + 1) * T, :], src=p)
        # pass 3: O = P V
        for i in range(nt):
            acc = nl.ndarray((T, d), dtype=nl.float32, buffer=nl.psum)
            for j in range(nt):
                p = nl.ndarray((T, T), dtype=v.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=p, src=probs[h, i * T:(i + 1) * T, j * T:(j + 1) * T])
                pT_ps = nl.ndarray((T, T), dtype=v.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=pT_ps, data=p)
                pT = nl.ndarray((T, T), dtype=v.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pT, src=pT_ps)
                vt = nl.ndarray((T, d), dtype=v.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=vt, src=v[h, j * T:(j + 1) * T, :])
                nisa.nc_matmul(dst=acc, stationary=pT, moving=vt, accumulate=(j > 0))
            o = nl.ndarray((T, d), dtype=v.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o, src=acc)
            nisa.dma_copy(dst=out[h, i * T:(i + 1) * T, :], src=o)
    return out


def _flash_tile_update(s_ps, vt, scale, m_run, l_run, o_run, diag, q0, k0, d, vdtype):
    """One online-softmax step for a [T queries x T keys] block.

    m_run, l_run: [T, 1] running max / sum (fp32, SBUF); o_run: [T, d] fp32.
    """
    if diag:
        s_sb = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s_sb, src=s_ps)
        s_src = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.sbuf)
        _causal_mask_tile(s_src, s_sb, q0, k0)
    else:
        s_src = s_ps  # Vector/Scalar engines read PSUM directly
    # m_new = max(m_run, scale * rowmax(s))
    tmax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=tmax, op=nl.max, data=s_src, axis=1)
    m_new = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    tmax_s = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tmax_s, data=tmax, op0=nl.multiply, operand0=scale)
    nisa.tensor_tensor(dst=m_new, data1=m_run, data2=tmax_s, op=nl.maximum)
    neg_m = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_m, data=m_new, op0=nl.multiply, operand0=-1.0)
    # p = exp(scale * s - m_new), rowsum in the same Scalar Engine instruction
    p = nl.ndarray((T, T), dtype=vdtype, buffer=nl.sbuf)
    rsum = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=p, op=nl.exp, data=s_src, scale=scale, bias=neg_m, reduce_op=nl.add,
                    reduce_res=rsum, reduce_cmd=nisa.reduce_cmd.reset_reduce)
    # correction = exp(m_run - m_new); l = l * corr + rsum
    corr = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=corr, op=nl.exp, data=m_run, bias=neg_m)
    l_new = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=l_new, data1=l_run, data2=corr, op=nl.multiply)
    nisa.tensor_tensor(dst=l_run, data1=l_new, data2=rsum, op=nl.add)
    nisa.tensor_copy(dst=m_run, src=m_new)
    # o = o * corr + P V   (P^T via Tensor Engine transpose)
    pT_ps = nl.ndarray((T, T), dtype=vdtype, buffer=nl.psum)
    nisa.nc_transpose(dst=pT_ps, data=p)
    pT = nl.ndarray((T, T), dtype=vdtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pT, src=pT_ps)
    pv = nl.ndarray((T, d), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=pv, stationary=pT, moving=vt)
    o_sc = nl.ndarray((T, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_sc, data=o_run, op0=nl.multiply, operand0=corr)
    nisa.tensor_tensor(dst=o_run, data1=o_sc, data2=pv, op=nl.add)


def _flash_finish(o_run, l_run, dst_hbm, d, vdtype):
    linv = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=linv, data=l_run)
    o = nl.ndarray((T, d), dtype=vdtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o, data=o_run, op0=nl.multiply, operand0=linv)
    nisa.dma_copy(dst=dst_hbm, src=o)


def _flash_init(d):
    m_run = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    l_run = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    o_run = nl.ndarray((T, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=m_run, value=NEG_INF)
    nisa.memset(dst=l_run, value=0.0)
    nisa.memset(dst=o_run, value=0.0)
    return m_run, l_run, o_run


@nki.jit
def flash_attention(qT, kT, v, scale: float):
    H, d, S = qT.shape
    assert S % T == 0 and d <= 128
    nt = S // T
    out = nl.ndarray((H, S, d), dtype=v.dtype, buffer=nl.shared_hbm)
    for h in range(H):
        for i in range(nt):
            q = nl.ndarray((d, T), dtype=qT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q, src=qT[h, :, i * T:(i + 1) * T])
            m_run, l_run, o_run = _flash_init(d)
            for j in range(i + 1):  # causal: key tiles after the diagonal are skipped
                k = nl.ndarray((d, T), dtype=kT.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=k, src=kT[h, :, j * T:(j + 1) * T])
                vt = nl.ndarray((T, d), dtype=v.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=vt, src=v[h, j * T:(j + 1) * T, :])
                s_ps = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=s_ps, stationary=q, moving=k)
                _flash_tile_update(s_ps, vt, scale, m_run, l_run, o_run,
                                   j == i, i * T, j * T, d, v.dtype)
            _flash_finish(o_run, l_run, out[h, i * T:(i + 1) * T, :], d, v.dtype)
    return out


@nki.jit
def flash_attention_kv_resident(qT, kT, v, scale: float):
    H, d, S = qT.shape
    assert S % T == 0 and d <= 128
    nt = S // T
    out = nl.ndarray((H, S, d), dtype=v.dtype, buffer=nl.shared_hbm)
    for h in range(H):
        # One DMA each: K^T as [d, S]; V as [T, nt, d] (key tile j at [:, j, :]).
        k_all = nl.ndarray((d, S), dtype=kT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_all, src=kT[h, :, :])
        v_all = nl.ndarray((T, nt, d), dtype=v.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_all, src=v[h, :, :].reshape((nt, T, d)).permute((1, 0, 2)))
        for i in range(nt):
            q = nl.ndarray((d, T), dtype=qT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q, src=qT[h, :, i * T:(i + 1) * T])
            m_run, l_run, o_run = _flash_init(d)
            for j in range(i + 1):
                s_ps = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=s_ps, stationary=q, moving=k_all[:, j * T:(j + 1) * T])
                _flash_tile_update(s_ps, v_all[:, j, :], scale, m_run, l_run, o_run,
                                   j == i, i * T, j * T, d, v.dtype)
            _flash_finish(o_run, l_run, out[h, i * T:(i + 1) * T, :], d, v.dtype)
    return out


def _flash_tile_update_balanced(s_ps, vt, scale, nm_run, l_run, o_run, diag, q0, k0, d, vdtype):
    """Engine-balanced online-softmax step. Returns the new (nm, l, o) tiles.

    Changes vs ``_flash_tile_update`` (found by reading the profiler trace,
    where the Vector Engine was the critical path on [128, 1] ops that each
    pay a fixed ~100-cycle issue cost):
      * keep the running max negated (nm = -m), so the new value is a single
        tensor_scalar: nm_new = min(rowmax * -scale, nm_run)
      * l and o updates are one fused instruction each instead of two
      * running state is rotated by rebinding tiles, not copied
      * PSUM->SBUF copies run on the Scalar Engine to offload Vector
    """
    if diag:
        s_sb = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s_sb, src=s_ps, engine=nisa.engine.scalar)
        s_src = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.sbuf)
        _causal_mask_tile(s_src, s_sb, q0, k0)
    else:
        s_src = s_ps
    tmax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=tmax, op=nl.max, data=s_src, axis=1)
    nm_new = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nm_new, data=tmax, op0=nl.multiply, operand0=-scale,
                       op1=nl.minimum, operand1=nm_run)
    p = nl.ndarray((T, T), dtype=vdtype, buffer=nl.sbuf)
    rsum = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=p, op=nl.exp, data=s_src, scale=scale, bias=nm_new, reduce_op=nl.add,
                    reduce_res=rsum, reduce_cmd=nisa.reduce_cmd.reset_reduce)
    corr = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)  # exp(m_old - m_new)
    nisa.activation(dst=corr, op=nl.exp, data=nm_run, scale=-1.0, bias=nm_new)
    l_new = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=l_new, data=l_run, op0=nl.multiply, operand0=corr,
                       op1=nl.add, operand1=rsum)
    pT_ps = nl.ndarray((T, T), dtype=vdtype, buffer=nl.psum)
    nisa.nc_transpose(dst=pT_ps, data=p)
    pT = nl.ndarray((T, T), dtype=vdtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pT, src=pT_ps, engine=nisa.engine.scalar)
    pv = nl.ndarray((T, d), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=pv, stationary=pT, moving=vt)
    o_new = nl.ndarray((T, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=o_new, data=o_run, op0=nl.multiply, operand0=corr,
                              op1=nl.add, operand1=pv)
    return nm_new, l_new, o_new


@nki.jit
def flash_attention_balanced(qT, kT, v, scale: float):
    """KV-resident flash attention with fused, engine-balanced softmax updates."""
    H, d, S = qT.shape
    assert S % T == 0 and d <= 128
    nt = S // T
    out = nl.ndarray((H, S, d), dtype=v.dtype, buffer=nl.shared_hbm)
    for h in range(H):
        k_all = nl.ndarray((d, S), dtype=kT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_all, src=kT[h, :, :])
        v_all = nl.ndarray((T, nt, d), dtype=v.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_all, src=v[h, :, :].reshape((nt, T, d)).permute((1, 0, 2)))
        for i in range(nt):
            q = nl.ndarray((d, T), dtype=qT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q, src=qT[h, :, i * T:(i + 1) * T])
            nm_run = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
            l_run = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
            o_run = nl.ndarray((T, d), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=nm_run, value=-NEG_INF)
            nisa.memset(dst=l_run, value=0.0)
            nisa.memset(dst=o_run, value=0.0)
            for j in range(i + 1):
                s_ps = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=s_ps, stationary=q, moving=k_all[:, j * T:(j + 1) * T])
                nm_run, l_run, o_run = _flash_tile_update_balanced(
                    s_ps, v_all[:, j, :], scale, nm_run, l_run, o_run,
                    j == i, i * T, j * T, d, v.dtype)
            _flash_finish(o_run, l_run, out[h, i * T:(i + 1) * T, :], d, v.dtype)
    return out


@nki.jit
def flash_attention_pipelined(qT, kT, v, scale: float, ILP: int = 2):
    """Balanced flash attention with ILP query tiles interleaved in program order.

    Each query tile's online softmax is a serial chain (max -> exp -> rescale
    -> matmul). Engines execute their instruction streams in order, so a lone
    chain leaves most engines idle. Interleaving ILP independent query tiles,
    step by step, lets tile B's Tensor Engine work fill gaps while tile A
    waits on the Vector or Scalar Engine.
    """
    H, d, S = qT.shape
    assert S % T == 0 and d <= 128
    nt = S // T
    out = nl.ndarray((H, S, d), dtype=v.dtype, buffer=nl.shared_hbm)
    for h in range(H):
        k_all = nl.ndarray((d, S), dtype=kT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_all, src=kT[h, :, :])
        v_all = nl.ndarray((T, nt, d), dtype=v.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_all, src=v[h, :, :].reshape((nt, T, d)).permute((1, 0, 2)))
        for g in range(0, nt, ILP):
            tiles = list(range(g, min(g + ILP, nt)))
            state = {}
            for i in tiles:
                q = nl.ndarray((d, T), dtype=qT.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q, src=qT[h, :, i * T:(i + 1) * T])
                nm_run = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
                l_run = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
                o_run = nl.ndarray((T, d), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=nm_run, value=-NEG_INF)
                nisa.memset(dst=l_run, value=0.0)
                nisa.memset(dst=o_run, value=0.0)
                state[i] = [q, nm_run, l_run, o_run]
            for j in range(tiles[-1] + 1):
                for i in tiles:
                    if j > i:
                        continue  # causal skip
                    q, nm_run, l_run, o_run = state[i]
                    s_ps = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=s_ps, stationary=q, moving=k_all[:, j * T:(j + 1) * T])
                    state[i][1:] = _flash_tile_update_balanced(
                        s_ps, v_all[:, j, :], scale, nm_run, l_run, o_run,
                        j == i, i * T, j * T, d, v.dtype)
            for i in tiles:
                _, nm_run, l_run, o_run = state[i]
                _flash_finish(o_run, l_run, out[h, i * T:(i + 1) * T, :], d, v.dtype)
    return out
