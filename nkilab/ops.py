"""Host-side wrappers: layout, padding and block-size choice around the kernels.

Every op takes and returns NumPy arrays (float32 or ml_dtypes.bfloat16) and runs
the NKI kernel on the CPU simulator. Any op called inside a
``nkilab.profiler.profile()`` block is recorded automatically.
"""
from __future__ import annotations

import math

import nki
import numpy as np

from .kernels import attention as _attn
from .kernels import matmul as _mm
from .kernels import norm as _norm


def _pad_to(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if x.shape == (rows, cols):
        return np.ascontiguousarray(x)
    out = np.zeros((rows, cols), dtype=x.dtype)
    out[: x.shape[0], : x.shape[1]] = x
    return out


def choose_blocks(m_tiles: int, n_tiles: int) -> tuple[int, int]:
    """Largest BM x BN register block that divides the grid and fits 8 PSUM banks."""
    best = (1, 1)
    for bm in (4, 2, 1):
        for bn in (8, 4, 2, 1):
            if bm * bn <= 8 and m_tiles % bm == 0 and n_tiles % bn == 0:
                if bm * bn > best[0] * best[1] or (bm * bn == best[0] * best[1] and bn > best[1]):
                    best = (bm, bn)
    return best


def matmul(a: np.ndarray, b: np.ndarray, variant: str = "blocked") -> np.ndarray:
    """C = A @ B for any shapes; pads to tile multiples and crops the result."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, (a.shape, b.shape)
    Mp, Kp, Np = (math.ceil(M / _mm.TM) * _mm.TM, math.ceil(K / _mm.TK) * _mm.TK,
                  math.ceil(N / _mm.TN) * _mm.TN)
    lhsT = _pad_to(a, Mp, Kp).T.copy()
    rhs = _pad_to(b, Kp, Np)
    if variant == "naive":
        c = nki.simulate(_mm.matmul_naive)(lhsT, rhs)
    else:
        bm, bn = choose_blocks(Mp // _mm.TM, Np // _mm.TN)
        c = nki.simulate(_mm.matmul_blocked)(lhsT, rhs, BM=bm, BN=bn)
    return c[:M, :N]


def softmax(x: np.ndarray, fused: bool = True) -> np.ndarray:
    fn = _norm.softmax_fused if fused else _norm.softmax_unfused
    return nki.simulate(fn)(np.ascontiguousarray(x))


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return nki.simulate(_norm.rmsnorm_fused)(np.ascontiguousarray(x), np.ascontiguousarray(weight), eps)


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return nki.simulate(_norm.swiglu_fused)(np.ascontiguousarray(gate), np.ascontiguousarray(up))


ATTENTION_VARIANTS = {
    "naive": _attn.attention_naive,
    "flash": _attn.flash_attention,
    "flash_kv_resident": _attn.flash_attention_kv_resident,
    "flash_balanced": _attn.flash_attention_balanced,
    "flash_pipelined": _attn.flash_attention_pipelined,
}


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, variant: str = "flash_pipelined") -> np.ndarray:
    """Causal attention. q, k, v: [H, S, d]. S is padded to a multiple of 128.

    Padding sits at the end of the sequence, and the mask is causal, so real
    query rows never attend to padded keys; padded rows are cropped.
    """
    H, S, d = q.shape
    Sp = math.ceil(S / _attn.T) * _attn.T
    if Sp != S:
        pad = ((0, 0), (0, Sp - S), (0, 0))
        q, k, v = (np.pad(t, pad) for t in (q, k, v))
    qT = np.ascontiguousarray(q.transpose(0, 2, 1))
    kT = np.ascontiguousarray(k.transpose(0, 2, 1))
    o = nki.simulate(ATTENTION_VARIANTS[variant])(qT, kT, np.ascontiguousarray(v), 1.0 / math.sqrt(d))
    return o[:, :S, :]
