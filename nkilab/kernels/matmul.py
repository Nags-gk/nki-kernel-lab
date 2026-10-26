"""Matrix multiplication on the Tensor Engine: C[M, N] = A[M, K] @ B[K, N].

Layout: the Tensor Engine computes ``dst = stationary.T @ moving`` with the
contraction dimension K on the 128 SBUF partitions, so kernels take the left
operand already transposed (``lhsT`` of shape [K, M]). ``nkilab.ops.matmul``
does the transpose and padding on the host.

Tile limits (Trainium arch guide):
  * stationary: K <= 128 partitions, M <= 128 free
  * moving:     K <= 128 partitions, N <= 512 free
  * one matmul writes one PSUM bank: [128, 512] fp32; 8 banks per partition
"""
from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

TK = 128  # contraction tile (partitions)
TM = 128  # stationary free size -> output partitions
TN = 512  # moving free size -> one PSUM bank of fp32


@nki.jit
def matmul_naive(lhsT, rhs):
    """Baseline: reload both operand tiles from HBM for every (m, n, k) step."""
    K, M = lhsT.shape
    _, N = rhs.shape
    assert K % TK == 0 and M % TM == 0 and N % TN == 0
    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for m in range(M // TM):
        for n in range(N // TN):
            acc = nl.ndarray((TM, TN), dtype=nl.float32, buffer=nl.psum)
            for k in range(K // TK):
                a = nl.ndarray((TK, TM), dtype=lhsT.dtype, buffer=nl.sbuf)
                b = nl.ndarray((TK, TN), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=a, src=lhsT[k * TK:(k + 1) * TK, m * TM:(m + 1) * TM])
                nisa.dma_copy(dst=b, src=rhs[k * TK:(k + 1) * TK, n * TN:(n + 1) * TN])
                nisa.nc_matmul(dst=acc, stationary=a, moving=b, accumulate=(k > 0))
            res = nl.ndarray((TM, TN), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res, src=acc)
            nisa.dma_copy(dst=out[m * TM:(m + 1) * TM, n * TN:(n + 1) * TN], src=res)
    return out


@nki.jit
def matmul_blocked(lhsT, rhs, BM: int = 2, BN: int = 4):
    """Register-blocked matmul: one output block of BM x BN tiles stays in PSUM.

    For each K step, one DMA brings a [128, BM*128] stripe of lhsT and one
    brings a [128, BN*512] stripe of rhs; every loaded tile is reused across
    the block, so HBM reads drop by roughly BN x for lhsT and BM x for rhs.
    BM * BN accumulators must fit the 8 PSUM banks.
    """
    K, M = lhsT.shape
    _, N = rhs.shape
    assert K % TK == 0 and M % (TM * BM) == 0 and N % (TN * BN) == 0
    assert BM * BN <= 8, "BM*BN fp32 [128,512] accumulators must fit 8 PSUM banks"
    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for mb in range(M // (TM * BM)):
        for nb in range(N // (TN * BN)):
            accs = [[nl.ndarray((TM, TN), dtype=nl.float32, buffer=nl.psum) for _ in range(BN)]
                    for _ in range(BM)]
            for k in range(K // TK):
                a = nl.ndarray((TK, BM * TM), dtype=lhsT.dtype, buffer=nl.sbuf)
                b = nl.ndarray((TK, BN * TN), dtype=rhs.dtype, buffer=nl.sbuf)
                m0, n0 = mb * BM * TM, nb * BN * TN
                nisa.dma_copy(dst=a, src=lhsT[k * TK:(k + 1) * TK, m0:m0 + BM * TM])
                nisa.dma_copy(dst=b, src=rhs[k * TK:(k + 1) * TK, n0:n0 + BN * TN])
                for i in range(BM):
                    for j in range(BN):
                        nisa.nc_matmul(dst=accs[i][j],
                                       stationary=a[:, i * TM:(i + 1) * TM],
                                       moving=b[:, j * TN:(j + 1) * TN],
                                       accumulate=(k > 0))
            for i in range(BM):
                res = nl.ndarray((TM, BN * TN), dtype=lhsT.dtype, buffer=nl.sbuf)
                for j in range(BN):
                    nisa.tensor_copy(dst=res[:, j * TN:(j + 1) * TN], src=accs[i][j])
                r0 = (mb * BM + i) * TM
                nisa.dma_copy(dst=out[r0:r0 + TM, nb * BN * TN:(nb + 1) * BN * TN], src=res)
    return out
