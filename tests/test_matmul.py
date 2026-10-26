import ml_dtypes
import nki
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import rel_err
from nkilab import ops
from nkilab.kernels import matmul as mm

BF16 = ml_dtypes.bfloat16


@pytest.mark.parametrize("variant", ["naive", "blocked"])
@pytest.mark.parametrize("M,K,N", [(128, 128, 512), (256, 384, 1024), (200, 300, 130), (1, 7, 3)])
def test_matmul_fp32(rng, variant, M, K, N):
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    assert rel_err(ops.matmul(a, b, variant=variant), a @ b) < 1e-5


@pytest.mark.parametrize("variant", ["naive", "blocked"])
def test_matmul_bf16_accumulates_in_fp32(rng, variant):
    a = rng.standard_normal((256, 1024)).astype(BF16)
    b = rng.standard_normal((1024, 1024)).astype(BF16)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    out = ops.matmul(a, b, variant=variant)
    assert out.dtype == BF16
    assert rel_err(out, ref) < 1e-2  # only the final bf16 rounding of C


def test_blocked_rejects_psum_overflow(rng):
    lhsT = np.zeros((128, 384), np.float32)
    rhs = np.zeros((128, 1536), np.float32)
    with pytest.raises(Exception, match="PSUM"):
        nki.simulate(mm.matmul_blocked)(lhsT, rhs, BM=3, BN=3)


def test_choose_blocks_respects_psum_banks():
    for mt in range(1, 9):
        for nt in range(1, 17):
            bm, bn = ops.choose_blocks(mt, nt)
            assert bm * bn <= 8 and mt % bm == 0 and nt % bn == 0
    assert ops.choose_blocks(4, 4) in {(2, 4), (4, 2)}


@settings(max_examples=12, deadline=None)
@given(M=st.integers(1, 300), K=st.integers(1, 300), N=st.integers(1, 700))
def test_matmul_property_any_shape(M, K, N):
    rng = np.random.default_rng(M * 1000003 + K * 1009 + N)
    a = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    b = rng.uniform(-1, 1, (K, N)).astype(np.float32)
    out = ops.matmul(a, b)
    assert out.shape == (M, N)
    np.testing.assert_allclose(out, a @ b, rtol=1e-4, atol=1e-4)
