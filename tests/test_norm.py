import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import rel_err
from nkilab import ops


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("R,D", [(128, 64), (1000, 2048), (5, 3)])
def test_softmax_matches_torch_and_jax(rng, fused, R, D):
    x = (rng.standard_normal((R, D)) * 4).astype(np.float32)
    out = ops.softmax(x, fused=fused)
    np.testing.assert_allclose(out, torch.softmax(torch.from_numpy(x), -1).numpy(), atol=1e-6)
    np.testing.assert_allclose(out, np.asarray(jax.nn.softmax(jnp.asarray(x), axis=-1)), atol=1e-6)


def test_softmax_is_stable_for_large_logits(rng):
    x = (rng.standard_normal((256, 512)) * 1e4).astype(np.float32)
    out = ops.softmax(x)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(-1), 1.0, atol=1e-5)


@settings(max_examples=10, deadline=None)
@given(R=st.integers(1, 400), D=st.integers(1, 1024), s=st.floats(0.1, 50))
def test_softmax_property_rows_sum_to_one(R, D, s):
    x = (np.random.default_rng(R * 7 + D).standard_normal((R, D)) * s).astype(np.float32)
    out = ops.softmax(x)
    np.testing.assert_allclose(out.sum(-1), 1.0, atol=1e-5)
    assert (out >= 0).all()


@pytest.mark.parametrize("R,D", [(128, 256), (300, 1024), (7, 33)])
def test_rmsnorm_matches_torch(rng, R, D):
    x = rng.standard_normal((R, D)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, D).astype(np.float32)
    xt = torch.from_numpy(x)
    ref = (xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + 1e-6) * torch.from_numpy(w)).numpy()
    assert rel_err(ops.rmsnorm(x, w), ref) < 1e-5


def test_rmsnorm_bf16(rng):
    x = rng.standard_normal((256, 512)).astype(ml_dtypes.bfloat16)
    w = np.ones(512, ml_dtypes.bfloat16)
    xf = x.astype(np.float32)
    ref = xf / np.sqrt((xf ** 2).mean(-1, keepdims=True) + 1e-6)
    assert rel_err(ops.rmsnorm(x, w).astype(np.float32), ref) < 1e-2


def test_swiglu_matches_torch(rng):
    g = rng.standard_normal((300, 1024)).astype(np.float32)
    u = rng.standard_normal((300, 1024)).astype(np.float32)
    ref = (torch.nn.functional.silu(torch.from_numpy(g)) * torch.from_numpy(u)).numpy()
    assert rel_err(ops.swiglu(g, u), ref) < 1e-5
