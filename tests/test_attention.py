import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch

from conftest import rel_err
from nkilab import ops

VARIANTS = list(ops.ATTENTION_VARIANTS)


def _torch_ref(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(
        *map(torch.from_numpy, (q, k, v)), is_causal=True).numpy()


def _jax_ref(q, k, v):
    # jax layout: [batch, seq, heads, head_dim]
    f = lambda t: jnp.asarray(t.transpose(1, 0, 2))[None]
    o = jax.nn.dot_product_attention(f(q), f(k), f(v), is_causal=True)
    return np.asarray(o[0]).transpose(1, 0, 2)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("H,S,d", [(1, 128, 64), (2, 512, 64), (2, 200, 32), (1, 384, 128)])
def test_attention_matches_torch_and_jax(rng, variant, H, S, d):
    q, k, v = (rng.standard_normal((H, S, d)).astype(np.float32) for _ in range(3))
    out = ops.attention(q, k, v, variant=variant)
    assert out.shape == (H, S, d)
    np.testing.assert_allclose(out, _torch_ref(q, k, v), atol=2e-5)
    np.testing.assert_allclose(out, _jax_ref(q, k, v), atol=2e-5)


@pytest.mark.parametrize("variant", ["naive", "flash_pipelined"])
def test_attention_bf16(rng, variant):
    q, k, v = (rng.standard_normal((2, 256, 64)).astype(ml_dtypes.bfloat16) for _ in range(3))
    ref = _torch_ref(*(t.astype(np.float32) for t in (q, k, v)))
    assert rel_err(ops.attention(q, k, v, variant=variant).astype(np.float32), ref) < 3e-2


def test_first_token_attends_only_to_itself(rng):
    q, k, v = (rng.standard_normal((1, 256, 64)).astype(np.float32) for _ in range(3))
    out = ops.attention(q, k, v)
    np.testing.assert_allclose(out[0, 0], v[0, 0], atol=1e-6)


def test_variants_agree_bitwise_on_flash_family(rng):
    q, k, v = (rng.standard_normal((2, 512, 64)).astype(np.float32) for _ in range(3))
    outs = [ops.attention(q, k, v, variant=n) for n in ("flash_balanced", "flash_pipelined")]
    np.testing.assert_array_equal(outs[0], outs[1])  # interleaving must not change numerics
