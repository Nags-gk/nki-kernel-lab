import numpy as np
import pytest
import torch

from nkilab.model import BlockConfig, DecoderBlock, nki_forward


@pytest.mark.parametrize("S", [128, 300])
@pytest.mark.parametrize("variant", ["naive", "flash_pipelined"])
def test_decoder_block_matches_pytorch(S, variant):
    torch.manual_seed(0)
    block = DecoderBlock(BlockConfig()).eval()
    x = torch.randn(S, block.cfg.d_model)
    with torch.no_grad():
        ref = block(x).numpy()
    out = nki_forward(block, x.numpy(), attention_variant=variant)
    np.testing.assert_allclose(out, ref, atol=1e-5)
