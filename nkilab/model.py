"""A Llama-style decoder block run two ways: PyTorch eager and NKI kernels.

    x = x + Wo · causal_attention(split_heads(RMSNorm(x) · Wqkv))
    x = x + Wd · (silu(RMSNorm(x) · Wg) * (RMSNorm(x) · Wu))

On the NKI path every matmul, RMSNorm, attention and SwiGLU runs as an NKI
kernel on the Trainium2 CPU simulator. Residual adds, head split/merge and
weight layout stay on the host. Rotary embeddings are left out on both paths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ops


@dataclass
class BlockConfig:
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class DecoderBlock(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.cfg = cfg
        d, f = cfg.d_model, cfg.d_ff
        self.norm1, self.norm2 = RMSNorm(d, cfg.eps), RMSNorm(d, cfg.eps)
        self.wqkv = nn.Linear(d, 3 * d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.wg = nn.Linear(d, f, bias=False)
        self.wu = nn.Linear(d, f, bias=False)
        self.wd = nn.Linear(f, d, bias=False)
        with torch.no_grad():  # non-trivial norm weights so the test exercises them
            self.norm1.weight.uniform_(0.5, 1.5)
            self.norm2.weight.uniform_(0.5, 1.5)

    def forward(self, x):  # x: [S, d]
        S, d = x.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.wqkv(self.norm1(x)).split(d, dim=-1)
        q, k, v = (t.view(S, H, hd).transpose(0, 1) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(a.transpose(0, 1).reshape(S, d))
        h = self.norm2(x)
        return x + self.wd(F.silu(self.wg(h)) * self.wu(h))


def nki_forward(block: DecoderBlock, x: np.ndarray, attention_variant: str = "flash_pipelined") -> np.ndarray:
    """The same block, every heavy op on NKI kernels (CPU simulator)."""
    cfg = block.cfg
    S, d = x.shape
    H, hd = cfg.n_heads, cfg.head_dim

    def w(lin: nn.Linear) -> np.ndarray:  # nn.Linear stores [out, in]; we need [in, out]
        return lin.weight.detach().numpy().T.astype(x.dtype).copy()

    def g(norm: RMSNorm) -> np.ndarray:
        return norm.weight.detach().numpy().astype(x.dtype)

    h = ops.rmsnorm(x, g(block.norm1), cfg.eps)
    qkv = ops.matmul(h, w(block.wqkv))
    q, k, v = (qkv[:, i * d:(i + 1) * d].reshape(S, H, hd).transpose(1, 0, 2) for i in range(3))
    a = ops.attention(q, k, v, variant=attention_variant)
    x = x + ops.matmul(a.transpose(1, 0, 2).reshape(S, d), w(block.wo))
    h = ops.rmsnorm(x, g(block.norm2), cfg.eps)
    act = ops.swiglu(ops.matmul(h, w(block.wg)), ops.matmul(h, w(block.wu)))
    return x + ops.matmul(act, w(block.wd))
