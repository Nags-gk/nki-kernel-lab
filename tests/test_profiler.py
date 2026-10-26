import nki
import numpy as np
import pytest

from nkilab import ops
from nkilab.hwmodel import TRN2
from nkilab.kernels import matmul as mm
from nkilab.profiler import profile


def _run(fn, *args, **kw):
    with profile(fn.__name__ if hasattr(fn, "__name__") else "k") as p:
        nki.simulate(fn)(*args, **kw)
    return p


def test_naive_matmul_dma_bytes_are_exact():
    K, M, N = 256, 256, 1024
    p = _run(mm.matmul_naive, np.zeros((K, M), np.float32), np.zeros((K, N), np.float32))
    steps = (M // 128) * (N // 512) * (K // 128)
    expect_reads = steps * (128 * 128 + 128 * 512) * 4
    assert p.dma_bytes()["hbm_to_sbuf"] == expect_reads
    assert p.dma_bytes()["sbuf_to_hbm"] == M * N * 4
    assert p.flops() == 2 * M * N * K
    assert p.counts()["Tensor"] == steps


def test_blocking_cuts_operand_traffic():
    K, M, N = 512, 512, 2048
    a, b = np.zeros((K, M), np.float32), np.zeros((K, N), np.float32)
    naive = _run(mm.matmul_naive, a, b)
    blocked = _run(mm.matmul_blocked, a, b, BM=2, BN=4)
    assert blocked.flops() == naive.flops()
    assert blocked.dma_bytes()["hbm_to_sbuf"] * 2 < naive.dma_bytes()["hbm_to_sbuf"]
    assert blocked.summary()["peak_psum_bytes_per_partition"] == TRN2.psum_bytes_per_partition


@pytest.mark.parametrize("variant", list(ops.ATTENTION_VARIANTS))
def test_schedule_respects_lower_and_upper_bounds(variant):
    q = np.random.default_rng(0).standard_normal((1, 256, 64)).astype(np.float32)
    with profile(variant) as p:
        ops.attention(q, q, q, variant=variant)
    s = p.summary()
    assert s["roofline_bound_us"] <= s["modeled_latency_us"] + 1e-9
    assert s["modeled_latency_us"] <= s["serial_latency_us"] + 1e-9
    assert s["sbuf_capacity_ok"] and s["psum_capacity_ok"]
    for ins in p.instrs:
        assert ins.end_s >= ins.start_s >= 0


def test_flash_skips_masked_tiles():
    q = np.zeros((1, 512, 64), np.float32)
    with profile("naive") as pn:
        ops.attention(q, q, q, variant="naive")
    with profile("flash") as pf:
        ops.attention(q, q, q, variant="flash")
    nt = 4
    # QK^T matmuls: naive computes all nt*nt tiles, flash only the nt*(nt+1)/2 causal ones
    qk = lambda p: sum(1 for i in p.instrs if i.op == "nc_matmul" and i.elems == 128 and i.flops == 2 * 64 * 128 * 128)
    assert qk(pn) == nt * nt
    assert qk(pf) == nt * (nt + 1) // 2


def test_profiler_restores_isa_functions():
    import nki.isa as nisa
    before = nisa.dma_copy
    with profile():
        pass
    assert nisa.dma_copy is before


def test_profile_is_deterministic_across_runs():
    """Regression: id() reuse of freed tiles once merged buffers between runs."""
    q = np.zeros((1, 1024, 64), np.float32)
    lat = []
    for _ in range(3):
        with profile("flash") as p:
            ops.attention(q, q, q, variant="flash_pipelined")
        lat.append((p.modeled_latency_s(), len(p.storages)))
    assert lat[0] == lat[1] == lat[2]
