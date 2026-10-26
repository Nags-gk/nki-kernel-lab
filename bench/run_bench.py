"""Run every kernel under the profiler and write site/data/results.json.

Counts (instructions, DMA bytes, FLOPs, on-chip footprint) are exact, taken
from the instructions the kernels issue on the NKI CPU simulator. Latencies
are produced by nkilab.hwmodel's analytic Trainium2 model, not by hardware.

    python bench/run_bench.py            # full run (~1-2 min)
    python bench/run_bench.py --quick    # smaller shapes for CI
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nkilab  # noqa: E402  (sets the trn2 target)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ml_dtypes  # noqa: E402
import nki  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from nkilab import ops  # noqa: E402
from nkilab.hwmodel import TRN2  # noqa: E402
from nkilab.kernels import attention as attn  # noqa: E402
from nkilab.kernels import matmul as mm  # noqa: E402
from nkilab.kernels import norm  # noqa: E402
from nkilab.model import BlockConfig, DecoderBlock, nki_forward  # noqa: E402
from nkilab.profiler import profile  # noqa: E402

BF16 = ml_dtypes.bfloat16
RNG = np.random.default_rng(7)


def run(name, fn, args, kwargs=None, dtype="bfloat16", compulsory=None, trace=False, extra=None):
    kwargs = kwargs or {}
    t0 = time.perf_counter()
    with profile(name) as p:
        out = nki.simulate(fn)(*args, **kwargs)
    wall = time.perf_counter() - t0
    s = p.summary(dtype)
    if compulsory is not None:
        s["compulsory_hbm_bytes"] = int(compulsory)
        s["ideal_bound_us"] = max(s["flops"] / TRN2.peak_flops[dtype], compulsory / TRN2.hbm_bytes_per_s) * 1e6
    s["simulator_wall_s"] = round(wall, 3)
    s["dtype"] = dtype
    if extra:
        s.update(extra)
    if trace:
        s["trace"] = p.trace()
    print(f"  {name:42s} {s['modeled_latency_us']:9.1f} us  {s['hbm_bytes'] / 1e6:8.2f} MB  {s['instructions']:6d} instr")
    return s, out


def bench_matmul(quick):
    M = K = N = 1024 if not quick else 512
    a = RNG.standard_normal((M, K)).astype(BF16)
    b = RNG.standard_normal((K, N)).astype(BF16)
    lhsT = np.ascontiguousarray(a.T)
    comp = (M * K + K * N + M * N) * 2
    ref = a.astype(np.float32) @ b.astype(np.float32)
    rows = []
    s, o = run("matmul naive", mm.matmul_naive, (lhsT, b), compulsory=comp)
    s["rel_err_vs_numpy"] = float(np.abs(o.astype(np.float32) - ref).max() / np.abs(ref).max())
    rows.append(s)
    bm, bn = ops.choose_blocks(M // 128, N // 512)
    s, o = run(f"matmul blocked {bm}x{bn}", mm.matmul_blocked, (lhsT, b), {"BM": bm, "BN": bn},
               compulsory=comp)
    s["rel_err_vs_numpy"] = float(np.abs(o.astype(np.float32) - ref).max() / np.abs(ref).max())
    rows.append(s)
    sweep = []
    for n in ([512, 1024, 2048] if not quick else [512, 1024]):
        a = np.zeros((n, n), BF16)
        for nm, fn, kw in (("naive", mm.matmul_naive, {}),
                           ("blocked", mm.matmul_blocked, dict(zip(("BM", "BN"), ops.choose_blocks(n // 128, n // 512))))):
            s, _ = run(f"matmul {nm} {n}^3", fn, (a, a), kw, compulsory=3 * n * n * 2)
            sweep.append({"n": n, "variant": nm, **{k: s[k] for k in (
                "modeled_latency_us", "hbm_bytes", "flops", "arithmetic_intensity", "modeled_tflops", "ideal_bound_us")}})
    return {"shape": [M, K, N], "variants": rows, "sweep": sweep}


def bench_norm(quick):
    R, D = (4096, 2048) if not quick else (1024, 1024)
    x = (RNG.standard_normal((R, D)) * 3).astype(BF16)
    ref = torch.softmax(torch.from_numpy(x.astype(np.float32)), -1).numpy()
    rows = []
    for nm, fn in (("softmax unfused (3 HBM passes)", norm.softmax_unfused), ("softmax fused", norm.softmax_fused)):
        s, o = run(nm, fn, (x,), compulsory=2 * R * D * 2)
        s["max_abs_err_vs_torch"] = float(np.abs(o.astype(np.float32) - ref).max())
        rows.append(s)
    w = RNG.uniform(0.5, 1.5, D).astype(BF16)
    s, _ = run("rmsnorm fused", norm.rmsnorm_fused, (x, w), compulsory=2 * R * D * 2 + D * 2)
    rows.append(s)
    s, _ = run("swiglu fused", norm.swiglu_fused, (x, x), compulsory=3 * R * D * 2)
    rows.append(s)
    return {"shape": [R, D], "variants": rows}


def bench_attention(quick):
    H, S, d = (8, 2048, 128) if not quick else (2, 512, 64)
    q, k, v = (RNG.standard_normal((H, S, d)).astype(BF16) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(torch.from_numpy(t.astype(np.float32)) for t in (q, k, v)), is_causal=True).numpy()
    qT, kT = (np.ascontiguousarray(t.transpose(0, 2, 1)) for t in (q, k))
    sc = 1.0 / np.sqrt(d)
    causal_flops = 4 * H * d * (S // 128) * (S // 128 + 1) // 2 * 128 * 128
    comp = 4 * H * S * d * 2
    ladder = []
    steps = [
        ("naive (scores in HBM)", attn.attention_naive, {}),
        ("flash (online softmax, causal skip)", attn.flash_attention, {}),
        ("+ K/V resident in SBUF", attn.flash_attention_kv_resident, {}),
        ("+ fused, engine-balanced updates", attn.flash_attention_balanced, {}),
        ("+ 2 query tiles interleaved (ILP)", attn.flash_attention_pipelined, {"ILP": 2}),
    ]
    for nm, fn, kw in steps:
        s, o = run(nm, fn, (qT, kT, v, sc), kw, compulsory=comp,
                   extra={"useful_causal_flops": causal_flops})
        s["rel_err_vs_torch"] = float(np.abs(o.astype(np.float32) - ref).max() / np.abs(ref).max())
        ladder.append(s)
    scaling = []
    for s_len in ([512, 1024, 2048, 4096] if not quick else [256, 512]):
        qz = np.zeros((1, d, s_len), BF16)
        vz = np.zeros((1, s_len, d), BF16)
        for nm, fn in (("naive", attn.attention_naive), ("flash", attn.flash_attention_pipelined)):
            s, _ = run(f"attention {nm} S={s_len}", fn, (qz, qz, vz, sc))
            scaling.append({"S": s_len, "variant": nm, "hbm_bytes": s["hbm_bytes"],
                            "modeled_latency_us": s["modeled_latency_us"]})
    return {"shape": [H, S, d], "ladder": ladder, "scaling": scaling}


def bench_block(quick):
    torch.manual_seed(0)
    cfg = BlockConfig()
    block = DecoderBlock(cfg).eval()
    S = 512 if not quick else 256
    x = torch.randn(S, cfg.d_model)
    with torch.no_grad():
        ref = block(x).numpy()
    parts = []
    xs = x.numpy()
    with profile("decoder block") as p:
        out = nki_forward(block, xs)
    err = float(np.abs(out - ref).max())
    # per-op breakdown: rerun each op in its own profile
    h = ops.rmsnorm(xs, block.norm1.weight.detach().numpy())
    w = lambda lin: lin.weight.detach().numpy().T.copy()
    qkv = ops.matmul(h, w(block.wqkv))
    d = cfg.d_model
    q, k, v = (qkv[:, i * d:(i + 1) * d].reshape(S, cfg.n_heads, cfg.head_dim).transpose(1, 0, 2) for i in range(3))
    for nm, call in (
        ("rmsnorm", lambda: ops.rmsnorm(xs, block.norm1.weight.detach().numpy())),
        ("qkv matmul", lambda: ops.matmul(h, w(block.wqkv))),
        ("flash attention", lambda: ops.attention(q, k, v)),
        ("out-proj matmul", lambda: ops.matmul(xs, w(block.wo))),
        ("rmsnorm", lambda: ops.rmsnorm(xs, block.norm2.weight.detach().numpy())),
        ("gate + up matmuls", lambda: (ops.matmul(xs, w(block.wg)), ops.matmul(xs, w(block.wu)))),
        ("swiglu", lambda: ops.swiglu(np.zeros((S, cfg.d_ff), np.float32), np.zeros((S, cfg.d_ff), np.float32))),
        ("down-proj matmul", lambda: ops.matmul(np.zeros((S, cfg.d_ff), np.float32), w(block.wd))),
    ):
        with profile(nm) as pp:
            call()
        ss = pp.summary("float32")
        parts.append({"op": nm, "modeled_latency_us": ss["modeled_latency_us"], "hbm_bytes": ss["hbm_bytes"],
                      "instructions": ss["instructions"], "flops": ss["flops"]})
    return {"config": dataclasses.asdict(cfg), "seq_len": S, "dtype": "float32",
            "max_abs_err_vs_pytorch": err, "max_abs_ref": float(np.abs(ref).max()),
            "total_instructions": len(p.instrs), "parts": parts}


def small_traces():
    """Complete (untruncated) instruction traces at a small shape for the trace viewer."""
    out = {}
    H, S, d = 1, 512, 128
    q = RNG.standard_normal((H, S, d)).astype(BF16)
    qT = np.ascontiguousarray(q.transpose(0, 2, 1))
    for nm, fn, kw in (("attention naive", attn.attention_naive, {}),
                       ("attention flash", attn.flash_attention, {}),
                       ("attention flash, K/V resident", attn.flash_attention_kv_resident, {}),
                       ("attention flash, balanced", attn.flash_attention_balanced, {}),
                       ("attention flash, 2-tile ILP", attn.flash_attention_pipelined, {"ILP": 2})):
        with profile(nm) as p:
            nki.simulate(fn)(qT, qT, q, 1.0 / np.sqrt(d), **kw)
        out[nm] = {"shape": f"H={H} S={S} d={d} bf16", "latency_us": p.modeled_latency_s() * 1e6,
                   "events": p.trace(limit=10**6)}
    a = np.zeros((512, 512), BF16)
    for nm, fn, kw in (("matmul naive", mm.matmul_naive, {}), ("matmul blocked", mm.matmul_blocked, {"BM": 4, "BN": 1})):
        with profile(nm) as p:
            nki.simulate(fn)(a, a, **kw)
        out[nm] = {"shape": "512x512x512 bf16", "latency_us": p.modeled_latency_s() * 1e6, "events": p.trace(limit=10**6)}
    x = np.zeros((512, 2048), BF16)
    for nm, fn in (("softmax unfused", norm.softmax_unfused), ("softmax fused", norm.softmax_fused)):
        with profile(nm) as p:
            nki.simulate(fn)(x)
        out[nm] = {"shape": "512x2048 bf16", "latency_us": p.modeled_latency_s() * 1e6, "events": p.trace(limit=10**6)}
    for k, v in out.items():
        print(f"  trace {k:32s} {len(v['events']):5d} events")
    return out


def correctness_matrix():
    rows = []
    x = (RNG.standard_normal((300, 777)) * 4).astype(np.float32)
    ref_t = torch.softmax(torch.from_numpy(x), -1).numpy()
    ref_j = np.asarray(jax.nn.softmax(jnp.asarray(x), -1))
    o = ops.softmax(x)
    rows.append({"op": "softmax fused", "shape": "300x777", "vs_torch": float(np.abs(o - ref_t).max()),
                 "vs_jax": float(np.abs(o - ref_j).max())})
    q, k, v = (RNG.standard_normal((2, 200, 64)).astype(np.float32) for _ in range(3))
    rt = torch.nn.functional.scaled_dot_product_attention(*map(torch.from_numpy, (q, k, v)), is_causal=True).numpy()
    f = lambda t: jnp.asarray(t.transpose(1, 0, 2))[None]
    rj = np.asarray(jax.nn.dot_product_attention(f(q), f(k), f(v), is_causal=True)[0]).transpose(1, 0, 2)
    for nm in ops.ATTENTION_VARIANTS:
        o = ops.attention(q, k, v, variant=nm)
        rows.append({"op": f"attention {nm}", "shape": "2x200x64", "vs_torch": float(np.abs(o - rt).max()),
                     "vs_jax": float(np.abs(o - rj).max())})
    a, b = RNG.standard_normal((200, 300)).astype(np.float32), RNG.standard_normal((300, 130)).astype(np.float32)
    o = ops.matmul(a, b)
    rows.append({"op": "matmul blocked", "shape": "200x300 @ 300x130",
                 "vs_torch": float(np.abs(o - (torch.from_numpy(a) @ torch.from_numpy(b)).numpy()).max()),
                 "vs_jax": float(np.abs(o - np.asarray(jnp.asarray(a) @ jnp.asarray(b))).max())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "site", "data", "results.json"))
    a = ap.parse_args()
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = ""
    print("matmul"); mm_r = bench_matmul(a.quick)
    print("norm"); nr = bench_norm(a.quick)
    print("attention"); at = bench_attention(a.quick)
    print("decoder block"); bl = bench_block(a.quick)
    print("traces"); tr = small_traces()
    print("correctness"); cm = correctness_matrix()
    spec = dataclasses.asdict(TRN2)
    spec["peak_flops"] = TRN2.peak_flops
    result = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha, "quick": a.quick,
        "versions": {"python": platform.python_version(), "nki": nki.__version__ if hasattr(nki, "__version__") else "",
                     "torch": torch.__version__, "jax": jax.__version__, "numpy": np.__version__},
        "hardware_model": spec,
        "matmul": mm_r, "norm": nr, "attention": at, "decoder_block": bl, "correctness": cm, "traces": tr,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1, default=float)
    print("wrote", os.path.abspath(a.out), f"{os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
