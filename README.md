# NKI Kernel Lab

[![ci](https://github.com/Nags-gk/nki-kernel-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Nags-gk/nki-kernel-lab/actions/workflows/ci.yml)

**Live dashboard: https://nags-gk.github.io/nki-kernel-lab/**

Trainium2 kernels written in [NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html) (Neuron Kernel Interface): matmul, softmax, RMSNorm, SwiGLU and causal flash attention. They run on the Neuron SDK's CPU simulator, are checked against PyTorch and JAX, and are profiled instruction by instruction with a profiler and Trainium2 cost model built for this repo. A Llama-style decoder block runs end to end on the kernels and matches PyTorch to 6e-7.

No Trainium hardware was used. **Correctness and counts are measured; latencies are modeled.** See [How the numbers are made](#how-the-numbers-are-made).

## Results

### Causal attention, 8 heads × 2048 tokens × d=128, bf16

Each step changes one thing; all five produce the same output (max relative error vs. PyTorch SDPA 2.6e-3, i.e. bf16 rounding).

| # | Kernel | Modeled latency | HBM traffic | Vector instrs |
|---|---|---:|---:|---:|
| 1 | Naive: S×S scores and probabilities written to HBM | 4,425 µs | 545.3 MB | 4,608 |
| 2 | Flash: online softmax in SBUF, skip fully-masked key tiles | 1,818 µs | 79.7 MB | 11,648 |
| 3 | + K/V kept resident in SBUF per head | 1,829 µs | 16.8 MB | 11,648 |
| 4 | + fused, engine-balanced softmax updates | 1,201 µs | 16.8 MB | 4,992 |
| 5 | + two query tiles interleaved (ILP) | 1,147 µs | 16.8 MB | 4,992 |

**3.9x lower modeled latency, 32x less HBM traffic.** Step 3 is the instructive one: traffic drops 4.7x but latency doesn't move, because the trace shows the kernel is now Vector Engine bound (82% busy) on dozens of tiny `[128, 1]` ops that each pay a ~100-cycle fixed cost. Step 4 fuses those (`tensor_scalar` with two ops, `scalar_tensor_tensor`, a negated running max so the max update is one instruction) and moves PSUM→SBUF copies to the Scalar Engine: Vector instructions drop 57%.

Naive score traffic grows with S², flash with S: at S=4096 (one head) it's 270.5 MB vs. 4.2 MB.

### Matmul and row-wise kernels

| Kernel | Shape | HBM traffic | Modeled | Ideal bound |
|---|---|---:|---:|---:|
| Matmul, naive (reload every tile) | 1024³ bf16 | 23.1 MB | 79.4 µs | 27.3 µs |
| Matmul, 4×2 PSUM-blocked | 1024³ bf16 | 8.4 MB | 37.8 µs (56.8 TFLOPS) | 27.3 µs |
| Softmax, unfused (3 HBM passes) | 4096×2048 bf16 | 117.5 MB | 523.8 µs | 89.5 µs |
| Softmax, fused | 4096×2048 bf16 | 33.6 MB | 223.5 µs | 89.5 µs |
| RMSNorm, fused | 4096×2048 bf16 | 34.1 MB | 226.6 µs | 89.5 µs |
| SwiGLU, fused | 4096×2048 bf16 | 50.3 MB | 186.0 µs | 134.2 µs |

The blocked matmul keeps a BM×BN grid of fp32 accumulators in PSUM. One `nc_matmul` writes one PSUM bank (`[128, 512]` fp32) and a partition has 8 banks, so `BM·BN ≤ 8`; the kernel asserts it and the host picks the largest block that divides the problem.

## What's in here

```
nkilab/
  kernels/matmul.py     naive and PSUM-blocked Tensor Engine matmul
  kernels/norm.py       softmax (unfused / fused), RMSNorm, SwiGLU
  kernels/attention.py  naive, flash, K/V-resident, engine-balanced, interleaved
  profiler.py           intercepts nki.isa instructions; counts, liveness, schedule, trace
  hwmodel.py            Trainium2 NeuronCore-v3 cost model, every number cited or marked ASSUMPTION
  ops.py                host wrappers: transpose, padding, block-size choice
  model.py              Llama-style decoder block in PyTorch and on NKI kernels
bench/run_bench.py      runs everything, writes site/data/results.json
site/index.html         dashboard (GitHub Pages)
tests/                  64 tests: PyTorch + JAX references, Hypothesis shapes, profiler invariants
docs/STUDY_GUIDE.md     how each kernel works and the questions it answers
```

## Run it

Python 3.13 (the `nki` wheel on the Neuron repo is cp313).

```bash
pip install -r requirements.txt
pytest -q                        # 64 tests, ~15 s
python bench/run_bench.py        # ~1 min, writes site/data/results.json
python -m http.server -d site    # open http://localhost:8000
```

## How the numbers are made

**Measured** (exact, from the instructions the kernels issue on the NKI CPU simulator): numerical output; instructions per engine; HBM↔SBUF DMA bytes; Tensor Engine FLOPs; per-partition SBUF and PSUM liveness.

**Modeled** (every latency): the simulator reports no timing, so `profiler.py` gives each instruction a cost from `hwmodel.py` and list-schedules it onto in-order engine queues (Tensor, Vector, Scalar, GpSimd, 16 DMA queues sharing the core's HBM bandwidth), waiting on read-after-write, write-after-read and write-after-write hazards between buffers. Tests check that every modeled latency sits between the roofline bound and the no-overlap serial sum.

From AWS's [Trainium2 architecture guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html): Tensor Engine 79 TFLOPS BF16 / 20 TFLOPS FP32 at 2.4 GHz; Vector Engine 0.96 GHz, 512 (16-bit) or 256 elements/cycle; Scalar Engine 1.2 GHz, 128 elements/cycle; SBUF 224 KiB and PSUM 16 KiB per partition; 3 TB/s HBM per 8-core chip; 16 DMA engines per core; ~600 ns per hardware-DGE DMA. PSUM banking and fixed instruction overheads come from the [Trainium/Inferentia2 guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium_inferentia2_arch.html).

Assumptions (not documented): HBM bandwidth splits evenly across a chip's 8 cores (375 GB/s each); NeuronCore-v2 fixed overheads carry over to v3; GpSimd runs at Scalar Engine width; buffers are unlimited, so DMA prefetch overlaps as far as dependencies allow. Use the latencies to compare variants, not as hardware predictions.

## Bugs worth knowing about

- **`id()` reuse in the profiler.** Tiles allocated inside loops get freed; CPython can hand a new tile the same `id()`, which silently merged buffers and created false dependencies. Modeled latency differed between identical runs (e.g. 579 vs. 554 µs). Fix: keep each storage alive for the profile's lifetime; `test_profile_is_deterministic_across_runs` pins it.
- **DMA modeled on one queue.** The first model pinned each `dma_copy` to one DMA engine's 27 GiB/s, which made the modeled latency exceed the no-overlap serial bound. Descriptors spread across all 16 queues; the binding limit is the core's HBM share.

## Not done yet

- Run on real Trainium (trn1/trn2) to calibrate the cost model against hardware.
- Softmax and RMSNorm sit at 2.5x their bandwidth bound: they are Vector Engine bound; splitting work across Vector and GpSimd is the next step.
- Rotary embeddings and multi-query/grouped-query attention.
