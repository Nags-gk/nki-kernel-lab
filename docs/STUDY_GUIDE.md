# Study guide

Everything here should be explainable from first principles in an interview. Each section ends with questions to be ready for.

## 1. The NeuronCore in one page

- **Memories.** HBM (off-chip, large, slow) → **SBUF** (on-chip scratchpad, 128 partitions × 224 KiB on trn2) → engines. **PSUM** is a separate 2 MiB accumulation buffer (16 KiB per partition, 8 banks of 512 fp32) that only the Tensor Engine writes matmul results into.
- **Engines run in parallel, each in order.** Tensor (128×128 systolic array), Vector (reductions, elementwise), Scalar (activations like exp/silu/rsqrt, with a free per-row sum), GpSimd (programmable: iota, masks), plus DMA queues. Performance comes from keeping all of them busy at once.
- **Partition dimension.** Axis 0 of an on-chip tile maps to the 128 partitions. Engines work on all partitions in parallel and reduce only along the free axis. Layout decides everything: put the dimension you reduce over on the free axis.
- **`nc_matmul` computes `stationary.T @ moving`.** Contraction dim K is on partitions (≤128); stationary free size ≤128; moving free size ≤512 (one PSUM bank). That's why kernels take `lhsT` and why attention takes `qT`, `kT`.

Be ready for: Why does matmul want A transposed? What limits how many accumulators you can keep? Why is there a separate PSUM?

## 2. Matmul: naive → blocked

- Naive reloads both tiles for every (m, n, k) step: operand traffic ∝ M·N·K·(1/TN + 1/TM).
- Blocked keeps BM×BN output tiles in PSUM, loads one `[128, BM·128]` stripe of lhsT and one `[128, BN·512]` stripe of rhs per k step, and reuses each tile BN or BM times. At 1024³ bf16: 23.1 → 8.4 MB, 79 → 38 µs modeled.
- The constraint: BM·BN ≤ 8 PSUM banks. `choose_blocks` picks the largest block that divides the grid.
- Accumulation precision: bf16 inputs, fp32 accumulation in PSUM, a single rounding to bf16 on the way out (`test_matmul_bf16_accumulates_in_fp32`).

Be ready for: Derive the traffic formula. Why not BM=BN=8? What would you do if M isn't a multiple of 128 (host padding in `ops.matmul`; masking would avoid the padding cost)?

## 3. Softmax and RMSNorm: fusion

- Unfused softmax = 3 passes, each reading/writing HBM (max, exp+sum, divide): 117.5 MB. Fused = read once, write once: 33.6 MB.
- Tricks: `tensor_reduce(..., negate=True)` gives −max for free; `activation(exp, bias=-max, reduce_op=add)` computes exp and the row sum in one Scalar Engine instruction; `reciprocal` then one broadcast multiply instead of a divide per element.
- Numerical stability: subtracting the max keeps exp ≤ 1 (`test_softmax_is_stable_for_large_logits` uses logits ~1e4).
- RMSNorm: `activation(square, reduce add)` → `tensor_scalar(×1/D, +eps)` → `rsqrt` → scale. The weight row is broadcast across partitions once by DMA (`weight.reshape((1, D)).broadcast(0, 128)`).

Be ready for: Why is fused softmax still 2.5x off its bandwidth bound? (Vector Engine bound: the trace shows it.) How would you fix it?

## 4. Flash attention

- **Online softmax.** For each 128-query tile, stream 128-key tiles. Keep running max `m`, running sum `l`, running output `O`. For a new tile with scores `s`:
  `m' = max(m, rowmax(s))`, `p = exp(s − m')`, `c = exp(m − m')`, `l' = l·c + rowsum(p)`, `O' = O·c + p·V`. At the end `O/l`.
- **Causal skip.** Key tiles strictly after the diagonal are fully masked; the loop never computes them. Only the diagonal tile needs a mask (`affine_select` on GpSimd: keep where q_index − k_index ≥ 0). `test_flash_skips_masked_tiles` counts the matmuls: n(n+1)/2 vs n².
- **P·V needs Pᵀ as stationary.** `nc_transpose` (Tensor Engine, identity matmul) into PSUM, copy to SBUF, then matmul.
- **K/V resident.** Load each head's K (`[d, S]`) and V (`[128, S/128, d]`) once; every query tile reuses them. Traffic 79.7 → 16.8 MB. Latency unchanged, because the kernel was never DMA bound at that point.
- **Engine balancing (the profiler-driven step).** The Vector Engine was 82% busy on `[128, 1]` ops that each pay ~100 cycles fixed. Fixes: keep the running max negated so the update is one `tensor_scalar(×(−scale), min)`; one fused instruction each for `l` and `O`; rotate running state by rebinding Python names instead of copying tiles; move PSUM→SBUF copies to the Scalar Engine. Vector instructions 11,648 → 4,992; 1,829 → 1,201 µs.
- **Interleaving.** Each query tile is a serial dependency chain; engines are in-order, so a lone chain idles them. Interleaving two independent query tiles in program order lets one tile's matmul overlap the other's softmax. 1,201 → 1,147 µs. Diminishing returns past 2.

Be ready for: Prove the online softmax update is exact. Why −inf as −3e38, not `float('-inf')`? (exp(−inf − (−inf)) = NaN on the first tile if both are infinite.) Why does skipping causal tiles halve work, not quarter it? What breaks if S isn't a multiple of 128 (`ops.attention` pads at the end; causality means real queries never see padded keys)?

## 5. The profiler and cost model

- `profile()` temporarily replaces every `nki.isa` function with a wrapper that records the instruction (engine, read/write buffers, bytes, FLOPs) and then calls the original, so numerics are untouched (`test_profiler_restores_isa_functions`).
- Scheduling: walk instructions in program order; an instruction starts when its engine queue is free and all hazards on its buffers clear. DMA: 16 queues, ~600 ns fixed latency each, bytes share one 375 GB/s link.
- Liveness: a buffer is live from first to last access; sweeping gives peak SBUF/PSUM per partition, checked against capacity.
- Invariants tested: roofline bound ≤ modeled ≤ serial sum; exact DMA bytes for naive matmul against a closed-form formula; determinism across runs.

Be ready for: What does the model get wrong? (No bank conflicts, no buffer-count limits, assumed overheads, even HBM split.) How would you calibrate it on real hardware? (Neuron Explorer profiles on trn2; fit fixed overheads per engine.)

## 6. Decoder block

`model.nki_forward` runs RMSNorm → QKV matmul → flash attention → out-proj → residual → RMSNorm → gate/up matmuls → SwiGLU → down-proj → residual, with every heavy op on NKI. Matches `DecoderBlock` in PyTorch to 6e-7 at S=512 and on a non-aligned S=300.

Be ready for: Which op dominates and why (flash attention, then the MLP matmuls)? What would you fuse next (residual add into RMSNorm; gate and up matmuls into one with SwiGLU in the PSUM eviction)?
