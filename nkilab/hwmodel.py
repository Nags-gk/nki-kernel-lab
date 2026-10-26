"""First-order analytic hardware model of one Trainium2 NeuronCore-v3.

Every number that comes from AWS documentation is cited. Every number that
does not is marked ASSUMPTION. The CPU simulator gives exact numerics but no
timing, so all times produced from this model are *modeled*, not measured,
and the README says so wherever they appear.

Source: AWS Neuron docs, "Trainium2 Architecture Guide for NKI"
https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/architecture/trainium2_arch.html
"""
from __future__ import annotations

from dataclasses import dataclass, field

KiB = 1024
MiB = 1024 * KiB


@dataclass(frozen=True)
class CoreSpec:
    name: str = "trn2 NeuronCore-v3"
    partitions: int = 128
    # Doc: SBUF 28 MiB, 128 partitions of 224 KiB each.
    sbuf_bytes_per_partition: int = 224 * KiB
    # Doc: PSUM 2 MiB -> 16 KiB per partition.
    psum_bytes_per_partition: int = (2 * MiB) // 128
    # Doc (v2 guide; trn2 guide says PSUM is unchanged): 8 banks per partition,
    # each holding 512 fp32 values, and one nc_matmul writes a single bank.
    psum_banks: int = 8
    psum_bank_fp32: int = 512

    # Tensor Engine: 128x128 systolic array at 2.4 GHz.
    # Doc: 79 TFLOPS BF16/FP16/TF32, 20 TFLOPS FP32, 158 TFLOPS FP8 (dense).
    # 2*128*128*2.4e9 = 78.6e12 -> one moving column per cycle for BF16,
    # a quarter column for FP32, two columns for FP8.
    pe_clock_hz: float = 2.4e9
    pe_rows: int = 128
    pe_cols: int = 128
    pe_cols_per_cycle: dict = field(default_factory=lambda: {
        "bfloat16": 1.0, "float16": 1.0, "tfloat32": 1.0,
        "float32": 0.25,
        "float8_e4m3": 2.0, "float8_e5m2": 2.0, "float8_e4m3fn": 2.0,
    })

    # Doc: Vector Engine 0.96 GHz, 512 elem/cycle for BF16/FP16, 256 otherwise.
    vector_clock_hz: float = 0.96e9
    vector_elems_per_cycle_16b: int = 512
    vector_elems_per_cycle: int = 256
    # Doc: Scalar Engine 1.2 GHz, 128 elem/cycle in/out.
    scalar_clock_hz: float = 1.2e9
    scalar_elems_per_cycle: int = 128
    # Doc: GpSimd 1.2 GHz, eight programmable processors; width not stated.
    # ASSUMPTION: treat it like the Scalar Engine (128 elem/cycle).
    gpsimd_clock_hz: float = 1.2e9
    gpsimd_elems_per_cycle: int = 128

    # Fixed per-instruction overheads. Doc (NeuronCore-v2 guide): Tensor Engine
    # MM_INIT_LATENCY 64 cycles; Vector static overhead 100 cycles; Scalar
    # accumulator read ~64 cycles. ASSUMPTION: same cycle counts on v3.
    tensor_fixed_cycles: int = 64
    vector_fixed_cycles: int = 100
    scalar_fixed_cycles: int = 64
    gpsimd_fixed_cycles: int = 64  # ASSUMPTION: not documented

    # Doc: 3 TB/s HBM per chip, 8 NeuronCores per chip.
    # ASSUMPTION: even split -> 375 GB/s aggregate per NeuronCore.
    hbm_bytes_per_s: float = 3.0e12 / 8
    # Doc: 16 DMA engines paired with each NeuronCore-v3.
    dma_queues: int = 16
    # Doc (v2 guide): each DMA engine peaks at 27 GiB/s; 16 x 27 GiB/s exceeds
    # the 375 GB/s share, so the model assumes a dma_copy's descriptors spread
    # across queues and the core's HBM share is the binding limit.
    # Doc: a hardware-DGE DMA instruction takes about 600 ns.
    dma_fixed_s: float = 600e-9

    @property
    def peak_flops(self) -> dict:
        base = 2 * self.pe_rows * self.pe_cols * self.pe_clock_hz
        return {k: base * v for k, v in self.pe_cols_per_cycle.items()}


TRN2 = CoreSpec()
