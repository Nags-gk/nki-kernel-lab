"""Instruction-level profiler for NKI kernels running on the CPU simulator.

The NKI CPU simulator executes kernels with exact numerics but reports no
timing. This module intercepts every ``nki.isa`` instruction a kernel issues
while it runs under ``nki.simulate`` and records:

* exact counts: instructions per engine, HBM<->SBUF DMA bytes, Tensor Engine
  FLOPs, and per-partition SBUF/PSUM liveness (peak on-chip footprint);
* a *modeled* schedule: each instruction gets a cost from ``hwmodel`` and is
  list-scheduled in program order onto in-order engine queues, waiting on
  read-after-write / write-after-read / write-after-write hazards between
  buffers. The result is a timeline (like a profiler trace) and a modeled
  kernel latency.

Counts are measured. Times are modeled. Keep them apart when quoting results.
"""
from __future__ import annotations

import contextlib
import functools
import math
from dataclasses import dataclass, field

import nki.isa as nisa
import nki.language as nl
from nki.language._dtypes import dtype_size

from .hwmodel import TRN2, CoreSpec

DMA_OPS = {"dma_copy", "dma_transpose"}
TENSOR_OPS = {"nc_matmul", "nc_matmul_mx"}
SCALAR_OPS = {"activation", "activation_reduce", "activate2", "exponential"}
GPSIMD_OPS = {"iota", "affine_select", "range_select", "local_gather", "nc_n_gather"}
VECTOR_OPS = {
    "tensor_tensor", "tensor_reduce", "tensor_scalar", "tensor_copy", "memset",
    "reciprocal", "tensor_tensor_scan", "tensor_scalar_reduce", "scalar_tensor_tensor",
    "tensor_partition_reduce", "bn_stats", "bn_aggr", "max8", "select_reduce",
    "tensor_copy_predicated", "nc_transpose",
}
TRACKED = sorted(DMA_OPS | TENSOR_OPS | SCALAR_OPS | GPSIMD_OPS | VECTOR_OPS)


def _is_tensor(x) -> bool:
    return hasattr(x, "_storage") and hasattr(x, "buffer") and hasattr(x, "shape")


def _buf(t) -> str:
    b = str(t.buffer).lower()
    for k in ("psum", "sbuf", "hbm"):
        if k in b:
            return k
    return b


def _dtype_name(t) -> str:
    return str(t.dtype).split(".")[-1].lower()


def _nelem(t) -> int:
    return int(math.prod(t.shape))


def _nbytes(t) -> int:
    return _nelem(t) * dtype_size(t.dtype)


def _engine_name(e) -> str:
    return str(getattr(e, "name", e)).lower() if e is not None else "unknown"


@dataclass
class Instr:
    idx: int
    op: str
    engine: str
    reads: list
    writes: list
    elems: int = 0
    bytes: int = 0
    flops: int = 0
    direction: str = ""
    cost_s: float = 0.0
    start_s: float = 0.0
    end_s: float = 0.0
    queue: int = 0


@dataclass
class Storage:
    sid: int
    buffer: str
    bytes_per_partition: int
    total_bytes: int
    first: int = 10**12
    last: int = -1


@dataclass
class Profile:
    kernel: str
    spec: CoreSpec
    instrs: list = field(default_factory=list)
    storages: dict = field(default_factory=dict)

    # ---- exact counts -------------------------------------------------
    def counts(self) -> dict:
        c: dict = {}
        for i in self.instrs:
            c[i.engine] = c.get(i.engine, 0) + 1
        return c

    def dma_bytes(self) -> dict:
        d = {"hbm_to_sbuf": 0, "sbuf_to_hbm": 0, "other": 0}
        for i in self.instrs:
            if i.engine == "DMA":
                d[i.direction if i.direction in d else "other"] += i.bytes
        return d

    def hbm_bytes(self) -> int:
        d = self.dma_bytes()
        return d["hbm_to_sbuf"] + d["sbuf_to_hbm"]

    def flops(self) -> int:
        return sum(i.flops for i in self.instrs)

    def peak_onchip(self) -> dict:
        """Peak live bytes per partition for SBUF and PSUM (liveness sweep)."""
        out = {}
        for kind in ("sbuf", "psum"):
            events = []
            for s in self.storages.values():
                if s.buffer == kind and s.last >= 0:
                    events.append((s.first, +s.bytes_per_partition))
                    events.append((s.last + 1, -s.bytes_per_partition))
            events.sort(key=lambda e: (e[0], e[1]))
            cur = peak = 0
            for _, delta in events:
                cur += delta
                peak = max(peak, cur)
            out[kind] = peak
        return out

    # ---- modeled timing -----------------------------------------------
    def schedule(self) -> None:
        sp = self.spec
        eng_free = {"Tensor": 0.0, "Vector": 0.0, "Scalar": 0.0, "GpSimd": 0.0}
        q_free = [0.0] * sp.dma_queues
        link_free = 0.0
        last_write: dict = {}
        last_read: dict = {}
        hbm = {sid for sid, s in self.storages.items() if s.buffer == "hbm"}
        for ins in self.instrs:
            ready = 0.0
            for sid in ins.reads:
                ready = max(ready, last_write.get(sid, 0.0))
            for sid in ins.writes:
                if sid in hbm:
                    continue  # disjoint output regions; RAW is still enforced
                ready = max(ready, last_write.get(sid, 0.0), last_read.get(sid, 0.0))
            if ins.engine == "DMA":
                # Fixed descriptor latency overlaps across the 16 queues; the
                # bytes share the core's HBM bandwidth (one link, FIFO).
                q = min(range(sp.dma_queues), key=lambda j: q_free[j])
                start = max(ready, q_free[q])
                link_start = max(link_free, start + sp.dma_fixed_s)
                end = link_start + ins.bytes / sp.hbm_bytes_per_s
                link_free = end
                q_free[q] = end
                ins.queue = q
            else:
                start = max(ready, eng_free[ins.engine])
                end = start + ins.cost_s
                eng_free[ins.engine] = end
            ins.start_s, ins.end_s = start, end
            for sid in ins.reads:
                last_read[sid] = max(last_read.get(sid, 0.0), end)
            for sid in ins.writes:
                last_write[sid] = end

    def modeled_latency_s(self) -> float:
        return max((i.end_s for i in self.instrs), default=0.0)

    def serial_latency_s(self) -> float:
        """Upper bound: every instruction back to back, no overlap at all."""
        sp = self.spec
        t = 0.0
        for i in self.instrs:
            t += i.cost_s
        return t

    def busy(self) -> dict:
        lat = self.modeled_latency_s() or 1.0
        b: dict = {}
        for i in self.instrs:
            key = "DMA" if i.engine == "DMA" else i.engine
            b[key] = b.get(key, 0.0) + (i.end_s - i.start_s)
        out = {k: v / lat for k, v in b.items()}
        if "DMA" in out:
            out["DMA"] /= self.spec.dma_queues  # average over the 16 queues
        return out

    def roofline(self, dtype: str = "float32") -> dict:
        peak = self.spec.peak_flops.get(dtype, self.spec.peak_flops["float32"])
        fl, by = self.flops(), self.hbm_bytes()
        t_compute = fl / peak if peak else 0.0
        t_memory = by / self.spec.hbm_bytes_per_s
        return {
            "arithmetic_intensity": fl / by if by else None,
            "ridge_point": peak / self.spec.hbm_bytes_per_s,
            "bound_s": max(t_compute, t_memory),
            "bound": "compute" if t_compute >= t_memory else "memory",
            "peak_flops": peak,
        }

    def summary(self, dtype: str = "float32") -> dict:
        lat = self.modeled_latency_s()
        rf = self.roofline(dtype)
        onchip = self.peak_onchip()
        return {
            "kernel": self.kernel,
            "instructions": len(self.instrs),
            "per_engine": self.counts(),
            "dma_bytes": self.dma_bytes(),
            "hbm_bytes": self.hbm_bytes(),
            "flops": self.flops(),
            "modeled_latency_us": lat * 1e6,
            "serial_latency_us": self.serial_latency_s() * 1e6,
            "roofline_bound_us": rf["bound_s"] * 1e6,
            "roofline_regime": rf["bound"],
            "arithmetic_intensity": rf["arithmetic_intensity"],
            "modeled_tflops": (self.flops() / lat / 1e12) if lat else 0.0,
            "pct_of_roofline": (rf["bound_s"] / lat * 100) if lat else 0.0,
            "engine_busy_pct": {k: round(v * 100, 1) for k, v in self.busy().items()},
            "peak_sbuf_bytes_per_partition": onchip["sbuf"],
            "peak_psum_bytes_per_partition": onchip["psum"],
            "sbuf_capacity_ok": onchip["sbuf"] <= self.spec.sbuf_bytes_per_partition,
            "psum_capacity_ok": onchip["psum"] <= self.spec.psum_bytes_per_partition,
        }

    def trace(self, limit: int = 4000) -> list:
        """Timeline events (modeled) for the dashboard's trace viewer."""
        ev = []
        for i in self.instrs[:limit]:
            lane = f"DMA q{i.queue}" if i.engine == "DMA" else i.engine
            ev.append({"op": i.op, "lane": lane, "t0": round(i.start_s * 1e6, 4),
                       "t1": round(i.end_s * 1e6, 4), "bytes": i.bytes, "flops": i.flops})
        return ev


class _Recorder:
    def __init__(self, kernel: str, spec: CoreSpec):
        self.prof = Profile(kernel=kernel, spec=spec)
        # Hold every storage object for the profile's lifetime. Without this,
        # a tile freed inside a loop can be garbage-collected and its id()
        # reused by a new tile, silently merging two buffers (false hazards,
        # wrong liveness, run-to-run differences in modeled latency).
        self._keepalive: list = []

    def _storage(self, t) -> int:
        sid = id(t._storage)
        if sid not in self.prof.storages:
            self._keepalive.append(t._storage)
            shape = tuple(t._storage_shape)
            size = dtype_size(t._storage_dtype)
            per_part = int(math.prod(shape[1:]) * size) if len(shape) > 1 else size
            self.prof.storages[sid] = Storage(sid, _buf(t), per_part, int(math.prod(shape) * size))
        return sid

    def _touch(self, sid: int, idx: int) -> None:
        s = self.prof.storages[sid]
        s.first = min(s.first, idx)
        s.last = max(s.last, idx)

    def record(self, op: str, args: tuple, kwargs: dict) -> None:
        sp = self.prof.spec
        params = dict(kwargs)
        dst = params.pop("dst", None)
        if dst is None and args:
            dst, args = args[0], args[1:]
        inputs = [a for a in list(args) + list(params.values()) if _is_tensor(a)]
        idx = len(self.prof.instrs)

        if op in DMA_OPS:
            engine = "DMA"
        elif op in TENSOR_OPS:
            engine = "Tensor"
        elif op in SCALAR_OPS:
            engine = "Scalar"
        elif op in GPSIMD_OPS:
            engine = "GpSimd"
        else:
            engine = "Vector"
        requested = _engine_name(params.get("engine"))
        if op == "nc_transpose":
            engine = "Vector" if requested == "vector" else "Tensor"
        elif requested in ("scalar", "vector", "gpsimd") and engine not in ("DMA", "Tensor"):
            engine = {"scalar": "Scalar", "vector": "Vector", "gpsimd": "GpSimd"}[requested]

        reads = [self._storage(t) for t in inputs]
        writes = [self._storage(dst)] if _is_tensor(dst) else []
        if op in TENSOR_OPS and params.get("accumulate") is not False and writes:
            reads.append(writes[0])  # accumulate (or inferred) reads PSUM
        for sid in set(reads + writes):
            self._touch(sid, idx)

        ins = Instr(idx=idx, op=op, engine=engine, reads=reads, writes=writes)
        if engine == "DMA" and _is_tensor(dst):
            ins.bytes = _nbytes(dst)
            src = inputs[0] if inputs else None
            sb, db = (_buf(src) if src is not None else "?"), _buf(dst)
            ins.direction = {("hbm", "sbuf"): "hbm_to_sbuf", ("sbuf", "hbm"): "sbuf_to_hbm"}.get((sb, db), f"{sb}_to_{db}")
            ins.cost_s = sp.dma_fixed_s + ins.bytes / sp.hbm_bytes_per_s
        elif engine == "Tensor":
            if op == "nc_transpose":
                data = inputs[0]
                kdim, mdim = data.shape[0], _nelem(data) // data.shape[0]
                ncols, dt = kdim, _dtype_name(data)
                ins.flops = 0
            else:
                st = params.get("stationary", inputs[0] if inputs else None)
                mv = params.get("moving", inputs[1] if len(inputs) > 1 else None)
                kdim = st.shape[0]
                mdim = _nelem(st) // kdim
                ncols = _nelem(mv) // mv.shape[0]
                dt = _dtype_name(mv)
                ins.flops = 2 * kdim * mdim * ncols
            cpc = sp.pe_cols_per_cycle.get(dt, 1.0)
            ins.elems = ncols
            ins.cost_s = (ncols / cpc + sp.tensor_fixed_cycles) / sp.pe_clock_hz
        else:
            elems = max([_nelem(t) for t in inputs + ([dst] if _is_tensor(dst) else [])] or [0])
            ins.elems = elems
            if engine == "Vector":
                is16 = _is_tensor(dst) and dtype_size(dst.dtype) == 2
                width = sp.vector_elems_per_cycle_16b if is16 else sp.vector_elems_per_cycle
                ins.cost_s = (elems / width + sp.vector_fixed_cycles) / sp.vector_clock_hz
            elif engine == "Scalar":
                ins.cost_s = (elems / sp.scalar_elems_per_cycle + sp.scalar_fixed_cycles) / sp.scalar_clock_hz
            else:
                ins.cost_s = (elems / sp.gpsimd_elems_per_cycle + sp.gpsimd_fixed_cycles) / sp.gpsimd_clock_hz
        self.prof.instrs.append(ins)


@contextlib.contextmanager
def profile(kernel: str = "kernel", spec: CoreSpec = TRN2):
    """Record every nki.isa instruction issued inside the block.

    Usage::

        with profile("matmul") as p:
            nki.simulate(kernel)(a, b)
        p.schedule(); print(p.summary())
    """
    rec = _Recorder(kernel, spec)
    saved = {}
    for name in TRACKED:
        fn = getattr(nisa, name, None)
        if fn is None:
            continue
        saved[name] = fn

        def wrap(f, op):
            @functools.wraps(f)
            def inner(*a, **k):
                rec.record(op, a, k)
                return f(*a, **k)
            return inner

        setattr(nisa, name, wrap(fn, name))
    try:
        yield rec.prof
    finally:
        for name, fn in saved.items():
            setattr(nisa, name, fn)
        rec.prof.schedule()
