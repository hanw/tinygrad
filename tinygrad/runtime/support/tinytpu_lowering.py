"""TinyTPU instruction-selection pass and UOp-walking renderer.

Lowers a tinygrad UOp kernel graph into a typed ``TpuInst`` sequence and
serializes it to the ``SXU_PROGRAM`` descriptor consumed by ``TinyTPUProgram``.

This replaces the per-kernel ``_render_*_sxu_program`` pattern matchers in
``ops_tinytpu.py`` with a single graph walker that emits one TinyTPU
instruction per UOp. See ``doc/plan-tinytpu-instsel.md``.

The module is self-contained: it owns its instruction encoders and opcode
table so that ``ops_tinytpu.py`` can import it without an import cycle.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import PtrDType

# ---------------------------------------------------------------------------
# Tile geometry and opcode tables (kept in sync with ops_tinytpu.py)
# ---------------------------------------------------------------------------
_ROWS = 4
_COLS = 4
_TILE_ELEMS = _ROWS * _COLS

# VPU op codes — subset used by the elementwise walker.
_VPU = {"ADD": 0, "MUL": 1, "MAX": 3, "CMPLT": 5, "CMPNE": 6, "SUB": 7,
        "CMPEQ": 8, "SHL": 10, "SHR": 11, "MIN": 12, "DIV": 14,
        "AND": 15, "OR": 16, "XOR": 17}

# tinygrad ALU op -> VPU op name.
_ALU_TO_VPU = {Ops.ADD: "ADD", Ops.MUL: "MUL", Ops.SUB: "SUB", Ops.MAX: "MAX",
               Ops.CMPLT: "CMPLT", Ops.CMPNE: "CMPNE", Ops.CMPEQ: "CMPEQ",
               Ops.AND: "AND", Ops.OR: "OR", Ops.XOR: "XOR",
               Ops.SHL: "SHL", Ops.SHR: "SHR", Ops.IDIV: "DIV"}

_CMP_OPS = {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}
# Ops the walker emits an instruction for. LOAD/CONST are leaves.
_ALU_OPS = frozenset(_ALU_TO_VPU)
_DATA_OPS = _ALU_OPS | {Ops.WHERE}

# ---------------------------------------------------------------------------
# Typed instruction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TpuInst:
  """One TinyTPU instruction in typed form, before bundle encoding."""
  op: str                              # LOAD | STORE | VPU | SELECT | BROADCAST_SCALAR | HALT
  dst: int = 0                         # destination VREG or VMEM address
  srcs: tuple[int, ...] = ()            # source VREGs / VMEM address
  vpu_op: int | None = None            # _VPU code for VPU dispatches

  def encode(self) -> str:
    """Encode to a simulator bundle instruction line."""
    if self.op == "LOAD":  return f"2 0 {self.srcs[0]} {self.dst} 0 0 0 0 0 0"
    if self.op == "STORE": return f"2 1 {self.dst} 0 {self.srcs[0]} 0 0 0 0 0"
    if self.op == "VPU":
      vb = self.srcs[1] if len(self.srcs) > 1 else 0
      return f"2 2 0 {self.dst} {self.srcs[0]} {self.vpu_op} {vb} 0 0 0"
    if self.op == "SELECT":
      return f"2 8 0 {self.dst} {self.srcs[0]} 0 {self.srcs[1]} {self.srcs[2]} 0 0"
    if self.op == "BROADCAST_SCALAR":
      return f"2 9 0 {self.dst} {self.srcs[0]} 0 0 0 0 0"
    if self.op == "HALT":  return "2 7 0 0 0 0 0 0 0 0"
    raise ValueError(f"cannot encode TpuInst op {self.op!r}")

# ---------------------------------------------------------------------------
# Kernel: typed instruction list + memory plan
# ---------------------------------------------------------------------------
@dataclass
class TpuKernel:
  instructions: list[TpuInst] = field(default_factory=list)
  data_plan: list[dict] = field(default_factory=list)
  outputs: list[dict] = field(default_factory=list)
  out_arg: int = 0
  bool_out: bool = False
  primitives: set[str] = field(default_factory=set)   # diagnostic tags

  def to_sxu_descriptor(self) -> dict:
    """Serialize to the SXU_PROGRAM JSON descriptor TinyTPUProgram consumes."""
    desc: dict = {"op": "SXU_PROGRAM"}
    # A single architectural primitive is surfaced as a diagnostic tag,
    # matching the legacy descriptor (consumed by lowering-dump tooling).
    if len(self.primitives) == 1:
      desc["primitive"] = next(iter(self.primitives))
    desc.update({
      "instructions": [i.encode() for i in self.instructions],
      "data_plan": self.data_plan,
      "outputs": self.outputs,
      "num_output_tiles": len(self.outputs),
      "out": self.out_arg,
      "bool_out": self.bool_out,
    })
    return desc

# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
def _unique_param(u: UOp) -> int | None:
  """The single PARAM arg reachable from u, or None if not unique."""
  args = {n.arg for n in u.toposort() if n.op is Ops.PARAM}
  return next(iter(args)) if len(args) == 1 and isinstance(next(iter(args)), int) else None

def _data_dag(val: UOp) -> list[UOp]:
  """Walk the stored-value tree, returning data nodes in topological order.

  LOAD and CONST are leaves — their index subtrees are not entered, so
  index arithmetic never reaches the walker.
  """
  order: list[UOp] = []
  seen: set[UOp] = set()
  def visit(u: UOp) -> None:
    if u in seen: return
    seen.add(u)
    if u.op in _DATA_OPS:
      for s in u.src: visit(s)
    order.append(u)
  visit(val)
  return order

# ---------------------------------------------------------------------------
# can_lower — positive predicate selecting walker-owned kernels
# ---------------------------------------------------------------------------
def can_lower(uops: list[UOp]) -> bool:
  """True iff the elementwise walker fully owns this kernel.

  Scope (iteration 1): int32/bool elementwise — ALU and WHERE over equal-size
  or size-1 (scalar broadcast) operands. No float, cast, transcendental,
  reduction, WMMA, or structured broadcast.

  tinygrad delivers elementwise kernels per-element unrolled (one STORE per
  output lane, sometimes inside a RANGE). Every STORE must compute the same
  data-DAG shape so a single STORE can template the tiled program.
  """
  if any(u.op is Ops.WMMA for u in uops): return False
  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  stores = [u for u in uops if u.op is Ops.STORE]
  if not stores or not params: return False

  out_args = {_unique_param(s.src[0]) for s in stores}
  if len(out_args) != 1: return False
  out_arg = next(iter(out_args))
  if out_arg is None or out_arg not in params: return False

  out_dtype = params[out_arg].dtype
  if "float" in str(out_dtype): return False
  out_size = out_dtype.size
  if out_size <= 0: return False

  ref_shape: tuple | None = None
  for s in stores:
    val = s.src[1]
    if val.op not in _DATA_OPS: return False   # bare copy / const-fill stays elsewhere
    nodes = _data_dag(val)
    shape = tuple(n.op for n in nodes)
    if ref_shape is None: ref_shape = shape
    elif shape != ref_shape: return False      # non-uniform: not a plain elementwise map
    for n in nodes:
      if n.op in _DATA_OPS: continue
      if n.op is Ops.LOAD:
        p = _unique_param(n)
        if p is None or p not in params: return False
        psize = params[p].dtype.size
        if psize not in (out_size, 1): return False
        if "float" in str(params[p].dtype): return False
        continue
      if n.op is Ops.CONST:
        if isinstance(n.arg, float): return False
        continue
      return False   # unknown op (CAST, transcendental, VECTORIZE, ...)
  return True

# ---------------------------------------------------------------------------
# lower_kernel — the InstSel pass + UOp walker
# ---------------------------------------------------------------------------
def lower_kernel(uops: list[UOp]) -> dict:
  """Lower an elementwise kernel to an SXU_PROGRAM descriptor.

  Caller must have checked can_lower(uops) first.
  """
  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  store = next(u for u in uops if u.op is Ops.STORE)
  out_arg = _unique_param(store.src[0])
  out_param = params[out_arg]
  out_size = out_param.dtype.size
  out_is_bool = out_param.dtype.base.itemsize == 1

  nodes = _data_dag(store.src[1])
  leaves = [n for n in nodes if n.op in (Ops.LOAD, Ops.CONST)]
  interior = [n for n in nodes if n.op in _DATA_OPS]

  num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
  addrs_per_tile = len(leaves) + 1

  kern = TpuKernel(out_arg=out_arg, bool_out=out_is_bool)
  for tile_idx in range(num_tiles):
    offset = tile_idx * _TILE_ELEMS
    count = min(_TILE_ELEMS, out_size - offset)
    base = tile_idx * addrs_per_tile

    vreg: dict[UOp, int] = {}
    next_vreg = 0

    # --- leaves: plan VMEM + emit LOAD (+ scalar broadcast) ---
    for leaf_idx, leaf in enumerate(leaves):
      vmem = base + leaf_idx
      reg = next_vreg; next_vreg += 1
      vreg[leaf] = reg
      if leaf.op is Ops.CONST:
        kern.data_plan.append({"type": "VMEM", "addr": vmem, "layout": "broadcast_const",
                               "value": int(leaf.arg), "count": count, "dtype": "int32"})
        kern.instructions.append(TpuInst("LOAD", reg, (vmem,)))
        continue
      p = _unique_param(leaf)
      psize = params[p].dtype.size
      is_bool = params[p].dtype.base.itemsize == 1
      if psize == 1 and out_size > 1:
        entry = {"type": "VMEM", "addr": vmem, "param": p, "offset": 0,
                 "count": 1, "dtype": "int32"}
        if is_bool: entry["bool"] = True
        kern.data_plan.append(entry)
        kern.instructions.append(TpuInst("LOAD", reg, (vmem,)))
        kern.instructions.append(TpuInst("BROADCAST_SCALAR", reg, (reg,)))
        kern.primitives.add("BROADCAST_SCALAR")
      else:
        entry = {"type": "VMEM", "addr": vmem, "param": p, "offset": offset,
                 "count": count, "dtype": "int32"}
        if is_bool: entry["bool"] = True
        kern.data_plan.append(entry)
        kern.instructions.append(TpuInst("LOAD", reg, (vmem,)))

    # --- interior: emit one VPU/SELECT per node ---
    for node in interior:
      reg = next_vreg; next_vreg += 1
      vreg[node] = reg
      if node.op is Ops.WHERE:
        kern.instructions.append(TpuInst("SELECT", reg,
          (vreg[node.src[0]], vreg[node.src[1]], vreg[node.src[2]])))
        kern.primitives.add("SELECT")
      else:
        kern.instructions.append(TpuInst("VPU", reg,
          (vreg[node.src[0]], vreg[node.src[1]]), vpu_op=_VPU[_ALU_TO_VPU[node.op]]))

    out_vmem = base + len(leaves)
    kern.instructions.append(TpuInst("STORE", out_vmem, (vreg[interior[-1]],)))
    kern.outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})

  kern.instructions.append(TpuInst("HALT"))
  return kern.to_sxu_descriptor()
