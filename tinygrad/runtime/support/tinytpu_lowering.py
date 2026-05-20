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
from tinygrad.dtype import PtrDType, dtypes

# ---------------------------------------------------------------------------
# Tile geometry and opcode tables (kept in sync with ops_tinytpu.py)
# ---------------------------------------------------------------------------
_ROWS = 4
_COLS = 4
_TILE_ELEMS = _ROWS * _COLS

# VPU op codes — subset used by the elementwise walker.
_VPU = {"ADD": 0, "MUL": 1, "MAX": 3, "CMPLT": 5, "CMPNE": 6, "SUB": 7,
        "CMPEQ": 8, "SHL": 10, "SHR": 11, "MIN": 12, "DIV": 14,
        "AND": 15, "OR": 16, "XOR": 17,
        "FADD": 18, "FMUL": 19, "FSUB": 20, "FMAX": 21, "FCMPLT": 22}

# tinygrad ALU op -> integer VPU op name.
_ALU_TO_VPU = {Ops.ADD: "ADD", Ops.MUL: "MUL", Ops.SUB: "SUB", Ops.MAX: "MAX",
               Ops.CMPLT: "CMPLT", Ops.CMPNE: "CMPNE", Ops.CMPEQ: "CMPEQ",
               Ops.AND: "AND", Ops.OR: "OR", Ops.XOR: "XOR",
               Ops.SHL: "SHL", Ops.SHR: "SHR", Ops.IDIV: "DIV"}
# tinygrad ALU op -> float VPU op name (operands are float).
_FLOAT_VPU = {Ops.ADD: "FADD", Ops.MUL: "FMUL", Ops.SUB: "FSUB",
              Ops.MAX: "FMAX", Ops.CMPLT: "FCMPLT"}

# Ops the walker emits an instruction for. LOAD/CONST are leaves; GEP is
# transparent lane-selection that the walker sees through.
_ALU_OPS = frozenset(_ALU_TO_VPU)
_DATA_OPS = _ALU_OPS | {Ops.WHERE}

def _is_float(u: UOp) -> bool:
  return "float" in str(u.dtype)

def _float_operands(u: UOp) -> bool:
  """True if the op operates on float operands (so it needs an F-variant)."""
  return any(_is_float(s) for s in u.src)

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
def _canon(u: UOp) -> UOp:
  """Strip transparent wrappers the walker sees through:

  - GEP lane-selection — GEP(x, i) reduces to x.
  - value-preserving bool->int casts — a bool is physically 0/1 and is
    loaded straight into an int32 tile, so the cast is a no-op.
  """
  while True:
    if u.op is Ops.GEP:
      u = u.src[0]
    elif u.op is Ops.CAST and u.src[0].dtype == dtypes.bool and "float" not in str(u.dtype):
      u = u.src[0]
    else:
      return u

def _unique_param(u: UOp) -> int | None:
  """The single PARAM arg reachable from u, or None if not unique."""
  args = {n.arg for n in u.toposort() if n.op is Ops.PARAM}
  return next(iter(args)) if len(args) == 1 and isinstance(next(iter(args)), int) else None

def _data_dag(val: UOp) -> list[UOp]:
  """Walk a stored-value tree, returning data nodes in topological order.

  GEP is transparent; LOAD and CONST are leaves — their index subtrees are
  not entered, so index arithmetic never reaches the walker.
  """
  order: list[UOp] = []
  seen: set[UOp] = set()
  def visit(u: UOp) -> None:
    u = _canon(u)
    if u in seen: return
    seen.add(u)
    if u.op in _DATA_OPS:
      for s in u.src: visit(s)
    order.append(u)
  visit(val)
  return order

def _store_lanes(store: UOp) -> list[UOp]:
  """The per-lane value computations of a STORE.

  Float kernels store a VECTORIZE of N lane computations; int kernels store
  one scalar value (and are unrolled into many STOREs instead).
  """
  v = store.src[1]
  return list(v.src) if v.op is Ops.VECTORIZE else [v]

# ---------------------------------------------------------------------------
# can_lower — positive predicate selecting walker-owned kernels
# ---------------------------------------------------------------------------
def can_lower(uops: list[UOp]) -> bool:
  """True iff the elementwise walker fully owns this kernel.

  Scope: int32/bool/float elementwise — ALU and WHERE over equal-size or
  size-1 (scalar broadcast) operands. No cast, transcendental, reduction,
  WMMA, or structured row/column broadcast.

  Kernels arrive per-element unrolled (one STORE per lane) or vectorized
  (one STORE of a VECTORIZE). Every lane must compute the same data-DAG
  shape so one lane can template the tiled program.
  """
  if any(u.op is Ops.WMMA for u in uops): return False
  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  stores = [u for u in uops if u.op is Ops.STORE]
  if not stores or not params: return False

  out_args = {_unique_param(s.src[0]) for s in stores}
  if len(out_args) != 1: return False
  out_arg = next(iter(out_args))
  if out_arg is None or out_arg not in params: return False
  out_size = params[out_arg].dtype.size
  if out_size <= 0: return False

  ref_shape: tuple | None = None
  for s in stores:
    for lane in _store_lanes(s):
      if _canon(lane).op not in _DATA_OPS: return False  # bare copy / const-fill stays elsewhere
      nodes = _data_dag(lane)
      shape = tuple(n.op for n in nodes)
      if ref_shape is None: ref_shape = shape
      elif shape != ref_shape: return False              # non-uniform: not a plain elementwise map
      for n in nodes:
        if n.op in _DATA_OPS:
          # Float CMPNE/CMPEQ are valid as integer bit-compares; other float
          # arithmetic needs an F-variant opcode.
          if (n.op is not Ops.WHERE and _float_operands(n)
              and n.op not in _FLOAT_VPU and n.op not in (Ops.CMPNE, Ops.CMPEQ)):
            return False
          continue
        if n.op is Ops.LOAD:
          p = _unique_param(n)
          if p is None or p not in params: return False
          if params[p].dtype.size not in (out_size, 1): return False
          continue
        if n.op is Ops.CONST:
          continue
        return False   # unknown op (CAST, transcendental, ...)
  return True

# ---------------------------------------------------------------------------
# lower_kernel — the InstSel pass + UOp walker
# ---------------------------------------------------------------------------
def _const_bits(arg) -> int:
  """Encode a CONST operand as the int32 the broadcast tile carries."""
  if isinstance(arg, float):
    return int(np.frombuffer(np.float32(arg).tobytes(), dtype=np.int32)[0])
  return int(arg)

def lower_kernel(uops: list[UOp]) -> dict:
  """Lower an elementwise kernel to an SXU_PROGRAM descriptor.

  Caller must have checked can_lower(uops) first.
  """
  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  stores = [u for u in uops if u.op is Ops.STORE]
  out_arg = _unique_param(stores[0].src[0])
  out_param = params[out_arg]
  out_size = out_param.dtype.size
  out_is_bool = out_param.dtype.base.itemsize == 1

  # One lane templates the whole tiled program (can_lower proved uniformity).
  nodes = _data_dag(_store_lanes(stores[0])[0])
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
                               "value": _const_bits(leaf.arg), "count": count, "dtype": "int32"})
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
          tuple(vreg[_canon(s)] for s in node.src)))
        kern.primitives.add("SELECT")
      else:
        table = _FLOAT_VPU if _float_operands(node) and node.op in _FLOAT_VPU else _ALU_TO_VPU
        kern.instructions.append(TpuInst("VPU", reg,
          tuple(vreg[_canon(s)] for s in node.src), vpu_op=_VPU[table[node.op]]))

    out_vmem = base + len(leaves)
    kern.instructions.append(TpuInst("STORE", out_vmem, (vreg[interior[-1]],)))
    kern.outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})

  kern.instructions.append(TpuInst("HALT"))
  return kern.to_sxu_descriptor()
