"""TinyTPU instruction-selection shared infrastructure.

Shared types (TpuInst, TpuKernel), tile geometry constants, opcode tables,
graph helpers, and the InstSel PatternMatcher used by all lowerers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, graph_rewrite
from tinygrad.dtype import PtrDType, dtypes

# ---------------------------------------------------------------------------
# Tile geometry and opcode tables (kept in sync with ops_tinytpu.py)
# ---------------------------------------------------------------------------
_ROWS = 4
_COLS = 4
_TILE_ELEMS = _ROWS * _COLS
_NUM_VREGS = 16

# VPU op codes — the full opcode table for the lowering package
# (elementwise + reduction lowerers share this single table).
_VPU = {"ADD": 0, "MUL": 1, "MAX": 3, "SUM_REDUCE": 4, "CMPLT": 5, "CMPNE": 6,
        "SUB": 7, "CMPEQ": 8, "MAX_REDUCE": 9, "SHL": 10, "SHR": 11, "MIN": 12,
        "MIN_REDUCE": 13, "DIV": 14, "AND": 15, "OR": 16, "XOR": 17,
        "FADD": 18, "FMUL": 19, "FSUB": 20, "FMAX": 21, "FCMPLT": 22,
        "FRECIP": 23, "I2F": 24, "F2I": 25,
        "SUM_REDUCE_COL": 29, "MAX_REDUCE_COL": 30, "MIN_REDUCE_COL": 31,
        "SUM_REDUCE_TILE": 32, "MAX_REDUCE_TILE": 33, "MIN_REDUCE_TILE": 34,
        "MUL_REDUCE": 35, "MUL_REDUCE_COL": 36, "MUL_REDUCE_TILE": 37,
        "FSUM_REDUCE_TILE": 38, "FMAX_REDUCE_TILE": 39, "FMIN_REDUCE_TILE": 40,
        "FMIN": 41,
        "FSUM_REDUCE": 42, "FMAX_REDUCE": 43, "FMIN_REDUCE": 44,
        "FSUM_REDUCE_COL": 45, "FMAX_REDUCE_COL": 46, "FMIN_REDUCE_COL": 47,
        "FPROD_REDUCE_TILE": 48, "FPROD_REDUCE": 49, "FPROD_REDUCE_COL": 50,
        "EXP2": 51, "LOG2": 52, "SIN": 53}

# tinygrad ALU op -> integer VPU op name.
_ALU_TO_VPU = {Ops.ADD: "ADD", Ops.MUL: "MUL", Ops.SUB: "SUB", Ops.MAX: "MAX",
               Ops.CMPLT: "CMPLT", Ops.CMPNE: "CMPNE", Ops.CMPEQ: "CMPEQ",
               Ops.AND: "AND", Ops.OR: "OR", Ops.XOR: "XOR",
               Ops.SHL: "SHL", Ops.SHR: "SHR", Ops.IDIV: "DIV"}
# tinygrad ALU op -> float VPU op name (operands are float).
_FLOAT_VPU = {Ops.ADD: "FADD", Ops.MUL: "FMUL", Ops.SUB: "FSUB",
              Ops.MAX: "FMAX", Ops.CMPLT: "FCMPLT"}
# tinygrad unary op -> VPU op name (single hardware opcode).
_UNARY_VPU = {Ops.EXP2: "EXP2", Ops.LOG2: "LOG2", Ops.SIN: "SIN",
              Ops.RECIPROCAL: "FRECIP"}

# Ops the walker emits an instruction for. LOAD/CONST are leaves; GEP is
# transparent lane-selection that the walker sees through.
_ALU_OPS = frozenset(_ALU_TO_VPU)
# TRUNC has no single VPU opcode — the walker emits an F2I+I2F micro-pair.
# CAST is an interior unary: a value-converting int<->float CAST maps to one
# VPU opcode (I2F / F2I). Transparent bool->int casts are stripped by _canon
# before they ever reach the data DAG, so an interior CAST is always a convert.
_DATA_OPS = _ALU_OPS | {Ops.WHERE, Ops.TRUNC, Ops.CAST} | frozenset(_UNARY_VPU)

def _is_float(u: UOp) -> bool:
  return "float" in str(u.dtype)

def _cast_vpu(u: UOp) -> str | None:
  """VPU opcode name for a value-converting CAST node, or None if it has no
  per-element opcode (e.g. a same-class cast, which the walker rejects).

  bool->int casts never reach here — _canon strips them as transparent.
  """
  dst_float = _is_float(u)
  src_float = _is_float(u.src[0])
  if src_float and not dst_float: return "F2I"   # float -> int
  if dst_float and not src_float: return "I2F"   # int -> float
  return None                                     # float<->float / int<->int

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
# InstSel pass — graph rewrites expanding ops with no single VPU opcode
# ---------------------------------------------------------------------------
def _expand_sqrt(x: UOp) -> UOp:
  # sqrt(x) = exp2(0.5 * log2(x)); the VPU has no direct sqrt opcode.
  a = x.src[0]
  return (a.alu(Ops.LOG2) * a.const_like(0.5)).alu(Ops.EXP2)

def _expand_mod(x: UOp) -> UOp:
  # mod(a, b) = a - (a // b) * b; the VPU has no direct mod opcode.
  a, b = x.src[0], x.src[1]
  return a.alu(Ops.SUB, a.alu(Ops.IDIV, b).alu(Ops.MUL, b))

_INSTSEL = PatternMatcher([
  (UPat(Ops.SQRT, name="x"), _expand_sqrt),
  (UPat(Ops.MOD, name="x"), _expand_mod),
])
_INSTSEL_OPS = (Ops.SQRT, Ops.MOD)

def _run_instsel(uops: list[UOp]) -> list[UOp]:
  """Apply InstSel graph rewrites; return the (possibly rewritten) uop list."""
  if not any(u.op in _INSTSEL_OPS for u in uops):
    return uops
  sink = next(u for u in uops if u.op is Ops.SINK)
  return list(graph_rewrite(sink, _INSTSEL).toposort())

# ---------------------------------------------------------------------------
# lower_kernel helper: const encoding
# ---------------------------------------------------------------------------
def _const_bits(arg) -> int:
  """Encode a CONST operand as the int32 the broadcast tile carries."""
  if isinstance(arg, float):
    return int(np.frombuffer(np.float32(arg).tobytes(), dtype=np.int32)[0])
  return int(arg)
