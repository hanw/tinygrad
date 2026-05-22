"""TinyTPU instruction-selection shared infrastructure.

Shared types (TpuInst, TpuKernel), tile geometry constants, opcode tables,
graph helpers, and the InstSel PatternMatcher used by all lowerers.
"""
from __future__ import annotations
from collections import Counter
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
               Ops.SHL: "SHL", Ops.SHR: "SHR", Ops.CDIV: "DIV"}
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

def _find_unique_param_arg(u: UOp) -> int | None:
  """The single PARAM arg in u's source tree, or None if not unique.

  Shared by the GEMM lowerer and the structural recognizers in ops_tinytpu.py.
  """
  params = {node.arg for node in u.toposort() if node.op is Ops.PARAM}
  if len(params) != 1:
    return None
  arg = next(iter(params))
  return arg if isinstance(arg, int) else None

def _has_load_src(u: UOp) -> bool:
  """True if a UOp has a LOAD anywhere in its source tree (data-path)."""
  return any(n.op is Ops.LOAD for n in u.toposort())


def _data_alu_ops(uops: list[UOp]) -> Counter:
  """Count only data-path ALU ops (ones with LOAD in their source tree)."""
  return Counter(_ALU_TO_VPU[u.op] for u in uops
                 if u.op in _ALU_TO_VPU and _has_load_src(u))

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

  Float kernels store a STACK of N lane computations; int kernels store
  one scalar value (and are unrolled into many STOREs instead).
  """
  v = store.src[1]
  return list(v.src) if v.op is Ops.STACK else [v]

# ---------------------------------------------------------------------------
# InstSel pass — graph rewrites expanding ops with no single VPU opcode
# ---------------------------------------------------------------------------
def _expand_sqrt(x: UOp) -> UOp:
  # sqrt(x) = exp2(0.5 * log2(x)); the VPU has no direct sqrt opcode.
  a = x.src[0]
  return (a.alu(Ops.LOG2) * a.const_like(0.5)).alu(Ops.EXP2)

def _expand_mod(x: UOp) -> UOp:
  # mod(a, b) = a - (a CDIV b) * b; the VPU has no direct mod opcode.
  a, b = x.src[0], x.src[1]
  return a.alu(Ops.SUB, a.alu(Ops.CDIV, b).alu(Ops.MUL, b))

_INSTSEL = PatternMatcher([
  (UPat(Ops.SQRT, name="x"), _expand_sqrt),
  (UPat(Ops.CMOD, name="x"), _expand_mod),
])
_INSTSEL_OPS = (Ops.SQRT, Ops.CMOD)

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


# ---------------------------------------------------------------------------
# Compact TASM helpers for bundle construction
# ---------------------------------------------------------------------------
# These produce the wire-format integer lines consumed by TbTinyTPURuntime.
# See doc/tinytpu_asm.md for the full TASM specification. All lowerers and
# ops_tinytpu.py share this single set of encoders (ops_tinytpu re-exports
# them so existing imports keep working).

def _vmem(addr: int, vals: list[int]) -> str:
    return "5 " + str(addr) + " " + " ".join(str(v) for v in vals)

def _wmem(addr: int, vals: list[int]) -> str:
    return "0 " + str(addr) + " " + " ".join(str(v) for v in vals)

def _amem(addr: int, vals: list[int]) -> str:
    return "1 " + str(addr) + " " + " ".join(str(v) for v in vals)

def _load(vd: int, vmem_src: int) -> str:
    return f"2 0 {vmem_src} {vd} 0 0 0 0 0 0"

def _store(vmem_dst: int, vs: int) -> str:
    return f"2 1 {vmem_dst} 0 {vs} 0 0 0 0 0"

def _vpu(vd: int, va: int, op: int, vb: int = 0) -> str:
    return f"2 2 0 {vd} {va} {op} {vb} 0 0 0"

def _vpu_bg(vd: int, va: int, op: int, vb: int = 0) -> str:
    # DISPATCH_VPU_BG (opcode 41). Background-collect dual-issue path
    # for single-cycle VPU ops. Main FSM advances pc the same cycle;
    # a background rule retires the vreg write when vpu.isDone.
    return f"2 41 0 {vd} {va} {op} {vb} 0 0 0"

def _vpu_exp2(vd: int, va: int) -> str:
    # VPU_EXP2 (opcode 51). Multi-cycle walker — SXU stalls on vpu.isDone
    # until TranscUnit finishes its lane-by-lane Horner. ~80 cycles per tile.
    return _vpu(vd, va, _VPU["EXP2"])

def _vpu_log2(vd: int, va: int) -> str:
    # VPU_LOG2 (opcode 52). Range-reduced (split x = m * 2^e) polynomial
    # in the TranscUnit walker. Exact at powers of two, ~28% error at
    # worst-case fractional inputs. ~96 cycles per tile (6 steps/lane).
    return _vpu(vd, va, _VPU["LOG2"])

def _vpu_sin(vd: int, va: int) -> str:
    # VPU_SIN (opcode 53). Degree-5 Taylor through the TranscUnit walker.
    # Accurate for |x| <= π/2; diverges for |x| > π. Upstream range
    # reduction (mod 2π + quadrant fold) must be emitted by the renderer.
    return _vpu(vd, va, _VPU["SIN"])

def _select(vd: int, cond: int, lhs: int, rhs: int) -> str:
    return f"2 8 0 {vd} {cond} 0 {lhs} {rhs} 0 0"

def _broadcast_scalar(vd: int, vs: int, row: int = 0, col: int = 0) -> str:
    sel = ((row & 0x3) << 2) | (col & 0x3)
    return f"2 9 0 {vd} {vs} 0 {sel} 0 0 0"

def _broadcast_row(vd: int, vs: int, row: int = 0) -> str:
    return f"2 10 0 {vd} {vs} 0 {row} 0 0 0"

def _broadcast_col(vd: int, vs: int, col: int = 0) -> str:
    return f"2 11 0 {vd} {vs} 0 {col} 0 0 0"

def _broadcast(vn: int, lane: int = 0) -> str:
    return f"2 3 0 {vn} {vn} 0 {lane} 0 0 0"

def _mxu(wbase: int, abase: int, tiles: int,
         psum_addr: int = 0, psum_row: int = 0, psum_mode: int = 0) -> str:
    # PSUM target fields repurpose unused vreg slots in DISPATCH_MXU:
    #   vregDst=psum_addr, vregSrc=psum_row, vregSrc2=psum_mode
    # Mode encoding: 0=PSUM_OFF, 1=PSUM_WRITE, 2=PSUM_ACCUMULATE.
    return (f"2 4 0 {psum_addr} {psum_row} 0 {psum_mode} "
            f"{wbase} {abase} {tiles}")

def _mxu_psum_write(wbase: int, abase: int, tiles: int,
                    psum_addr: int, psum_row: int) -> str:
    return _mxu(wbase, abase, tiles, psum_addr, psum_row, 1)

def _mxu_psum_acc(wbase: int, abase: int, tiles: int,
                  psum_addr: int, psum_row: int) -> str:
    return _mxu(wbase, abase, tiles, psum_addr, psum_row, 2)

def _mxu_accumulate(wbase: int, abase: int, tiles: int) -> str:
    # SXU_DISPATCH_MXU_ACCUMULATE opcode = 23. Routes through
    # Controller.startAccumulate: WS feed path with drain-time PE
    # clear skipped, so consecutive dispatches sum into the same PE
    # accumulator (multi-K-tile GEMM). Not a distinct dataflow — the
    # PE still holds a preloaded weight. PSUM plumbing not available.
    return f"2 23 0 0 0 0 0 {wbase} {abase} {tiles}"

def _mxu_clear() -> str:
    # SXU_MXU_CLEAR opcode = 24. Zeroes the systolic-array PE
    # accumulators. Needed between accumulate epochs and when
    # re-entering WS from a previous accumulate/OS dispatch.
    return "2 24 0 0 0 0 0 0 0 0"

def _mxu_os(wbase: int, abase: int, klen: int) -> str:
    # SXU_DISPATCH_MXU_OS opcode = 25. Routes through Controller.startOS:
    # real output-stationary — weights + activations both stream as a
    # staircase, each PE holds its own psum, full (rows x cols) psum
    # drained via resultsMatrix(). klen reuses the MXU tileLen field
    # (<= rows for the current single-tile weight SRAM read).
    return f"2 25 0 0 0 0 0 {wbase} {abase} {klen}"

def _load_mxu_matrix_row(vd: int, row: int) -> str:
    # SXU_LOAD_MXU_MATRIX_ROW opcode = 26. Copies ctrl.resultsMatrix[row]
    # into row 0 of vd. Intended for draining an OS dispatch row-by-row.
    return f"2 26 0 {vd} {row} 0 0 0 0 0"

def _read_cycle(vd: int) -> str:
    # SXU_READ_CYCLE opcode = 27. Writes the SXU's free-running cycle
    # counter as Int#(32) into row 0, lane 0 of vd (other lanes zeroed).
    # Pair two READ_CYCLE calls with a STORE + host parse to measure
    # the span of a program region from inside the bundle itself.
    return f"2 27 0 {vd} 0 0 0 0 0 0"

def _load_loop_depth(vd: int) -> str:
    # SXU_LOAD_LOOP_DEPTH opcode = 36. Writes the current LOOP stack
    # depth (0..4) into row 0, lane 0 of vd (other lanes zeroed).
    # Intended for debug/test of nested SXU_LOOP frames.
    return f"2 36 0 {vd} 0 0 0 0 0 0"

def _xlu_rotate(vd: int, vs: int, amount: int) -> str:
    # SXU_DISPATCH_XLU_ROTATE opcode = 37. Cyclic lane rotation via the
    # XLU: vd[s][i] = vs[s][(i + amount) mod lanes]. Dual-issue like
    # other XLU dispatches. `amount` stored in vregSrc2 low bits.
    return f"2 37 0 {vd} {vs} 0 {amount} 0 0 0"

def _loop_begin(count: int) -> str:
    # SXU_LOOP_BEGIN opcode = 28. Sets loopCounter := count and marks
    # the next instruction as the loop-return pc. Count must be 1..255.
    assert 1 <= count <= 255, "loop count out of range"
    return f"2 28 0 0 0 0 0 0 0 {count}"

def _loop_end() -> str:
    # SXU_LOOP_END opcode = 29. Decrements loopCounter; jumps back to
    # the instruction after LOOP_BEGIN if more iterations remain.
    return "2 29 0 0 0 0 0 0 0 0"

def _vzero(vd: int) -> str:
    # SXU_VZERO opcode = 30. One-cycle tile-of-zeros into vd. Skips
    # the "preload zero in VMEM + LOAD" two-instruction dance.
    return f"2 30 0 {vd} 0 0 0 0 0 0"

def _vfill(vd: int, imm_i8: int) -> str:
    # SXU_VFILL opcode = 31. Broadcast a signed 8-bit constant to all
    # 16 lanes of vd. Encoded with imm in mxuWBase (unsigned byte).
    assert -128 <= imm_i8 <= 127, "VFILL immediate out of int8 range"
    enc = imm_i8 & 0xFF
    return f"2 31 0 {vd} 0 0 0 {enc} 0 0"

def _vmov(vd: int, vs: int) -> str:
    # SXU_VMOV opcode = 32. vd := vs in one cycle.
    return f"2 32 0 {vd} {vs} 0 0 0 0 0"

def _vneg(vd: int, vs: int) -> str:
    # SXU_VNEG opcode = 34. vd := -vs lane-wise in one cycle.
    return f"2 34 0 {vd} {vs} 0 0 0 0 0"

def _vabs(vd: int, vs: int) -> str:
    # SXU_VABS opcode = 35. vd := |vs| lane-wise in one cycle.
    return f"2 35 0 {vd} {vs} 0 0 0 0 0"

def _mxu_os_accumulate(wbase: int, abase: int, klen: int) -> str:
    # SXU_DISPATCH_MXU_OS_ACCUMULATE opcode = 33. Routes through
    # Controller.startOsAccumulate: real-OS dispatch that skips the
    # drain-time clearAll, so consecutive dispatches add another
    # kLen worth of psums into the same matrix. Lets multi-K-tile OS
    # scale past K == rows.
    return f"2 33 0 0 0 0 0 {wbase} {abase} {klen}"

def _psum_read(vd: int, psum_addr: int) -> str:
    # SXU_PSUM_READ opcode = 17; vmemAddr doubles as PSUM bucket index.
    return f"2 17 {psum_addr} {vd} 0 0 0 0 0 0"

def _psum_read_row(vd: int, psum_addr: int, psum_row: int) -> str:
    # SXU_PSUM_READ_ROW opcode = 18. Reads one row of a bucket into
    # row 0 of vd with the other rows zeroed — same shape as
    # LOAD_MXU_RESULT, so downstream bias/relu/store don't care that
    # the row came from PSUM.
    return f"2 18 {psum_addr} {vd} {psum_row} 0 0 0 0 0"

def _psum_clear(psum_addr: int) -> str:
    # SXU_PSUM_CLEAR opcode = 19. Zeros the whole bucket in one cycle
    # without touching any vreg, so multi-K-tile GEMM can drop the
    # "preload zero tile + LOAD v15 + PSUM_WRITE" boilerplate.
    return f"2 19 {psum_addr} 0 0 0 0 0 0 0"

def _psum_clear_all() -> str:
    # SXU_PSUM_CLEAR_ALL opcode = 38. Multi-cycle walker inside SXU
    # zeroes every bucket (psumDepth cycles total, one instruction).
    # Replaces the 8-instruction PSUM_CLEAR sweep a multi-K-tile GEMM
    # does before re-using buckets for a new K chain.
    return f"2 38 0 0 0 0 0 0 0 0"

def _set_pred_if_zero(vs: int) -> str:
    # SXU_SET_PRED_IF_ZERO opcode = 20; pred := (vs[0][0] == 0).
    return f"2 20 0 0 {vs} 0 0 0 0 0"

def _skip_if_pred() -> str:
    # SXU_SKIP_IF_PRED opcode = 21; if pred, skip the next instruction.
    return f"2 21 0 0 0 0 0 0 0 0"

def _set_pred_ne_zero(vs: int) -> str:
    # SXU_SET_PRED_NE_ZERO opcode = 39; pred := (vs[0][0] != 0).
    return f"2 39 0 0 {vs} 0 0 0 0 0"

def _skip_if_not_pred() -> str:
    # SXU_SKIP_IF_NOT_PRED opcode = 40; if !pred, skip the next instruction.
    return f"2 40 0 0 0 0 0 0 0 0"

def _psum_accumulate_row(vs: int, psum_addr: int, psum_row: int) -> str:
    # SXU_PSUM_ACCUMULATE_ROW opcode = 22; accumulate row 0 of vs into
    # psum[psum_addr][psum_row]. VPU-side row-granular deposit,
    # symmetric with the MXU dispatch's psum_acc path.
    return f"2 22 {psum_addr} {psum_row} {vs} 0 0 0 0 0"

def _wait_mxu() -> str: return "2 5 0 0 0 0 0 0 0 0"
def _load_mxu_result(vd: int) -> str: return f"2 6 0 {vd} 0 0 0 0 0 0"

def _load_vpu_result(vd: int) -> str:
    # SXU_LOAD_VPU_RESULT opcode = 13; copies vpu.resultReg (linger
    # register) into vd so subsequent ops can reuse the last VPU
    # output without re-dispatch.
    return f"2 13 0 {vd} 0 0 0 0 0 0"

def _load_xlu_result(vd: int) -> str:
    # SXU_LOAD_XLU_RESULT opcode = 14; same pattern for the XLU output
    # register.
    return f"2 14 0 {vd} 0 0 0 0 0 0"
def _halt()     -> str: return "2 7 0 0 0 0 0 0 0 0"
def _output_mxu()           -> str: return "3 1"
def _output_vmem(addr: int) -> str: return f"6 {addr}"
def _end()      -> str: return "4"

def _bundle(*lines: str) -> str:
    return "\n".join(lines) + "\n"
