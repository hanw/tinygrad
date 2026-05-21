"""TinyTPU reduction lowerer.

Consolidates the three legacy structural recognizers
(`_render_reduction_sxu_program`, `_render_rowreduce_sxu_program`,
`_render_colreduce_sxu_program`) into one classify-then-emit lowerer.

A reduction kernel has an input of size S and an output of size O < S whose
stored value is an associative-op tree (ADD/MAX/MUL/XOR) over loads of the
input. The lowerer classifies the reduce op (ADD->SUM, MAX->MAX, MUL->PROD,
MAX+XOR->MIN, float-min-via-negation->FMIN) and the axis (scalar O==1, row,
column), then emits VPU_*_REDUCE_TILE / _REDUCE / _REDUCE_COL with reduction-
identity padding for partial tiles, multi-tile combine, and post-op folding.

The behavioral spec is the three legacy recognizers; this lowerer must produce
descriptors identical to them for every kernel they accept.
"""
from __future__ import annotations
from collections import Counter
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import PtrDType
# Shared infrastructure — one opcode table / geometry / encoders for the package.
from tinygrad.runtime.support.tinytpu_lowering.common import (
  _ROWS, _COLS, _TILE_ELEMS, _VPU, _ALU_TO_VPU, _const_bits,
  _load, _store, _vpu, _halt)

# Reduction-identity bit patterns used to pad partial tiles.
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_FLOAT_NEG_INF_BITS = -(1 << 23)   # 0xFF800000 as signed int32
_FLOAT_POS_INF_BITS = 0x7F800000
_FLOAT_ONE_BITS = 0x3F800000       # 1.0


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
def _has_load_src(u: UOp) -> bool:
  """True if a UOp has a LOAD anywhere in its source tree (data-path)."""
  return any(n.op is Ops.LOAD for n in u.toposort())


def _data_alu_ops(uops: list[UOp]) -> Counter:
  """Count only data-path ALU ops (ones with LOAD in their source tree)."""
  return Counter(_ALU_TO_VPU[u.op] for u in uops
                 if u.op in _ALU_TO_VPU and _has_load_src(u))


def _is_float_min_negation(uops: list[UOp]) -> bool:
  """True if the kernel matches tinygrad's float-MIN decomposition:
     MUL(MAX(..., MUL(load_i, -1.0), ...), -1.0).
  """
  stores = [u for u in uops if u.op is Ops.STORE]
  if len(stores) != 1:
    return False
  sv = stores[0].src[1]
  def _is_neg_one(u):
    return u.op is Ops.CONST and isinstance(u.arg, float) and float(u.arg) == -1.0
  outer_ok = (sv.op is Ops.MUL and len(sv.src) == 2
              and any(_is_neg_one(s) for s in sv.src))
  data_muls = [u for u in uops if u.op is Ops.MUL and _has_load_src(u)]
  inner_ok = all(any(_is_neg_one(s) for s in u.src) for u in data_muls)
  return outer_ok and inner_ok and len(data_muls) >= 1


def _detect_reduce_op(op_counts: Counter, data_alu: Counter | None = None) -> str | None:
  """Detect SUM/MAX/MIN/PROD from UOp op counts (axis row/col path)."""
  nloads = op_counts.get("LOAD", 0)
  if op_counts.get("ADD", 0) > nloads - 1 and op_counts.get("MAX", 0) == 0:
    return "SUM"
  if (op_counts.get("MAX", 0) >= nloads - 1 and op_counts.get("MAX", 0) > 0
      and op_counts.get("XOR", 0) == 0):
    return "MAX"
  if (op_counts.get("MAX", 0) >= nloads - 1 and op_counts.get("MAX", 0) > 0
      and op_counts.get("XOR", 0) > 0):
    return "MIN"
  if data_alu is not None:
    data_mul = data_alu.get("MUL", 0)
    if data_mul > 0 and data_alu.get("ADD", 0) == 0 and data_alu.get("MAX", 0) == 0:
      return "PROD"
  return None


# ---------------------------------------------------------------------------
# Reduction classification — shared by is_reduction() and lower_reduction().
# A reduction "plan" is a small dict; None means "not a reduction we handle".
# ---------------------------------------------------------------------------
def _classify_reduction(uops: list[UOp]) -> dict | None:
  """Classify a kernel as a scalar / row / column reduction, or None.

  Returns a plan dict with: axis ('scalar'|'row'|'col'), out_arg, src_arg,
  out_size, src_size, reduce_op ('SUM'|'MAX'|'MIN'|'PROD'), is_float,
  src_is_bool, post_op (None | ('ADD'|'MUL', const)). None when the kernel is
  not a reduction the legacy recognizers accept.
  """
  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  all_params = [u for u in uops if u.op is Ops.PARAM]
  # Exactly two PtrDType params and no non-pointer params.
  if len(params) != 2 or len(all_params) != 2:
    return None
  if 0 not in params:
    return None
  out_arg = 0
  src_arg = next((k for k in params if k != 0), None)
  if src_arg is None:
    return None
  out_size = params[out_arg].dtype.size
  src_size = params[src_arg].dtype.size
  if not (src_size > out_size and out_size >= 1):
    return None

  op_counts = Counter(u.op.name for u in uops)
  if op_counts.get("STORE", 0) != 1:
    return None
  is_float = any("float" in str(params[p].dtype) for p in params)

  if out_size == 1:
    return _classify_scalar(uops, op_counts, params, out_arg, src_arg,
                            out_size, src_size, is_float)
  # Row / column reductions share the divisibility + single-RANGE shape.
  if src_size % out_size != 0:
    return None
  if op_counts.get("RANGE", 0) != 1:
    return None
  return _classify_axis(uops, op_counts, params, out_arg, src_arg,
                        out_size, src_size, is_float)


def _classify_scalar(uops, op_counts, params, out_arg, src_arg,
                      out_size, src_size, is_float) -> dict | None:
  """Scalar reduction (out_size == 1). Mirrors _render_reduction_sxu_program."""
  # Reject trivial 1->1 kernels (e.g. Tensor([5])+1): nothing to reduce.
  if src_size <= 1:
    return None

  has_add = op_counts.get("ADD", 0) > 0
  has_max = op_counts.get("MAX", 0) > 0
  has_xor = op_counts.get("XOR", 0) > 0

  # Reject pre-reduction transcendental / conditional ops the reducer would
  # silently drop (giving wrong results).
  if any(op_counts.get(n, 0) for n in ("EXP2", "LOG2", "SIN", "SQRT", "RECIPROCAL")):
    return None
  if any(op_counts.get(n, 0) > 0 for n in ("WHERE", "CMPLT", "CMPEQ", "CMPNE")):
    return None

  # Integer MIN decomposes to XOR+MAX; float MIN decomposes via negation.
  if is_float and has_xor:
    return None
  is_float_min = (is_float and has_max
                  and _data_alu_ops(uops).get("MUL", 0) > 0
                  and _is_float_min_negation(uops))
  if (is_float and has_max
      and _data_alu_ops(uops).get("MUL", 0) > 0
      and not is_float_min):
    return None

  # Post-reduction scalar op: ADD(tree, CONST) / MUL(tree, CONST).
  post_op = None
  store = next(u for u in uops if u.op is Ops.STORE)
  val = store.src[1]
  if val.op in (Ops.ADD, Ops.MUL):
    const_src = next((s for s in val.src if s.op is Ops.CONST
                      and not isinstance(s.arg, bool)), None)
    tree_src = next((s for s in val.src if s is not const_src), None)
    if const_src is not None and tree_src is not None and _has_load_src(tree_src):
      const_data_uses = sum(1 for u in uops
                            if any(s is const_src for s in u.src)
                            and u.op is not Ops.INDEX)
      if const_data_uses == 1 and val.op is Ops.ADD:
        post_op = ("ADD", const_src.arg)
      elif const_data_uses == 1 and val.op is Ops.MUL:
        post_op = ("MUL", const_src.arg)
  # An outer ADD(reduce, const) is the post-op, not part of the reduction.
  if post_op is not None and post_op[0] == "ADD":
    has_add = (op_counts.get("ADD", 0) - 1) > 0

  data_alu = _data_alu_ops(uops)
  has_data_mul = data_alu.get("MUL", 0) > 0
  # Reject SUM/MAX with un-foldable pre-reduction MULs (sum(x*x), sum(-x)).
  if has_data_mul and has_add and post_op is None:
    return None
  if has_data_mul and has_max and not is_float_min and post_op is None:
    return None

  # Classify the reduce op (mirrors the legacy if/elif ladder exactly).
  if is_float and has_data_mul and not has_add and not has_max and not is_float_min:
    reduce_op = "PROD"
  elif is_float_min:
    reduce_op = "MIN"
  elif is_float and has_add and not has_max:
    reduce_op = "SUM"
  elif is_float and has_max and not has_xor:
    reduce_op = "MAX"
  elif has_data_mul and not has_add and not has_max:
    reduce_op = "PROD"
  elif has_add and not has_max:
    reduce_op = "SUM"
  elif has_max and not has_xor:
    reduce_op = "MAX"
  elif has_max and has_xor:
    reduce_op = "MIN"
  else:
    return None

  src_is_bool = (params[src_arg].dtype.base.itemsize == 1
                 and "bool" in str(params[src_arg].dtype))
  return {"axis": "scalar", "out_arg": out_arg, "src_arg": src_arg,
          "out_size": out_size, "src_size": src_size, "reduce_op": reduce_op,
          "is_float": is_float, "is_float_min": is_float_min,
          "src_is_bool": src_is_bool, "post_op": post_op}


def _classify_axis(uops, op_counts, params, out_arg, src_arg,
                    out_size, src_size, is_float) -> dict | None:
  """Row / column reduction (out_size > 1). Mirrors the row/col recognizers."""
  data_alu = _data_alu_ops(uops)
  nloads = op_counts.get("LOAD", 0)
  total_mul = op_counts.get("MUL", 0)
  data_mul = data_alu.get("MUL", 0)

  # Distinguish row vs column reduction. Row-reduce has exactly one stride
  # multiply in the index path (total MUL >= 1, index MUL > data MUL);
  # col-reduce has none (total MUL == data MUL).
  is_col = (total_mul == data_mul)
  is_row = (total_mul >= 1 and total_mul != data_mul)
  if not (is_col or is_row):
    return None

  reduce_op = _detect_reduce_op(op_counts, data_alu)
  if reduce_op is None:
    return None

  if is_col:
    ncols = out_size
    nrows = src_size // ncols
    # Col-reduce requires out_size > 1 (a 1-col output is a scalar reduce).
    if not (out_size > 1):
      return None
  else:
    nrows = out_size
    ncols = src_size // nrows
    if ncols < 2:
      return None
    # Tinygrad may fully unroll or keep a RANGE loop; match both.
    if nloads != ncols and nloads != 1:
      return None

  # Float MIN via negation-decomposition rewrites a MAX classification.
  if is_float:
    if reduce_op == "MAX" and data_mul > 0:
      if _is_float_min_negation(uops):
        reduce_op = "MIN"
      else:
        return None

  # Post-reduction scalar mul for float SUM (mean = sum * (1/N)).
  post_op = None
  if is_float and reduce_op == "SUM":
    for u in uops:
      if u.op is Ops.MUL and _has_load_src(u):
        cst = next((s for s in u.src if s.op is Ops.CONST and isinstance(s.arg, float)), None)
        if cst is not None:
          post_op = ("MUL", float(cst.arg))
          break

  # Reject un-foldable pre-reduction data-path MULs (sum(x*x, axis), max(-x)).
  if data_mul > 0 and post_op is None and reduce_op in ("SUM", "MAX"):
    return None

  return {"axis": "col" if is_col else "row", "out_arg": out_arg,
          "src_arg": src_arg, "out_size": out_size, "src_size": src_size,
          "reduce_op": reduce_op, "is_float": is_float, "nrows": nrows,
          "ncols": ncols, "post_op": post_op}


# ---------------------------------------------------------------------------
# Reduce-op opcode tables
# ---------------------------------------------------------------------------
def _scalar_opcodes(reduce_op: str, is_float: bool, is_float_min: bool):
  """(tile_reduce_op, combine_op, pad_value, reduce_tag) for a scalar reduce."""
  if is_float and reduce_op == "PROD":
    return _VPU["FPROD_REDUCE_TILE"], _VPU["FMUL"], _FLOAT_ONE_BITS, "prod"
  if is_float_min:
    return _VPU["FMIN_REDUCE_TILE"], _VPU["FMIN"], _FLOAT_POS_INF_BITS, "min"
  if is_float and reduce_op == "SUM":
    # +0.0 has bit pattern 0x00000000 — same as integer zero pad.
    return _VPU["FSUM_REDUCE_TILE"], _VPU["FADD"], 0, "sum"
  if is_float and reduce_op == "MAX":
    return _VPU["FMAX_REDUCE_TILE"], _VPU["FMAX"], _FLOAT_NEG_INF_BITS, "max"
  if reduce_op == "PROD":
    return _VPU["MUL_REDUCE_TILE"], _VPU["MUL"], 1, "prod"
  if reduce_op == "SUM":
    return _VPU["SUM_REDUCE_TILE"], _VPU["ADD"], 0, "sum"
  if reduce_op == "MAX":
    return _VPU["MAX_REDUCE_TILE"], _VPU["MAX"], _INT32_MIN, "max"
  # integer MIN
  return _VPU["MIN_REDUCE_TILE"], _VPU["MIN"], _INT32_MAX, "min"


_AXIS_OPCODES_INT = {
    "row": {"SUM":  (_VPU["SUM_REDUCE"], _VPU["ADD"], 0),
            "MAX":  (_VPU["MAX_REDUCE"], _VPU["MAX"], _INT32_MIN),
            "MIN":  (_VPU["MIN_REDUCE"], _VPU["MIN"], _INT32_MAX),
            "PROD": (_VPU["MUL_REDUCE"], _VPU["MUL"], 1)},
    "col": {"SUM":  (_VPU["SUM_REDUCE_COL"], _VPU["ADD"], 0),
            "MAX":  (_VPU["MAX_REDUCE_COL"], _VPU["MAX"], _INT32_MIN),
            "MIN":  (_VPU["MIN_REDUCE_COL"], _VPU["MIN"], _INT32_MAX),
            "PROD": (_VPU["MUL_REDUCE_COL"], _VPU["MUL"], 1)},
}
_AXIS_OPCODES_FLOAT = {
    "row": {"SUM":  (_VPU["FSUM_REDUCE"], _VPU["FADD"], 0),
            "MAX":  (_VPU["FMAX_REDUCE"], _VPU["FMAX"], _FLOAT_NEG_INF_BITS),
            "MIN":  (_VPU["FMIN_REDUCE"], _VPU["FMIN"], _FLOAT_POS_INF_BITS),
            "PROD": (_VPU["FPROD_REDUCE"], _VPU["FMUL"], _FLOAT_ONE_BITS)},
    "col": {"SUM":  (_VPU["FSUM_REDUCE_COL"], _VPU["FADD"], 0),
            "MAX":  (_VPU["FMAX_REDUCE_COL"], _VPU["FMAX"], _FLOAT_NEG_INF_BITS),
            "MIN":  (_VPU["FMIN_REDUCE_COL"], _VPU["FMIN"], _FLOAT_POS_INF_BITS),
            "PROD": (_VPU["FPROD_REDUCE_COL"], _VPU["FMUL"], _FLOAT_ONE_BITS)},
}


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------
def _emit_scalar(plan: dict) -> dict:
  """Emit a scalar (out_size == 1) reduction SXU_PROGRAM descriptor."""
  src_arg, out_arg = plan["src_arg"], plan["out_arg"]
  src_size = plan["src_size"]
  vpu_op, combine_op, pad_value, reduce_tag = _scalar_opcodes(
      plan["reduce_op"], plan["is_float"], plan["is_float_min"])
  is_float_reduce = vpu_op in (_VPU["FSUM_REDUCE_TILE"], _VPU["FMAX_REDUCE_TILE"],
                               _VPU["FMIN_REDUCE_TILE"], _VPU["FPROD_REDUCE_TILE"])

  num_tiles = (src_size + _TILE_ELEMS - 1) // _TILE_ELEMS
  all_instrs: list[str] = []
  data_plan: list[dict] = []

  for tile_idx in range(num_tiles):
    offset = tile_idx * _TILE_ELEMS
    count = min(_TILE_ELEMS, src_size - offset)
    vmem_addr = tile_idx
    entry = {"type": "VMEM", "addr": vmem_addr, "param": src_arg,
             "offset": offset, "count": count, "dtype": "int32"}
    if plan["src_is_bool"]:
      entry["bool"] = True
    if pad_value != 0:
      entry["pad_value"] = pad_value
    data_plan.append(entry)
    src_vreg = tile_idx * 2
    dst_vreg = tile_idx * 2 + 1
    all_instrs.append(_load(src_vreg, vmem_addr))
    all_instrs.append(_vpu(dst_vreg, src_vreg, vpu_op))

  if num_tiles > 1:
    acc_vreg = 1
    for tile_idx in range(1, num_tiles):
      tile_result_vreg = tile_idx * 2 + 1
      next_vreg = num_tiles * 2 + tile_idx
      all_instrs.append(_vpu(next_vreg, acc_vreg, combine_op, tile_result_vreg))
      acc_vreg = next_vreg
  else:
    acc_vreg = 1

  post_op = plan["post_op"]
  if post_op is not None:
    op_name, const_val = post_op
    c_bits = _const_bits(const_val)
    const_addr = num_tiles
    data_plan.append({"type": "VMEM", "addr": const_addr, "layout": "broadcast_const",
                      "value": c_bits, "count": _TILE_ELEMS, "dtype": "int32"})
    const_vreg = num_tiles * 2 + 10
    result_vreg = const_vreg + 1
    all_instrs.append(_load(const_vreg, const_addr))
    if is_float_reduce:
      post_vpu = _VPU["FADD"] if op_name == "ADD" else _VPU["FMUL"]
    else:
      post_vpu = _VPU["ADD"] if op_name == "ADD" else _VPU["MUL"]
    all_instrs.append(_vpu(result_vreg, acc_vreg, post_vpu, const_vreg))
    acc_vreg = result_vreg
    out_vmem = num_tiles + 1
  else:
    out_vmem = num_tiles if num_tiles > 1 else 1

  all_instrs.append(_store(out_vmem, acc_vreg))
  all_instrs.append(_halt())
  outputs = [{"addr": out_vmem, "param": out_arg, "offset": 0, "count": 1}]

  return {
      "op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
      "outputs": outputs, "num_output_tiles": 1, "out": out_arg,
      "reduce": reduce_tag,
  }


def _emit_axis(plan: dict) -> dict:
  """Emit a row / column reduction SXU_PROGRAM descriptor."""
  axis = plan["axis"]
  src_arg, out_arg = plan["src_arg"], plan["out_arg"]
  nrows, ncols = plan["nrows"], plan["ncols"]
  table = _AXIS_OPCODES_FLOAT if plan["is_float"] else _AXIS_OPCODES_INT
  vpu_op, combine_op, pad_value = table[axis][plan["reduce_op"]]

  num_row_tiles = (nrows + _ROWS - 1) // _ROWS
  num_col_tiles = (ncols + _COLS - 1) // _COLS
  post_op = plan["post_op"]

  data_plan: list[dict] = []
  all_instrs: list[str] = []
  outputs: list[dict] = []
  src_addr = 0
  vreg = 0

  # reserved_slots: post-reduction const sits one past the src+output tiles.
  if axis == "col":
    reserved_slots = num_col_tiles * (num_row_tiles + 1)
  else:
    reserved_slots = num_row_tiles * (num_col_tiles + 1)

  post_const_vreg = None
  if post_op is not None:
    post_const_addr = reserved_slots
    c_bits = _const_bits(post_op[1])
    data_plan.append({"type": "VMEM", "addr": post_const_addr,
                      "layout": "broadcast_const", "value": c_bits,
                      "count": _TILE_ELEMS, "dtype": "int32"})
    post_const_vreg = vreg; vreg += 1
    all_instrs.append(_load(post_const_vreg, post_const_addr))

  # Iteration order: col-reduce iterates col-tiles (outer) x row-tiles;
  # row-reduce iterates row-tiles (outer) x col-tiles. Within an outer tile
  # the per-inner-tile partial reductions are combined.
  if axis == "col":
    outer_count, inner_count = num_col_tiles, num_row_tiles
  else:
    outer_count, inner_count = num_row_tiles, num_col_tiles

  for outer in range(outer_count):
    per_tile_red_vregs: list[int] = []
    if axis == "col":
      col_base = outer * _COLS
      tile_cols = min(_COLS, ncols - col_base)
    else:
      row_base = outer * _ROWS
      tile_rows = min(_ROWS, nrows - row_base)
    for inner in range(inner_count):
      if axis == "col":
        row_base = inner * _ROWS
        tile_rows = min(_ROWS, nrows - row_base)
      else:
        col_base = inner * _COLS
        tile_cols = min(_COLS, ncols - col_base)
      entry = {
          "type": "VMEM", "addr": src_addr, "param": src_arg,
          "mode": "MATRIX_TILE", "matrix_nrows": nrows, "matrix_ncols": ncols,
          "row_base": row_base, "col_base": col_base,
          "tile_rows": tile_rows, "tile_cols": tile_cols,
          "offset": 0, "count": tile_rows * tile_cols, "dtype": "int32",
      }
      if pad_value != 0:
        entry["pad_value"] = pad_value
      data_plan.append(entry)
      src_vreg = vreg; vreg += 1
      red_vreg = vreg; vreg += 1
      all_instrs.append(_load(src_vreg, src_addr))
      all_instrs.append(_vpu(red_vreg, src_vreg, vpu_op))
      per_tile_red_vregs.append(red_vreg)
      src_addr += 1
    # Combine per-inner-tile partial reductions.
    if len(per_tile_red_vregs) == 1:
      final_vreg = per_tile_red_vregs[0]
    else:
      acc = per_tile_red_vregs[0]
      for nxt in per_tile_red_vregs[1:]:
        out_vreg = vreg; vreg += 1
        all_instrs.append(_vpu(out_vreg, acc, combine_op, nxt))
        acc = out_vreg
      final_vreg = acc
    # Post-reduction scalar op (currently only FMUL).
    if post_op is not None:
      scaled_vreg = vreg; vreg += 1
      post_vpu = _VPU["FMUL"] if post_op[0] == "MUL" else _VPU["FADD"]
      all_instrs.append(_vpu(scaled_vreg, final_vreg, post_vpu, post_const_vreg))
      final_vreg = scaled_vreg
    out_addr = src_addr; src_addr += 1
    all_instrs.append(_store(out_addr, final_vreg))
    if axis == "col":
      outputs.append({"addr": out_addr, "param": out_arg,
                      "offset": col_base, "count": tile_cols})
    else:
      outputs.append({"addr": out_addr, "param": out_arg,
                      "offset": row_base, "count": tile_rows,
                      "extract": "row_heads"})

  all_instrs.append(_halt())
  # The "reduce" tag is intentionally omitted for axis descriptors: the legacy
  # row/col recognizers emit no such tag (only the scalar path carries it).
  return {
      "op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
      "outputs": outputs, "num_output_tiles": outer_count, "out": out_arg,
  }


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
def is_reduction(uops: list[UOp]) -> bool:
  """True if the kernel is a reduction the lowerer handles.

  Positive predicate: exactly two PtrDType params, output size < the input
  param size, the single STORE's value is an associative-op tree
  (ADD/MAX/MUL/XOR) over loads of the input, with no transcendental / WHERE /
  compare op on the data path.
  """
  return _classify_reduction(uops) is not None


def lower_reduction(uops: list[UOp]) -> dict:
  """Lower a reduction kernel to an SXU_PROGRAM descriptor.

  Caller must have confirmed ``is_reduction(uops)`` is True (the classifier
  routes REDUCTION kernels here).
  """
  plan = _classify_reduction(uops)
  assert plan is not None, "lower_reduction called on a non-reduction kernel"
  if plan["axis"] == "scalar":
    return _emit_scalar(plan)
  return _emit_axis(plan)
