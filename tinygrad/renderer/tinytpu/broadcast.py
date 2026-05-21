"""TinyTPU broadcast lowerer.

Consolidates the three legacy structural recognizers
(`_render_rowbc_sxu_program`, `_render_colbc_sxu_program`,
`_render_colbc_where_sxu_program`) into one classify-then-emit lowerer.

A broadcast kernel has three pointer params: one output, one "full" operand
whose size equals the output, and one smaller operand replicated along an
axis into the output. The lowerer classifies the replication axis from the
smaller operand's index relationship to the output ranges:

  * row broadcast — a length-`ncols` operand replicated down every row;
  * column broadcast — a length-`nrows` operand replicated across every
    column;
  * column-broadcast-where — a column broadcast feeding a
    ``WHERE(full < col_bcast, full, full * const)`` select.

The behavioral spec is the three legacy recognizers; this lowerer must
produce descriptors identical to them for every kernel they accept.
"""
from __future__ import annotations
from collections import Counter
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import PtrDType
# Shared infrastructure — one opcode table / geometry / encoders / graph helpers.
from tinygrad.renderer.tinytpu.common import (
  _ROWS, _COLS, _TILE_ELEMS, _VPU,
  _load, _store, _vpu, _select, _broadcast_row, _broadcast_col, _halt,
  _has_load_src)

# VPU op codes that produce a boolean (0/1) tile.
_VPU_BOOL_OPS = {_VPU["CMPLT"], _VPU["CMPNE"], _VPU["CMPEQ"]}
# Integer VPU op name -> float variant, used when operands are float.
_FLOAT_REMAP = {"ADD": "FADD", "SUB": "FSUB", "MUL": "FMUL",
                "MAX": "FMAX", "MIN": "FMIN", "CMPLT": "FCMPLT"}


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
def _unique_param_arg(u: UOp) -> int | None:
  """The single PARAM arg reachable from u, or None if not unique."""
  args = {n.arg for n in u.toposort() if n.op is Ops.PARAM}
  if len(args) != 1:
    return None
  arg = next(iter(args))
  return arg if isinstance(arg, int) else None


def _classify_broadcast_axis(uops: list[UOp], param_arg: int) -> str | None:
  """Infer row/column broadcast orientation for a short 2D operand."""
  # TODO(InstSel): faithful legacy port — the chunk==1 address-arithmetic
  # heuristic below should later be replaced with a principled index-relationship
  # check between the bias index and the store index.
  param_indices = [u.src[1] for u in uops
                   if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and u.src[0].arg == param_arg]
  if not param_indices:
    return None
  if all(idx.op is Ops.RANGE for idx in param_indices):
    return "col"
  if all(idx.op is Ops.CONST for idx in param_indices):
    vals = [int(idx.arg) for idx in param_indices]
    uniq = sorted(set(vals))
    if uniq and len(vals) % len(uniq) == 0:
      chunk = len(vals) // len(uniq)
      if chunk > 1 and vals == [v for u in uniq for v in [u] * chunk]:
        return "col"
      if chunk > 1 and vals == uniq * chunk:
        return "row"
    # chunk==1 (all-unique param indices) is ambiguous from val ordering
    # alone. Correlate each bias CONST with the store addresses that
    # consume it: bias[k] used at addrs satisfying addr % bias_size == k
    # is a row broadcast; addr // stride == k is a col broadcast.
    bias_size = len(uniq) if uniq else 0
    idx_uops = [u for u in uops if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM
                and u.src[0].arg == param_arg and u.src[1].op is Ops.CONST]
    stores = [s for s in uops if s.op is Ops.STORE and s.src[0].op is Ops.INDEX
              and s.src[0].src[1].op is Ops.CONST]
    def _depends(node: UOp, target: UOp) -> bool:
      # UOp is hashable and supports .toposort(); membership is exact.
      return target in node.toposort()
    pairs: list[tuple[int, int]] = []  # (bias_idx, output_addr)
    for idx_uop in idx_uops:
      k = int(idx_uop.src[1].arg)
      for s in stores:
        if _depends(s.src[1], idx_uop):
          pairs.append((k, int(s.src[0].src[1].arg)))
    if bias_size > 0 and pairs:
      row_ok = all(a % bias_size == k for k, a in pairs)
      all_addrs = [a for _, a in pairs]
      col_stride_candidates = {a // k for k, a in pairs if k != 0 and a >= k and a % k == 0}
      col_stride_candidates |= {(max(all_addrs) + 1) // bias_size} if all_addrs else set()
      col_ok = any(stride > 0 and all(a // stride == k for k, a in pairs) for stride in col_stride_candidates)
      if row_ok and not col_ok:
        return "row"
      if col_ok and not row_ok:
        return "col"
    return "row"

  store = next((u for u in uops if u.op is Ops.STORE and u.src[0].op is Ops.INDEX), None)
  if store is None:
    return None
  out_idx = store.src[0].src[1]
  row_range = col_range = None
  if out_idx.op is Ops.ADD:
    for src in out_idx.src:
      if src.op is Ops.MUL:
        row_range = next((s for s in src.src if s.op is Ops.RANGE), None)
      elif src.op is Ops.RANGE:
        col_range = src
  if row_range is None or col_range is None:
    return None

  uses_row = any(idx is row_range for idx in param_indices)
  uses_col = any(idx is col_range for idx in param_indices)
  if uses_row and not uses_col:
    return "col"
  if uses_col and not uses_row:
    return "row"
  return None


# ---------------------------------------------------------------------------
# Broadcast classification — shared by is_broadcast() and lower_broadcast().
# A broadcast "plan" is a small dict; None means "not a broadcast we handle".
# ---------------------------------------------------------------------------
def _classify_colbc_where(uops: list[UOp]) -> dict | None:
  """Classify WHERE(full < col_bcast, full, full * const), or None.

  Mirrors `_render_colbc_where_sxu_program`.
  """
  op_counts = Counter(u.op.name for u in uops)
  if op_counts.get("WHERE", 0) == 0 or op_counts.get("CMPLT", 0) == 0 or op_counts.get("MUL", 0) == 0:
    return None

  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  if len(params) != 3:
    return None
  out_arg = 0
  out_size = params[out_arg].dtype.size
  if out_size <= 0 or out_size > _TILE_ELEMS:
    return None

  input_args = sorted(arg for arg in params if arg != out_arg)
  full_args = [arg for arg in input_args if params[arg].dtype.size == out_size]
  col_args = [arg for arg in input_args
              if 0 < params[arg].dtype.size < out_size
              and params[arg].dtype.size <= _ROWS
              and _classify_broadcast_axis(uops, arg) == "col"]
  if len(full_args) != 1 or len(col_args) != 1:
    return None
  full_arg, col_arg = full_args[0], col_args[0]

  where_uops = [u for u in uops if u.op is Ops.WHERE]
  if not where_uops:
    return None

  mul_consts: set[int] = set()
  for where_uop in where_uops:
    if len(where_uop.src) != 3:
      return None
    cond_uop, true_uop, false_uop = where_uop.src
    if cond_uop.op is not Ops.CMPLT or true_uop.op is not Ops.LOAD or false_uop.op is not Ops.MUL:
      return None
    if _unique_param_arg(true_uop) != full_arg:
      return None
    mul_loads = [src for src in false_uop.src if src.op is Ops.LOAD]
    mul_const_nodes = [src for src in false_uop.src if src.op is Ops.CONST and isinstance(src.arg, int)]
    if len(mul_loads) != 1 or len(mul_const_nodes) != 1 or _unique_param_arg(mul_loads[0]) != full_arg:
      return None
    mul_consts.add(int(mul_const_nodes[0].arg))

    cmplt_full = [src for src in cond_uop.src if src.op is Ops.LOAD and _unique_param_arg(src) == full_arg]
    cmplt_col = [src for src in cond_uop.src if src.op is Ops.LOAD and _unique_param_arg(src) == col_arg]
    if len(cmplt_full) != 1 or len(cmplt_col) != 1:
      return None

  if len(mul_consts) != 1:
    return None
  return {"kind": "colbc_where", "out_arg": out_arg, "full_arg": full_arg,
          "col_arg": col_arg, "out_size": out_size,
          "col_size": params[col_arg].dtype.size,
          "mul_const": next(iter(mul_consts))}


def _classify_colbc(uops: list[UOp]) -> dict | None:
  """Classify a single-tile 2D column-broadcast binary op, or None.

  Mirrors `_render_colbc_sxu_program`.
  """
  op_counts = Counter(u.op.name for u in uops)
  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  if len(params) != 3:
    return None

  out_arg = 0
  out_size = params[out_arg].dtype.size
  input_args = sorted(arg for arg in params if arg != out_arg)
  if out_size <= 0 or len(input_args) != 2 or out_size > _TILE_ELEMS:
    return None

  full_args = [arg for arg in input_args if params[arg].dtype.size == out_size]
  col_args = [arg for arg in input_args
              if 0 < params[arg].dtype.size < out_size
              and params[arg].dtype.size <= _ROWS
              and _classify_broadcast_axis(uops, arg) == "col"]
  if len(full_args) != 1 or len(col_args) != 1:
    return None

  lhs_arg = full_args[0]
  rhs_arg = col_args[0]
  nrows = params[rhs_arg].dtype.size
  # A size-1 broadcast operand is a scalar broadcast, not a column broadcast.
  if nrows <= 1 or out_size % nrows != 0:
    return None
  ncols = out_size // nrows
  if ncols > _COLS:
    return None

  # Detect SUB lowered as ADD(a, MUL(b, -1)). op_counts alone would pick
  # MUL and silently emit the wrong kernel for col-broadcast subtract.
  neg_mul = next((u for u in uops if u.op is Ops.MUL and _has_load_src(u)
                  and any(s.op is Ops.CONST and s.arg == -1 for s in u.src)), None)
  neg_add = None
  if neg_mul is not None:
    neg_add = next((u for u in uops if u.op is Ops.ADD and _has_load_src(u)
                    and any(s is neg_mul for s in u.src)), None)

  col_is_lhs = False
  if neg_add is not None:
    vpu_name = "SUB"
    unneg_src = next(s for s in neg_add.src if s is not neg_mul)
    neg_src = next((s for s in neg_mul.src if s.op is not Ops.CONST), None)
    unneg_param = _unique_param_arg(unneg_src)
    neg_param = _unique_param_arg(neg_src) if neg_src is not None else None
    if unneg_param == rhs_arg and neg_param == lhs_arg:
      col_is_lhs = True
    elif unneg_param == lhs_arg and neg_param == rhs_arg:
      col_is_lhs = False
    else:
      return None
  else:
    non_comm_ops = {"CMPLT": Ops.CMPLT, "CMPNE": Ops.CMPNE, "SUB": Ops.SUB}
    vpu_name = None
    for name in ("CMPLT", "CMPNE", "CMPEQ", "MAX", "MIN", "SUB", "MUL", "ADD"):
      if op_counts.get(name, 0):
        vpu_name = name
        break
    if vpu_name is None:
      return None
    if vpu_name in non_comm_ops:
      op_uop = next((u for u in uops if u.op is non_comm_ops[vpu_name] and _has_load_src(u)), None)
      if op_uop is None:
        return None
      lhs_param = _unique_param_arg(op_uop.src[0])
      rhs_param = _unique_param_arg(op_uop.src[1])
      if lhs_param == rhs_arg and rhs_param == lhs_arg:
        col_is_lhs = True
      elif lhs_param == lhs_arg and rhs_param == rhs_arg:
        col_is_lhs = False
      else:
        return None

  is_float = any("float" in str(params[p].dtype) for p in (out_arg, lhs_arg, rhs_arg))
  if is_float and vpu_name in _FLOAT_REMAP:
    vpu_name = _FLOAT_REMAP[vpu_name]
  return {"kind": "colbc", "out_arg": out_arg, "lhs_arg": lhs_arg,
          "rhs_arg": rhs_arg, "out_size": out_size, "nrows": nrows,
          "vpu_name": vpu_name, "col_is_lhs": col_is_lhs}


def _classify_rowbc(uops: list[UOp]) -> dict | None:
  """Classify a row-broadcast binary op, or None.

  Mirrors `_render_rowbc_sxu_program`.
  """
  op_counts = Counter(u.op.name for u in uops)
  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  if len(params) != 3:
    return None

  has_bool_logic = op_counts.get("AND", 0) > 0 or op_counts.get("OR", 0) > 0 or op_counts.get("XOR", 0) > 0
  has_fused_cmp = op_counts.get("CMPLT", 0) > 0 and op_counts.get("WHERE", 0) > 0
  if not (op_counts.get("GROUP", 0) == 1 and op_counts.get("RANGE", 0) == 1 and
          op_counts.get("STORE", 0) == 4 and op_counts.get("LOAD", 0) == 8 and
          not has_bool_logic and not has_fused_cmp):
    return None

  if op_counts.get("CMPNE", 0):
    op_name = "CMPNE"
  elif op_counts.get("CMPLT", 0):
    op_name = "CMPLT"
  elif op_counts.get("MAX", 0):
    op_name = "MAX"
  elif op_counts.get("MUL", 0) > 1:
    op_name = "MUL"
  elif op_counts.get("ADD", 0):
    op_name = "ADD"
  else:
    return None

  out_arg = 0
  out_size = params[out_arg].dtype.size
  input_args = sorted(arg for arg in params if arg != out_arg)
  if out_size <= 0 or len(input_args) != 2:
    return None

  full_args = [arg for arg in input_args if params[arg].dtype.size == out_size]
  row_args = [arg for arg in input_args
              if 0 < params[arg].dtype.size < out_size
              and out_size % params[arg].dtype.size == 0
              and params[arg].dtype.size <= _TILE_ELEMS]
  if len(full_args) != 1 or len(row_args) != 1:
    return None

  lhs_arg = full_args[0]
  rhs_arg = row_args[0]
  ncols = params[rhs_arg].dtype.size
  nrows = out_size // ncols

  is_float = any("float" in str(params[p].dtype) for p in (out_arg, lhs_arg, rhs_arg))
  if is_float and op_name in _FLOAT_REMAP:
    op_name = _FLOAT_REMAP[op_name]
  return {"kind": "rowbc", "out_arg": out_arg, "lhs_arg": lhs_arg,
          "rhs_arg": rhs_arg, "out_size": out_size, "ncols": ncols,
          "nrows": nrows, "vpu_name": op_name}


def _classify_broadcast(uops: list[UOp]) -> dict | None:
  """Classify a broadcast kernel, or None.

  Order mirrors `_render_sxu_program`: the colbc-where shape (which also
  carries a column broadcast) is recognized before plain colbc; rowbc last.
  """
  if (plan := _classify_colbc_where(uops)) is not None:
    return plan
  if (plan := _classify_colbc(uops)) is not None:
    return plan
  return _classify_rowbc(uops)


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------
def _emit_colbc_where(plan: dict) -> dict:
  """Emit a column-broadcast-where SXU_PROGRAM descriptor."""
  out_arg, full_arg, col_arg = plan["out_arg"], plan["full_arg"], plan["col_arg"]
  out_size, col_size, mul_const = plan["out_size"], plan["col_size"], plan["mul_const"]
  data_plan: list[dict] = [
    {"type": "VMEM", "addr": 0, "param": col_arg, "offset": 0, "count": col_size, "dtype": "int32",
     "mode": "MATRIX_TILE", "matrix_nrows": col_size, "matrix_ncols": 1,
     "row_base": 0, "col_base": 0, "tile_rows": col_size, "tile_cols": 1},
    {"type": "VMEM", "addr": 1, "param": full_arg, "offset": 0, "count": out_size, "dtype": "int32"},
    {"type": "VMEM", "addr": 2, "layout": "broadcast_const", "value": mul_const, "count": out_size, "dtype": "int32"},
  ]
  instructions = [
    _load(0, 1),
    _load(1, 0),
    _broadcast_col(2, 1, 0),
    _vpu(3, 0, _VPU["CMPLT"], 2),
    _load(4, 2),
    _vpu(5, 0, _VPU["MUL"], 4),
    _select(6, 3, 0, 5),
    _store(3, 6),
    _halt(),
  ]
  outputs = [{"addr": 3, "param": out_arg, "offset": 0, "count": out_size}]
  return {
    "op": "SXU_PROGRAM",
    "primitive": "BROADCAST_COL_SELECT",
    "instructions": instructions,
    "data_plan": data_plan,
    "outputs": outputs,
    "num_output_tiles": 1,
    "out": out_arg,
  }


def _emit_colbc(plan: dict) -> dict:
  """Emit a single-tile column-broadcast SXU_PROGRAM descriptor."""
  out_arg, lhs_arg, rhs_arg = plan["out_arg"], plan["lhs_arg"], plan["rhs_arg"]
  out_size, nrows = plan["out_size"], plan["nrows"]
  vpu_op = _VPU[plan["vpu_name"]]
  data_plan: list[dict] = [
    {"type": "VMEM", "addr": 0, "param": rhs_arg, "offset": 0, "count": nrows, "dtype": "int32",
     "mode": "MATRIX_TILE", "matrix_nrows": nrows, "matrix_ncols": 1,
     "row_base": 0, "col_base": 0, "tile_rows": nrows, "tile_cols": 1},
    {"type": "VMEM", "addr": 1, "param": lhs_arg, "offset": 0, "count": out_size, "dtype": "int32"},
  ]
  va, vb = (2, 0) if plan["col_is_lhs"] else (0, 2)
  instructions = [
    _load(0, 1),
    _load(1, 0),
    _broadcast_col(2, 1, 0),
    _vpu(3, va, vpu_op, vb),
    _store(2, 3),
    _halt(),
  ]
  outputs = [{"addr": 2, "param": out_arg, "offset": 0, "count": out_size}]
  return {
    "op": "SXU_PROGRAM",
    "primitive": "BROADCAST_COL",
    "instructions": instructions,
    "data_plan": data_plan,
    "outputs": outputs,
    "num_output_tiles": 1,
    "out": out_arg,
    "bool_out": vpu_op in _VPU_BOOL_OPS,
  }


def _emit_rowbc(plan: dict) -> dict:
  """Emit a row-broadcast SXU_PROGRAM descriptor (one VMEM tile per row chunk)."""
  out_arg, lhs_arg, rhs_arg = plan["out_arg"], plan["lhs_arg"], plan["rhs_arg"]
  ncols, nrows = plan["ncols"], plan["nrows"]
  vpu_op = _VPU[plan["vpu_name"]]
  rhs_addr = 0

  data_plan: list[dict] = [{
    "type": "VMEM", "addr": rhs_addr, "param": rhs_arg,
    "offset": 0, "count": ncols, "dtype": "int32",
  }]
  instructions: list[str] = []
  outputs: list[dict] = []
  rows_per_tile = _ROWS
  num_chunks = (nrows + rows_per_tile - 1) // rows_per_tile
  out_base = 1 + num_chunks

  for chunk_idx in range(num_chunks):
    row = chunk_idx * rows_per_tile
    rows_this_chunk = min(rows_per_tile, nrows - row)
    lhs_addr = 1 + chunk_idx
    out_addr = out_base + chunk_idx
    offset = row * ncols
    chunk_count = rows_this_chunk * ncols
    data_plan.append({
      "type": "VMEM", "addr": lhs_addr, "param": lhs_arg,
      "offset": offset, "count": chunk_count, "dtype": "int32",
    })
    instructions += [
      _load(0, lhs_addr),
      _load(1, rhs_addr),
      _broadcast_row(2, 1, 0),
      _vpu(3, 0, vpu_op, 2),
      _store(out_addr, 3),
    ]
    outputs.append({
      "addr": out_addr, "param": out_arg,
      "offset": offset, "count": chunk_count,
    })

  instructions.append(_halt())
  return {
    "op": "SXU_PROGRAM",
    "primitive": "BROADCAST_ROW",
    "instructions": instructions,
    "data_plan": data_plan,
    "outputs": outputs,
    "num_output_tiles": num_chunks,
    "out": out_arg,
    "bool_out": vpu_op in _VPU_BOOL_OPS,
  }


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
def is_broadcast(uops: list[UOp]) -> bool:
  """True if the kernel is a broadcast the lowerer handles.

  Positive predicate: exactly three PtrDType params (output + a full operand
  + a smaller operand replicated along an axis), where the smaller operand's
  index relationship to the output ranges classifies the replication axis as
  a row, column, or column-broadcast-where pattern.
  """
  return _classify_broadcast(uops) is not None


def lower_broadcast(uops: list[UOp]) -> dict:
  """Lower a broadcast kernel to an SXU_PROGRAM descriptor.

  Caller must have confirmed ``is_broadcast(uops)`` is True (the classifier
  routes BROADCAST kernels here).
  """
  plan = _classify_broadcast(uops)
  assert plan is not None, "lower_broadcast called on a non-broadcast kernel"
  if plan["kind"] == "colbc_where":
    return _emit_colbc_where(plan)
  if plan["kind"] == "colbc":
    return _emit_colbc(plan)
  return _emit_rowbc(plan)
