"""TinyTPU movement lowerer (Branch B — renderer-side instruction selection).

Consolidates the three legacy structural recognizers
(`_render_pad_sxu_program`, `_render_transpose_sxu_program`,
`_render_rowbc_copy_sxu_program`) into one classify-then-emit lowerer.

Movement ops (pad / flip / non-affine permute, the 4x4 transpose, and
single-input row-broadcast copy) genuinely need renderer-side lowering:
rangeify dissolves the movement op itself, but the recognizers recover a
non-affine tile access pattern the SXU model cannot express generically
(transpose -> SXU_DISPATCH_XLU_TRANSPOSE; pad -> a PAD_FILL scatter data
plan; row-broadcast copy -> SXU_BROADCAST_ROW). They are real instruction
selection, relocated not deleted.

The behavioral spec is the three legacy recognizers; this lowerer produces
descriptors identical to them for every kernel they accept.
"""
from __future__ import annotations
from collections import Counter
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import PtrDType
# Shared infrastructure — one geometry / encoders for the package.
from tinygrad.runtime.support.tinytpu_lowering.common import (
  _COLS, _TILE_ELEMS, _ALU_TO_VPU,
  _load, _store, _halt, _find_unique_param_arg)
from tinygrad.runtime.support.tinytpu_lowering.elementwise import can_lower


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


# ---------------------------------------------------------------------------
# Pad / flip / non-affine permute  (legacy `_render_pad_sxu_program`)
# ---------------------------------------------------------------------------
def _lower_pad(uops: list[UOp]) -> dict | None:
  """Render single-tile unrolled movement kernels (PAD / FLIP / permute)
  as a LOAD/STORE with a PAD_FILL VMEM preload that scatters source
  positions into an arbitrary output layout, zero-filling positions
  whose STORE value is CONST(0).

  Pattern: N STOREs whose values are each either LOAD(src[i]) or
  CONST(0). No ALU ops. Covers PAD (mix of LOAD+CONST), FLIP (all
  LOADs, reversed indices), and unrolled non-affine permutations
  that the affine copy renderer rejects.
  """
  op_counts = Counter(u.op.name for u in uops)
  if not op_counts.get("STORE") or not op_counts.get("LOAD"):
    return None
  # Pure movement only: no data-path ALU, ternary, cast, or WMMA.
  data_alu = _data_alu_ops(uops)
  if sum(data_alu.values()) > 0:
    return None
  for n in ("WHERE", "MOD", "RECIP", "RECIPROCAL", "TRUNC", "WMMA", "MULACC",
            "SELECT", "CAST"):
    if op_counts.get(n, 0):
      return None

  stores = [u for u in uops if u.op is Ops.STORE]

  # Extract per-STORE info: (dst_pos, src_pos or None for zero-fill).
  def _const_tail(idx_uop):
    # Walk an INDEX chain and return the final integer CONST position.
    if idx_uop.op is not Ops.INDEX:
      return None
    last = idx_uop.src[-1]
    if last.op is Ops.CONST and isinstance(last.arg, int):
      return int(last.arg)
    return None

  def _load_src_pos(load_uop):
    if load_uop.op is not Ops.LOAD:
      return None
    return _const_tail(load_uop.src[0])

  pad_map: list[tuple[int, int | None]] = []  # (dst_pos, src_pos or None)
  out_params: set = set()
  src_params: set = set()
  for s in stores:
    dst = _const_tail(s.src[0])
    if dst is None:
      return None
    out_param = _find_unique_param_arg(s.src[0])
    out_params.add(out_param)
    val = s.src[1]
    if val.op is Ops.CONST:
      if val.arg != 0:
        return None  # only zero-pad for now
      pad_map.append((dst, None))
    elif val.op is Ops.LOAD:
      sp = _load_src_pos(val)
      if sp is None:
        return None
      pad_map.append((dst, sp))
      src_params.add(_find_unique_param_arg(val.src[0]))
    else:
      return None

  if len(out_params) != 1 or len(src_params) != 1:
    return None
  out_arg = next(iter(out_params))
  src_arg = next(iter(src_params))

  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  out_size = params[out_arg].dtype.size
  src_size = params[src_arg].dtype.size
  if out_size > _TILE_ELEMS or src_size > _TILE_ELEMS:
    return None  # single-tile only for now

  # Pad all CONST stores to 0 slot, LOAD stores to source. Preloaded VMEM
  # tile is the padded output; the SXU program copies it straight through.
  data_plan = [{
    "type": "VMEM", "addr": 0, "param": src_arg,
    "mode": "PAD_FILL", "pad_map": pad_map,
    "offset": 0, "count": _TILE_ELEMS, "dtype": "int32",
  }]
  all_instrs = [_load(0, 0), _store(1, 0), _halt()]
  outputs = [{"addr": 1, "param": out_arg, "offset": 0, "count": out_size}]
  return {
    "op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
    "outputs": outputs, "num_output_tiles": 1, "out": out_arg,
  }


# ---------------------------------------------------------------------------
# Single-input row-broadcast copy  (legacy `_render_rowbc_copy_sxu_program`)
# ---------------------------------------------------------------------------
def _lower_rowbc_copy(uops: list[UOp]) -> dict | None:
  """Render a single-input row-broadcast copy as LOAD + BROADCAST_ROW.

  Covers ``Tensor([[v0, .., vM]]).expand(N, M)``: a small source row (<=_COLS
  elements, no RANGE dependence) replicated down every output row. This is a
  structured broadcast, not a per-element map, so it stays a recognizer.
  """
  op_counts = Counter(u.op.name for u in uops)
  if not op_counts.get("STORE") or not op_counts.get("LOAD"):
    return None
  if sum(_data_alu_ops(uops).values()) > 0:
    return None
  for n in ("WHERE", "MOD", "RECIP", "RECIPROCAL", "TRUNC", "WMMA", "MULACC",
            "SELECT", "CAST"):
    if op_counts.get(n, 0):
      return None

  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  if len(params) != 2:
    return None
  stores = [u for u in uops if u.op is Ops.STORE]
  out_params = {_find_unique_param_arg(s.src[0]) for s in stores}
  out_params.discard(None)
  if len(out_params) != 1:
    return None
  out_arg = next(iter(out_params))
  src_params = [k for k in params if k != out_arg]
  if len(src_params) != 1:
    return None
  src_arg = src_params[0]
  out_size = params[out_arg].dtype.size
  src_size = params[src_arg].dtype.size
  if out_size <= 0 or src_size <= 0:
    return None
  if params[out_arg].dtype.base.itemsize != params[src_arg].dtype.base.itemsize:
    return None

  def _index_of(addr_uop):
    return addr_uop.src[1] if addr_uop.op is Ops.INDEX else None

  def _has_range(u, seen=None):
    if seen is None: seen = set()
    if id(u) in seen: return False
    seen.add(id(u))
    if u.op is Ops.RANGE: return True
    return any(_has_range(s, seen) for s in u.src)

  load_idxs_all = [_index_of(s.src[1].src[0]) for s in stores if s.src[1].op is Ops.LOAD]
  if len(load_idxs_all) != len(stores):
    return None
  all_loads_no_range = all(li is not None and not _has_range(li) for li in load_idxs_all)
  # Row-broadcast: every LOAD index is range-independent, the output is a
  # multiple of a small source row. The InstSel walker rejects this — its
  # source-offset relation to the STORE index is not a constant.
  if not (all_loads_no_range and src_size <= _COLS and out_size % src_size == 0
          and out_size > src_size):
    return None

  num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
  data_plan = [{
    "type": "VMEM", "addr": 0, "param": src_arg,
    "offset": 0, "count": src_size, "dtype": "int32",
  }]
  all_instrs = [_load(0, 0), "2 10 0 1 0 0 0 0 0 0"]  # SXU_BROADCAST_ROW vd=1 vs=0 srcRow=0
  outputs = []
  for t in range(num_tiles):
    offset = t * _TILE_ELEMS
    count = min(_TILE_ELEMS, out_size - offset)
    out_addr = 1 + t
    all_instrs.append(_store(out_addr, 1))
    outputs.append({"addr": out_addr, "param": out_arg, "offset": offset, "count": count})
  all_instrs.append(_halt())
  return {
    "op": "SXU_PROGRAM", "instructions": all_instrs,
    "data_plan": data_plan, "outputs": outputs,
    "num_output_tiles": num_tiles, "out": out_arg,
  }


# ---------------------------------------------------------------------------
# 4x4 transpose  (legacy `_render_transpose_sxu_program`)
# ---------------------------------------------------------------------------
def _lower_transpose(uops: list[UOp]) -> dict | None:
  """Render a 4x4 2D permute(1,0) as LOAD + XLU_TRANSPOSE + STORE.

  Detects the GROUP-of-4-STOREs kernel tinygrad emits for contiguous
  .permute(1,0) over a 4x4 tile: one RANGE loop over rows, 4 STOREs per
  iteration. Each STORE writes a LOAD whose index transposes the store
  index. No data-path ALU ops.
  """
  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  if len(params) != 2:
    return None
  data_alu = _data_alu_ops(uops)
  if sum(data_alu.values()) > 0:
    return None
  op_counts = Counter(u.op.name for u in uops)
  # Two shapes match:
  #   - int32 GROUP(4)-unrolled: 4 STOREs + 4 LOADs + 1 RANGE + 1 MUL
  #   - float32 VECTORIZE-unrolled: 1 STORE + 4 LOADs + 1 VECTORIZE + 1 RANGE + 1 MUL
  shape_int = (op_counts.get("STORE", 0) == 4 and op_counts.get("LOAD", 0) == 4
               and op_counts.get("RANGE", 0) == 1 and op_counts.get("MUL", 0) == 1)
  shape_float = (op_counts.get("STORE", 0) == 1 and op_counts.get("LOAD", 0) == 4
                 and op_counts.get("VECTORIZE", 0) == 1 and op_counts.get("RANGE", 0) == 1
                 and op_counts.get("MUL", 0) == 1)
  if not (shape_int or shape_float):
    return None

  stores = [u for u in uops if u.op is Ops.STORE]
  out_params = {_find_unique_param_arg(s.src[0]) for s in stores}
  out_params.discard(None)
  if len(out_params) != 1:
    return None
  out_arg = next(iter(out_params))
  src_params = [k for k in params if k != out_arg]
  if len(src_params) != 1:
    return None
  src_arg = src_params[0]
  out_size = params[out_arg].dtype.size
  src_size = params[src_arg].dtype.size
  if out_size != 16 or src_size != 16:
    return None
  if params[out_arg].dtype.base.itemsize != params[src_arg].dtype.base.itemsize:
    return None
  # Each STORE value must be a LOAD of src_arg, or a VECTORIZE of LOADs
  # (possibly wrapped in CAST for float bitcasting).
  loads = []
  store_idx_uops = []
  for s in stores:
    val = s.src[1]
    # Unwrap CAST on the address side too (float store addresses get bitcast wrapped).
    addr = s.src[0]
    while addr.op is Ops.CAST:
      addr = addr.src[0]
    store_idx_uops.append(addr.src[1] if addr.op is Ops.INDEX else None)
    while val.op is Ops.CAST:
      val = val.src[0]
    if val.op is Ops.VECTORIZE:
      for l in val.src:
        while l.op is Ops.CAST:
          l = l.src[0]
        if l.op is not Ops.LOAD or _find_unique_param_arg(l) != src_arg:
          return None
        loads.append(l)
    elif val.op is Ops.LOAD:
      if _find_unique_param_arg(val) != src_arg:
        return None
      loads.append(val)
    else:
      return None
  if len(loads) != 4:
    return None

  # Distinguish transpose from reshape/copy by comparing CONST offsets in
  # load vs store index expressions. Reshape: matching const sets (e.g. 0/1/2/3
  # each side). Transpose: store consts are {0,1,2,3} while load consts are
  # multiples of row stride {0,4,8,12}.
  def _consts_in(u, seen=None):
    if seen is None: seen = set()
    if id(u) in seen: return []
    seen.add(id(u))
    if u.op is Ops.CONST and isinstance(u.arg, int):
      return [u.arg]
    out = []
    for s in u.src: out += _consts_in(s, seen)
    return out
  load_idx_uops = [l.src[0].src[1] if l.src[0].op is Ops.INDEX else None for l in loads]
  if any(u is None for u in store_idx_uops) or any(u is None for u in load_idx_uops):
    return None
  store_consts = sorted(set(sum([_consts_in(u) for u in store_idx_uops], [])))
  load_consts = sorted(set(sum([_consts_in(u) for u in load_idx_uops], [])))
  if not ({4, 8, 12} <= set(load_consts)):
    return None
  if len({id(u) for u in load_idx_uops}) != 4:
    return None
  if set(load_consts) == set(store_consts):
    return None

  # Emit: LOAD VMEM[0]→VREG 0, XLU_TRANSPOSE VREG 1 = transpose(VREG 0),
  # STORE VREG 1 → VMEM[1].
  data_plan = [{
    "type": "VMEM", "addr": 0, "param": src_arg,
    "offset": 0, "count": 16, "dtype": "int32",
  }]
  instructions = [
    _load(0, 0),
    f"2 12 0 1 0 0 0 0 0 0",  # SXU_DISPATCH_XLU_TRANSPOSE vd=1, vs=0
    _store(1, 1),
    _halt(),
  ]
  outputs = [{"addr": 1, "param": out_arg, "offset": 0, "count": 16}]
  return {
    "op": "SXU_PROGRAM",
    "primitive": "TRANSPOSE",
    "instructions": instructions,
    "data_plan": data_plan,
    "outputs": outputs,
    "num_output_tiles": 1,
    "out": out_arg,
  }


# ---------------------------------------------------------------------------
# Classifier + public surface
# ---------------------------------------------------------------------------
# Dispatch order mirrors the legacy `_render_sxu_program`: pad first, then
# transpose, then row-broadcast copy.
_MOVEMENT_LOWERERS = (_lower_pad, _lower_transpose, _lower_rowbc_copy)


def _classify_movement(uops: list[UOp]) -> dict | None:
  """Return the SXU_PROGRAM descriptor for the first matching movement
  recognizer, or None if no movement pattern applies.

  Mirrors the legacy dispatch order: in `render()` the three recognizers
  were a *fallback* reached only after `lower_kernel` (the elementwise
  walker) declined the kernel. So a kernel the InstSel walker can lower
  (`can_lower` True) — e.g. a scalar `expand` lowered as BROADCAST_SCALAR —
  never reached the recognizers and must not be claimed here either. The
  recognizer logic itself is unchanged; this is purely the legacy ordering
  gate, relocated."""
  if can_lower(uops):
    return None
  for lower in _MOVEMENT_LOWERERS:
    if (desc := lower(uops)) is not None:
      return desc
  return None


def is_movement(uops: list[UOp]) -> bool:
  """True for exactly the kernels the three movement recognizers handle:
  pad / flip / non-affine permute, the 4x4 transpose, and single-input
  row-broadcast copy."""
  return _classify_movement(uops) is not None


def lower_movement(uops: list[UOp]) -> dict:
  """Lower a movement kernel to its SXU_PROGRAM descriptor.

  Must be called only when `is_movement(uops)` is True.
  """
  desc = _classify_movement(uops)
  assert desc is not None, "lower_movement called on a non-movement kernel"
  return desc
