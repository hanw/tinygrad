"""TinyTPU elementwise lowerer.

Lowers int32/bool/float elementwise kernels to an SXU_PROGRAM descriptor via a
UOp-walking linear-scan register allocator.

Scope (a kernel the walker fully owns):
  - ALU / WHERE / unary-transcendental maps over equal-size operands;
  - degenerate maps — a bare ``LOAD`` (copy) or bare ``CONST`` (const-fill);
  - ``CAST`` value converts — int<->float via the ``I2F`` / ``F2I`` VPU ops,
    bool->int handled transparently in ``_canon``;
  - per-``LOAD`` constant source offsets, so a contiguous slice / shrink copy
    (``LOAD`` index shifted from the ``STORE`` index by a constant) is a copy.

Out of scope: transcendentals with no opcode, reductions, WMMA, and structured
row/column broadcast — those route to their own lowerers via ``classify``.
"""
from __future__ import annotations
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import PtrDType
from tinygrad.renderer.tinytpu.common import (
  _ALU_OPS, _ALU_TO_VPU, _FLOAT_VPU, _UNARY_VPU, _VPU, _DATA_OPS,
  _NUM_VREGS, _TILE_ELEMS,
  TpuInst, TpuKernel,
  _canon, _unique_param, _data_dag, _store_lanes, _float_operands,
  _const_bits, _cast_vpu, _run_instsel,
)

# ---------------------------------------------------------------------------
# LOAD index analysis — constant source offsets and scalar broadcasts
# ---------------------------------------------------------------------------
def _addr_index(node: UOp) -> UOp | None:
  """The element-index expression of a LOAD / STORE address.

  The address is an ``INDEX(PARAM, idx)``; a vectorized (float) kernel wraps it
  in a ``CAST`` to the vector pointer type, which is transparent here.
  """
  addr = node.src[0]
  while addr.op is Ops.CAST:
    addr = addr.src[0]
  return addr.src[1] if addr.op is Ops.INDEX and len(addr.src) > 1 else None

def _copy_offset(store: UOp) -> int | None:
  """Constant source offset of a degenerate copy (a ``STORE`` of a bare ``LOAD``).

  A copy's LOAD and STORE indices share the same RANGE structure, so their
  difference is a constant: 0 for a plain reshape-as-copy, K for a contiguous
  slice / shrink. Returns that constant, or None when the relationship is not
  a constant shift (transpose / flip / strided gather) — not a foldable copy.
  Only meaningful for a kernel with no interior op; an elementwise kernel's
  operands are always read at offset 0.
  """
  store_idx = _addr_index(store)
  load = _canon(store.src[1])
  if load.op is not Ops.LOAD or store_idx is None: return None
  load_idx = _addr_index(load)
  if load_idx is None: return None
  delta = (load_idx - store_idx).simplify()
  return int(delta.arg) if delta.op is Ops.CONST and isinstance(delta.arg, int) else None

# ---------------------------------------------------------------------------
# can_lower — positive predicate selecting walker-owned kernels
# ---------------------------------------------------------------------------
def can_lower(uops: list[UOp]) -> bool:
  """True iff the elementwise walker fully owns this kernel.

  Kernels arrive per-element unrolled (one STORE per lane) or vectorized
  (one STORE of a VECTORIZE). Every lane must compute the same data-DAG
  shape so one lane can template the tiled program.
  """
  if any(u.op is Ops.WMMA for u in uops): return False
  uops = _run_instsel(uops)
  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  stores = [u for u in uops if u.op is Ops.STORE]
  if not stores or not params: return False

  out_args = {_unique_param(s.src[0]) for s in stores}
  if len(out_args) != 1: return False
  out_arg = next(iter(out_args))
  if out_arg is None or out_arg not in params: return False
  out_size = params[out_arg].dtype.size
  if out_size <= 0: return False

  # A degenerate copy writes a bare LOAD from every STORE; a copy may read a
  # larger source (a contiguous slice), so the load-size check below is relaxed
  # only for copies. Everything else is a per-element map.
  copy_lanes = [_canon(lane) for s in stores for lane in _store_lanes(s)]
  is_copy = all(c.op is Ops.LOAD for c in copy_lanes)

  # A const-fill writes a bare CONST from every STORE. The walker owns the
  # single-tile fill (the shape tinygrad emits with literal-CONST indexing); a
  # fill needing RANGE-loop index arithmetic (MUL/ADD over a RANGE) stays
  # unsupported — it never had a renderer, and folding it would also silently
  # accept a zero-sized-contraction GEMM that must be reported instead.
  is_fill = all(c.op is Ops.CONST for c in copy_lanes)
  if is_fill and any(u.op in (Ops.MUL, Ops.ADD) for u in uops): return False

  # One lane templates the whole tiled program: every lane of every STORE must
  # compute the same data-DAG shape, and every node must be a leaf (LOAD/CONST)
  # or an interior op the VPU can issue. A bare-LOAD DAG is a copy and a bare
  # CONST DAG is a const-fill — both degenerate elementwise maps.
  ref_shape: tuple | None = None
  for s in stores:
    for lane in _store_lanes(s):
      nodes = _data_dag(lane)
      shape = tuple(n.op for n in nodes)
      if ref_shape is None: ref_shape = shape
      elif shape != ref_shape: return False              # non-uniform: not a plain elementwise map
      for n in nodes:
        if n.op is Ops.CAST:
          if _cast_vpu(n) is None: return False           # no per-element opcode
          continue
        if n.op in _DATA_OPS:
          # Float arithmetic ALU ops need an F-variant opcode. Float
          # CMPNE/CMPEQ are valid as integer bit-compares; transcendentals
          # and WHERE have their own opcodes and are exempt.
          if (n.op in _ALU_OPS and _float_operands(n)
              and n.op not in _FLOAT_VPU and n.op not in (Ops.CMPNE, Ops.CMPEQ)):
            return False
          continue
        if n.op is Ops.LOAD:
          p = _unique_param(n)
          if p is None or p not in params: return False
          # An interior-op kernel reads operands lane-for-lane: a LOAD buffer
          # must match the output size (or be a size-1 scalar broadcast). A
          # mismatched size is a gather / GEMM-as-elementwise the walker must
          # not own. A copy may legitimately read a larger sliced source.
          if not is_copy and params[p].dtype.size not in (out_size, 1): return False
          continue
        if n.op is Ops.CONST:
          continue
        return False   # unknown op (transcendental with no opcode, ...)

  # A copy from a full-size buffer may carry a constant source offset — a
  # contiguous slice / shrink. The offset must be a uniform constant across
  # STOREs; a non-constant relationship (transpose / flip / strided gather) is
  # a movement kernel the walker does not own. A size-1 copy source is a scalar
  # broadcast, not a slice, so its offset is not consulted. A negative offset is
  # rejected too: it preserves the legacy _render_copy_sxu_program guard, so the
  # walker never emits a negative VMEM source offset.
  if is_copy and not all(params[_unique_param(c)].dtype.size == 1 for c in copy_lanes):
    offs = {_copy_offset(s) for s in stores}
    if len(offs) != 1 or None in offs or next(iter(offs)) < 0: return False
  return True

# ---------------------------------------------------------------------------
# lower_kernel — the InstSel pass + UOp walker
# ---------------------------------------------------------------------------
def lower_kernel(uops: list[UOp]) -> dict:
  """Lower an elementwise kernel to an SXU_PROGRAM descriptor.

  Caller must have checked can_lower(uops) first.
  """
  uops = _run_instsel(uops)
  params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
  stores = [u for u in uops if u.op is Ops.STORE]
  out_arg = _unique_param(stores[0].src[0])
  out_param = params[out_arg]
  out_size = out_param.dtype.size
  out_is_bool = out_param.dtype.base.itemsize == 1

  # One lane templates the whole tiled program (can_lower proved uniformity).
  templ_store = stores[0]
  lane = _store_lanes(templ_store)[0]
  nodes = _data_dag(lane)
  leaves = [n for n in nodes if n.op in (Ops.LOAD, Ops.CONST)]
  interior = [n for n in nodes if n.op in _DATA_OPS]
  # A degenerate copy (no interior op) from a full-size buffer may carry a
  # constant source offset — a contiguous slice / shrink. A size-1 source is a
  # scalar broadcast (offset unused); elementwise kernels read at offset 0.
  copy_offset = (_copy_offset(templ_store) or 0) if not interior else 0

  num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
  addrs_per_tile = len(leaves) + 1

  # --- linear-scan register allocation over the per-lane DAG ---
  # seq is the emission order; a value's VREG is freed after its last use,
  # so a vreg is reused rather than burned monotonically.
  seq = leaves + interior
  pos = {n: i for i, n in enumerate(seq)}
  last_use: dict[UOp, int] = {}
  for node in interior:
    for s in node.src:
      c = _canon(s)
      if c in pos: last_use[c] = max(last_use.get(c, -1), pos[node])
  # The final value is consumed by the STORE; for a degenerate copy / fill the
  # DAG is a single leaf and that leaf is itself the stored value.
  result = interior[-1] if interior else leaves[-1]
  last_use[result] = len(seq)

  kern = TpuKernel(out_arg=out_arg, bool_out=out_is_bool)
  for tile_idx in range(num_tiles):
    offset = tile_idx * _TILE_ELEMS
    count = min(_TILE_ELEMS, out_size - offset)
    base = tile_idx * addrs_per_tile
    free = list(range(_NUM_VREGS - 1, -1, -1))   # pop() hands out low indices first
    vreg: dict[UOp, int] = {}

    for i, node in enumerate(seq):
      # free operand VREGs whose last use is this node, before allocating dst
      if node.op not in (Ops.LOAD, Ops.CONST):
        for s in node.src:
          c = _canon(s)
          if last_use.get(c) == i and vreg[c] not in free:
            free.append(vreg[c])
      if not free:
        raise RuntimeError("TinyTPU InstSel walker: kernel live set exceeds 16 VREGs")
      reg = free.pop()
      vreg[node] = reg

      if node.op is Ops.CONST:
        # A bare-CONST DAG is a const-fill. The legacy "primitive": "CONST_FILL"
        # diagnostic tag is intentionally not emitted — the walker only tags
        # architectural primitives, and no test depends on the CONST_FILL tag.
        kern.data_plan.append({"type": "VMEM", "addr": base + i, "layout": "broadcast_const",
                               "value": _const_bits(node.arg), "count": count, "dtype": "int32"})
        kern.instructions.append(TpuInst("LOAD", reg, (base + i,)))
      elif node.op is Ops.LOAD:
        p = _unique_param(node)
        is_bool = params[p].dtype.base.itemsize == 1
        if params[p].dtype.size == 1 and out_size > 1:
          # Size-1 buffer: load the single element and broadcast over the tile.
          entry = {"type": "VMEM", "addr": base + i, "param": p, "offset": 0,
                   "count": 1, "dtype": "int32"}
          if is_bool: entry["bool"] = True
          kern.data_plan.append(entry)
          kern.instructions.append(TpuInst("LOAD", reg, (base + i,)))
          kern.instructions.append(TpuInst("BROADCAST_SCALAR", reg, (reg,)))
          kern.primitives.add("BROADCAST_SCALAR")
        else:
          # Full-size load: read the tile, shifted by the constant source
          # offset (0 for a plain map, K for a contiguous slice copy).
          entry = {"type": "VMEM", "addr": base + i, "param": p, "offset": offset + copy_offset,
                   "count": count, "dtype": "int32"}
          if is_bool: entry["bool"] = True
          kern.data_plan.append(entry)
          kern.instructions.append(TpuInst("LOAD", reg, (base + i,)))
      elif node.op is Ops.WHERE:
        kern.instructions.append(TpuInst("SELECT", reg,
          tuple(vreg[_canon(s)] for s in node.src)))
        kern.primitives.add("SELECT")
      elif node.op is Ops.CAST:
        kern.instructions.append(TpuInst("VPU", reg,
          (vreg[_canon(node.src[0])],), vpu_op=_VPU[_cast_vpu(node)]))
      elif node.op in _UNARY_VPU:
        kern.instructions.append(TpuInst("VPU", reg,
          (vreg[_canon(node.src[0])],), vpu_op=_VPU[_UNARY_VPU[node.op]]))
      elif node.op is Ops.TRUNC:
        # trunc(x) = (float)(int)x — round toward zero via an int round-trip.
        kern.instructions.append(TpuInst("VPU", reg,
          (vreg[_canon(node.src[0])],), vpu_op=_VPU["F2I"]))
        kern.instructions.append(TpuInst("VPU", reg, (reg,), vpu_op=_VPU["I2F"]))
      else:
        table = _FLOAT_VPU if _float_operands(node) and node.op in _FLOAT_VPU else _ALU_TO_VPU
        kern.instructions.append(TpuInst("VPU", reg,
          tuple(vreg[_canon(s)] for s in node.src), vpu_op=_VPU[table[node.op]]))

    out_vmem = base + len(leaves)
    kern.instructions.append(TpuInst("STORE", out_vmem, (vreg[result],)))
    kern.outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})

  kern.instructions.append(TpuInst("HALT"))
  return kern.to_sxu_descriptor()
