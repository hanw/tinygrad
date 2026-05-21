"""TinyTPU elementwise lowerer.

Lowers int32/bool/float elementwise kernels (ALU, WHERE, unary transcendentals)
to an SXU_PROGRAM descriptor via a UOp-walking linear-scan register allocator.
"""
from __future__ import annotations
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import PtrDType
from tinygrad.runtime.support.tinytpu_lowering.common import (
  _ALU_OPS, _ALU_TO_VPU, _FLOAT_VPU, _UNARY_VPU, _VPU, _DATA_OPS,
  _NUM_VREGS, _TILE_ELEMS,
  TpuInst, TpuKernel,
  _canon, _unique_param, _data_dag, _store_lanes, _float_operands,
  _const_bits, _run_instsel,
)

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
          if params[p].dtype.size not in (out_size, 1): return False
          continue
        if n.op is Ops.CONST:
          continue
        return False   # unknown op (CAST, transcendental, ...)
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
  nodes = _data_dag(_store_lanes(stores[0])[0])
  leaves = [n for n in nodes if n.op in (Ops.LOAD, Ops.CONST)]
  interior = [n for n in nodes if n.op in _DATA_OPS]

  num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
  addrs_per_tile = len(leaves) + 1

  # --- linear-scan register allocation over the per-lane DAG ---
  # seq is the emission order; a value's VREG is freed after its last use,
  # so a vreg is reused rather than burned monotonically (deep activation
  # and divmod graphs have far more than 16 nodes but a small live set).
  seq = leaves + interior
  pos = {n: i for i, n in enumerate(seq)}
  last_use: dict[UOp, int] = {}
  for node in interior:
    for s in node.src:
      c = _canon(s)
      if c in pos: last_use[c] = max(last_use.get(c, -1), pos[node])
  last_use[interior[-1]] = len(seq)   # the final value is consumed by the STORE

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
        kern.data_plan.append({"type": "VMEM", "addr": base + i, "layout": "broadcast_const",
                               "value": _const_bits(node.arg), "count": count, "dtype": "int32"})
        kern.instructions.append(TpuInst("LOAD", reg, (base + i,)))
      elif node.op is Ops.LOAD:
        p = _unique_param(node)
        is_bool = params[p].dtype.base.itemsize == 1
        if params[p].dtype.size == 1 and out_size > 1:
          entry = {"type": "VMEM", "addr": base + i, "param": p, "offset": 0,
                   "count": 1, "dtype": "int32"}
          if is_bool: entry["bool"] = True
          kern.data_plan.append(entry)
          kern.instructions.append(TpuInst("LOAD", reg, (base + i,)))
          kern.instructions.append(TpuInst("BROADCAST_SCALAR", reg, (reg,)))
          kern.primitives.add("BROADCAST_SCALAR")
        else:
          entry = {"type": "VMEM", "addr": base + i, "param": p, "offset": offset,
                   "count": count, "dtype": "int32"}
          if is_bool: entry["bool"] = True
          kern.data_plan.append(entry)
          kern.instructions.append(TpuInst("LOAD", reg, (base + i,)))
      elif node.op is Ops.WHERE:
        kern.instructions.append(TpuInst("SELECT", reg,
          tuple(vreg[_canon(s)] for s in node.src)))
        kern.primitives.add("SELECT")
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
    kern.instructions.append(TpuInst("STORE", out_vmem, (vreg[interior[-1]],)))
    kern.outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})

  kern.instructions.append(TpuInst("HALT"))
  return kern.to_sxu_descriptor()
