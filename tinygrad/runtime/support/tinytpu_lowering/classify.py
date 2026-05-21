"""TinyTPU kernel classifier.

Classifies an incoming UOp list into one of the kernel classes so that
``render()`` can dispatch to the right lowering path without duplicating
predicate logic.
"""
from enum import Enum, auto
from tinygrad.uop.ops import Ops, UOp
from tinygrad.runtime.support.tinytpu_lowering.elementwise import can_lower
from tinygrad.runtime.support.tinytpu_lowering.reduction import is_reduction
from tinygrad.runtime.support.tinytpu_lowering.broadcast import is_broadcast


class KernelClass(Enum):
  ELEMENTWISE = auto()
  REDUCTION = auto()
  BROADCAST = auto()
  GEMM = auto()
  UNSUPPORTED = auto()


def classify(uops: list[UOp]) -> KernelClass:
  # Classification is most-specific-first: GEMM (WMMA), REDUCTION, and
  # BROADCAST are checked before the broad ELEMENTWISE predicate.
  if any(u.op is Ops.WMMA for u in uops): return KernelClass.GEMM
  if is_reduction(uops): return KernelClass.REDUCTION
  if is_broadcast(uops): return KernelClass.BROADCAST
  if can_lower(uops): return KernelClass.ELEMENTWISE
  return KernelClass.UNSUPPORTED
