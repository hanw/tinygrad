"""TinyTPU kernel classifier.

Classifies an incoming UOp list into one of three kernel classes so that
``render()`` can dispatch to the right lowering path without duplicating
predicate logic.
"""
from enum import Enum, auto
from tinygrad.uop.ops import Ops
from tinygrad.runtime.support.tinytpu_lowering.elementwise import can_lower


class KernelClass(Enum):
  ELEMENTWISE = auto()
  GEMM = auto()
  UNSUPPORTED = auto()


def classify(uops):
  if any(u.op is Ops.WMMA for u in uops): return KernelClass.GEMM
  if can_lower(uops): return KernelClass.ELEMENTWISE
  return KernelClass.UNSUPPORTED
