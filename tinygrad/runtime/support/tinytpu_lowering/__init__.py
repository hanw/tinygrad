"""TinyTPU instruction-selection package.

Public surface re-exported here so ``ops_tinytpu.py`` keeps working unchanged.
"""
from tinygrad.runtime.support.tinytpu_lowering.elementwise import can_lower, lower_kernel
from tinygrad.runtime.support.tinytpu_lowering.reduction import is_reduction, lower_reduction
from tinygrad.runtime.support.tinytpu_lowering.classify import classify, KernelClass

__all__ = ["can_lower", "lower_kernel", "is_reduction", "lower_reduction",
           "classify", "KernelClass"]
