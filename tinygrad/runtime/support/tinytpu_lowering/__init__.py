"""TinyTPU instruction-selection package.

Public surface re-exported here so ``ops_tinytpu.py`` keeps working unchanged.
"""
from tinygrad.runtime.support.tinytpu_lowering.elementwise import can_lower, lower_kernel
from tinygrad.runtime.support.tinytpu_lowering.reduction import is_reduction, lower_reduction
from tinygrad.runtime.support.tinytpu_lowering.broadcast import is_broadcast, lower_broadcast
from tinygrad.runtime.support.tinytpu_lowering.gemm import lower_gemm, lower_gemm_fallback
from tinygrad.runtime.support.tinytpu_lowering.classify import classify, KernelClass

__all__ = ["can_lower", "lower_kernel", "is_reduction", "lower_reduction",
           "is_broadcast", "lower_broadcast", "lower_gemm", "lower_gemm_fallback",
           "classify", "KernelClass"]
