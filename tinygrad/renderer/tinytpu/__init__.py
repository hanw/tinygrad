"""TinyTPU instruction-selection package.

Public surface re-exported here so ``ops_tinytpu.py`` keeps working unchanged.
"""
from tinygrad.renderer.tinytpu.elementwise import can_lower, lower_kernel
from tinygrad.renderer.tinytpu.reduction import is_reduction, lower_reduction
from tinygrad.renderer.tinytpu.broadcast import is_broadcast, lower_broadcast
from tinygrad.renderer.tinytpu.movement import is_movement, lower_movement
from tinygrad.renderer.tinytpu.gemm import lower_gemm, lower_gemm_fallback
from tinygrad.renderer.tinytpu.classify import classify, KernelClass

__all__ = ["can_lower", "lower_kernel", "is_reduction", "lower_reduction",
           "is_broadcast", "lower_broadcast", "is_movement", "lower_movement",
           "lower_gemm", "lower_gemm_fallback", "classify", "KernelClass"]
