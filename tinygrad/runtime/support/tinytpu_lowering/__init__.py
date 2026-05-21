"""TinyTPU instruction-selection package.

Public surface re-exported here so ``ops_tinytpu.py`` keeps working unchanged.
"""
from tinygrad.runtime.support.tinytpu_lowering.elementwise import can_lower, lower_kernel

__all__ = ["can_lower", "lower_kernel"]
