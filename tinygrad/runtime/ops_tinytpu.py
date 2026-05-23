"""TinyTPU tinygrad runtime device.

Implements a tinygrad Compiled device that drives the BSV TinyTPU simulator.
The renderer classifies tinygrad UOps into TinyTPU lowering classes
(elementwise, reduction, broadcast, movement, GEMM) and emits an SXU_PROGRAM
descriptor. The runtime binds host buffers into that descriptor, builds the
simulator bundle, launches the simulator, and copies VMEM results back into
the tinygrad output buffer. Unsupported kernels raise NotImplementedError with
lowering diagnostics.

The BSV simulator binary is located via the TINYTPU_SIM environment variable
(default: <repo_root>/build/mkTbTinyTPURuntime.bexe).
"""

from __future__ import annotations
import os, json
from collections import Counter
import numpy as np
from tinygrad.device import Compiled, Allocator, BufferSpec
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import dtypes
from tinygrad.codegen.opt.tc import TensorCore
from tinygrad.renderer.tinytpu import (
    lower_kernel, lower_reduction, lower_broadcast, lower_movement,
    lower_gemm, lower_gemm_fallback, classify, KernelClass)
# Bundle-instruction encoders, shared graph helpers, and GEMM tiling helpers
# now live in the tinytpu_lowering package. They are re-imported here so the
# long-standing `from tinygrad.runtime.ops_tinytpu import _vmem, ...` imports
# in tests/ and scripts/ keep working unchanged.
from tinygrad.renderer.tinytpu.common import (
    _vmem, _wmem, _amem, _load, _store, _vpu, _vpu_bg,
    _vpu_exp2, _vpu_log2, _vpu_sin, _select,
    _broadcast_scalar, _broadcast_row, _broadcast_col, _broadcast,
    _mxu, _mxu_psum_write, _mxu_psum_acc, _mxu_accumulate, _mxu_clear,
    _mxu_os, _mxu_os_accumulate, _load_mxu_matrix_row,
    _read_cycle, _load_loop_depth, _xlu_rotate, _loop_begin, _loop_end,
    _vzero, _vfill, _vmov, _vneg, _vabs,
    _psum_read, _psum_read_row, _psum_clear, _psum_clear_all,
    _psum_accumulate_row, _set_pred_if_zero, _skip_if_pred,
    _set_pred_ne_zero, _skip_if_not_pred,
    _wait_mxu, _load_mxu_result, _load_vpu_result, _load_xlu_result,
    _mxu_epilogue, _load_epilogue_stat,
    _set_requant_config, _mxu_requant, _output_asram,
    _halt, _output_mxu, _output_vmem, _end, _bundle, _find_unique_param_arg)
from tinygrad.renderer.tinytpu.gemm import _infer_tiling, _tiling_failure_note
from tinygrad.runtime.support.compiler_tinytpu import (
    TinyTPUCompiler, TinyTPUKernel, SUPPORTED_TINYTPU_OPS, unsupported_message)
from tinygrad.runtime.support import tinytpu as tinytpu_rt
from tinygrad.runtime.support.tinytpu_isa import (
    ROWS as _ROWS, COLS as _COLS, TILE_ELEMS as _TILE_ELEMS,
    BYTES_PER_ELEM as _BYTES_PER_ELEM, VPU_OPS as _VPU_OPS,
    VPU_BOOL_OPS as _VPU_BOOL_OPS, SXU_OPS as _SXU_OPS)

# ---------------------------------------------------------------------------
# Constants matching the BSV TensorCore#(4,4,16) prototype
# ---------------------------------------------------------------------------

def _sim_path() -> str:
    if (p := os.environ.get("TINYTPU_SIM")):
        return p
    # default: <repo_root>/build/mkTbTinyTPURuntime.bexe
    # this file is at  <repo_root>/tinygrad/tinygrad/runtime/ops_tinytpu.py
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "..", "..", "build", "mkTbTinyTPURuntime.bexe")


# ---------------------------------------------------------------------------
# Allocator — buffers are plain bytearray objects stored on the Python heap
# ---------------------------------------------------------------------------
class TinytpuAllocator(Allocator["TinytpuDevice"]):
    def _alloc(self, size: int, options: BufferSpec) -> bytearray:
        return bytearray(size)

    def _copyin(self, dest: bytearray, src: memoryview) -> None:
        dest[:] = src

    def _copyout(self, dest: memoryview, src: bytearray) -> None:
        dest[:] = src

    def _free(self, buf: bytearray, options: BufferSpec) -> None:
        pass  # GC handles it

    def _offset(self, buf: bytearray, offset: int, size: int) -> bytearray:
        return buf[offset : offset + size]


class TinyTPURenderer(Renderer):
    """Renderer for TinyTPU SXU_PROGRAM descriptors.

    The renderer does not emit textual source. It dispatches UOps to focused
    TinyTPU lowerers and returns a JSON descriptor consumed by TinyTPUProgram.
    """
    compiler = TinyTPUCompiler()
    has_local   = False
    has_threads = False
    global_max  = (1,) * 3
    local_max   = (1,) * 3
    # Declaring transcendental ops here tells tinygrad's codegen that this
    # backend has hardware support, so the default xexp2/xlog2/xsin poly
    # decompositions are skipped and EXP2/LOG2/SIN pass through to the
    # SXU renderer as single UOps.
    code_for_op = {Ops.EXP2: (lambda *_a, **_k: None),
                   Ops.LOG2: (lambda *_a, **_k: None),
                   Ops.SIN:  (lambda *_a, **_k: None),
                   Ops.SQRT: (lambda *_a, **_k: None)}
    tensor_cores = [TensorCore(
        dims=(4, 4, 4),
        threads=1,
        elements_per_thread=(16, 16, 16),
        dtype_in=dtypes.int,
        dtype_out=dtypes.int,
        opts=("u0", "u0", "u1", "u1"),
        swizzle=(((), ("u2", "u3", "r0", "r1"), ("u0", "u1")),
                 ((), ("u0", "u1", "r0", "r1"), ("u2", "u3"))),
    )]

    def render(self, uops: list[UOp]) -> str:  # type: ignore[override]
        # Classify the kernel, then dispatch to the matching lowerer in the
        # tinytpu_lowering package. All kernel lowering lives behind classify().
        klass = classify(uops)
        if klass is KernelClass.ELEMENTWISE:
            return _dump_lowering(json.dumps(lower_kernel(uops)))
        if klass is KernelClass.REDUCTION:
            return _dump_lowering(json.dumps(lower_reduction(uops)))
        if klass is KernelClass.BROADCAST:
            return _dump_lowering(json.dumps(lower_broadcast(uops)))
        if klass is KernelClass.MOVEMENT:
            return _dump_lowering(json.dumps(lower_movement(uops)))
        if klass is KernelClass.GEMM and (gemm_desc := lower_gemm(uops)) is not None:
            return _dump_lowering(json.dumps(gemm_desc))
        # Non-WMMA matmul kernels are not classified as GEMM (classify only
        # tags WMMA kernels GEMM); they fall through to the GEMM fallback
        # lowerer, which was previously reached after _render_sxu_program.
        if (gemm_desc := lower_gemm_fallback(uops)) is not None:
            return _dump_lowering(json.dumps(gemm_desc))
        op_counts = dict(sorted(Counter(u.op.name for u in uops).items()))
        return _dump_lowering(json.dumps({
            "op": "UNSUPPORTED",
            "reason": "no SXU_PROGRAM renderer matched",
            "missing_instructions": [],
            "notes": [],
            "op_counts": op_counts,
        }))


def _dump_lowering(desc:str) -> str:
    target = os.environ.get("TINYTPU_DUMP_LOWERING")
    if not target:
        return desc
    if target == "1":
        print(desc)
    else:
        with open(target, "a", encoding="utf-8") as f:
            f.write(desc + "\n")
    return desc

# ---------------------------------------------------------------------------
# Compatibility exports for tests/scripts. Runtime code calls support modules
# directly; keep these names stable until downstream imports move.
# ---------------------------------------------------------------------------

def _build_vpu_binary_bundle(lhs_i32: np.ndarray, rhs_i32: np.ndarray, num_elems: int, vpu_op: int,
                             lhs_broadcast: bool = False, rhs_broadcast: bool = False) -> str:
    return tinytpu_rt.build_vpu_binary_bundle(lhs_i32, rhs_i32, num_elems, vpu_op, lhs_broadcast, rhs_broadcast)

def _build_vpu_unary_bundle(src_i32: np.ndarray, num_elems: int, vpu_op: int) -> str:
    return tinytpu_rt.build_vpu_unary_bundle(src_i32, num_elems, vpu_op)

def _build_vpu_program_bundle(inputs: list[np.ndarray], num_elems: int, steps: list[dict], output_reg: int,
                              input_broadcasts: list[bool] | None = None) -> str:
    return tinytpu_rt.build_vpu_program_bundle(inputs, num_elems, steps, output_reg, input_broadcasts)

def _parse_result_line(line: str, prefix: str, expected_count: int) -> list[int]:
    return tinytpu_rt.parse_result_line(line, prefix, expected_count)

def _parse_sim_output(stdout: str) -> list[int] | None:
    return tinytpu_rt.parse_sim_output(stdout)

def _parse_vmem_output(stdout: str) -> list[int] | None:
    return tinytpu_rt.parse_vmem_output(stdout)

def _parse_asram_output(stdout: str) -> list[int] | None:
    return tinytpu_rt.parse_asram_output(stdout)

def _parse_multi_vmem_output(stdout: str) -> list[list[int]]:
    return tinytpu_rt.parse_multi_vmem_output(stdout)

def _build_full_gemm_bundle(act_rows_i8: np.ndarray, weight_matrix_i8: np.ndarray,
                            num_vecs: int, num_k_tiles: int, num_weight_tiles: int,
                            bias_i32: np.ndarray | None = None, relu: bool = False) -> str:
    return tinytpu_rt.build_full_gemm_bundle(act_rows_i8, weight_matrix_i8, num_vecs, num_k_tiles,
                                             num_weight_tiles, _VPU_OPS, bias_i32, relu)

def _run_bundle(sim: str, bundle_text: str) -> str:
    return tinytpu_rt.run_bundle(sim, bundle_text)

def _require_int8_range(name: str, values: np.ndarray) -> None:
    from tinygrad.runtime.support.compiler_tinytpu import _require_int8_range as require_int8_range
    require_int8_range(name, values)

def _unsupported_message(prog: dict) -> str:
    return unsupported_message(prog)

# ---------------------------------------------------------------------------
# Program — drives the BSV simulator
# ---------------------------------------------------------------------------
_SUPPORTED_OPS = SUPPORTED_TINYTPU_OPS

class TinyTPUProgram:
    def __init__(self, name: str, lib: bytes, *args, **kwargs):
        self.name = name
        self.kernel = TinyTPUKernel.from_json(lib)
        self.sim = _sim_path()

    def _run(self, bundle_text: str) -> str:
        return tinytpu_rt.run_bundle(self.sim, bundle_text)

    def _run_vmem(self, bundle_text: str) -> list[int]:
        """Run bundle and parse the first vmem_result line."""
        result = tinytpu_rt.parse_vmem_output(self._run(bundle_text))
        if result is None:
            raise RuntimeError("TinyTPU sim produced no vmem_result")
        return result

    def __call__(self, *bufs: bytearray,
                 global_size: tuple = (1, 1, 1),
                 local_size: tuple | None = None,
                 vals: tuple = (),
                 wait: bool = False,
                 **kwargs) -> float | None:
        prog = self.kernel.desc
        op = self.kernel.op
        if op not in _SUPPORTED_OPS:
            raise NotImplementedError(unsupported_message(prog))
        return getattr(self, f"_exec_{op.lower()}")(bufs)

    def _exec_sxu_program(self, bufs):
        bundle_text = self.kernel.build_bundle(bufs)
        stdout = self._run(bundle_text)
        vmem_results = tinytpu_rt.parse_multi_vmem_output(stdout)
        try:
            self.kernel.write_outputs(bufs, vmem_results)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}\n{stdout[:500]}") from exc
        return 1e-3

# ---------------------------------------------------------------------------
# Device — top-level tinygrad device
# ---------------------------------------------------------------------------
class TinytpuDevice(Compiled):
    def __init__(self, device: str):
        super().__init__(
            device,
            allocator=TinytpuAllocator(self),
            renderers=[TinyTPURenderer],
            runtime=TinyTPUProgram,
        )
