"""
TinyTPU tinygrad runtime device.

Implements a tinygrad Compiled device that drives the BSV TensorCore simulation
for 4x4 GEMM operations.  Other ops raise NotImplementedError.

The BSV simulator binary is located via the TINYTPU_SIM environment variable
(default: <repo_root>/build/mkTbTinyTPURuntime.bexe).
"""

from __future__ import annotations
import os, json, subprocess, tempfile
import numpy as np
from tinygrad.device import Compiled, Allocator, BufferSpec
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import PtrDType
from tinygrad.helpers import Target

# ---------------------------------------------------------------------------
# Constants matching the BSV TensorCore#(4,4,16) prototype
# ---------------------------------------------------------------------------
_ROWS   = 4
_COLS   = 4
_BYTES_PER_ELEM = 4           # Int#(32) = 4 bytes

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


# ---------------------------------------------------------------------------
# Renderer — detects 4x4 GEMM from UOps, emits JSON descriptor
# ---------------------------------------------------------------------------
class TinyTPURenderer(Renderer):
    """
    Minimal renderer for the TinyTPU prototype.

    Supports only the 4×4 GEMM pattern produced by tinygrad for:
        Tensor(shape=(1,4)) @ Tensor(shape=(4,4))

    Emits a JSON descriptor that TinyTPUProgram uses at call time.
    """
    has_local   = False
    has_threads = False
    global_max  = (1,) * 3
    local_max   = (1,) * 3

    def render(self, uops: list[UOp]) -> str:  # type: ignore[override]
        params     = [u for u in uops if u.op is Ops.PARAM]
        has_mulacc = any(u.op is Ops.MULACC for u in uops)
        # UNROLL optimization replaces MULACC with chained MUL+ADD inside a RANGE
        has_mul    = any(u.op is Ops.MUL    for u in uops)
        has_range  = any(u.op is Ops.RANGE  for u in uops)
        has_store  = any(u.op is Ops.STORE  for u in uops)
        is_gemm    = has_mulacc or (has_mul and has_range)

        # Must be exactly 3 params (out, act, weight) with a matmul pattern.
        if len(params) != 3 or not is_gemm or not has_store:
            return json.dumps({"op": "UNSUPPORTED",
                               "reason": f"params={len(params)} gemm={is_gemm}"})

        # Identify roles by PtrDType size (number of elements).
        param_sizes = {}
        for p in params:
            if isinstance(p.dtype, PtrDType):
                param_sizes[p.arg] = p.dtype.size
            else:
                return json.dumps({"op": "UNSUPPORTED", "reason": "non-ptr param"})

        sizes = sorted(param_sizes.values())
        # Expected: [4, 4, 16] for (1×4 output, 1×4 activation, 4×4 weight)
        if sizes != [4, 4, 16]:
            return json.dumps({"op": "UNSUPPORTED",
                               "reason": f"unexpected param sizes {sizes}"})

        # Find the weight param (size 16) and output param (arg 0 by tinygrad convention)
        weight_arg = next(arg for arg, sz in param_sizes.items() if sz == 16)
        out_arg    = 0  # tinygrad always puts the output buffer first
        act_arg    = next(arg for arg in param_sizes if arg != weight_arg and arg != out_arg)

        return json.dumps({"op": "GEMM4x4",
                           "out": out_arg,
                           "act": act_arg,
                           "weight": weight_arg})


# ---------------------------------------------------------------------------
# Helpers: build text bundle, parse BSV sim output
# ---------------------------------------------------------------------------
def _build_gemm_bundle(weight_i8: np.ndarray, act_i8: np.ndarray) -> str:
    """
    weight_i8 : int8 array, shape (4, 4), row-major
    act_i8    : int8 array, shape (4,)
    Returns the numeric text bundle that TbTinyTPURuntime reads.
    """
    lines: list[str] = []
    # Record 0: WEIGHT_TILE at SRAM addr 0 — 16 values row-major
    w_flat = weight_i8.flatten()
    lines.append("0 0 " + " ".join(str(int(x)) for x in w_flat))
    # Record 1: ACT_TILE at SRAM addr 1 — 4 values
    lines.append("1 1 " + " ".join(str(int(x)) for x in act_i8))
    # Record 2: SXU_DISPATCH_MXU (opcode=3): wBase=0 aBase=1 tLen=1
    lines.append("2 3 0 0 0 0 0 0 1 1")
    # Record 2: SXU_WAIT_MXU (opcode=4)
    lines.append("2 4 0 0 0 0 0 0 0 0")
    # Record 2: SXU_HALT (opcode=5)
    lines.append("2 5 0 0 0 0 0 0 0 0")
    # Record 3: OUTPUT_MXU = 1
    lines.append("3 1")
    # Record 4: END
    lines.append("4")
    return "\n".join(lines) + "\n"


def _parse_sim_output(stdout: str) -> list[int] | None:
    """Extract mxu_result from BSV sim stdout.  Returns None if not found."""
    for line in stdout.splitlines():
        if line.startswith("mxu_result "):
            return [int(x) for x in line.split()[1:]]
    return None


# ---------------------------------------------------------------------------
# Program — drives the BSV simulator
# ---------------------------------------------------------------------------
class TinyTPUProgram:
    def __init__(self, name: str, lib: bytes, *args, **kwargs):
        self.name = name
        self.prog = json.loads(lib)

    def __call__(self, *bufs: bytearray,
                 global_size: tuple = (1, 1, 1),
                 local_size: tuple | None = None,
                 vals: tuple = (),
                 wait: bool = False,
                 **kwargs) -> float | None:
        prog = self.prog
        if prog.get("op") != "GEMM4x4":
            raise NotImplementedError(
                f"TinyTPU: unsupported op '{prog.get('op')}' "
                f"(reason: {prog.get('reason', 'n/a')})"
            )

        out_buf    = bufs[prog["out"]]
        act_buf    = bufs[prog["act"]]
        weight_buf = bufs[prog["weight"]]

        # Decode int32 data from the raw bytearrays
        act_i32    = np.frombuffer(bytes(act_buf),    dtype="<i4")       # shape (4,)
        weight_i32 = np.frombuffer(bytes(weight_buf), dtype="<i4")       # shape (16,)

        # Downcast to int8 (hardware operand type)
        act_i8    = act_i32.astype(np.int8)
        weight_i8 = weight_i32.reshape(_ROWS, _COLS).astype(np.int8)

        # Write text bundle to a temp file
        bundle_text = _build_gemm_bundle(weight_i8, act_i8)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(bundle_text)
            bundle_path = f.name

        try:
            sim = _sim_path()
            env = {**os.environ, "TINYTPU_BUNDLE": bundle_path}
            proc = subprocess.run([sim], env=env,
                                  capture_output=True, text=True, timeout=30)
        finally:
            os.unlink(bundle_path)

        if proc.returncode != 0:
            raise RuntimeError(
                f"TinyTPU sim exited {proc.returncode}\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )

        result = _parse_sim_output(proc.stdout)
        if result is None:
            raise RuntimeError(
                f"TinyTPU sim produced no mxu_result\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )

        # Write int32 results back to the output buffer
        out_i32 = np.array(result, dtype="<i4")
        out_buf[: len(out_i32) * _BYTES_PER_ELEM] = out_i32.tobytes()
        return 1e-3  # placeholder timing (seconds)


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
