"""
TinyTPU tinygrad runtime device.

Implements a tinygrad Compiled device that drives the BSV TensorCore simulation
for 4x4 GEMM operations.  Other ops raise NotImplementedError.

The BSV simulator binary is located via the TINYTPU_SIM environment variable
(default: <repo_root>/build/mkTbTinyTPURuntime.bexe).
"""

from __future__ import annotations
import os, json, subprocess, tempfile
from collections import Counter
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
        diag = analyze_tinytpu_uops(uops)
        if diag["supported"]:
            return json.dumps({"op": "GEMM4x4",
                               "out": diag["out_arg"],
                               "act": diag["act_arg"],
                               "weight": diag["weight_arg"],
                               "num_vecs": diag["num_vecs"],
                               "num_weight_tiles": diag["num_weight_tiles"]})
        return json.dumps({
            "op": "UNSUPPORTED",
            "reason": diag["reason"],
            "missing_instructions": diag["missing_instructions"],
            "notes": diag["notes"],
            "op_counts": diag["op_counts"],
        })

def analyze_tinytpu_uops(uops:list[UOp]) -> dict:
    params = [u for u in uops if u.op is Ops.PARAM]
    op_counts = Counter(u.op.name for u in uops)
    has_mulacc = any(u.op is Ops.MULACC for u in uops)
    has_mul = any(u.op is Ops.MUL for u in uops)
    has_range = any(u.op is Ops.RANGE for u in uops)
    has_store = any(u.op is Ops.STORE for u in uops)
    is_gemm = has_mulacc or (has_mul and has_range)

    diag = {
        "supported": False,
        "reason": "",
        "missing_instructions": [],
        "notes": [],
        "op_counts": dict(sorted(op_counts.items())),
        "out_arg": None, "act_arg": None, "weight_arg": None,
        "num_vecs": None,
        "num_weight_tiles": None,
    }

    param_sizes: dict[int, int] = {}
    for p in params:
        if not isinstance(p.dtype, PtrDType):
            diag["reason"] = "non-ptr param"
            diag["notes"].append("TinyTPU kernels currently expect pointer-backed buffers only.")
            return diag
        param_sizes[p.arg] = p.dtype.size

    if len(params) == 3 and is_gemm and has_store:
        sizes = sorted(param_sizes.values())
        candidate_weights = [arg for arg, sz in param_sizes.items() if sz >= 16 and sz % 16 == 0]
        for weight_arg in candidate_weights:
            weight_size = param_sizes[weight_arg]
            num_weight_tiles = weight_size // 16
            non_weight = {arg: sz for arg, sz in param_sizes.items() if arg != weight_arg}
            out_arg = 0
            if out_arg not in non_weight:
                continue
            act_arg = next((arg for arg in non_weight if arg != out_arg), None)
            if act_arg is None:
                continue
            out_size = non_weight.get(out_arg)
            act_size = non_weight.get(act_arg)
            if out_size is None or act_size is None or act_size < 4 or act_size % 4 != 0:
                continue
            if out_size == act_size * num_weight_tiles:
                diag.update({
                    "supported": True,
                    "reason": "supported gemm4x4",
                    "out_arg": out_arg,
                    "act_arg": act_arg,
                    "weight_arg": weight_arg,
                    "num_vecs": act_size // 4,
                    "num_weight_tiles": num_weight_tiles,
                })
                return diag
        diag["reason"] = f"unexpected param sizes {sizes}"
        diag["notes"].append("Current TinyTPU backend only handles int32 matmul cases where the weight buffer is one or more 4x4 tiles and the output buffer matches activation_rows x output_tile_count.")
    else:
        diag["reason"] = f"params={len(params)} gemm={is_gemm}"

    missing: list[str] = []
    notes: list[str] = []
    uses_load_store = op_counts.get("LOAD", 0) > 0 or op_counts.get("STORE", 0) > 0
    uses_cmp_select = op_counts.get("CMPLT", 0) > 0 or op_counts.get("WHERE", 0) > 0
    uses_group = op_counts.get("GROUP", 0) > 0
    multi_store = op_counts.get("STORE", 0) > 1

    if is_gemm and (uses_cmp_select or multi_store or uses_group):
        missing.extend(["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"])
        notes.append("This looks like a fused MXU kernel with a pointwise epilogue. TinyTPU only lowers the bare GEMM today.")
        notes.append("To support this, the compiler needs to split or lower the epilogue and provide a path from MXU results into the VPU/VMEM pipeline.")
    elif uses_load_store:
        missing.extend(["SXU_LOAD_VREG", "SXU_STORE_VREG"])
        notes.append("General VMEM<->VReg movement kernels are not lowered yet.")

    if uses_cmp_select:
        missing.append("SXU_DISPATCH_VPU")
        notes.append("Compare/select UOps are present. For ReLU-like cases this likely maps to a VPU epilogue.")

    if op_counts.get("CMPLT", 0) == 4 and op_counts.get("WHERE", 0) == 4:
        notes.append("The UOp pattern matches a lane-wise ReLU/select epilogue.")

    if uses_group:
        notes.append("GROUP indicates multi-store/vector pack behavior that the TinyTPU backend does not currently lower.")

    if not missing:
        missing.append("unknown lowering gap")
        notes.append("Inspect the op counts and UOps for a new lowering rule.")

    diag["missing_instructions"] = sorted(set(missing))
    diag["notes"].extend(notes)
    return diag


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


def _run_gemm_vec(sim: str, weight_i8: np.ndarray, act_i8: np.ndarray) -> list[int]:
    bundle_text = _build_gemm_bundle(weight_i8, act_i8)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(bundle_text)
        bundle_path = f.name

    try:
        env = {**os.environ, "TINYTPU_BUNDLE": bundle_path}
        proc = subprocess.run([sim], env=env, capture_output=True, text=True, timeout=30)
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
    return result


def _require_int8_range(name: str, values: np.ndarray) -> None:
    if values.size == 0:
        return
    min_val = int(values.min())
    max_val = int(values.max())
    if min_val < -128 or max_val > 127:
        raise ValueError(f"TinyTPU {name} values must fit in signed int8, got range [{min_val}, {max_val}]")


def _unsupported_message(prog: dict) -> str:
    parts = [f"TinyTPU: unsupported op '{prog.get('op')}' (reason: {prog.get('reason', 'n/a')})"]
    missing = prog.get("missing_instructions") or []
    notes = prog.get("notes") or []
    if missing:
        parts.append("missing instructions: " + ", ".join(str(x) for x in missing))
    if notes:
        parts.append("notes: " + " | ".join(str(x) for x in notes))
    return "; ".join(parts)


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
            raise NotImplementedError(_unsupported_message(prog))

        out_buf    = bufs[prog["out"]]
        act_buf    = bufs[prog["act"]]
        weight_buf = bufs[prog["weight"]]

        # Decode int32 data from the raw bytearrays
        act_i32    = np.frombuffer(bytes(act_buf),    dtype="<i4")
        weight_i32 = np.frombuffer(bytes(weight_buf), dtype="<i4")
        num_vecs   = int(prog.get("num_vecs", max(1, act_i32.size // _ROWS)))
        num_weight_tiles = int(prog.get("num_weight_tiles", max(1, weight_i32.size // (_ROWS * _COLS))))
        out_cols = num_weight_tiles * _COLS

        if act_i32.size != num_vecs * _ROWS:
            raise RuntimeError(f"TinyTPU activation buffer size {act_i32.size} does not match num_vecs={num_vecs}")
        if weight_i32.size != num_weight_tiles * _ROWS * _COLS:
            raise RuntimeError(f"TinyTPU weight buffer size {weight_i32.size} does not match num_weight_tiles={num_weight_tiles}")
        if len(out_buf) < num_vecs * out_cols * _BYTES_PER_ELEM:
            raise RuntimeError(f"TinyTPU output buffer too small for shape=({num_vecs}, {out_cols})")

        _require_int8_range("weight", weight_i32)
        _require_int8_range("activation", act_i32)

        # Downcast to int8 (hardware operand type)
        weight_matrix = weight_i32.reshape(_ROWS, out_cols)
        weight_tiles = [weight_matrix[:, tile_idx * _COLS : (tile_idx + 1) * _COLS].astype(np.int8)
                        for tile_idx in range(num_weight_tiles)]
        sim = _sim_path()
        out_i32 = np.empty(num_vecs * out_cols, dtype="<i4")
        act_rows = act_i32.reshape(num_vecs, _ROWS).astype(np.int8)
        for i, act_i8 in enumerate(act_rows):
            row_base = i * out_cols
            for tile_idx, weight_i8 in enumerate(weight_tiles):
                col_base = row_base + tile_idx * _COLS
                out_i32[col_base : col_base + _COLS] = _run_gemm_vec(sim, weight_i8, act_i8)
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
