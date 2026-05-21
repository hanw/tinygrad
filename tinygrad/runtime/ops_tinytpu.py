"""
TinyTPU tinygrad runtime device.

Implements a tinygrad Compiled device that drives the BSV TensorCore simulation
for 4x4 GEMM operations.  Other ops raise NotImplementedError.

The BSV simulator binary is located via the TINYTPU_SIM environment variable
(default: <repo_root>/build/mkTbTinyTPURuntime.bexe).
"""

from __future__ import annotations
import os, json, subprocess, tempfile, math
from collections import Counter
import numpy as np
from tinygrad.device import Compiled, Allocator, BufferSpec, Compiler
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.codegen.opt.tc import TensorCore
from tinygrad.runtime.support.tinytpu_lowering import can_lower, lower_kernel, lower_reduction, lower_broadcast, classify, KernelClass

# ---------------------------------------------------------------------------
# Constants matching the BSV TensorCore#(4,4,16) prototype
# ---------------------------------------------------------------------------
_ROWS   = 4
_COLS   = 4
_BYTES_PER_ELEM = 4           # Int#(32) = 4 bytes
_TILE_ELEMS = _ROWS * _COLS   # 16 elements per VMEM tile
_VPU_OPS = {"ADD": 0, "MUL": 1, "MAX": 3, "SUM_REDUCE": 4, "CMPLT": 5, "CMPNE": 6, "SUB": 7, "CMPEQ": 8, "MAX_REDUCE": 9, "SHL": 10, "SHR": 11, "MIN": 12, "MIN_REDUCE": 13, "DIV": 14, "AND": 15, "OR": 16, "XOR": 17,
             "FADD": 18, "FMUL": 19, "FSUB": 20, "FMAX": 21, "FCMPLT": 22, "FRECIP": 23, "I2F": 24, "F2I": 25, "NOT": 26, "SELECT": 27, "COPY": 28,
             "SUM_REDUCE_COL": 29, "MAX_REDUCE_COL": 30, "MIN_REDUCE_COL": 31,
             "SUM_REDUCE_TILE": 32, "MAX_REDUCE_TILE": 33, "MIN_REDUCE_TILE": 34,
             "MUL_REDUCE": 35, "MUL_REDUCE_COL": 36, "MUL_REDUCE_TILE": 37,
             "FSUM_REDUCE_TILE": 38, "FMAX_REDUCE_TILE": 39, "FMIN_REDUCE_TILE": 40,
             "FMIN": 41,
             "FSUM_REDUCE": 42, "FMAX_REDUCE": 43, "FMIN_REDUCE": 44,
             "FSUM_REDUCE_COL": 45, "FMAX_REDUCE_COL": 46, "FMIN_REDUCE_COL": 47,
             "FPROD_REDUCE_TILE": 48, "FPROD_REDUCE": 49, "FPROD_REDUCE_COL": 50,
             "EXP2": 51, "LOG2": 52, "SIN": 53, "COS": 54,
             "PACKED_I8_ADD": 55, "PACKED_I8_SUB": 56,
             "PACKED_I8_MAX": 57, "PACKED_I8_MIN": 58,
             "PACKED_I8_NEG": 59, "PACKED_I8_RELU": 60,
             "PACKED_I8_CMPLT": 61, "PACKED_I8_CMPEQ": 62,
             "PACKED_I8_MUL_LOW": 63, "PACKED_I8_MUL_HIGH": 64,
             "PACKED_I8_ABS": 65, "SIGN": 66, "PACKED_I8_SIGN": 67,
             "FSIGN": 68, "ARGMIN": 69, "ARGMAX": 70,
             "CLZ": 71, "POPCOUNT": 72, "CTZ": 73, "BYTE_REVERSE": 74,
             "SAT_ADD_I32": 75, "SAT_SUB_I32": 76,
             "ABS_DIFF_I32": 77, "PACKED_I8_ABS_DIFF": 78,
             "FABS": 79, "ROTL": 80, "ROTR": 81,
             "MIN_U32": 82, "MAX_U32": 83}
_VPU_BOOL_OPS = {_VPU_OPS["CMPLT"], _VPU_OPS["CMPNE"], _VPU_OPS["CMPEQ"]}
_SXU_OPS = {"LOAD_VREG": 0, "STORE_VREG": 1, "DISPATCH_VPU": 2, "DISPATCH_XLU_BROADCAST": 3, "DISPATCH_MXU": 4, "WAIT_MXU": 5, "LOAD_MXU_RESULT": 6, "HALT": 7, "DISPATCH_SELECT": 8, "BROADCAST_SCALAR": 9, "BROADCAST_ROW": 10, "BROADCAST_COL": 11, "DISPATCH_XLU_TRANSPOSE": 12, "LOAD_VPU_RESULT": 13, "LOAD_XLU_RESULT": 14, "PSUM_WRITE": 15, "PSUM_ACCUMULATE": 16, "PSUM_READ": 17}

_ALU_OPS = {Ops.ADD: "ADD", Ops.MUL: "MUL", Ops.SUB: "SUB", Ops.MAX: "MAX",
            Ops.CMPLT: "CMPLT", Ops.CMPNE: "CMPNE", Ops.CMPEQ: "CMPEQ",
            Ops.AND: "AND", Ops.OR: "OR", Ops.XOR: "XOR",
            Ops.SHL: "SHL", Ops.SHR: "SHR", Ops.IDIV: "DIV"}

def _uop_contains(u: UOp, target_op, seen: set | None = None) -> bool:
    """True if the UOp's source tree contains any UOp with op == target_op."""
    if seen is None: seen = set()
    if id(u) in seen: return False
    seen.add(id(u))
    if u.op is target_op: return True
    return any(_uop_contains(s, target_op, seen) for s in u.src)

def _has_load_src(u: UOp, visited: set | None = None) -> bool:
    """Check if a UOp has a LOAD anywhere in its source tree (data-path, not index)."""
    if visited is None: visited = set()
    if id(u) in visited: return False
    visited.add(id(u))
    if u.op is Ops.LOAD: return True
    return any(_has_load_src(s, visited) for s in u.src)

def _data_alu_ops(uops: list[UOp]) -> Counter:
    """Count only data-path ALU ops (ones with LOAD in their source tree)."""
    return Counter(_ALU_OPS[u.op] for u in uops if u.op in _ALU_OPS and _has_load_src(u))

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
# Compiler — JSON descriptor pass-through (bundle building happens at
# call time because it needs buffer data). Future: move bundle building
# here once we have a buffer-independent program format.
# ---------------------------------------------------------------------------
class TinyTPUCompiler(Compiler):
    pass


# ---------------------------------------------------------------------------
# Legacy descriptor renderer — handles patterns not yet migrated to SXU_PROGRAM
# ---------------------------------------------------------------------------

def _render_gemm_fallback_sxu_program(uops: list[UOp]) -> dict | None:
    """Render MULACC or scalar MUL+RANGE GEMMs (no WMMA UOp) as SXU_PROGRAM.

    Same structure as the WMMA SXU path but triggered by the non-WMMA lowering
    pattern. No epilogue support (bias/relu) — that still requires the WMMA
    UOp path in `_render_sxu_program`.
    """
    op_counts = Counter(u.op.name for u in uops)
    params = [u for u in uops if u.op is Ops.PARAM]
    param_sizes: dict[int, int] = {}
    for p in params:
        if not isinstance(p.dtype, PtrDType):
            return None
        param_sizes[p.arg] = p.dtype.size

    has_mulacc = any(u.op is Ops.MULACC for u in uops)
    has_store = op_counts.get("STORE", 0) > 0
    is_gemm = has_mulacc or (len(params) == 3 and op_counts.get("MUL", 0) > 0
                              and op_counts.get("RANGE", 0) > 0 and has_store)
    if is_gemm and len(param_sizes) == 3 and op_counts.get("GROUP", 0) == 0:
        tiling = _infer_tiling(param_sizes.get(0), param_sizes.get(1), param_sizes.get(2, 0))
        if tiling is not None:
            num_vecs, num_k_tiles, num_weight_tiles = tiling
            out_arg, act_arg, weight_arg = 0, 1, 2
            out_cols = num_weight_tiles * _COLS
            k_cols = num_k_tiles * _ROWS
            total_weight_tiles = num_k_tiles * num_weight_tiles
            data_plan = [
                {"type": "WMEM", "addr": 0, "param": weight_arg,
                 "offset": 0, "count": total_weight_tiles * _ROWS * _COLS,
                 "dtype": "int8", "layout": "weight_tiles",
                 "num_k_tiles": num_k_tiles, "num_weight_tiles": num_weight_tiles},
                {"type": "AMEM", "addr": 0, "param": act_arg,
                 "offset": 0, "count": num_vecs * k_cols,
                 "dtype": "int8", "layout": "act_tiles",
                 "num_vecs": num_vecs, "num_k_tiles": num_k_tiles},
            ]
            instructions = _generate_gemm_sxu_instructions(
                num_vecs, num_k_tiles, num_weight_tiles,
                has_bias=False, bias_vmem_base=0, has_relu=False,
            )
            outputs = []
            for row in range(num_vecs):
                for tile_idx in range(num_weight_tiles):
                    out_addr = row * num_weight_tiles + tile_idx
                    outputs.append({
                        "addr": out_addr, "param": out_arg,
                        "offset": row * out_cols + tile_idx * _COLS,
                        "count": _COLS,
                    })
            return {
                "op": "SXU_PROGRAM",
                "instructions": instructions,
                "data_plan": data_plan,
                "outputs": outputs,
                "num_output_tiles": num_vecs * num_weight_tiles,
                "num_vecs": num_vecs,
                "num_k_tiles": num_k_tiles,
                "num_weight_tiles": num_weight_tiles,
                "out": out_arg,
            }

    return None


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
        # Classify the kernel first; ELEMENTWISE goes to the UOp-walking
        # lowerer, everything else falls through to the structural recognizers.
        klass = classify(uops)
        if klass is KernelClass.ELEMENTWISE:
            return _dump_lowering(json.dumps(lower_kernel(uops)))
        if klass is KernelClass.REDUCTION:
            return _dump_lowering(json.dumps(lower_reduction(uops)))
        if klass is KernelClass.BROADCAST:
            return _dump_lowering(json.dumps(lower_broadcast(uops)))
        if (sxu_desc := _render_sxu_program(uops)) is not None:
            return _dump_lowering(json.dumps(sxu_desc))
        if (gemm_desc := _render_gemm_fallback_sxu_program(uops)) is not None:
            return _dump_lowering(json.dumps(gemm_desc))
        op_counts = dict(sorted(Counter(u.op.name for u in uops).items()))
        return _dump_lowering(json.dumps({
            "op": "UNSUPPORTED",
            "reason": "no SXU_PROGRAM renderer matched",
            "missing_instructions": [],
            "notes": [],
            "op_counts": op_counts,
        }))


def _render_sxu_program(uops: list[UOp]) -> dict | None:
    """Render a kernel as an SXU_PROGRAM descriptor.

    Returns a dict with op="SXU_PROGRAM", pre-built SXU instructions, and a
    data_plan that maps buffer param indices to WMEM/AMEM/VMEM addresses.
    The runtime fills in actual data at call time.

    Handles: WMMA GEMM kernels, elementwise binary/unary VPU kernels.
    Returns None if the kernel pattern is not recognized.
    """
    wmmas = [u for u in uops if u.op is Ops.WMMA]
    if not wmmas:
        # Broadcast kernels (row / column / column-where) are classified up
        # front in render() via is_broadcast() and dispatched to
        # lower_broadcast(); they never reach this fallback renderer.
        if (pad_desc := _render_pad_sxu_program(uops)) is not None:
            return pad_desc
        if (trans_desc := _render_transpose_sxu_program(uops)) is not None:
            return trans_desc
        if (rowbc_desc := _render_rowbc_copy_sxu_program(uops)) is not None:
            return rowbc_desc
        # Plain elementwise kernels — including degenerate copy, cast, and
        # const-fill maps — are owned by the InstSel walker, selected up front
        # in render() via can_lower(). Anything reaching here is an
        # unrecognized kernel — surface it rather than silently mishandling.
        return None

    wmma = wmmas[0]

    # Extract param mappings (same logic as _render_wmma_descriptor)
    out_params = {_find_unique_param_arg(store.src[0]) for store in uops if store.op is Ops.STORE}
    out_params.discard(None)
    src0_param = _find_unique_param_arg(wmma.src[0])
    src1_param = _find_unique_param_arg(wmma.src[1])
    if len(out_params) != 1 or src0_param is None or src1_param is None:
        return None  # fall back to old path

    out_arg = next(iter(out_params))
    act_arg, weight_arg = src0_param, src1_param
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if out_arg not in params or act_arg not in params or weight_arg not in params:
        return None

    out_size = params[out_arg].dtype.size
    act_size = params[act_arg].dtype.size
    weight_size = params[weight_arg].dtype.size
    if (tiling := _infer_tiling(out_size, act_size, weight_size)) is None:
        return None

    num_vecs, num_k_tiles, num_weight_tiles = tiling
    out_cols = num_weight_tiles * _COLS
    k_cols = num_k_tiles * _ROWS

    # Detect epilogue (bias add, relu)
    epilogue, epilogue_error = _extract_wmma_epilogue(uops, params, out_arg, act_arg, weight_arg, out_size, out_cols)
    if epilogue_error is not None:
        return None

    has_bias = any(step["op"] == "ADD" for step in epilogue)
    has_relu = any(step["op"] == "RELU" for step in epilogue)
    bias_arg = None
    bias_mode = None
    if has_bias:
        bias_step = next(s for s in epilogue if s["op"] == "ADD")
        bias_arg = bias_step["arg"]
        bias_mode = bias_step["mode"]

    # Build data_plan: describe which buffers map to which memory addresses
    data_plan: list[dict] = []

    # Weight tiles → WMEM: address = k * num_weight_tiles + tile_idx
    total_weight_tiles = num_k_tiles * num_weight_tiles
    data_plan.append({
        "type": "WMEM",
        "addr": 0,
        "param": weight_arg,
        "offset": 0,
        "count": total_weight_tiles * _ROWS * _COLS,
        "dtype": "int8",
        "layout": "weight_tiles",
        "num_k_tiles": num_k_tiles,
        "num_weight_tiles": num_weight_tiles,
    })

    # Activation rows → AMEM: address = row * num_k_tiles + k
    data_plan.append({
        "type": "AMEM",
        "addr": 0,
        "param": act_arg,
        "offset": 0,
        "count": num_vecs * k_cols,
        "dtype": "int8",
        "layout": "act_tiles",
        "num_vecs": num_vecs,
        "num_k_tiles": num_k_tiles,
    })

    # Bias → VMEM (if present)
    bias_vmem_base = 0
    if has_bias:
        bias_size = params[bias_arg].dtype.size
        data_plan.append({
            "type": "VMEM",
            "addr": bias_vmem_base,
            "param": bias_arg,
            "offset": 0,
            "count": bias_size,
            "dtype": "int32",
            "layout": "bias",
            "mode": bias_mode,
            "num_weight_tiles": num_weight_tiles,
        })

    # Output VMEM addresses
    out_vmem_base = num_weight_tiles if has_bias else 0

    # PSUM accumulation path eliminates the VPU_ADD chain and per-K
    # LOAD_MXU_RESULT for multi-K-tile GEMM. SXU_PSUM_CLEAR zeroes the
    # bucket in one cycle, so no zero-tile preload is needed.
    use_psum = num_k_tiles > 1

    # Generate SXU instructions
    instructions = _generate_gemm_sxu_instructions(
        num_vecs, num_k_tiles, num_weight_tiles,
        has_bias=has_bias, bias_vmem_base=bias_vmem_base,
        has_relu=has_relu,
        use_psum=use_psum,
    )
    outputs: list[dict] = []
    for row in range(num_vecs):
        for tile_idx in range(num_weight_tiles):
            out_addr = out_vmem_base + row * num_weight_tiles + tile_idx
            outputs.append({
                "addr": out_addr,
                "param": out_arg,
                "offset": (row * out_cols + tile_idx * _COLS),
                "count": _COLS,
            })

    return {
        "op": "SXU_PROGRAM",
        "instructions": instructions,
        "data_plan": data_plan,
        "outputs": outputs,
        "num_output_tiles": num_vecs * num_weight_tiles,
        "num_vecs": num_vecs,
        "num_k_tiles": num_k_tiles,
        "num_weight_tiles": num_weight_tiles,
        "out": out_arg,
    }


def _render_pad_sxu_program(uops: list[UOp]) -> dict | None:
    """Render single-tile unrolled movement kernels (PAD / FLIP / permute)
    as a LOAD/STORE with a PAD_FILL VMEM preload that scatters source
    positions into an arbitrary output layout, zero-filling positions
    whose STORE value is CONST(0).

    Pattern: N STOREs whose values are each either LOAD(src[i]) or
    CONST(0). No ALU ops. Covers PAD (mix of LOAD+CONST), FLIP (all
    LOADs, reversed indices), and unrolled non-affine permutations
    that the affine copy renderer rejects.
    """
    op_counts = Counter(u.op.name for u in uops)
    if not op_counts.get("STORE") or not op_counts.get("LOAD"):
        return None
    # Pure movement only: no data-path ALU, ternary, cast, or WMMA.
    data_alu = _data_alu_ops(uops)
    if sum(data_alu.values()) > 0:
        return None
    for n in ("WHERE", "MOD", "RECIP", "RECIPROCAL", "TRUNC", "WMMA", "MULACC",
              "SELECT", "CAST"):
        if op_counts.get(n, 0):
            return None

    stores = [u for u in uops if u.op is Ops.STORE]

    # Extract per-STORE info: (dst_pos, src_pos or None for zero-fill).
    def _const_tail(idx_uop):
        # Walk an INDEX chain and return the final integer CONST position.
        if idx_uop.op is not Ops.INDEX:
            return None
        last = idx_uop.src[-1]
        if last.op is Ops.CONST and isinstance(last.arg, int):
            return int(last.arg)
        return None

    def _load_src_pos(load_uop):
        if load_uop.op is not Ops.LOAD:
            return None
        return _const_tail(load_uop.src[0])

    pad_map: list[tuple[int, int | None]] = []  # (dst_pos, src_pos or None)
    out_params: set = set()
    src_params: set = set()
    for s in stores:
        dst = _const_tail(s.src[0])
        if dst is None:
            return None
        out_param = _find_unique_param_arg(s.src[0])
        out_params.add(out_param)
        val = s.src[1]
        if val.op is Ops.CONST:
            if val.arg != 0:
                return None  # only zero-pad for now
            pad_map.append((dst, None))
        elif val.op is Ops.LOAD:
            sp = _load_src_pos(val)
            if sp is None:
                return None
            pad_map.append((dst, sp))
            src_params.add(_find_unique_param_arg(val.src[0]))
        else:
            return None

    if len(out_params) != 1 or len(src_params) != 1:
        return None
    out_arg = next(iter(out_params))
    src_arg = next(iter(src_params))

    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    out_size = params[out_arg].dtype.size
    src_size = params[src_arg].dtype.size
    if out_size > _TILE_ELEMS or src_size > _TILE_ELEMS:
        return None  # single-tile only for now

    # Pad all CONST stores to 0 slot, LOAD stores to source. Preloaded VMEM
    # tile is the padded output; the SXU program copies it straight through.
    data_plan = [{
        "type": "VMEM", "addr": 0, "param": src_arg,
        "mode": "PAD_FILL", "pad_map": pad_map,
        "offset": 0, "count": _TILE_ELEMS, "dtype": "int32",
    }]
    all_instrs = [_load(0, 0), _store(1, 0), _halt()]
    outputs = [{"addr": 1, "param": out_arg, "offset": 0, "count": out_size}]
    return {
        "op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
        "outputs": outputs, "num_output_tiles": 1, "out": out_arg,
    }


def _render_rowbc_copy_sxu_program(uops: list[UOp]) -> dict | None:
    """Render a single-input row-broadcast copy as LOAD + BROADCAST_ROW.

    Covers ``Tensor([[v0, .., vM]]).expand(N, M)``: a small source row (<=_COLS
    elements, no RANGE dependence) replicated down every output row. This is a
    structured broadcast, not a per-element map, so it stays a recognizer.
    """
    op_counts = Counter(u.op.name for u in uops)
    if not op_counts.get("STORE") or not op_counts.get("LOAD"):
        return None
    if sum(_data_alu_ops(uops).values()) > 0:
        return None
    for n in ("WHERE", "MOD", "RECIP", "RECIPROCAL", "TRUNC", "WMMA", "MULACC",
              "SELECT", "CAST"):
        if op_counts.get(n, 0):
            return None

    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2:
        return None
    stores = [u for u in uops if u.op is Ops.STORE]
    out_params = {_find_unique_param_arg(s.src[0]) for s in stores}
    out_params.discard(None)
    if len(out_params) != 1:
        return None
    out_arg = next(iter(out_params))
    src_params = [k for k in params if k != out_arg]
    if len(src_params) != 1:
        return None
    src_arg = src_params[0]
    out_size = params[out_arg].dtype.size
    src_size = params[src_arg].dtype.size
    if out_size <= 0 or src_size <= 0:
        return None
    if params[out_arg].dtype.base.itemsize != params[src_arg].dtype.base.itemsize:
        return None

    def _index_of(addr_uop):
        return addr_uop.src[1] if addr_uop.op is Ops.INDEX else None

    def _has_range(u, seen=None):
        if seen is None: seen = set()
        if id(u) in seen: return False
        seen.add(id(u))
        if u.op is Ops.RANGE: return True
        return any(_has_range(s, seen) for s in u.src)

    load_idxs_all = [_index_of(s.src[1].src[0]) for s in stores if s.src[1].op is Ops.LOAD]
    if len(load_idxs_all) != len(stores):
        return None
    all_loads_no_range = all(li is not None and not _has_range(li) for li in load_idxs_all)
    # Row-broadcast: every LOAD index is range-independent, the output is a
    # multiple of a small source row. The InstSel walker rejects this — its
    # source-offset relation to the STORE index is not a constant.
    if not (all_loads_no_range and src_size <= _COLS and out_size % src_size == 0
            and out_size > src_size):
        return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan = [{
        "type": "VMEM", "addr": 0, "param": src_arg,
        "offset": 0, "count": src_size, "dtype": "int32",
    }]
    all_instrs = [_load(0, 0), "2 10 0 1 0 0 0 0 0 0"]  # SXU_BROADCAST_ROW vd=1 vs=0 srcRow=0
    outputs = []
    for t in range(num_tiles):
        offset = t * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)
        out_addr = 1 + t
        all_instrs.append(_store(out_addr, 1))
        outputs.append({"addr": out_addr, "param": out_arg, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {
        "op": "SXU_PROGRAM", "instructions": all_instrs,
        "data_plan": data_plan, "outputs": outputs,
        "num_output_tiles": num_tiles, "out": out_arg,
    }


def _render_transpose_sxu_program(uops: list[UOp]) -> dict | None:
    """Render a 4x4 2D permute(1,0) as LOAD + XLU_TRANSPOSE + STORE.

    Detects the GROUP-of-4-STOREs kernel tinygrad emits for contiguous
    .permute(1,0) over a 4x4 tile: one RANGE loop over rows, 4 STOREs per
    iteration. Each STORE writes a LOAD whose index transposes the store
    index. No data-path ALU ops.
    """
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2:
        return None
    data_alu = _data_alu_ops(uops)
    if sum(data_alu.values()) > 0:
        return None
    op_counts = Counter(u.op.name for u in uops)
    # Two shapes match:
    #   - int32 GROUP(4)-unrolled: 4 STOREs + 4 LOADs + 1 RANGE + 1 MUL
    #   - float32 VECTORIZE-unrolled: 1 STORE + 4 LOADs + 1 VECTORIZE + 1 RANGE + 1 MUL
    shape_int = (op_counts.get("STORE", 0) == 4 and op_counts.get("LOAD", 0) == 4
                 and op_counts.get("RANGE", 0) == 1 and op_counts.get("MUL", 0) == 1)
    shape_float = (op_counts.get("STORE", 0) == 1 and op_counts.get("LOAD", 0) == 4
                   and op_counts.get("VECTORIZE", 0) == 1 and op_counts.get("RANGE", 0) == 1
                   and op_counts.get("MUL", 0) == 1)
    if not (shape_int or shape_float):
        return None

    stores = [u for u in uops if u.op is Ops.STORE]
    out_params = {_find_unique_param_arg(s.src[0]) for s in stores}
    out_params.discard(None)
    if len(out_params) != 1:
        return None
    out_arg = next(iter(out_params))
    src_params = [k for k in params if k != out_arg]
    if len(src_params) != 1:
        return None
    src_arg = src_params[0]
    out_size = params[out_arg].dtype.size
    src_size = params[src_arg].dtype.size
    if out_size != 16 or src_size != 16:
        return None
    if params[out_arg].dtype.base.itemsize != params[src_arg].dtype.base.itemsize:
        return None
    # Each STORE value must be a LOAD of src_arg, or a VECTORIZE of LOADs
    # (possibly wrapped in CAST for float bitcasting).
    loads = []
    store_idx_uops = []
    for s in stores:
        val = s.src[1]
        # Unwrap CAST on the address side too (float store addresses get bitcast wrapped).
        addr = s.src[0]
        while addr.op is Ops.CAST:
            addr = addr.src[0]
        store_idx_uops.append(addr.src[1] if addr.op is Ops.INDEX else None)
        while val.op is Ops.CAST:
            val = val.src[0]
        if val.op is Ops.VECTORIZE:
            for l in val.src:
                while l.op is Ops.CAST:
                    l = l.src[0]
                if l.op is not Ops.LOAD or _find_unique_param_arg(l) != src_arg:
                    return None
                loads.append(l)
        elif val.op is Ops.LOAD:
            if _find_unique_param_arg(val) != src_arg:
                return None
            loads.append(val)
        else:
            return None
    if len(loads) != 4:
        return None

    # Distinguish transpose from reshape/copy by comparing CONST offsets in
    # load vs store index expressions. Reshape: matching const sets (e.g. 0/1/2/3
    # each side). Transpose: store consts are {0,1,2,3} while load consts are
    # multiples of row stride {0,4,8,12}.
    def _consts_in(u, seen=None):
        if seen is None: seen = set()
        if id(u) in seen: return []
        seen.add(id(u))
        if u.op is Ops.CONST and isinstance(u.arg, int):
            return [u.arg]
        out = []
        for s in u.src: out += _consts_in(s, seen)
        return out
    load_idx_uops = [l.src[0].src[1] if l.src[0].op is Ops.INDEX else None for l in loads]
    if any(u is None for u in store_idx_uops) or any(u is None for u in load_idx_uops):
        return None
    store_consts = sorted(set(sum([_consts_in(u) for u in store_idx_uops], [])))
    load_consts = sorted(set(sum([_consts_in(u) for u in load_idx_uops], [])))
    if not ({4, 8, 12} <= set(load_consts)):
        return None
    if len({id(u) for u in load_idx_uops}) != 4:
        return None
    if set(load_consts) == set(store_consts):
        return None

    # Emit: LOAD VMEM[0]→VREG 0, XLU_TRANSPOSE VREG 1 = transpose(VREG 0),
    # STORE VREG 1 → VMEM[1].
    data_plan = [{
        "type": "VMEM", "addr": 0, "param": src_arg,
        "offset": 0, "count": 16, "dtype": "int32",
    }]
    instructions = [
        _load(0, 0),
        f"2 12 0 1 0 0 0 0 0 0",  # SXU_DISPATCH_XLU_TRANSPOSE vd=1, vs=0
        _store(1, 1),
        _halt(),
    ]
    outputs = [{"addr": 1, "param": out_arg, "offset": 0, "count": 16}]
    return {
        "op": "SXU_PROGRAM",
        "primitive": "TRANSPOSE",
        "instructions": instructions,
        "data_plan": data_plan,
        "outputs": outputs,
        "num_output_tiles": 1,
        "out": out_arg,
    }

def _generate_gemm_sxu_instructions(num_vecs: int, num_k_tiles: int, num_weight_tiles: int,
                                     *, has_bias: bool = False, bias_vmem_base: int = 0,
                                     has_relu: bool = False,
                                     use_psum: bool = False) -> list[str]:
    """Generate SXU instruction strings for a GEMM kernel.

    When use_psum=True and num_k_tiles>1, accumulate K-tiles in the
    PSUM bucket bank instead of reading each partial into a vreg and
    chaining VPU_ADDs. This eliminates num_k_tiles-1 VPU_ADDs and
    num_k_tiles LOAD_MXU_RESULT instructions per output tile.
    """
    out_vmem_base = num_weight_tiles if has_bias else 0
    prog_lines: list[str] = []

    psum_path = use_psum and num_k_tiles > 1

    for row in range(num_vecs):
        for tile_idx in range(num_weight_tiles):
            if psum_path:
                # Zero bucket 0 in one cycle, accumulate every K-tile
                # into row 0, then extract the accumulated row into v0.
                prog_lines.append(_psum_clear(0))
                for k in range(num_k_tiles):
                    wmem_addr = k * num_weight_tiles + tile_idx
                    amem_addr = row * num_k_tiles + k
                    prog_lines.append(_mxu_psum_acc(wmem_addr, amem_addr, 1, 0, 0))
                    prog_lines.append(_wait_mxu())
                prog_lines.append(_psum_read_row(0, 0, 0))  # v0 := psum[0].row[0]
                cur = 0
            else:
                # MXU dispatches for K-tile accumulation (legacy VPU path)
                for k in range(num_k_tiles):
                    wmem_addr = k * num_weight_tiles + tile_idx
                    amem_addr = row * num_k_tiles + k
                    vreg_k = k
                    prog_lines.append(_mxu(wmem_addr, amem_addr, 1))
                    prog_lines.append(_wait_mxu())
                    prog_lines.append(_load_mxu_result(vreg_k))

                # Accumulate K-tiles
                if num_k_tiles == 1:
                    cur = 0
                else:
                    acc = num_k_tiles
                    prog_lines.append(_vpu(acc, 0, _VPU_OPS["ADD"], 1))
                    cur = acc
                    for k in range(2, num_k_tiles):
                        nxt = cur + 1
                        prog_lines.append(_vpu(nxt, cur, _VPU_OPS["ADD"], k))
                        cur = nxt

            # Bias epilogue
            if has_bias:
                bias_vreg = cur + 1
                prog_lines.append(_load(bias_vreg, bias_vmem_base + tile_idx))
                result_vreg = bias_vreg + 1
                prog_lines.append(_vpu(result_vreg, cur, _VPU_OPS["ADD"], bias_vreg))
                cur = result_vreg

            # ReLU epilogue
            if has_relu:
                nxt = cur + 1
                prog_lines.append(_vpu(nxt, cur, 2))  # VPU_RELU opcode
                cur = nxt

            # Store result
            out_addr = out_vmem_base + row * num_weight_tiles + tile_idx
            prog_lines.append(_store(out_addr, cur))

    prog_lines.append(_halt())
    return prog_lines


def _find_unique_param_arg(u: UOp) -> int | None:
    params = {node.arg for node in u.toposort() if node.op is Ops.PARAM}
    if len(params) != 1:
        return None
    arg = next(iter(params))
    return arg if isinstance(arg, int) else None

def _extract_wmma_epilogue(uops: list[UOp], params: dict[int, UOp], out_arg: int, act_arg: int, weight_arg: int,
                           out_size: int, out_cols: int) -> tuple[list[dict], str | None]:
    op_counts = Counter(u.op.name for u in uops)
    extra_params = sorted(k for k in params if k not in {out_arg, act_arg, weight_arg})
    epilogue: list[dict] = []

    if op_counts.get("ADD", 0) and len(extra_params) > 0:
        if len(extra_params) != 1:
            return [], f"wmma add epilogue expected one extra param, found {len(extra_params)}"
        bias_arg = extra_params[0]
        bias_size = params[bias_arg].dtype.size
        if bias_size == out_cols:
            epilogue.append({"op": "ADD", "arg": bias_arg, "mode": "ROW_BROADCAST"})
        elif bias_size == out_size:
            epilogue.append({"op": "ADD", "arg": bias_arg, "mode": "FULL"})
        else:
            return [], f"wmma add epilogue unsupported bias size {bias_size}"

    if op_counts.get("WHERE", 0) or op_counts.get("CMPLT", 0):
        if op_counts.get("WHERE", 0) != out_size or op_counts.get("CMPLT", 0) != out_size:
            return [], f"wmma relu epilogue expected {out_size} lane ops, got where={op_counts.get('WHERE', 0)} cmplt={op_counts.get('CMPLT', 0)}"
        epilogue.append({"op": "RELU"})

    unsupported = {name for name, count in op_counts.items() if count and name in {"MAX", "CMPNE", "CMPEQ"}}
    if unsupported:
        return [], f"wmma epilogue present: {', '.join(sorted(unsupported))}"
    return epilogue, None


def _apply_gemm_epilogue(bufs: tuple[bytearray, ...], out_i32: np.ndarray, prog: dict) -> np.ndarray:
    out = out_i32
    num_vecs = int(prog["num_vecs"])
    out_cols = int(prog["num_weight_tiles"]) * _COLS
    out_size = num_vecs * out_cols
    for step in prog.get("epilogue", []):
        if step["op"] == "ADD":
            raw = np.frombuffer(bytes(bufs[int(step["arg"])]), dtype="<i4")
            if step["mode"] == "ROW_BROADCAST":
                if raw.size < out_cols:
                    raise RuntimeError(f"TinyTPU row-broadcast bias expected at least {out_cols} elements, got {raw.size}")
                out = (out.reshape(num_vecs, out_cols) + raw[:out_cols].reshape(1, out_cols)).reshape(out_size)
            elif step["mode"] == "FULL":
                if raw.size < out_size:
                    raise RuntimeError(f"TinyTPU full bias expected at least {out_size} elements, got {raw.size}")
                out = out + raw[:out_size]
            else:
                raise RuntimeError(f"unknown TinyTPU GEMM epilogue mode {step['mode']}")
        elif step["op"] == "RELU":
            out = np.maximum(out, 0)
        else:
            raise RuntimeError(f"unknown TinyTPU GEMM epilogue op {step['op']}")
    return np.asarray(out, dtype=np.int32)


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

def analyze_tinytpu_uops(uops:list[UOp]) -> dict:
    """Slim legacy analyzer — only handles VPU_BINARY, VPU_PROGRAM, VPU_ROWBC_BINARY patterns
    not yet migrated to SXU_PROGRAM (scalar-const DIV/MIN/MOD)."""
    params = [u for u in uops if u.op is Ops.PARAM]
    op_counts = Counter(u.op.name for u in uops)
    # Reject complex float math (sqrt, log2, sin, exp2) — these decompose into
    # many ops including BITCAST and would otherwise match DIV/RECIP patterns.
    if op_counts.get("BITCAST", 0) > 0:
        return {"supported": False, "kind": None, "reason": "complex float math not supported", "notes": [],
                "out_arg": None, "lhs_arg": None, "lhs_const": None, "rhs_arg": None, "rhs_const": None,
                "num_elems": None, "vpu_op": None, "inputs": None, "steps": None, "output_reg": None,
                "lhs_broadcast": False, "rhs_broadcast": False}

    diag = {
        "supported": False, "kind": None, "reason": "", "notes": [],
        "out_arg": None, "lhs_arg": None, "lhs_const": None, "rhs_arg": None, "rhs_const": None,
        "num_elems": None, "vpu_op": None, "inputs": None, "steps": None, "output_reg": None,
        "lhs_broadcast": False, "rhs_broadcast": False,
    }

    param_sizes: dict[int, int] = {}
    for p in params:
        if not isinstance(p.dtype, PtrDType):
            return diag
        param_sizes[p.arg] = p.dtype.size

    binary_vpu_ops = _VPU_OPS
    matched_single_binary_ops = [name for name in binary_vpu_ops if op_counts.get(name, 0) in {1, 4}]
    matched_grouped_binary_ops = [("CMPNE" if op_counts.get("CMPNE", 0) else "CMPLT" if op_counts.get("CMPLT", 0) else "MAX" if op_counts.get("MAX", 0) else "MUL" if op_counts.get("MUL", 0) > 1 else "ADD")] if any(op_counts.get(name, 0) for name in binary_vpu_ops) else []
    out_is_bool = any(p.arg == 0 and "bool" in str(p.dtype) for p in params)
    in_is_bool = any(p.arg != 0 and "bool" in str(p.dtype) for p in params)
    scalar_const_binary_ops = [name for name in binary_vpu_ops
                               if op_counts.get(name, 0) in {1, 4} and _find_scalar_const_binary(uops, name) is not None]
    if out_is_bool:
        scalar_const_binary_ops = [name for name in scalar_const_binary_ops if name in {"CMPLT", "CMPNE", "XOR"}]
    else:
        scalar_const_binary_ops = [name for name in scalar_const_binary_ops if name in {"ADD", "MUL", "MAX", "SUB", "SHL", "SHR", "XOR"}]
    if out_is_bool and "XOR" in scalar_const_binary_ops and _find_scalar_const_binary(uops, "XOR") == 1:
        scalar_const_binary_ops = ["XOR"]
    if len(scalar_const_binary_ops) > 1:
        scalar_const_binary_ops = [max(scalar_const_binary_ops, key=lambda n: op_counts.get(n, 0))]
    scalar_const = _find_scalar_const_binary(uops, scalar_const_binary_ops[0]) if len(scalar_const_binary_ops) == 1 else None
    reverse_sub_const = _find_reverse_sub_const(uops)
    eq_scalar_const = _find_eq_scalar_const(uops)
    is_eq_from_cmpne = _has_eq_from_cmpne(uops)
    clip_consts = _find_clip_consts(uops)
    bool_cast = _find_bool_to_int_cast(uops)
    divmod_pattern = _classify_divmod_pattern(uops)
    # tinygrad may leave pointer reads as INDEX nodes for a fully upcast 16-lane
    # tile, while smaller tiles materialize explicit LOAD UOps.
    _has_bool_logic_op = op_counts.get("AND", 0) > 0 or op_counts.get("OR", 0) > 0 or op_counts.get("XOR", 0) > 0
    _has_complex_op = any(op_counts.get(x, 0) for x in ("IDIV", "MOD", "RECIP"))
    is_single_binary = len(params) == 3 and len(matched_single_binary_ops) == 1 and op_counts.get("LOAD", 0) in {0, 2} and op_counts.get("STORE", 0) == 1 and not _has_bool_logic_op
    _has_fused_cmp = op_counts.get("CMPLT", 0) > 0 and op_counts.get("WHERE", 0) > 0
    is_grouped_binary = len(params) == 3 and len(matched_grouped_binary_ops) == 1 and op_counts.get("STORE", 0) == 4 and op_counts.get("GROUP", 0) == 1 and not _has_bool_logic_op and not _has_fused_cmp
    # 2-param grouped scalar-const: RANGE+GROUP vectorised unary-like op.
    # MUL=5 (4 elem + 1 stride) or ADD/SUB/MAX with count not in {1,4}.
    _is_grouped_sc = (len(params) == 2 and op_counts.get("GROUP", 0) == 1
                      and op_counts.get("STORE", 0) == 4 and op_counts.get("LOAD", 0) == 4
                      and op_counts.get("RANGE", 0) == 1 and not _has_bool_logic_op)
    if len(params) in {2, 3} and divmod_pattern is not None and divmod_pattern[0] == "IDIV":
        _, rhs_const = divmod_pattern
        out_size, input_args = _resolve_binary_io(param_sizes)
        if out_size is not None and len(input_args) >= 1 and 0 < out_size and all(param_sizes[arg] in {1, out_size} for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu div",
                "out_arg": 0,
                "lhs_arg": input_args[0],
                "lhs_const": None,
                "rhs_arg": input_args[1] if len(input_args) > 1 else None,
                "rhs_const": rhs_const,
                "lhs_broadcast": _arg_needs_broadcast(input_args[0], param_sizes, out_size),
                "rhs_broadcast": _arg_needs_broadcast(input_args[1], param_sizes, out_size) if len(input_args) > 1 else False,
                "num_elems": out_size,
                "vpu_op": binary_vpu_ops["DIV"],
            })
            return diag
        diag["reason"] = f"unsupported vpu div sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU DIV lowering handles int32 elementwise outputs with optional scalar broadcasting.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif len(params) in {2, 3} and divmod_pattern is not None and divmod_pattern[0] == "MOD":
        _, rhs_const = divmod_pattern
        out_size, input_args = _resolve_binary_io(param_sizes)
        if out_size is not None and len(input_args) >= 1 and 0 < out_size and all(param_sizes[arg] in {1, out_size} for arg in input_args):
            inputs = [{"arg": input_args[0], "bool": False, "broadcast": _arg_needs_broadcast(input_args[0], param_sizes, out_size)}]
            rhs_ref: int
            if rhs_const is not None:
                inputs.append({"const": rhs_const})
                rhs_ref = 1
            else:
                inputs.append({"arg": input_args[1], "bool": False, "broadcast": _arg_needs_broadcast(input_args[1], param_sizes, out_size)})
                rhs_ref = 1
            diag.update({
                "supported": True,
                "kind": "vpu_program",
                "reason": "supported mod via div/mul/sub bundle",
                "out_arg": 0,
                "num_elems": out_size,
                "inputs": inputs,
                "steps": [
                    {"op": binary_vpu_ops["DIV"], "lhs": 0, "rhs": rhs_ref, "dst": 2},
                    {"op": binary_vpu_ops["MUL"], "lhs": 2, "rhs": rhs_ref, "dst": 3},
                    {"op": binary_vpu_ops["SUB"], "lhs": 0, "rhs": 3, "dst": 4},
                ],
                "output_reg": 4,
            })
            return diag
        diag["reason"] = f"unsupported mod sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU MOD lowering handles int32 elementwise outputs with optional scalar broadcasting.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif len(params) == 2 and bool_cast:
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < src_size:
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported bool to int32 cast",
                "out_arg": 0,
                "lhs_arg": 1,
                "lhs_const": None,
                "rhs_arg": None,
                "rhs_const": 1,
                "lhs_broadcast": False,
                "rhs_broadcast": False,
                "num_elems": src_size,
                "vpu_op": binary_vpu_ops["MUL"],
                "bool_in": True,
            })
            return diag
        diag["reason"] = f"unsupported bool to int32 cast sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU bool->int32 cast lowering handles one bool VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif len(params) == 2 and clip_consts is not None and op_counts.get("STORE", 0) in {1, 4}:
        lo_const, hi_const = clip_consts
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < src_size:
            diag.update({
                "supported": True,
                "kind": "vpu_program",
                "reason": "supported clip via min/max bundle",
                "out_arg": 0,
                "num_elems": src_size,
                "inputs": [
                    {"arg": 1, "bool": False, "broadcast": False},
                    {"const": hi_const},
                    {"const": lo_const},
                ],
                "steps": [
                    {"op": binary_vpu_ops["MIN"], "lhs": 0, "rhs": 1, "dst": 3},
                    {"op": binary_vpu_ops["MAX"], "lhs": 3, "rhs": 2, "dst": 4},
                ],
                "output_reg": 4,
            })
            return diag
        diag["reason"] = f"unsupported clip sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU clip lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 2 and not _has_complex_op and op_counts.get("MUL", 0) > 0 and op_counts.get("CMPLT", 0) > 0 and
          op_counts.get("CMPNE", 0) > 0 and op_counts.get("WHERE", 0) >= 2 and op_counts.get("STORE", 0) in {1, 4}):
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < src_size:
            diag.update({
                "supported": True,
                "kind": "vpu_program",
                "reason": "supported abs via sub/max bundle",
                "out_arg": 0,
                "num_elems": src_size,
                "inputs": [
                    {"arg": 1, "bool": False, "broadcast": False},
                    {"const": 0},
                ],
                "steps": [
                    {"op": binary_vpu_ops["SUB"], "lhs": 1, "rhs": 0, "dst": 2},
                    {"op": binary_vpu_ops["MAX"], "lhs": 0, "rhs": 2, "dst": 3},
                ],
                "output_reg": 3,
            })
            return diag
        diag["reason"] = f"unsupported abs sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU abs lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 3 and op_counts.get("ADD", 0) > 0 and op_counts.get("CMPLT", 0) > 0 and
          op_counts.get("WHERE", 0) > 0 and op_counts.get("STORE", 0) in {1, 4}):
        out_size = param_sizes.get(0)
        input_args = [arg for arg in sorted(param_sizes) if arg != 0]
        if out_size is not None and len(input_args) == 2 and 0 < out_size and all(param_sizes[arg] in {1, out_size} for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_program",
                "reason": "supported fused add relu",
                "out_arg": 0,
                "num_elems": out_size,
                "inputs": [
                    {"arg": input_args[0], "bool": False, "broadcast": _arg_needs_broadcast(input_args[0], param_sizes, out_size)},
                    {"arg": input_args[1], "bool": False, "broadcast": _arg_needs_broadcast(input_args[1], param_sizes, out_size)},
                ],
                "steps": [
                    {"op": binary_vpu_ops["ADD"], "lhs": 0, "rhs": 1, "dst": 2},
                    {"op": 2, "lhs": 2, "dst": 3},
                ],
                "output_reg": 3,
            })
            return diag
        diag["reason"] = f"unsupported fused add relu sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU fused ADD+RELU lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 2 and reverse_sub_const is not None and not _has_complex_op and op_counts.get("STORE", 0) in {1, 4} and
          (op_counts.get("STORE", 0) == 4 or op_counts.get("LOAD", 0) == 1)):
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < src_size:
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu reverse sub const",
                "out_arg": 0,
                "lhs_arg": None,
                "lhs_const": reverse_sub_const,
                "rhs_arg": 1,
                "rhs_const": None,
                "lhs_broadcast": False,
                "rhs_broadcast": False,
                "num_elems": src_size,
                "vpu_op": binary_vpu_ops["SUB"],
            })
            return diag
        diag["reason"] = f"unsupported vpu reverse sub const sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU reverse SUB constant lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 2 and in_is_bool and op_counts.get("XOR", 0) > 0 and _find_scalar_const_binary(uops, "XOR") == 1 and
          not _has_complex_op and op_counts.get("STORE", 0) in {1, 4} and
          (op_counts.get("STORE", 0) == 4 or op_counts.get("LOAD", 0) == 1)):
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < src_size:
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu not",
                "out_arg": 0,
                "lhs_arg": 1,
                "lhs_const": None,
                "rhs_arg": None,
                "rhs_const": 1,
                "lhs_broadcast": False,
                "rhs_broadcast": False,
                "num_elems": src_size,
                "vpu_op": binary_vpu_ops["XOR"],
                "bool_in": True,
                "bool_out": True,
            })
            return diag
        diag["reason"] = f"unsupported vpu not sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU NOT lowering handles one bool VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 2 and len(scalar_const_binary_ops) == 1 and scalar_const is not None and not _has_complex_op and op_counts.get("STORE", 0) in {1, 4} and
          (op_counts.get("STORE", 0) == 4 or op_counts.get("LOAD", 0) == 1)):
        op_name = scalar_const_binary_ops[0]
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < src_size:
            bool_out = bool(in_is_bool and op_name == "XOR" and scalar_const == 1)
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": f"supported vpu {op_name.lower()} const",
                "out_arg": 0,
                "lhs_arg": 1,
                "lhs_const": None,
                "rhs_arg": None,
                "rhs_const": scalar_const,
                "lhs_broadcast": False,
                "rhs_broadcast": False,
                "num_elems": src_size,
                "vpu_op": binary_vpu_ops[op_name],
                "bool_in": in_is_bool,
                "bool_out": bool_out,
            })
            return diag
        diag["reason"] = f"unsupported vpu {op_name.lower()} const sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append(f"Current TinyTPU VPU {op_name} constant lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 2 and eq_scalar_const is not None and not _has_complex_op and op_counts.get("STORE", 0) in {1, 4} and
          (op_counts.get("STORE", 0) == 4 or op_counts.get("LOAD", 0) == 1)):
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < src_size:
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu cmpeq const",
                "out_arg": 0,
                "lhs_arg": 1,
                "lhs_const": None,
                "rhs_arg": None,
                "rhs_const": eq_scalar_const,
                "lhs_broadcast": False,
                "rhs_broadcast": False,
                "num_elems": src_size,
                "vpu_op": binary_vpu_ops["CMPEQ"],
            })
            return diag
        diag["reason"] = f"unsupported vpu cmpeq const sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU CMPEQ constant lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 3 and is_eq_from_cmpne and op_counts.get("STORE", 0) in {1, 4} and
          (op_counts.get("STORE", 0) == 4 or op_counts.get("LOAD", 0) in {0, 2})):
        out_size = param_sizes.get(0)
        input_args = [arg for arg in sorted(param_sizes) if arg != 0]
        if out_size is not None and len(input_args) == 2 and 0 < out_size and all(param_sizes[arg] in {1, out_size} for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu cmpeq",
                "out_arg": 0,
                "lhs_arg": input_args[0],
                "lhs_const": None,
                "rhs_arg": input_args[1],
                "rhs_const": None,
                "lhs_broadcast": _arg_needs_broadcast(input_args[0], param_sizes, out_size),
                "rhs_broadcast": _arg_needs_broadcast(input_args[1], param_sizes, out_size),
                "num_elems": out_size,
                "vpu_op": binary_vpu_ops["CMPEQ"],
            })
            return diag
        diag["reason"] = f"unsupported vpu cmpeq sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU CMPEQ lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 3 and op_counts.get("ADD", 0) > 0 and op_counts.get("MUL", 0) > 0 and
          op_counts.get("STORE", 0) in {1, 4} and
          (op_counts.get("STORE", 0) == 4 or op_counts.get("LOAD", 0) in {0, 2}) and
          any(u.op is Ops.MUL and _contains_const_int(u, -1) for u in uops)):
        out_size = param_sizes.get(0)
        input_args = [arg for arg in sorted(param_sizes) if arg != 0]
        if out_size is not None and len(input_args) == 2 and 0 < out_size and all(param_sizes[arg] in {1, out_size} for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu sub",
                "out_arg": 0,
                "lhs_arg": input_args[0],
                "lhs_const": None,
                "rhs_arg": input_args[1],
                "rhs_const": None,
                "lhs_broadcast": _arg_needs_broadcast(input_args[0], param_sizes, out_size),
                "rhs_broadcast": _arg_needs_broadcast(input_args[1], param_sizes, out_size),
                "num_elems": out_size,
                "vpu_op": binary_vpu_ops["SUB"],
            })
            return diag
        diag["reason"] = f"unsupported vpu sub sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU SUB lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif _is_grouped_sc and len(matched_grouped_binary_ops) == 1:
        # 2-param grouped scalar-const: e.g. x*-1 (neg), x*2, x+5 for 2D/large tensors.
        op_name = matched_grouped_binary_ops[0]
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        gsc = _find_grouped_elementwise_const(uops, op_name)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < out_size and gsc is not None:
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": f"supported grouped-scalar-const vpu {op_name.lower()}",
                "out_arg": 0,
                "lhs_arg": 1,
                "lhs_const": None,
                "rhs_arg": None,
                "rhs_const": gsc,
                "lhs_broadcast": False,
                "rhs_broadcast": False,
                "num_elems": out_size,
                "vpu_op": binary_vpu_ops[op_name],
                "bool_out": False,
                "bool_in": False,
            })
            return diag
        diag["reason"] = f"unsupported grouped-scalar-const {op_name} sizes {dict(sorted(param_sizes.items()))}"
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif is_single_binary or is_grouped_binary:
        op_name = (matched_single_binary_ops if is_single_binary else matched_grouped_binary_ops)[0]
        out_size = param_sizes.get(0)
        input_args = [arg for arg in sorted(param_sizes) if arg != 0]
        if out_size is not None and len(input_args) == 2 and 0 < out_size and all(param_sizes[arg] in {1, out_size} for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": f"supported vpu {op_name.lower()}",
                "out_arg": 0,
                "lhs_arg": input_args[0],
                "lhs_const": None,
                "rhs_arg": input_args[1],
                "rhs_const": None,
                "lhs_broadcast": _arg_needs_broadcast(input_args[0], param_sizes, out_size),
                "rhs_broadcast": _arg_needs_broadcast(input_args[1], param_sizes, out_size),
                "num_elems": out_size,
                "vpu_op": binary_vpu_ops[op_name],
            })
            return diag
        diag["reason"] = f"unsupported vpu {op_name.lower()} sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append(f"Current TinyTPU VPU {op_name} lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif len(params) == 3 and op_counts.get("XOR", 0) > 0 and op_counts.get("MAX", 0) > 0 and not in_is_bool:
        out_size = param_sizes.get(0)
        input_args = [arg for arg in sorted(param_sizes) if arg != 0]
        if out_size is not None and len(input_args) == 2 and 0 < out_size and all(param_sizes[arg] in {1, out_size} for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu min (via minimum decomposition)",
                "out_arg": 0,
                "lhs_arg": input_args[0],
                "lhs_const": None,
                "rhs_arg": input_args[1],
                "rhs_const": None,
                "lhs_broadcast": _arg_needs_broadcast(input_args[0], param_sizes, out_size),
                "rhs_broadcast": _arg_needs_broadcast(input_args[1], param_sizes, out_size),
                "num_elems": out_size,
                "vpu_op": binary_vpu_ops["MIN"],
            })
            return diag
        diag["reason"] = f"unsupported vpu min sizes {dict(sorted(param_sizes.items()))}"
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif len(params) == 3 and op_counts.get("AND", 0) > 0 and in_is_bool and op_counts.get("STORE", 0) >= 1:
        out_size = param_sizes.get(0)
        input_args = [arg for arg in sorted(param_sizes) if arg != 0]
        if out_size is not None and len(input_args) == 2 and 0 < out_size and all(param_sizes[arg] in {1, out_size} for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu and",
                "out_arg": 0,
                "lhs_arg": input_args[0],
                "lhs_const": None,
                "rhs_arg": input_args[1],
                "rhs_const": None,
                "lhs_broadcast": _arg_needs_broadcast(input_args[0], param_sizes, out_size),
                "rhs_broadcast": _arg_needs_broadcast(input_args[1], param_sizes, out_size),
                "num_elems": out_size,
                "vpu_op": binary_vpu_ops["AND"],
                "bool_out": True,
            })
            return diag
        diag["reason"] = f"unsupported vpu and sizes {dict(sorted(param_sizes.items()))}"
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif len(params) == 3 and op_counts.get("XOR", 0) > 0 and in_is_bool and op_counts.get("STORE", 0) >= 1:
        out_size = param_sizes.get(0)
        input_args = [arg for arg in sorted(param_sizes) if arg != 0]
        if out_size is not None and len(input_args) == 2 and 0 < out_size and all(param_sizes[arg] in {1, out_size} for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu xor",
                "out_arg": 0,
                "lhs_arg": input_args[0],
                "lhs_const": None,
                "rhs_arg": input_args[1],
                "rhs_const": None,
                "lhs_broadcast": _arg_needs_broadcast(input_args[0], param_sizes, out_size),
                "rhs_broadcast": _arg_needs_broadcast(input_args[1], param_sizes, out_size),
                "num_elems": out_size,
                "vpu_op": binary_vpu_ops["XOR"],
                "bool_out": True,
            })
            return diag
        diag["reason"] = f"unsupported vpu xor sizes {dict(sorted(param_sizes.items()))}"
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif len(params) == 3 and op_counts.get("OR", 0) > 0 and in_is_bool and op_counts.get("STORE", 0) >= 1:
        out_size = param_sizes.get(0)
        input_args = [arg for arg in sorted(param_sizes) if arg != 0]
        if out_size is not None and len(input_args) == 2 and 0 < out_size and all(param_sizes[arg] in {1, out_size} for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu or",
                "out_arg": 0,
                "lhs_arg": input_args[0],
                "lhs_const": None,
                "rhs_arg": input_args[1],
                "rhs_const": None,
                "lhs_broadcast": _arg_needs_broadcast(input_args[0], param_sizes, out_size),
                "rhs_broadcast": _arg_needs_broadcast(input_args[1], param_sizes, out_size),
                "num_elems": out_size,
                "vpu_op": binary_vpu_ops["OR"],
                "bool_out": True,
            })
            return diag
        diag["reason"] = f"unsupported vpu or sizes {dict(sorted(param_sizes.items()))}"
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    return diag


def _find_scalar_const_binary(uops:list[UOp], op_name:str) -> int | None:
    for u in uops:
        if u.op.name != op_name:
            continue
        consts = [s for s in u.src if s.op is Ops.CONST]
        non_consts = [s for s in u.src if s.op is not Ops.CONST]
        if len(consts) == 1 and len(non_consts) == 1:
            # Skip address-stride computations (RANGE × stride) in grouped kernels
            if non_consts[0].op is Ops.RANGE:
                continue
            return int(consts[0].arg)
    return None


def _find_grouped_elementwise_const(uops:list[UOp], op_name:str) -> int | None:
    """For GROUP-vectorised 2-param kernels: find the scalar const in element-wise
    ops whose non-const source is a LOAD result, skipping address computations."""
    load_ids = {id(u) for u in uops if u.op.name == "LOAD"}
    for u in uops:
        if u.op.name != op_name:
            continue
        consts = [s for s in u.src if s.op is Ops.CONST]
        non_consts = [s for s in u.src if s.op is not Ops.CONST]
        if len(consts) == 1 and len(non_consts) == 1 and id(non_consts[0]) in load_ids:
            return int(consts[0].arg)
    return None


def _find_reverse_sub_const(uops:list[UOp]) -> int | None:
    for u in uops:
        if u.op is not Ops.ADD:
            continue
        add_consts = [s for s in u.src if s.op is Ops.CONST]
        muls = [s for s in u.src if s.op is Ops.MUL]
        if len(add_consts) != 1 or len(muls) != 1:
            continue
        if _contains_const_int(muls[0], -1):
            return int(add_consts[0].arg)
    return None


def _contains_const_int(u:UOp, value:int, seen:set[UOp]|None=None) -> bool:
    seen = set() if seen is None else seen
    if u in seen:
        return False
    seen.add(u)
    if u.op is Ops.CONST and not isinstance(u.arg, bool) and int(u.arg) == value:
        return True
    return any(_contains_const_int(s, value, seen) for s in u.src)


def _has_eq_from_cmpne(uops:list[UOp]) -> bool:
    for u in uops:
        if u.op is not Ops.CMPNE:
            continue
        if any(s.op is Ops.CONST and s.arg is True for s in u.src) and any(s.op is Ops.CMPNE for s in u.src):
            return True
    return False


def _find_eq_scalar_const(uops:list[UOp]) -> int | None:
    for u in uops:
        if u.op is not Ops.CMPNE:
            continue
        if not any(s.op is Ops.CONST and s.arg is True for s in u.src):
            continue
        inner = next((s for s in u.src if s.op is Ops.CMPNE), None)
        if inner is None:
            continue
        consts = [s for s in inner.src if s.op is Ops.CONST]
        non_bool_consts = [s for s in consts if not isinstance(s.arg, bool)]
        if len(non_bool_consts) == 1:
            return int(non_bool_consts[0].arg)
    return None


def _find_clip_consts(uops:list[UOp]) -> tuple[int, int] | None:
    for u in uops:
        if u.op is not Ops.WHERE:
            continue
        outer_cond, outer_true, inner = u.src
        if outer_cond.op is not Ops.CMPLT or outer_true.op is not Ops.CONST or inner.op is not Ops.WHERE:
            continue
        inner_cond, inner_true, inner_false = inner.src
        if inner_cond.op is not Ops.CMPLT or inner_true.op is not Ops.CONST or inner_false.op not in {Ops.INDEX, Ops.LOAD}:
            continue
        hi_cmp_lhs, hi_cmp_rhs = outer_cond.src
        lo_cmp_lhs, lo_cmp_rhs = inner_cond.src
        if hi_cmp_lhs.op is Ops.CONST and hi_cmp_rhs is inner and lo_cmp_lhs is inner_false and lo_cmp_rhs.op is Ops.CONST:
            return int(lo_cmp_rhs.arg), int(hi_cmp_lhs.arg)
    return None


def _find_bool_to_int_cast(uops:list[UOp]) -> bool:
    return any(u.op is Ops.CAST and "bool" in str(u.src[0].dtype) and "int" in str(u.dtype) for u in uops)


def _classify_divmod_pattern(uops:list[UOp]) -> tuple[str, int | None] | None:
    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None
    values = [s.src[1] for s in stores]
    if any(u.op is Ops.IDIV for u in uops) and all(v.op is Ops.WHERE for v in values):
        return "IDIV", _find_scalar_const_binary(uops, "IDIV")
    if any(u.op is Ops.MOD for u in uops) and all(v.op is Ops.ADD for v in values):
        return "MOD", _find_scalar_const_binary(uops, "MOD")
    return None


def _resolve_binary_io(param_sizes: dict[int, int]) -> tuple[int | None, list[int]]:
    out_size = param_sizes.get(0)
    return out_size, [arg for arg in sorted(param_sizes) if arg != 0]


def _arg_needs_broadcast(arg: int | None, param_sizes: dict[int, int], out_size: int | None) -> bool:
    return arg is not None and out_size is not None and out_size > 1 and param_sizes.get(arg) == 1


# ---------------------------------------------------------------------------
# Compact TASM helpers for bundle construction
# ---------------------------------------------------------------------------
# These produce the wire-format integer lines consumed by TbTinyTPURuntime.
# See doc/tinytpu_asm.md for the full TASM specification.

def _vmem(addr: int, vals: list[int]) -> str:
    return "5 " + str(addr) + " " + " ".join(str(v) for v in vals)

def _wmem(addr: int, vals: list[int]) -> str:
    return "0 " + str(addr) + " " + " ".join(str(v) for v in vals)

def _amem(addr: int, vals: list[int]) -> str:
    return "1 " + str(addr) + " " + " ".join(str(v) for v in vals)

def _load(vd: int, vmem_src: int) -> str:
    return f"2 0 {vmem_src} {vd} 0 0 0 0 0 0"

def _store(vmem_dst: int, vs: int) -> str:
    return f"2 1 {vmem_dst} 0 {vs} 0 0 0 0 0"

def _vpu(vd: int, va: int, op: int, vb: int = 0) -> str:
    return f"2 2 0 {vd} {va} {op} {vb} 0 0 0"

def _vpu_bg(vd: int, va: int, op: int, vb: int = 0) -> str:
    # DISPATCH_VPU_BG (opcode 41). Background-collect dual-issue path
    # for single-cycle VPU ops. Main FSM advances pc the same cycle;
    # a background rule retires the vreg write when vpu.isDone.
    return f"2 41 0 {vd} {va} {op} {vb} 0 0 0"

def _vpu_exp2(vd: int, va: int) -> str:
    # VPU_EXP2 (opcode 51). Multi-cycle walker — SXU stalls on vpu.isDone
    # until TranscUnit finishes its lane-by-lane Horner. ~80 cycles per tile.
    return _vpu(vd, va, _VPU_OPS["EXP2"])

def _vpu_log2(vd: int, va: int) -> str:
    # VPU_LOG2 (opcode 52). Range-reduced (split x = m * 2^e) polynomial
    # in the TranscUnit walker. Exact at powers of two, ~28% error at
    # worst-case fractional inputs. ~96 cycles per tile (6 steps/lane).
    return _vpu(vd, va, _VPU_OPS["LOG2"])

def _vpu_sin(vd: int, va: int) -> str:
    # VPU_SIN (opcode 53). Degree-5 Taylor through the TranscUnit walker.
    # Accurate for |x| <= π/2; diverges for |x| > π. Upstream range
    # reduction (mod 2π + quadrant fold) must be emitted by the renderer.
    return _vpu(vd, va, _VPU_OPS["SIN"])

def _select(vd: int, cond: int, lhs: int, rhs: int) -> str:
    return f"2 8 0 {vd} {cond} 0 {lhs} {rhs} 0 0"

def _broadcast_scalar(vd: int, vs: int, row: int = 0, col: int = 0) -> str:
    sel = ((row & 0x3) << 2) | (col & 0x3)
    return f"2 9 0 {vd} {vs} 0 {sel} 0 0 0"

def _broadcast_row(vd: int, vs: int, row: int = 0) -> str:
    return f"2 10 0 {vd} {vs} 0 {row} 0 0 0"

def _broadcast_col(vd: int, vs: int, col: int = 0) -> str:
    return f"2 11 0 {vd} {vs} 0 {col} 0 0 0"

def _broadcast(vn: int, lane: int = 0) -> str:
    return f"2 3 0 {vn} {vn} 0 {lane} 0 0 0"

def _mxu(wbase: int, abase: int, tiles: int,
         psum_addr: int = 0, psum_row: int = 0, psum_mode: int = 0) -> str:
    # PSUM target fields repurpose unused vreg slots in DISPATCH_MXU:
    #   vregDst=psum_addr, vregSrc=psum_row, vregSrc2=psum_mode
    # Mode encoding: 0=PSUM_OFF, 1=PSUM_WRITE, 2=PSUM_ACCUMULATE.
    return (f"2 4 0 {psum_addr} {psum_row} 0 {psum_mode} "
            f"{wbase} {abase} {tiles}")

def _mxu_psum_write(wbase: int, abase: int, tiles: int,
                    psum_addr: int, psum_row: int) -> str:
    return _mxu(wbase, abase, tiles, psum_addr, psum_row, 1)

def _mxu_psum_acc(wbase: int, abase: int, tiles: int,
                  psum_addr: int, psum_row: int) -> str:
    return _mxu(wbase, abase, tiles, psum_addr, psum_row, 2)

def _mxu_accumulate(wbase: int, abase: int, tiles: int) -> str:
    # SXU_DISPATCH_MXU_ACCUMULATE opcode = 23. Routes through
    # Controller.startAccumulate: WS feed path with drain-time PE
    # clear skipped, so consecutive dispatches sum into the same PE
    # accumulator (multi-K-tile GEMM). Not a distinct dataflow — the
    # PE still holds a preloaded weight. PSUM plumbing not available.
    return f"2 23 0 0 0 0 0 {wbase} {abase} {tiles}"

def _mxu_clear() -> str:
    # SXU_MXU_CLEAR opcode = 24. Zeroes the systolic-array PE
    # accumulators. Needed between accumulate epochs and when
    # re-entering WS from a previous accumulate/OS dispatch.
    return "2 24 0 0 0 0 0 0 0 0"

def _mxu_os(wbase: int, abase: int, klen: int) -> str:
    # SXU_DISPATCH_MXU_OS opcode = 25. Routes through Controller.startOS:
    # real output-stationary — weights + activations both stream as a
    # staircase, each PE holds its own psum, full (rows x cols) psum
    # drained via resultsMatrix(). klen reuses the MXU tileLen field
    # (<= rows for the current single-tile weight SRAM read).
    return f"2 25 0 0 0 0 0 {wbase} {abase} {klen}"

def _load_mxu_matrix_row(vd: int, row: int) -> str:
    # SXU_LOAD_MXU_MATRIX_ROW opcode = 26. Copies ctrl.resultsMatrix[row]
    # into row 0 of vd. Intended for draining an OS dispatch row-by-row.
    return f"2 26 0 {vd} {row} 0 0 0 0 0"

def _read_cycle(vd: int) -> str:
    # SXU_READ_CYCLE opcode = 27. Writes the SXU's free-running cycle
    # counter as Int#(32) into row 0, lane 0 of vd (other lanes zeroed).
    # Pair two READ_CYCLE calls with a STORE + host parse to measure
    # the span of a program region from inside the bundle itself.
    return f"2 27 0 {vd} 0 0 0 0 0 0"

def _load_loop_depth(vd: int) -> str:
    # SXU_LOAD_LOOP_DEPTH opcode = 36. Writes the current LOOP stack
    # depth (0..4) into row 0, lane 0 of vd (other lanes zeroed).
    # Intended for debug/test of nested SXU_LOOP frames.
    return f"2 36 0 {vd} 0 0 0 0 0 0"

def _xlu_rotate(vd: int, vs: int, amount: int) -> str:
    # SXU_DISPATCH_XLU_ROTATE opcode = 37. Cyclic lane rotation via the
    # XLU: vd[s][i] = vs[s][(i + amount) mod lanes]. Dual-issue like
    # other XLU dispatches. `amount` stored in vregSrc2 low bits.
    return f"2 37 0 {vd} {vs} 0 {amount} 0 0 0"

def _loop_begin(count: int) -> str:
    # SXU_LOOP_BEGIN opcode = 28. Sets loopCounter := count and marks
    # the next instruction as the loop-return pc. Count must be 1..255.
    assert 1 <= count <= 255, "loop count out of range"
    return f"2 28 0 0 0 0 0 0 0 {count}"

def _loop_end() -> str:
    # SXU_LOOP_END opcode = 29. Decrements loopCounter; jumps back to
    # the instruction after LOOP_BEGIN if more iterations remain.
    return "2 29 0 0 0 0 0 0 0 0"

def _vzero(vd: int) -> str:
    # SXU_VZERO opcode = 30. One-cycle tile-of-zeros into vd. Skips
    # the "preload zero in VMEM + LOAD" two-instruction dance.
    return f"2 30 0 {vd} 0 0 0 0 0 0"

def _vfill(vd: int, imm_i8: int) -> str:
    # SXU_VFILL opcode = 31. Broadcast a signed 8-bit constant to all
    # 16 lanes of vd. Encoded with imm in mxuWBase (unsigned byte).
    assert -128 <= imm_i8 <= 127, "VFILL immediate out of int8 range"
    enc = imm_i8 & 0xFF
    return f"2 31 0 {vd} 0 0 0 {enc} 0 0"

def _vmov(vd: int, vs: int) -> str:
    # SXU_VMOV opcode = 32. vd := vs in one cycle.
    return f"2 32 0 {vd} {vs} 0 0 0 0 0"

def _vneg(vd: int, vs: int) -> str:
    # SXU_VNEG opcode = 34. vd := -vs lane-wise in one cycle.
    return f"2 34 0 {vd} {vs} 0 0 0 0 0"

def _vabs(vd: int, vs: int) -> str:
    # SXU_VABS opcode = 35. vd := |vs| lane-wise in one cycle.
    return f"2 35 0 {vd} {vs} 0 0 0 0 0"

def _mxu_os_accumulate(wbase: int, abase: int, klen: int) -> str:
    # SXU_DISPATCH_MXU_OS_ACCUMULATE opcode = 33. Routes through
    # Controller.startOsAccumulate: real-OS dispatch that skips the
    # drain-time clearAll, so consecutive dispatches add another
    # kLen worth of psums into the same matrix. Lets multi-K-tile OS
    # scale past K == rows.
    return f"2 33 0 0 0 0 0 {wbase} {abase} {klen}"

def _psum_read(vd: int, psum_addr: int) -> str:
    # SXU_PSUM_READ opcode = 17; vmemAddr doubles as PSUM bucket index.
    return f"2 17 {psum_addr} {vd} 0 0 0 0 0 0"

def _psum_read_row(vd: int, psum_addr: int, psum_row: int) -> str:
    # SXU_PSUM_READ_ROW opcode = 18. Reads one row of a bucket into
    # row 0 of vd with the other rows zeroed — same shape as
    # LOAD_MXU_RESULT, so downstream bias/relu/store don't care that
    # the row came from PSUM.
    return f"2 18 {psum_addr} {vd} {psum_row} 0 0 0 0 0"

def _psum_clear(psum_addr: int) -> str:
    # SXU_PSUM_CLEAR opcode = 19. Zeros the whole bucket in one cycle
    # without touching any vreg, so multi-K-tile GEMM can drop the
    # "preload zero tile + LOAD v15 + PSUM_WRITE" boilerplate.
    return f"2 19 {psum_addr} 0 0 0 0 0 0 0"

def _psum_clear_all() -> str:
    # SXU_PSUM_CLEAR_ALL opcode = 38. Multi-cycle walker inside SXU
    # zeroes every bucket (psumDepth cycles total, one instruction).
    # Replaces the 8-instruction PSUM_CLEAR sweep a multi-K-tile GEMM
    # does before re-using buckets for a new K chain.
    return f"2 38 0 0 0 0 0 0 0 0"

def _set_pred_if_zero(vs: int) -> str:
    # SXU_SET_PRED_IF_ZERO opcode = 20; pred := (vs[0][0] == 0).
    return f"2 20 0 0 {vs} 0 0 0 0 0"

def _skip_if_pred() -> str:
    # SXU_SKIP_IF_PRED opcode = 21; if pred, skip the next instruction.
    return f"2 21 0 0 0 0 0 0 0 0"

def _set_pred_ne_zero(vs: int) -> str:
    # SXU_SET_PRED_NE_ZERO opcode = 39; pred := (vs[0][0] != 0).
    return f"2 39 0 0 {vs} 0 0 0 0 0"

def _skip_if_not_pred() -> str:
    # SXU_SKIP_IF_NOT_PRED opcode = 40; if !pred, skip the next instruction.
    return f"2 40 0 0 0 0 0 0 0 0"

def _psum_accumulate_row(vs: int, psum_addr: int, psum_row: int) -> str:
    # SXU_PSUM_ACCUMULATE_ROW opcode = 22; accumulate row 0 of vs into
    # psum[psum_addr][psum_row]. VPU-side row-granular deposit,
    # symmetric with the MXU dispatch's psum_acc path.
    return f"2 22 {psum_addr} {psum_row} {vs} 0 0 0 0 0"

def _wait_mxu() -> str: return "2 5 0 0 0 0 0 0 0 0"
def _load_mxu_result(vd: int) -> str: return f"2 6 0 {vd} 0 0 0 0 0 0"

def _load_vpu_result(vd: int) -> str:
    # SXU_LOAD_VPU_RESULT opcode = 13; copies vpu.resultReg (linger
    # register) into vd so subsequent ops can reuse the last VPU
    # output without re-dispatch.
    return f"2 13 0 {vd} 0 0 0 0 0 0"

def _load_xlu_result(vd: int) -> str:
    # SXU_LOAD_XLU_RESULT opcode = 14; same pattern for the XLU output
    # register.
    return f"2 14 0 {vd} 0 0 0 0 0 0"
def _halt()     -> str: return "2 7 0 0 0 0 0 0 0 0"
def _output_mxu()           -> str: return "3 1"
def _output_vmem(addr: int) -> str: return f"6 {addr}"
def _end()      -> str: return "4"

def _bundle(*lines: str) -> str:
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Helpers: build text bundle, parse BSV sim output
# ---------------------------------------------------------------------------

def _build_vpu_binary_bundle(lhs_i32: np.ndarray, rhs_i32: np.ndarray, num_elems: int, vpu_op: int,
                             lhs_broadcast: bool = False, rhs_broadcast: bool = False) -> str:
    """VMEM[0]=lhs, VMEM[1]=rhs, VPU v2=OP(v0,v1), OUTPUT_VMEM VMEM[2]."""
    def tile(vals: np.ndarray) -> list[int]:
        padded = np.zeros(_ROWS * _COLS, dtype=np.int32)
        padded[:num_elems] = vals[:num_elems]
        return [int(x) for x in padded]

    lines = [
        _vmem(0, tile(lhs_i32)),   # VMEM[0] = lhs tile
        _vmem(1, tile(rhs_i32)),   # VMEM[1] = rhs tile
        _load(0, 0),               # LOAD v0, VMEM[0]
    ]
    if lhs_broadcast:
        lines.append(_broadcast(0))              # BROADCAST v0
    lines.append(_load(1, 1))                    # LOAD v1, VMEM[1]
    if rhs_broadcast:
        lines.append(_broadcast(1))              # BROADCAST v1
    lines += [
        _vpu(2, 0, vpu_op, 1),  # VPU v2 = OP(v0, v1)
        _store(2, 2),            # STORE VMEM[2], v2
        _halt(),                 # HALT
        _output_vmem(2),         # OUTPUT_VMEM VMEM[2]
        _end(),                  # END
    ]
    return _bundle(*lines)


def _build_vpu_unary_bundle(src_i32: np.ndarray, num_elems: int, vpu_op: int) -> str:
    """VMEM[0]=src, VPU v1=OP(v0), OUTPUT_VMEM VMEM[2]."""
    padded = np.zeros(_ROWS * _COLS, dtype=np.int32)
    padded[:num_elems] = src_i32[:num_elems]
    return _bundle(
        _vmem(0, [int(x) for x in padded]),  # VMEM[0] = src tile
        _load(0, 0),                          # LOAD v0, VMEM[0]
        _vpu(1, 0, vpu_op),                   # VPU v1 = OP(v0)  [unary: vb=0]
        _store(2, 1),                         # STORE VMEM[2], v1
        _halt(),                              # HALT
        _output_vmem(2),                      # OUTPUT_VMEM VMEM[2]
        _end(),                               # END
    )


def _build_vpu_program_bundle(inputs: list[np.ndarray], num_elems: int, steps: list[dict], output_reg: int,
                              input_broadcasts: list[bool] | None = None) -> str:
    """Multi-step VPU program: VMEM[0..N-1]=inputs, execute steps, VMEM[N]=output_reg."""
    def tile(vals: np.ndarray) -> list[int]:
        padded = np.zeros(_TILE_ELEMS, dtype=np.int32)
        padded[:num_elems] = vals[:num_elems]
        return [int(x) for x in padded]

    if input_broadcasts is None:
        input_broadcasts = [False] * len(inputs)

    lines: list[str] = []
    for idx, vals in enumerate(inputs):
        lines.append(_vmem(idx, tile(vals)))          # VMEM[idx] = input
    for idx in range(len(inputs)):
        lines.append(_load(idx, idx))                 # LOAD v{idx}, VMEM[idx]
        if input_broadcasts[idx]:
            lines.append(_broadcast(idx))             # BROADCAST v{idx}
    for step in steps:
        lhs = int(step["lhs"])
        dst = int(step["dst"])
        vb  = int(step.get("rhs", 0))
        lines.append(_vpu(dst, lhs, int(step["op"]), vb))  # VPU v{dst} = OP(v{lhs}, v{vb})
    lines += [
        _store(len(inputs), output_reg),  # STORE VMEM[N], v{output_reg}
        _halt(),                          # HALT
        _output_vmem(len(inputs)),        # OUTPUT_VMEM VMEM[N]
        _end(),                           # END
    ]
    return _bundle(*lines)


def _parse_result_line(line: str, prefix: str, expected_count: int) -> list[int]:
    """Parse a single sim output line like 'mxu_result v0 v1 ...' or 'vmem_result v0 v1 ...'."""
    vals = line.split()[1:]
    if len(vals) != expected_count:
        raise ValueError(f"{prefix} expects {expected_count} values, got {len(vals)}")
    try:
        return [int(x) for x in vals]
    except ValueError as exc:
        bad = next((x for x in vals if not x.lstrip("-").isdigit()), vals[0])
        raise ValueError(f"invalid {prefix} integer {bad!r}") from exc

def _parse_sim_output(stdout: str) -> list[int] | None:
    """Extract mxu_result from BSV sim stdout. Returns None if not found."""
    for line in stdout.splitlines():
        if line.strip().startswith("mxu_result "):
            return _parse_result_line(line.strip(), "mxu_result", _COLS)
    return None

def _parse_vmem_output(stdout: str) -> list[int] | None:
    """Extract first vmem_result from BSV sim stdout. Returns None if not found."""
    for line in stdout.splitlines():
        if line.strip().startswith("vmem_result "):
            return _parse_result_line(line.strip(), "vmem_result", _ROWS * _COLS)
    return None

def _parse_multi_vmem_output(stdout: str) -> list[list[int]]:
    """Extract all vmem_result lines from BSV sim stdout."""
    return [_parse_result_line(line.strip(), "vmem_result", _ROWS * _COLS)
            for line in stdout.splitlines() if line.strip().startswith("vmem_result ")]


def _build_full_gemm_bundle(act_rows_i8: np.ndarray, weight_matrix_i8: np.ndarray,
                            num_vecs: int, num_k_tiles: int, num_weight_tiles: int,
                            bias_i32: np.ndarray | None = None, relu: bool = False) -> str:
    """Build a single SXU program that computes all rows×tiles of a GEMM + epilogue.

    Preloads all weight tiles into WMEM, all activation rows into AMEM,
    and bias into VMEM. Runs one MXU dispatch per (row, k_tile), accumulates
    via VPU, applies epilogue, and stores each row's result to a separate
    VMEM address for output.
    """
    data_lines: list[str] = []

    # Preload weight tiles into WMEM: address = k*num_weight_tiles + tile_idx
    for k in range(num_k_tiles):
        for t in range(num_weight_tiles):
            w_tile = weight_matrix_i8[k * _ROWS : (k + 1) * _ROWS,
                                      t * _COLS : (t + 1) * _COLS]
            wmem_addr = k * num_weight_tiles + t
            data_lines.append(_wmem(wmem_addr, [int(x) for x in w_tile.flatten()]))

    # Preload activation rows into AMEM: address = row * num_k_tiles + k
    for row in range(num_vecs):
        for k in range(num_k_tiles):
            a_tile = act_rows_i8[row, k * _ROWS : (k + 1) * _ROWS]
            amem_addr = row * num_k_tiles + k
            data_lines.append(_amem(amem_addr, [int(x) for x in a_tile]))

    # Preload bias tiles into VMEM if needed (one tile per weight_tile)
    bias_vmem_base = 0
    if bias_i32 is not None:
        for t in range(num_weight_tiles):
            bias_tile = [0] * _TILE_ELEMS
            for i in range(_COLS):
                bias_tile[i] = int(bias_i32[t * _COLS + i])
            data_lines.append(_vmem(bias_vmem_base + t, bias_tile))

    # Output VMEM addresses: one per (row, weight_tile)
    # Start after bias tiles
    out_vmem_base = (num_weight_tiles if bias_i32 is not None else 0)

    # Build SXU program
    prog_lines: list[str] = []
    for row in range(num_vecs):
        for tile_idx in range(num_weight_tiles):
            # MXU dispatches for K-tile accumulation
            for k in range(num_k_tiles):
                wmem_addr = k * num_weight_tiles + tile_idx
                amem_addr = row * num_k_tiles + k
                vreg_k = k  # v0, v1, ... for K-tile partials
                prog_lines.append(_mxu(wmem_addr, amem_addr, 1))
                prog_lines.append(_wait_mxu())
                prog_lines.append(_load_mxu_result(vreg_k))

            # Accumulate K-tiles
            if num_k_tiles == 1:
                cur = 0
            else:
                acc = num_k_tiles  # first free vreg after K-tile results
                prog_lines.append(_vpu(acc, 0, _VPU_OPS["ADD"], 1))
                cur = acc
                for k in range(2, num_k_tiles):
                    nxt = cur + 1
                    prog_lines.append(_vpu(nxt, cur, _VPU_OPS["ADD"], k))
                    cur = nxt

            # Bias epilogue
            if bias_i32 is not None:
                bias_vreg = cur + 1
                prog_lines.append(_load(bias_vreg, bias_vmem_base + tile_idx))
                result_vreg = bias_vreg + 1
                prog_lines.append(_vpu(result_vreg, cur, _VPU_OPS["ADD"], bias_vreg))
                cur = result_vreg

            # ReLU epilogue
            if relu:
                nxt = cur + 1
                prog_lines.append(_vpu(nxt, cur, 2))  # VPU_RELU
                cur = nxt

            # Store result
            out_addr = out_vmem_base + row * num_weight_tiles + tile_idx
            prog_lines.append(_store(out_addr, cur))

    prog_lines.append(_halt())

    # Output records
    output_lines: list[str] = []
    for row in range(num_vecs):
        for tile_idx in range(num_weight_tiles):
            out_addr = out_vmem_base + row * num_weight_tiles + tile_idx
            output_lines.append(_output_vmem(out_addr))
    output_lines.append(_end())

    return _bundle(*(data_lines + prog_lines + output_lines))


def _run_bundle(sim: str, bundle_text: str) -> str:
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
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("FAIL:") or line.startswith("ERROR:"):
            raise RuntimeError(
                f"TinyTPU simulator reported failure: {line}\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )
    if "status ok" not in {line.strip() for line in proc.stdout.splitlines()}:
        raise RuntimeError(
            f"TinyTPU simulator did not report `status ok`\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc.stdout


def _infer_tiling(out_size: int | None, act_size: int | None, weight_size: int) -> tuple[int, int, int] | None:
    if out_size is None or act_size is None:
        return None
    if out_size <= 0 or act_size <= 0 or weight_size <= 0:
        return None
    if act_size % _ROWS != 0 or out_size % _COLS != 0 or weight_size % (_ROWS * _COLS) != 0:
        return None
    act_quads = act_size // _ROWS
    out_quads = out_size // _COLS
    weight_tiles = weight_size // (_ROWS * _COLS)
    numer = act_quads * out_quads
    if numer % weight_tiles != 0:
        return None
    num_vecs_sq = numer // weight_tiles
    num_vecs = math.isqrt(num_vecs_sq)
    if num_vecs <= 0 or num_vecs * num_vecs != num_vecs_sq:
        return None
    if act_quads % num_vecs != 0 or out_quads % num_vecs != 0:
        return None
    num_k_tiles = act_quads // num_vecs
    num_n_tiles = out_quads // num_vecs
    if num_k_tiles <= 0 or num_n_tiles <= 0 or num_k_tiles * num_n_tiles != weight_tiles:
        return None
    return num_vecs, num_k_tiles, num_n_tiles


def _tiling_failure_note(out_size: int | None, act_size: int | None, weight_size: int) -> str:
    issues: list[str] = []
    if act_size is None or out_size is None:
        return "missing activation or output buffer size for GEMM factoring"
    if act_size <= 0 or out_size <= 0 or weight_size <= 0:
        return "zero-sized GEMM buffers are not lowered through the current TinyTPU path"
    if act_size % _ROWS != 0:
        issues.append(f"activation size {act_size} is not divisible by {_ROWS}")
    if out_size % _COLS != 0:
        issues.append(f"output size {out_size} is not divisible by {_COLS}")
    if weight_size % (_ROWS * _COLS) != 0:
        issues.append(f"weight size {weight_size} is not divisible by {_ROWS * _COLS}")
    if not issues:
        issues.append(f"sizes out={out_size}, act={act_size}, weight={weight_size} do not factor into MxK, KxN, MxN tiles")
    return "; ".join(issues)


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
    op_counts = prog.get("op_counts") or {}
    if missing:
        parts.append("missing instructions: " + ", ".join(str(x) for x in missing))
    if notes:
        parts.append("notes: " + " | ".join(str(x) for x in notes))
    if op_counts:
        parts.append("op_counts: " + ", ".join(f"{k}={v}" for k, v in sorted(op_counts.items())))
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Program — drives the BSV simulator
# ---------------------------------------------------------------------------
_SUPPORTED_OPS = {"SXU_PROGRAM"}

class TinyTPUProgram:
    def __init__(self, name: str, lib: bytes, *args, **kwargs):
        self.name = name
        self.prog = json.loads(lib)
        self.sim = _sim_path()

    def _run(self, bundle_text: str) -> str:
        return _run_bundle(self.sim, bundle_text)

    def _run_vmem(self, bundle_text: str) -> list[int]:
        """Run bundle and parse the first vmem_result line."""
        result = _parse_vmem_output(self._run(bundle_text))
        if result is None:
            raise RuntimeError("TinyTPU sim produced no vmem_result")
        return result

    def _run_tiled_vpu(self, out_buf: bytearray, num_elems: int,
                       build_fn, *, out_dtype: np.dtype = np.dtype("<i4")) -> float:
        """Run a VPU op in VMEM-tile chunks, writing results to out_buf."""
        elem_bytes = out_dtype.itemsize
        out_offset = 0
        for chunk_start in range(0, num_elems, _TILE_ELEMS):
            chunk_end = min(chunk_start + _TILE_ELEMS, num_elems)
            chunk_size = chunk_end - chunk_start
            result = self._run_vmem(build_fn(chunk_start, chunk_end, chunk_size))
            chunk_out = np.array(result[:chunk_size], dtype=out_dtype)
            out_buf[out_offset : out_offset + len(chunk_out) * elem_bytes] = chunk_out.tobytes()
            out_offset += len(chunk_out) * elem_bytes
        return 1e-3

    def __call__(self, *bufs: bytearray,
                 global_size: tuple = (1, 1, 1),
                 local_size: tuple | None = None,
                 vals: tuple = (),
                 wait: bool = False,
                 **kwargs) -> float | None:
        prog = self.prog
        op = prog.get("op")
        if op not in _SUPPORTED_OPS:
            raise NotImplementedError(_unsupported_message(prog))
        return getattr(self, f"_exec_{op.lower()}")(bufs)

    def _exec_sxu_program(self, bufs):
        prog = self.prog
        data_plan = prog["data_plan"]
        instructions = prog["instructions"]
        outputs = prog["outputs"]
        is_bool_out = prog.get("bool_out", False)

        # Build bundle from data_plan + instructions + outputs
        data_lines: list[str] = []
        for entry in data_plan:
            mem_type = entry["type"]
            if entry.get("layout") == "broadcast_const":
                val = int(entry["value"])
                data_lines.append(_vmem(int(entry["addr"]), [val] * _TILE_ELEMS))
                continue
            param_idx = int(entry["param"])
            buf_data = bufs[param_idx]

            if mem_type == "WMEM":
                weight_i32 = np.frombuffer(bytes(buf_data), dtype="<i4")
                _require_int8_range("weight", weight_i32)
                weight_i8 = weight_i32.astype(np.int8)
                nk, nwt = entry["num_k_tiles"], entry["num_weight_tiles"]
                weight_matrix = weight_i8.reshape(nk * _ROWS, nwt * _COLS)
                for k in range(nk):
                    for t in range(nwt):
                        w_tile = weight_matrix[k*_ROWS:(k+1)*_ROWS, t*_COLS:(t+1)*_COLS]
                        data_lines.append(_wmem(k*nwt+t, [int(x) for x in w_tile.flatten()]))

            elif mem_type == "AMEM":
                act_i32 = np.frombuffer(bytes(buf_data), dtype="<i4")
                _require_int8_range("activation", act_i32)
                act_i8 = act_i32.astype(np.int8)
                nv, nk = entry["num_vecs"], entry["num_k_tiles"]
                act_rows = act_i8.reshape(nv, nk * _ROWS)
                for row in range(nv):
                    for k in range(nk):
                        a_tile = act_rows[row, k*_ROWS:(k+1)*_ROWS]
                        data_lines.append(_amem(row*nk+k, [int(x) for x in a_tile]))

            elif mem_type == "VMEM":
                is_bool = entry.get("bool", False)
                is_broadcast = entry.get("broadcast", False)
                raw = np.frombuffer(bytes(buf_data), dtype=np.bool_ if is_bool else "<i4")
                if is_bool:
                    raw = raw.astype(np.int32)
                addr = int(entry["addr"])
                offset = int(entry.get("offset", 0))
                count = int(entry.get("count", _TILE_ELEMS))
                if is_broadcast:
                    val = int(raw[0]) if len(raw) > 0 else 0
                    data_lines.append(_vmem(addr, [val] * _TILE_ELEMS))
                    continue
                mode = entry.get("mode", "TILE")
                if mode == "ROW_BROADCAST":
                    nwt = entry.get("num_weight_tiles", 1)
                    for t in range(nwt):
                        tile = [0] * _TILE_ELEMS
                        for i in range(_COLS):
                            tile[i] = int(raw[t*_COLS+i])
                        data_lines.append(_vmem(addr+t, tile))
                elif mode == "PAD_FILL":
                    # Scatter source positions into the output tile per the
                    # renderer-supplied dst->src map; unlisted positions stay
                    # at zero (pad_value defaults to 0). Single-tile only.
                    pad_val = int(entry.get("pad_value", 0))
                    tile = [pad_val] * _TILE_ELEMS
                    for dst, src in entry["pad_map"]:
                        if src is None:
                            continue
                        if 0 <= src < len(raw):
                            tile[dst] = int(raw[src])
                    data_lines.append(_vmem(addr, tile))
                elif mode == "MATRIX_TILE":
                    # Pack matrix[row_base:row_base+tile_rows, col_base:col_base+tile_cols]
                    # into a 4x4 tile with pad_value fill for out-of-bounds cells.
                    pad_val = int(entry.get("pad_value", 0))
                    nrows_mat = int(entry["matrix_nrows"])
                    ncols_mat = int(entry["matrix_ncols"])
                    row_base  = int(entry.get("row_base", 0))
                    col_base  = int(entry.get("col_base", 0))
                    tile_rows = int(entry.get("tile_rows", _ROWS))
                    tile_cols = int(entry.get("tile_cols", _COLS))
                    tile = [pad_val] * _TILE_ELEMS
                    for r in range(tile_rows):
                        mr = row_base + r
                        if mr >= nrows_mat: break
                        for c in range(tile_cols):
                            mc = col_base + c
                            if mc >= ncols_mat: break
                            tile[r * _COLS + c] = int(raw[mr * ncols_mat + mc])
                    data_lines.append(_vmem(addr, tile))
                else:
                    pad_val = int(entry.get("pad_value", 0))
                    tile = [pad_val] * _TILE_ELEMS
                    chunk = raw[offset:offset+count]
                    for i in range(min(count, len(chunk))):
                        tile[i] = int(chunk[i])
                    data_lines.append(_vmem(addr, tile))

        # Output records
        output_lines = [_output_vmem(int(o["addr"])) for o in outputs] + [_end()]
        bundle_text = _bundle(*(data_lines + instructions + output_lines))

        # Run sim
        stdout = self._run(bundle_text)
        vmem_results = _parse_multi_vmem_output(stdout)
        expected = int(prog["num_output_tiles"])
        if len(vmem_results) != expected:
            raise RuntimeError(f"SXU_PROGRAM expected {expected} vmem tiles, got {len(vmem_results)}\n{stdout[:500]}")

        # Write results to output buffer
        out_buf = bufs[int(prog["out"])]
        out_dtype = np.dtype(np.bool_) if is_bool_out else np.dtype("<i4")
        reduce_mode = prog.get("reduce")
        out_offset = 0
        for idx, out_entry in enumerate(outputs):
            count = int(out_entry["count"])
            tile_data = vmem_results[idx]
            if reduce_mode and count == 1:
                # Scalar reductions use VPU_*_REDUCE_TILE, so the scalar is
                # broadcast to every position of the output tile.
                chunk_out = np.array([tile_data[0]], dtype=out_dtype)
            elif out_entry.get("extract") == "row_heads":
                # Row reductions: each sublane holds a reduction broadcast
                # across its row; extract position [r * _COLS] for r in range(count).
                chunk_out = np.array([tile_data[r * _COLS] for r in range(count)], dtype=out_dtype)
            else:
                chunk_out = np.array(tile_data[:count], dtype=out_dtype)
            out_buf[out_offset:out_offset+len(chunk_out)*out_dtype.itemsize] = chunk_out.tobytes()
            out_offset += len(chunk_out) * out_dtype.itemsize
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
