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
             "EXP2": 51, "LOG2": 52, "SIN": 53, "COS": 54}
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

def _is_float_min_negation(uops: list[UOp]) -> bool:
    """True if the UOp kernel matches tinygrad's float-MIN decomposition:
       MUL(MAX(..., MUL(load_i, -1.0), ...), -1.0). Used by scalar/row/col
       reducers to rewrite to FMIN_REDUCE{,_COL} instead of FMAX.
    """
    stores = [u for u in uops if u.op is Ops.STORE]
    if len(stores) != 1:
        return False
    sv = stores[0].src[1]
    def _is_neg_one(u):
        return (u.op is Ops.CONST and isinstance(u.arg, float)
                and float(u.arg) == -1.0)
    outer_ok = (sv.op is Ops.MUL and len(sv.src) == 2
                and any(_is_neg_one(s) for s in sv.src))
    data_muls = [u for u in uops if u.op is Ops.MUL and _has_load_src(u)]
    inner_ok = all(any(_is_neg_one(s) for s in u.src) for u in data_muls)
    return outer_ok and inner_ok and len(data_muls) >= 1


def _detect_reduce_op(op_counts: Counter, data_alu: Counter | None = None) -> str | None:
    """Detect SUM/MAX/MIN/PROD from UOp op counts.

    For PROD we consult data-path ALU counts (if provided) to avoid matching
    the stride multiplies in row-reduce index arithmetic. The other reductions
    keep their historical (op_counts, nloads) threshold rules.
    """
    nloads = op_counts.get("LOAD", 0)
    if op_counts.get("ADD", 0) > nloads - 1 and op_counts.get("MAX", 0) == 0:
        return "SUM"
    if (op_counts.get("MAX", 0) >= nloads - 1
        and op_counts.get("MAX", 0) > 0
        and op_counts.get("XOR", 0) == 0):
        return "MAX"
    if (op_counts.get("MAX", 0) >= nloads - 1
        and op_counts.get("MAX", 0) > 0
        and op_counts.get("XOR", 0) > 0):
        return "MIN"
    if data_alu is not None:
        data_mul = data_alu.get("MUL", 0)
        if data_mul > 0 and data_alu.get("ADD", 0) == 0 and data_alu.get("MAX", 0) == 0:
            return "PROD"
    return None

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
        if (red_desc := _render_reduction_sxu_program(uops)) is not None:
            return red_desc
        if (colbc_where_desc := _render_colbc_where_sxu_program(uops)) is not None:
            return colbc_where_desc
        if (where_desc := _render_where_sxu_program(uops)) is not None:
            return where_desc
        if (min_const_desc := _render_min_const_sxu_program(uops)) is not None:
            return min_const_desc
        if (multi_desc := _render_multistep_sxu_program(uops)) is not None:
            return multi_desc
        # PAD before row-broadcast: pad kernels have mixed LOAD / CONST(0)
        # stores, which the row-broadcast renderer would otherwise try to
        # match and emit a wrong BROADCAST_ROW program for.
        if (pad_desc := _render_pad_sxu_program(uops)) is not None:
            return pad_desc
        if (colbc_desc := _render_colbc_sxu_program(uops)) is not None:
            return colbc_desc
        if (rowbc_desc := _render_rowbc_sxu_program(uops)) is not None:
            return rowbc_desc
        if (colred_desc := _render_colreduce_sxu_program(uops)) is not None:
            return colred_desc
        if (rowred_desc := _render_rowreduce_sxu_program(uops)) is not None:
            return rowred_desc
        if (fill_desc := _render_const_fill_sxu_program(uops)) is not None:
            return fill_desc
        if (trans_desc := _render_transpose_sxu_program(uops)) is not None:
            return trans_desc
        if (cast_desc := _render_cast_sxu_program(uops)) is not None:
            return cast_desc
        if (copy_desc := _render_copy_sxu_program(uops)) is not None:
            return copy_desc
        if (recip_desc := _render_reciprocal_sxu_program(uops)) is not None:
            return recip_desc
        if (tanh_desc := _render_tanh_sxu_program(uops)) is not None:
            return tanh_desc
        if (clip_desc := _render_clip_sxu_program(uops)) is not None:
            return clip_desc
        if (clamp_sb_desc := _render_clamp_single_bound_sxu_program(uops)) is not None:
            return clamp_sb_desc
        if (leaky_relu_desc := _render_leaky_relu_sxu_program(uops)) is not None:
            return leaky_relu_desc
        if (softsign_desc := _render_softsign_sxu_program(uops)) is not None:
            return softsign_desc
        if (swish_desc := _render_swish_sxu_program(uops)) is not None:
            return swish_desc
        if (sigmoid_desc := _render_sigmoid_sxu_program(uops)) is not None:
            return sigmoid_desc
        if (scaled_exp2_desc := _render_scaled_exp2_sxu_program(uops)) is not None:
            return scaled_exp2_desc
        if (exp2_desc := _render_exp2_sxu_program(uops)) is not None:
            return exp2_desc
        if (scaled_log2_desc := _render_scaled_log2_sxu_program(uops)) is not None:
            return scaled_log2_desc
        if (log2_desc := _render_log2_sxu_program(uops)) is not None:
            return log2_desc
        if (scaled_sin_desc := _render_scaled_sin_sxu_program(uops)) is not None:
            return scaled_sin_desc
        if (sin_desc := _render_sin_sxu_program(uops)) is not None:
            return sin_desc
        if (self_cube_desc := _render_self_cube_sxu_program(uops)) is not None:
            return self_cube_desc
        if (self_sq_desc := _render_self_square_sxu_program(uops)) is not None:
            return self_sq_desc
        if (rsqrt_desc := _render_rsqrt_sxu_program(uops)) is not None:
            return rsqrt_desc
        if (sqrt_desc := _render_sqrt_sxu_program(uops)) is not None:
            return sqrt_desc
        if (trunc_desc := _render_trunc_sxu_program(uops)) is not None:
            return trunc_desc
        if (divmod_desc := _render_scalar_const_divmod_sxu_program(uops)) is not None:
            return divmod_desc
        if (chain_desc := _render_chained_const_sxu_program(uops)) is not None:
            return chain_desc
        return _render_elementwise_sxu_program(uops)

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


def _render_reduction_sxu_program(uops: list[UOp]) -> dict | None:
    """Render a reduction (scalar or row-wise SUM/MAX/MIN) as SXU_PROGRAM.

    Uses VPU_SUM_REDUCE (4), VPU_MAX_REDUCE (9), or VPU_MIN_REDUCE (13).
    Scalar reduce: out_size=1, combines across sublanes.
    Row reduce: out_size=nrows, ncols divides _COLS, per-row reduce.
    """
    op_counts = Counter(u.op.name for u in uops)
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2:
        return None

    out_arg = 0
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    out_size = params[out_arg].dtype.size
    src_size = params[src_arg].dtype.size

    # Only handle scalar reductions (out_size=1) for now.
    # Row/column reductions (out_size>1) stay on old path which distinguishes axis.
    if out_size != 1:
        return None
    # Reject trivial 1-elem-to-1-elem kernels (e.g. Tensor([5])+1): there is
    # nothing to reduce and the elementwise path gives the correct result.
    if src_size <= 1:
        return None

    # Integer VPU_*_REDUCE ops treat bits as Int#(32), so we route float
    # reductions to dedicated float reducer opcodes (currently only
    # VPU_FSUM_REDUCE_TILE). Any float reduction kind we can't handle yet
    # is rejected here so the caller reports unsupported.
    is_float = any("float" in str(params[p].dtype) for p in params)

    # Detect reduction type from UOp tree
    has_add = op_counts.get("ADD", 0) > 0
    has_max = op_counts.get("MAX", 0) > 0
    has_xor = op_counts.get("XOR", 0) > 0
    has_store = op_counts.get("STORE", 0) > 0
    if not has_store:
        return None
    # Reject kernels where pre-reduction data-path processing sits
    # between the LOAD(s) and the reduction. The reduce renderer only
    # streams raw loaded tiles through VPU_*_REDUCE_TILE; any unhandled
    # pre-reduction op (transcendental, reciprocal, WHERE-based abs,
    # conditional selection) would be silently dropped and the reduction
    # would run over the unprocessed input (giving wrong results).
    if any(op_counts.get(n, 0) for n in ("EXP2", "LOG2", "SIN", "SQRT", "RECIPROCAL")):
        return None
    if op_counts.get("WHERE", 0) > 0 or op_counts.get("CMPLT", 0) > 0 or op_counts.get("CMPEQ", 0) > 0 or op_counts.get("CMPNE", 0) > 0:
        return None

    # Float reductions: sum, max, min supported today. Prod still needs its
    # own reducer opcode.
    # - Integer MIN decomposes to XOR+MAX (has_xor catches it).
    # - Float MIN decomposes via negation: tinygrad emits
    #     MUL(MAX(..., MUL(load_i, -1.0), ...), -1.0)
    #   i.e. negate each leaf, take max, negate the result. We detect this
    #   signature so we can rewrite to FMIN_REDUCE_TILE directly (and fall
    #   back to "unsupported" if the pattern doesn't match).
    if is_float and has_xor:
        return None
    is_float_min = (is_float and has_max
                    and _data_alu_ops(uops).get("MUL", 0) > 0
                    and _is_float_min_negation(uops))
    if (is_float and has_max
        and _data_alu_ops(uops).get("MUL", 0) > 0
        and not is_float_min):
        # Has data-path MULs on a float MAX kernel but not the negation
        # signature — we can't safely lower this without producing wrong
        # results (e.g. float-min wrapped around something we don't yet
        # understand). Bail out.
        return None

    # If the stored value is ADD(tree, CONST) or MUL(tree, CONST) where the tree
    # does the reduction and the CONST is a post-reduction scalar op, detect
    # and emit reduction + post-op. Otherwise use the original path which
    # treats any ADD as part of the reduction tree.
    post_op = None  # ("ADD"|"MUL", const_val) applied after reduction
    stores = [u for u in uops if u.op is Ops.STORE]
    if len(stores) == 1:
        val = stores[0].src[1]
        if val.op in (Ops.ADD, Ops.MUL):
            const_src = next((s for s in val.src if s.op is Ops.CONST
                              and not isinstance(s.arg, bool)), None)
            tree_src = next((s for s in val.src if s is not const_src), None)
            if const_src is not None and tree_src is not None and _has_load_src(tree_src):
                # The CONST must not be reused anywhere on the data path besides
                # the outer ADD/MUL. Index-path usage (e.g. CONST(2) as an INDEX
                # position) is unrelated and should be ignored so the post-op
                # detection still fires when the post-op constant happens to
                # collide with an index literal.
                const_data_uses = sum(1 for u in uops
                                      if any(s is const_src for s in u.src)
                                      and u.op is not Ops.INDEX)
                if const_data_uses == 1 and val.op is Ops.ADD:
                    post_op = ("ADD", const_src.arg)
                elif const_data_uses == 1 and val.op is Ops.MUL:
                    post_op = ("MUL", const_src.arg)
    # When the outer op is ADD(reduce, const) we must NOT count it as part of
    # the reduction (it is the post-op). Adjust op_counts for detection.
    if post_op is not None and post_op[0] == "ADD":
        has_add = (op_counts.get("ADD", 0) - 1) > 0
    if post_op is not None and post_op[0] == "MUL":
        pass  # leave has_add alone (MUL reduction detection uses data_alu)
    # Data-path MUL is the MUL-reduction signature (no index-arithmetic MULs).
    data_alu = _data_alu_ops(uops)
    has_data_mul = data_alu.get("MUL", 0) > 0
    # Reject SUM/MAX reductions with pre-reduction MULs we can't fold into
    # a post-op. Without this, sum(x*x) and sum(-x) silently return sum(x)
    # because the reducer ignores the inner MULs. PROD reductions (no ADD
    # in data path) still route through MUL_REDUCE_TILE below.
    if has_data_mul and has_add and post_op is None:
        return None
    # Float-min negation uses a MUL(-1) decomposition that's part of the
    # reduction, not a pre-op — allow it through the existing path.
    if has_data_mul and has_max and not is_float_min and post_op is None:
        return None

    # INT32 identity bounds used as padding so tile-reduce produces correct
    # results on partial last tiles.
    _INT32_MIN = -(1 << 31)
    _INT32_MAX = (1 << 31) - 1
    pad_value = 0
    _FLOAT_NEG_INF_BITS = -(1 << 23)  # 0xFF800000 as signed int32 = -8388608
    _FLOAT_POS_INF_BITS = 0x7F800000  # +inf as unsigned 32-bit
    _FLOAT_ONE_BITS     = 0x3F800000  # 1.0 (multiplicative identity)
    if is_float and has_data_mul and not has_add and not has_max and not is_float_min:
        # Float prod reduction: MUL on the data path, no MAX or ADD. Same
        # shape as integer prod but uses the FpReducer via FPROD_REDUCE_TILE
        # and FMUL for multi-tile combine.
        vpu_op = _VPU_OPS["FPROD_REDUCE_TILE"]
        combine_op = _VPU_OPS["FMUL"]
        pad_value = _FLOAT_ONE_BITS
    elif is_float_min:
        # Float min via negation-around-max. We read the ORIGINAL data
        # (pre-negation) and reduce with FMIN_REDUCE_TILE; the inner MUL(-1)
        # cancels the outer MUL(-1), so the emitted kernel is just a plain
        # float min with +inf pad. Multi-tile combine uses VPU_FMIN.
        vpu_op = _VPU_OPS["FMIN_REDUCE_TILE"]
        combine_op = _VPU_OPS["FMIN"]
        pad_value = _FLOAT_POS_INF_BITS
    elif is_float and has_add and not has_max:
        # Float tile-sum pads with +0.0, which has bit pattern 0x00000000
        # — same as integer zero — so pad_value stays 0.
        vpu_op = _VPU_OPS["FSUM_REDUCE_TILE"]
        combine_op = _VPU_OPS["FADD"]
    elif is_float and has_max and not has_xor:
        # Float tile-max pads with -inf (bit pattern 0xFF800000) so partial
        # tiles don't pull the max down. Combine across tiles with FMAX.
        vpu_op = _VPU_OPS["FMAX_REDUCE_TILE"]
        combine_op = _VPU_OPS["FMAX"]
        pad_value = _FLOAT_NEG_INF_BITS
    elif has_data_mul and not has_add and not has_max:
        vpu_op = _VPU_OPS["MUL_REDUCE_TILE"]
        combine_op = _VPU_OPS["MUL"]
        pad_value = 1  # multiplicative identity
    elif has_add and not has_max:
        vpu_op = _VPU_OPS["SUM_REDUCE_TILE"]
        combine_op = _VPU_OPS["ADD"]
    elif has_max and not has_xor:
        vpu_op = _VPU_OPS["MAX_REDUCE_TILE"]
        combine_op = _VPU_OPS["MAX"]
        pad_value = _INT32_MIN
    elif has_max and has_xor:
        vpu_op = _VPU_OPS["MIN_REDUCE_TILE"]
        combine_op = _VPU_OPS["MIN"]
        pad_value = _INT32_MAX
    else:
        return None

    _REDUCE_COMBINE = {_VPU_OPS["SUM_REDUCE_TILE"]: "sum",
                      _VPU_OPS["MAX_REDUCE_TILE"]: "max",
                      _VPU_OPS["MIN_REDUCE_TILE"]: "min",
                      _VPU_OPS["MUL_REDUCE_TILE"]: "prod",
                      _VPU_OPS["FSUM_REDUCE_TILE"]: "sum",
                      _VPU_OPS["FMAX_REDUCE_TILE"]: "max",
                      _VPU_OPS["FMIN_REDUCE_TILE"]: "min",
                      _VPU_OPS["FPROD_REDUCE_TILE"]: "prod"}

    # Scalar reduction
    num_tiles = (src_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    all_instrs: list[str] = []
    data_plan: list[dict] = []

    src_is_bool = (isinstance(params[src_arg].dtype, PtrDType)
                   and params[src_arg].dtype.base.itemsize == 1
                   and "bool" in str(params[src_arg].dtype))
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, src_size - offset)
        vmem_addr = tile_idx
        entry = {"type": "VMEM", "addr": vmem_addr, "param": src_arg,
                 "offset": offset, "count": count, "dtype": "int32"}
        if src_is_bool:
            entry["bool"] = True
        if pad_value != 0:
            entry["pad_value"] = pad_value
        data_plan.append(entry)
        src_vreg = tile_idx * 2
        dst_vreg = tile_idx * 2 + 1
        all_instrs.append(_load(src_vreg, vmem_addr))
        all_instrs.append(_vpu(dst_vreg, src_vreg, vpu_op))

    if num_tiles > 1:
        acc_vreg = 1
        for tile_idx in range(1, num_tiles):
            tile_result_vreg = tile_idx * 2 + 1
            next_vreg = num_tiles * 2 + tile_idx
            all_instrs.append(_vpu(next_vreg, acc_vreg, combine_op, tile_result_vreg))
            acc_vreg = next_vreg
    else:
        acc_vreg = 1

    # Post-reduction scalar op: reduce_result (op) const
    if post_op is not None:
        op_name, const_val = post_op
        c_bits = int(np.frombuffer(np.float32(const_val).tobytes(), dtype=np.int32)[0]) \
                 if isinstance(const_val, float) else int(const_val)
        const_addr = num_tiles  # next VMEM slot
        data_plan.append({"type": "VMEM", "addr": const_addr, "layout": "broadcast_const",
                          "value": c_bits, "count": _TILE_ELEMS, "dtype": "int32"})
        const_vreg = num_tiles * 2 + 10
        result_vreg = const_vreg + 1
        all_instrs.append(_load(const_vreg, const_addr))
        # Float reductions (FSUM/FMAX/FMIN/FPROD) must use the float VPU
        # variants for the post-op; integer reductions stay on the int ops.
        is_float_reduce = vpu_op in (_VPU_OPS["FSUM_REDUCE_TILE"],
                                     _VPU_OPS["FMAX_REDUCE_TILE"],
                                     _VPU_OPS["FMIN_REDUCE_TILE"],
                                     _VPU_OPS["FPROD_REDUCE_TILE"])
        if is_float_reduce:
            post_vpu = _VPU_OPS["FADD"] if op_name == "ADD" else _VPU_OPS["FMUL"]
        else:
            post_vpu = _VPU_OPS["ADD"] if op_name == "ADD" else _VPU_OPS["MUL"]
        all_instrs.append(_vpu(result_vreg, acc_vreg, post_vpu, const_vreg))
        acc_vreg = result_vreg
        out_vmem = num_tiles + 1
    else:
        out_vmem = num_tiles if num_tiles > 1 else 1

    all_instrs.append(_store(out_vmem, acc_vreg))
    all_instrs.append(_halt())
    outputs = [{"addr": out_vmem, "param": out_arg, "offset": 0, "count": 1}]

    return {
        "op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
        "outputs": outputs, "num_output_tiles": 1, "out": out_arg,
        "reduce": _REDUCE_COMBINE.get(vpu_op, "sum"),
    }


def _render_colreduce_sxu_program(uops: list[UOp]) -> dict | None:
    """Column-wise reduction (axis=0) via VPU_*_REDUCE_COL primitives.

    Iter 4 scope: N<=4 rows, M=4 cols, single 4x4 tile, SUM only.
    Tinygrad pattern: RANGE=1, STORE, out_size>1, MUL=0, LOAD=1.
    """
    op_counts = Counter(u.op.name for u in uops)
    params = [u for u in uops if u.op is Ops.PARAM]
    if len(params) != 2: return None
    for p in params:
        if not isinstance(p.dtype, PtrDType): return None
    out_size = params[0].dtype.size
    src_size = params[1].dtype.size
    if not (out_size > 1 and src_size > out_size and src_size % out_size == 0): return None
    if op_counts.get("RANGE", 0) != 1 or op_counts.get("STORE", 0) != 1: return None
    # Col-reduce has no stride multiply in the index path. PROD adds data-path
    # MULs; accept either zero MUL (SUM/MAX/MIN) or data-path MULs only (PROD).
    data_alu = _data_alu_ops(uops)
    if op_counts.get("MUL", 0) != data_alu.get("MUL", 0):
        return None
    reduce_op = _detect_reduce_op(op_counts, data_alu)
    if reduce_op is None: return None
    ncols = out_size
    nrows = src_size // ncols

    is_float_col = any("float" in str(p.dtype) for p in params)
    # Float col-reductions: SUM/MAX/PROD lower directly. Float MIN uses the
    # MUL(-1.0)+MAX+MUL(-1.0) negation decomposition.
    if is_float_col:
        if reduce_op == "MAX" and data_alu.get("MUL", 0) > 0:
            if _is_float_min_negation(uops):
                reduce_op = "MIN"
            else:
                return None

    _INT32_MIN = -(1 << 31)
    _INT32_MAX = (1 << 31) - 1
    _FLOAT_NEG_INF_BITS = -(1 << 23)     # 0xFF800000
    _FLOAT_POS_INF_BITS = 0x7F800000
    _REDUCE_VPU_INT = {
        "SUM":  (_VPU_OPS["SUM_REDUCE_COL"], _VPU_OPS["ADD"], 0),
        "MAX":  (_VPU_OPS["MAX_REDUCE_COL"], _VPU_OPS["MAX"], _INT32_MIN),
        "MIN":  (_VPU_OPS["MIN_REDUCE_COL"], _VPU_OPS["MIN"], _INT32_MAX),
        "PROD": (_VPU_OPS["MUL_REDUCE_COL"], _VPU_OPS["MUL"], 1),
    }
    _FLOAT_ONE_BITS = 0x3F800000
    _REDUCE_VPU_FLOAT = {
        "SUM":  (_VPU_OPS["FSUM_REDUCE_COL"], _VPU_OPS["FADD"], 0),
        "MAX":  (_VPU_OPS["FMAX_REDUCE_COL"], _VPU_OPS["FMAX"], _FLOAT_NEG_INF_BITS),
        "MIN":  (_VPU_OPS["FMIN_REDUCE_COL"], _VPU_OPS["FMIN"], _FLOAT_POS_INF_BITS),
        "PROD": (_VPU_OPS["FPROD_REDUCE_COL"], _VPU_OPS["FMUL"], _FLOAT_ONE_BITS),
    }
    _REDUCE_VPU = _REDUCE_VPU_FLOAT if is_float_col else _REDUCE_VPU_INT
    vpu_op, combine_op, pad_value = _REDUCE_VPU[reduce_op]

    # Post-reduction scalar mul (e.g. mean = sum * (1/N)).
    # Only detect the pattern for FLOAT SUM reductions: one extra
    # data-path MUL with a float CONST combines the reduction output
    # with a scalar. Skipping other cases keeps int PROD / MIN / MAX
    # reductions routing through the existing code paths.
    post_op_name = None
    post_const = None
    if is_float_col and reduce_op == "SUM":
        post_mul_uops = [u for u in uops if u.op is Ops.MUL and _has_load_src(u)]
        for u in post_mul_uops:
            cst = next((s for s in u.src if s.op is Ops.CONST and isinstance(s.arg, float)), None)
            if cst is not None:
                post_op_name = "MUL"
                post_const = float(cst.arg)
                break

    # Reject col-reduce kernels with pre-reduction data-path MUL that we
    # can't fold into a post-op (sum(x*x, axis=0), sum(-x, axis=0), etc.).
    # Float MIN uses a MUL(-1) negation decomp and stays on this path;
    # guard fires only for SUM / MAX.
    has_data_mul = data_alu.get("MUL", 0) > 0
    if has_data_mul and post_op_name is None and reduce_op in ("SUM", "MAX"):
        return None

    out_arg, src_arg = 0, 1
    num_row_tiles = (nrows + _ROWS - 1) // _ROWS
    num_col_tiles = (ncols + _COLS - 1) // _COLS
    # VMEM slot layout: slots 0..M-1 hold src col-tiles + output tiles, and
    # the post-reduction constant (if any) sits one past the end so we
    # don't collide with either.
    reserved_slots = num_col_tiles * (num_row_tiles + 1)
    data_plan: list[dict] = []
    all_instrs: list[str] = []
    outputs: list[dict] = []

    src_addr = 0
    vreg = 0

    # Pre-load scaling constant (broadcast tile) if post-op is in play.
    post_const_vreg = None
    post_const_addr = None
    if post_op_name is not None:
        post_const_addr = reserved_slots
        c_bits = int(np.frombuffer(np.float32(post_const).tobytes(), dtype=np.int32)[0])
        data_plan.append({"type": "VMEM", "addr": post_const_addr,
                          "layout": "broadcast_const", "value": c_bits,
                          "count": _TILE_ELEMS, "dtype": "int32"})
        post_const_vreg = vreg; vreg += 1
        all_instrs.append(_load(post_const_vreg, post_const_addr))

    # Each col-tile gets its own output; within a col-tile we reduce all row-tiles.
    for ct in range(num_col_tiles):
        col_base = ct * _COLS
        tile_cols = min(_COLS, ncols - col_base)
        per_tile_red_vregs: list[int] = []
        for rt in range(num_row_tiles):
            row_base = rt * _ROWS
            tile_rows = min(_ROWS, nrows - row_base)
            entry = {
                "type": "VMEM", "addr": src_addr, "param": src_arg,
                "mode": "MATRIX_TILE", "matrix_nrows": nrows, "matrix_ncols": ncols,
                "row_base": row_base, "col_base": col_base,
                "tile_rows": tile_rows, "tile_cols": tile_cols,
                "offset": 0, "count": tile_rows * tile_cols, "dtype": "int32",
            }
            if pad_value != 0:
                entry["pad_value"] = pad_value
            data_plan.append(entry)
            src_vreg = vreg; vreg += 1
            red_vreg = vreg; vreg += 1
            all_instrs.append(_load(src_vreg, src_addr))
            all_instrs.append(_vpu(red_vreg, src_vreg, vpu_op))
            per_tile_red_vregs.append(red_vreg)
            src_addr += 1
        # Combine row-tile reductions for this col-tile.
        if len(per_tile_red_vregs) == 1:
            final_vreg = per_tile_red_vregs[0]
        else:
            acc = per_tile_red_vregs[0]
            for nxt in per_tile_red_vregs[1:]:
                out_vreg = vreg; vreg += 1
                all_instrs.append(_vpu(out_vreg, acc, combine_op, nxt))
                acc = out_vreg
            final_vreg = acc
        # Apply post-reduction scalar op (currently only FMUL).
        if post_op_name is not None:
            scaled_vreg = vreg; vreg += 1
            post_vpu = _VPU_OPS["FMUL"] if post_op_name == "MUL" else _VPU_OPS["FADD"]
            all_instrs.append(_vpu(scaled_vreg, final_vreg, post_vpu, post_const_vreg))
            final_vreg = scaled_vreg
        out_addr = src_addr; src_addr += 1
        all_instrs.append(_store(out_addr, final_vreg))
        outputs.append({
            "addr": out_addr, "param": out_arg,
            "offset": col_base, "count": tile_cols,
        })

    all_instrs.append(_halt())
    return {
        "op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
        "outputs": outputs, "num_output_tiles": num_col_tiles, "out": out_arg,
    }


def _render_rowreduce_sxu_program(uops: list[UOp]) -> dict | None:
    """Row-wise reduction (axis=1) via VPU_{SUM,MAX,MIN}_REDUCE primitives.

    Iter 1 scope: single-tile N<=4, M<=4, SUM only. Uses VPU_SUM_REDUCE which
    broadcasts each row's sum to all lanes of that row; output parser extracts
    position [r * _COLS] for r in range(tile_rows).
    """
    op_counts = Counter(u.op.name for u in uops)
    params = [u for u in uops if u.op is Ops.PARAM]
    if len(params) != 2: return None
    for p in params:
        if not isinstance(p.dtype, PtrDType): return None
    out_size = params[0].dtype.size
    src_size = params[1].dtype.size
    if not (out_size >= 1 and src_size > out_size and src_size % out_size == 0): return None
    if op_counts.get("RANGE", 0) != 1 or op_counts.get("STORE", 0) != 1: return None
    # Row-reduce has one stride-multiply in the index expression. PROD kernels
    # add data-path MULs so total MUL = 1 + nloads-1 when fully unrolled, or
    # 1 + 1 = 2 for a RANGE loop. Accept total MUL >= 1.
    if op_counts.get("MUL", 0) < 1: return None
    data_alu = _data_alu_ops(uops)
    reduce_op = _detect_reduce_op(op_counts, data_alu)
    if reduce_op is None: return None
    nrows = out_size
    ncols = src_size // nrows
    if ncols < 2: return None
    # Tinygrad may fully unroll or keep a RANGE loop; match both.
    nloads = op_counts.get("LOAD", 0)
    if nloads != ncols and nloads != 1: return None
    is_float_row = any("float" in str(p.dtype) for p in params)
    # Float row reductions: SUM/MAX/PROD lower directly; MIN via the
    # negation-decomp rewrite shared with the scalar/col paths.
    if is_float_row:
        data_alu = _data_alu_ops(uops)
        if reduce_op == "MAX" and data_alu.get("MUL", 0) > 0:
            if _is_float_min_negation(uops):
                reduce_op = "MIN"
            else:
                return None
    _INT32_MIN = -(1 << 31)
    _INT32_MAX = (1 << 31) - 1
    _FLOAT_NEG_INF_BITS = -(1 << 23)     # 0xFF800000
    _FLOAT_POS_INF_BITS = 0x7F800000
    _REDUCE_VPU_INT = {
        "SUM":  (_VPU_OPS["SUM_REDUCE"], _VPU_OPS["ADD"], 0),
        "MAX":  (_VPU_OPS["MAX_REDUCE"], _VPU_OPS["MAX"], _INT32_MIN),
        "MIN":  (_VPU_OPS["MIN_REDUCE"], _VPU_OPS["MIN"], _INT32_MAX),
        "PROD": (_VPU_OPS["MUL_REDUCE"], _VPU_OPS["MUL"], 1),
    }
    _FLOAT_ONE_BITS = 0x3F800000
    _REDUCE_VPU_FLOAT = {
        "SUM":  (_VPU_OPS["FSUM_REDUCE"], _VPU_OPS["FADD"], 0),
        "MAX":  (_VPU_OPS["FMAX_REDUCE"], _VPU_OPS["FMAX"], _FLOAT_NEG_INF_BITS),
        "MIN":  (_VPU_OPS["FMIN_REDUCE"], _VPU_OPS["FMIN"], _FLOAT_POS_INF_BITS),
        "PROD": (_VPU_OPS["FPROD_REDUCE"], _VPU_OPS["FMUL"], _FLOAT_ONE_BITS),
    }
    _REDUCE_VPU = _REDUCE_VPU_FLOAT if is_float_row else _REDUCE_VPU_INT
    vpu_op, combine_op, pad_value = _REDUCE_VPU[reduce_op]

    # Pre-reduction data-path MUL count (reused by both post-op detection
    # and the guard just below).
    data_alu = _data_alu_ops(uops)

    # Post-reduction scalar mul for float SUM (mean = sum * (1/ncols)).
    post_op_name = None
    post_const = None
    if is_float_row and reduce_op == "SUM":
        for u in uops:
            if u.op is Ops.MUL and _has_load_src(u):
                cst = next((s for s in u.src if s.op is Ops.CONST and isinstance(s.arg, float)), None)
                if cst is not None:
                    post_op_name = "MUL"
                    post_const = float(cst.arg)
                    break

    # Reject row-reduce kernels with pre-reduction data-path MUL that we
    # can't fold into a post-op (sum(x*x, axis=1), max(-x, axis=1), etc.).
    # Float MIN uses a MUL(-1) negation decomposition and stays on this
    # path; guard only fires for SUM / MAX combos.
    has_data_mul = data_alu.get("MUL", 0) > 0
    if has_data_mul and post_op_name is None and reduce_op in ("SUM", "MAX"):
        return None

    out_arg, src_arg = 0, 1
    num_row_tiles = (nrows + _ROWS - 1) // _ROWS
    num_col_tiles = (ncols + _COLS - 1) // _COLS
    reserved_slots = num_row_tiles * (num_col_tiles + 1)
    data_plan: list[dict] = []
    all_instrs: list[str] = []
    outputs: list[dict] = []

    src_addr = 0
    vreg = 0

    post_const_vreg = None
    if post_op_name is not None:
        post_const_addr = reserved_slots
        c_bits = int(np.frombuffer(np.float32(post_const).tobytes(), dtype=np.int32)[0])
        data_plan.append({"type": "VMEM", "addr": post_const_addr,
                          "layout": "broadcast_const", "value": c_bits,
                          "count": _TILE_ELEMS, "dtype": "int32"})
        post_const_vreg = vreg; vreg += 1
        all_instrs.append(_load(post_const_vreg, post_const_addr))

    for rt in range(num_row_tiles):
        row_base = rt * _ROWS
        tile_rows = min(_ROWS, nrows - row_base)
        per_ct_red_vregs: list[int] = []
        for ct in range(num_col_tiles):
            col_base = ct * _COLS
            tile_cols = min(_COLS, ncols - col_base)
            entry = {
                "type": "VMEM", "addr": src_addr, "param": src_arg,
                "mode": "MATRIX_TILE", "matrix_nrows": nrows, "matrix_ncols": ncols,
                "row_base": row_base, "col_base": col_base,
                "tile_rows": tile_rows, "tile_cols": tile_cols,
                "offset": 0, "count": tile_rows * tile_cols, "dtype": "int32",
            }
            if pad_value != 0:
                entry["pad_value"] = pad_value
            data_plan.append(entry)
            src_vreg = vreg; vreg += 1
            red_vreg = vreg; vreg += 1
            all_instrs.append(_load(src_vreg, src_addr))
            all_instrs.append(_vpu(red_vreg, src_vreg, vpu_op))
            per_ct_red_vregs.append(red_vreg)
            src_addr += 1
        # Combine col-tile partial row-reductions (each broadcast across row).
        if len(per_ct_red_vregs) == 1:
            final_vreg = per_ct_red_vregs[0]
        else:
            acc = per_ct_red_vregs[0]
            for nxt in per_ct_red_vregs[1:]:
                out_vreg = vreg; vreg += 1
                all_instrs.append(_vpu(out_vreg, acc, combine_op, nxt))
                acc = out_vreg
            final_vreg = acc
        if post_op_name is not None:
            scaled_vreg = vreg; vreg += 1
            post_vpu = _VPU_OPS["FMUL"] if post_op_name == "MUL" else _VPU_OPS["FADD"]
            all_instrs.append(_vpu(scaled_vreg, final_vreg, post_vpu, post_const_vreg))
            final_vreg = scaled_vreg
        out_addr = src_addr; src_addr += 1
        all_instrs.append(_store(out_addr, final_vreg))
        outputs.append({
            "addr": out_addr, "param": out_arg,
            "offset": row_base, "count": tile_rows, "extract": "row_heads",
        })

    all_instrs.append(_halt())
    return {
        "op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
        "outputs": outputs, "num_output_tiles": num_row_tiles, "out": out_arg,
    }


_ALU_OP_NAMES = {"ADD", "SUB", "MUL", "MAX", "MIN", "IDIV", "MOD", "AND", "OR",
                 "XOR", "NOT", "CMPLT", "CMPEQ", "CMPNE", "SHL", "SHR", "WHERE",
                 "RECIP", "RECIPROCAL", "TRUNC", "SELECT", "WMMA", "MULACC"}


def _render_cast_sxu_program(uops: list[UOp]) -> dict | None:
    """Render int32↔float32 CAST kernels as SXU_PROGRAM using VPU_I2F / VPU_F2I.

    Pattern: 2 params (out, src), exactly one CAST op, int↔float type pair.
    bool↔int casts are handled by the legacy analyzer (bit-level, not value conversion).
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("CAST", 0) == 0:
        return None
    # Reject if anything beyond CAST/movement/indexing ops is present
    _allowed = {"CAST", "CONST", "INDEX", "LOAD", "STORE", "PARAM", "SINK", "GROUP",
                "END", "RANGE", "VECTORIZE", "GEP", "MUL", "ADD"}
    if any(c > 0 and n not in _allowed for n, c in op_counts.items()):
        return None
    # The MUL/ADD must be index arithmetic (no LOAD in source tree).
    for u in uops:
        if u.op in (Ops.MUL, Ops.ADD) and _has_load_src(u):
            return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    out_dtype = str(params[0].dtype)
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    src_dtype = str(params[src_arg].dtype)
    # Only handle the value-conversion cases (not bool/int bitwidth changes).
    src_is_bool = "bool" in src_dtype
    out_is_bool = "bool" in out_dtype
    if "float" in out_dtype and ("int" in src_dtype and not src_is_bool):
        vpu_op = _VPU_OPS["I2F"]
    elif "int" in out_dtype and not out_is_bool and "float" in src_dtype:
        vpu_op = _VPU_OPS["F2I"]
    elif "int" in out_dtype and not out_is_bool and src_is_bool:
        # bool → int32: LOAD lifts the bool into an int32 register; COPY passes it through.
        vpu_op = _VPU_OPS["COPY"]
    elif ("float" in out_dtype and "float" in src_dtype) or (
            "int" in out_dtype and not out_is_bool and
            "int" in src_dtype and not src_is_bool):
        # Same-dtype cast chain (e.g. int→float→int fused): emit identity via COPY
        vpu_op = _VPU_OPS["COPY"]
    else:
        return None

    out_size = params[0].dtype.size
    src_size = params[src_arg].dtype.size
    if out_size != src_size or out_size <= 0:
        return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    all_instrs: list[str] = []
    data_plan: list[dict] = []
    outputs: list[dict] = []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)
        base = tile_idx * 2  # src, out
        entry = {"type": "VMEM", "addr": base, "param": src_arg,
                 "offset": offset, "count": count, "dtype": "int32"}
        if src_is_bool:
            entry["bool"] = True
        data_plan.append(entry)
        out_vmem = base + 1
        all_instrs += [
            _load(0, base),         # v0 = src (int bits)
            _vpu(1, 0, vpu_op),     # v1 = I2F(v0) or F2I(v0) or COPY(v0)
            _store(out_vmem, 1),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


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


def _render_copy_sxu_program(uops: list[UOp]) -> dict | None:
    """Render a pure data-movement kernel (no ALU) as a LOAD/STORE SXU_PROGRAM.

    Covers the simple reshape-as-copy pattern tinygrad emits: 2 params (out, src),
    matching sizes, a LOAD/STORE pair per tile with no arithmetic. Each tile
    LOADs from VMEM[tile_in] and STOREs to VMEM[tile_out] without a VPU op.
    """
    op_counts = Counter(u.op.name for u in uops)
    if not op_counts.get("STORE") or not op_counts.get("LOAD"):
        return None
    # Only block on data-path ALU ops (index arithmetic is allowed).
    data_alu = _data_alu_ops(uops)
    if sum(data_alu.values()) > 0:
        return None
    # Guard against other op classes that must still block copy-detection.
    for n in ("WHERE", "MOD", "RECIP", "RECIPROCAL", "TRUNC", "WMMA", "MULACC", "SELECT"):
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
    # Require matching element dtype — dtype conversions (bool<->int32 casts)
    # need the VPU_BINARY widening path.
    out_base = params[out_arg].dtype.base.itemsize
    src_base = params[src_arg].dtype.base.itemsize
    if out_base != src_base:
        return None
    # Require LOAD and STORE index expressions to differ by a constant offset
    # (possibly zero). This accepts reshape (offset=0) and contiguous slice /
    # shrink (offset=K) while rejecting permute/transpose/flip/stride.
    def _index_of(addr_uop):
        if addr_uop.op is Ops.INDEX:
            return addr_uop.src[1]
        return None

    def _split_const(idx):
        # Return (base_expr_or_None, const) if idx == ADD(base, CONST) or CONST.
        if idx.op is Ops.CONST and isinstance(idx.arg, int):
            return (None, idx.arg)
        if idx.op is Ops.ADD:
            # ADD(x, CONST) — tinygrad canonicalizes with const on the right.
            a, b = idx.src
            if b.op is Ops.CONST and isinstance(b.arg, int):
                return (a, b.arg)
            if a.op is Ops.CONST and isinstance(a.arg, int):
                return (b, a.arg)
        return (idx, 0)

    # Detect scalar-broadcast: LOAD index is a literal CONST (no RANGE
    # dependence). Emit a single load + BROADCAST_SCALAR per tile.
    def _has_range(u, seen=None):
        if seen is None: seen = set()
        if id(u) in seen: return False
        seen.add(id(u))
        if u.op is Ops.RANGE: return True
        return any(_has_range(s, seen) for s in u.src)

    load_idxs_all = [_index_of(s.src[1].src[0]) for s in stores if s.src[1].op is Ops.LOAD]
    load_idxs = set(load_idxs_all)
    all_loads_no_range = all(li is not None and not _has_range(li) for li in load_idxs_all)
    if len(load_idxs) == 1 and all_loads_no_range:
        li = load_idxs_all[0]
        # Scalar broadcast path: single LOAD index, constant
        load_const = li.arg if li.op is Ops.CONST else None
        if load_const is not None:
            num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
            data_plan = [{
                "type": "VMEM", "addr": 0, "param": src_arg,
                "offset": load_const, "count": 1, "dtype": "int32",
            }]
            all_instrs = [_load(0, 0), f"2 9 0 1 0 0 0 0 0 0"]  # SXU_BROADCAST_SCALAR vd=1 vs=0 row=0 col=0
            outputs = []
            for t in range(num_tiles):
                offset = t * _TILE_ELEMS
                count = min(_TILE_ELEMS, out_size - offset)
                out_addr = 1 + t
                all_instrs.append(_store(out_addr, 1))
                outputs.append({
                    "addr": out_addr, "param": out_arg,
                    "offset": offset, "count": count,
                })
            all_instrs.append(_halt())
            return {
                "op": "SXU_PROGRAM", "instructions": all_instrs,
                "data_plan": data_plan, "outputs": outputs,
                "num_output_tiles": num_tiles, "out": out_arg,
            }
    # Row-broadcast: all LOAD indices are CONSTs (no RANGE), out_size is a
    # multiple of src_size, and src_size is small (<=_COLS). Emit
    # LOAD + BROADCAST_ROW + STOREs per tile. This handles
    # Tensor([[v0, v1, ..., vM]]).expand(N, M).
    if (all_loads_no_range and src_size <= _COLS and out_size % src_size == 0
            and out_size > src_size):
        num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
        data_plan = [{
            "type": "VMEM", "addr": 0, "param": src_arg,
            "offset": 0, "count": src_size, "dtype": "int32",
        }]
        all_instrs = [_load(0, 0), f"2 10 0 1 0 0 0 0 0 0"]  # SXU_BROADCAST_ROW vd=1 vs=0 srcRow=0
        outputs = []
        for t in range(num_tiles):
            offset = t * _TILE_ELEMS
            count = min(_TILE_ELEMS, out_size - offset)
            out_addr = 1 + t
            all_instrs.append(_store(out_addr, 1))
            outputs.append({
                "addr": out_addr, "param": out_arg,
                "offset": offset, "count": count,
            })
        all_instrs.append(_halt())
        return {
            "op": "SXU_PROGRAM", "instructions": all_instrs,
            "data_plan": data_plan, "outputs": outputs,
            "num_output_tiles": num_tiles, "out": out_arg,
        }

    # Non-broadcast copy path requires src_size >= out_size.
    if src_size < out_size:
        return None

    src_offset = None  # constant offset (elements) from out to in
    for store in stores:
        out_addr = store.src[0]
        val = store.src[1]
        if val.op is not Ops.LOAD:
            return None
        in_addr = val.src[0]
        out_idx = _index_of(out_addr)
        in_idx = _index_of(in_addr)
        if out_idx is None or in_idx is None:
            return None
        if out_idx is in_idx:
            k = 0
        else:
            out_base, out_k = _split_const(out_idx)
            in_base,  in_k  = _split_const(in_idx)
            if out_base is not in_base:
                return None
            k = in_k - out_k
        if src_offset is None:
            src_offset = k
        elif src_offset != k:
            return None
    if src_offset is None or src_offset < 0:
        return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    all_instrs: list[str] = []
    data_plan: list[dict] = []
    outputs: list[dict] = []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)
        in_addr = tile_idx
        out_addr = num_tiles + tile_idx
        data_plan.append({
            "type": "VMEM", "addr": in_addr, "param": src_arg,
            "offset": offset + src_offset, "count": count, "dtype": "int32",
        })
        all_instrs.append(_load(0, in_addr))
        all_instrs.append(_store(out_addr, 0))
        outputs.append({
            "addr": out_addr, "param": out_arg,
            "offset": offset, "count": count,
        })
    all_instrs.append(_halt())
    return {
        "op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
        "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg,
    }


def _render_multistep_sxu_program(uops: list[UOp]) -> dict | None:
    """Render multi-step VPU patterns (abs, clip, MOD, CMPEQ) as SXU_PROGRAM.

    These patterns require 2-3 VPU instructions per tile but use existing VPU opcodes.
    """
    op_counts = Counter(u.op.name for u in uops)
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}

    out_params = set()
    for s in uops:
        if s.op is Ops.STORE:
            p = _find_unique_param_arg(s.src[0])
            if p is not None: out_params.add(p)
    if len(out_params) != 1:
        return None
    out_arg = next(iter(out_params))
    out_size = params[out_arg].dtype.size
    src_params = sorted(k for k in params if k != out_arg)

    # Use data-path ALU counts (exclude index arithmetic MUL/ADD)
    data_alu = _data_alu_ops(uops)
    has_where = op_counts.get("WHERE", 0) > 0
    has_mul = data_alu.get("MUL", 0) > 0
    has_max = data_alu.get("MAX", 0) > 0
    has_cmplt = data_alu.get("CMPLT", 0) > 0 or op_counts.get("CMPLT", 0) > 0
    has_cmpne = data_alu.get("CMPNE", 0) > 0 or op_counts.get("CMPNE", 0) > 0
    has_idiv = data_alu.get("DIV", 0) > 0
    has_mod = op_counts.get("MOD", 0) > 0

    SUB_OP, MAX_OP, MIN_OP = _VPU_OPS["SUB"], _VPU_OPS["MAX"], _VPU_OPS["MIN"]
    DIV_OP, MUL_OP, ADD_OP = _VPU_OPS["DIV"], _VPU_OPS["MUL"], _VPU_OPS["ADD"]
    CMPNE_OP = _VPU_OPS["CMPNE"]
    FMUL_OP, FRECIP_OP = _VPU_OPS["FMUL"], _VPU_OPS["FRECIP"]

    # --- Float tensor-tensor divide: 3 params, MUL + RECIPROCAL, float dtypes ---
    # Pattern: a / b = a * (1/b) = FMUL(FRECIP(b), a)
    has_recip = op_counts.get("RECIPROCAL", 0) > 0
    all_float = all("float" in str(params[p].dtype) for p in [out_arg] + src_params)
    if (len(src_params) == 2 and has_recip and has_mul and all_float
            and not has_where and not has_cmplt):
        # Trace operands: RECIPROCAL applies to one LOAD, MUL combines that with the other LOAD
        recip_uops = [u for u in uops if u.op is Ops.RECIPROCAL and _has_load_src(u)]
        if recip_uops:
            recip_src_param = _find_unique_param_arg(recip_uops[0])
            # The other src param is the numerator
            other_params = [p for p in src_params if p != recip_src_param]
            if recip_src_param is not None and len(other_params) == 1:
                num_arg = other_params[0]
                denom_arg = recip_src_param
                num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
                addrs_per_tile = 3  # num, denom, out
                all_instrs, data_plan, outputs = [], [], []
                for tile_idx in range(num_tiles):
                    base = tile_idx * addrs_per_tile
                    offset = tile_idx * _TILE_ELEMS
                    count = min(_TILE_ELEMS, out_size - offset)
                    data_plan.append({"type": "VMEM", "addr": base, "param": num_arg,
                                      "offset": offset, "count": count, "dtype": "int32"})
                    data_plan.append({"type": "VMEM", "addr": base + 1, "param": denom_arg,
                                      "offset": offset, "count": count, "dtype": "int32"})
                    out_vmem = base + 2
                    all_instrs += [
                        _load(0, base),              # v0 = numerator
                        _load(1, base + 1),          # v1 = denominator
                        _vpu(2, 1, FRECIP_OP),       # v2 = 1 / denominator (unary)
                        _vpu(3, 0, FMUL_OP, 2),      # v3 = v0 * v2 = a / b
                        _store(out_vmem, 3),
                    ]
                    outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
                all_instrs.append(_halt())
                return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
                        "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg}

    # --- Float scalar-numerator divide: 2 params (x, out), MUL + RECIPROCAL, float const numerator ---
    # Pattern: c / x = c * (1/x) = FMUL(broadcast(c_bits), FRECIP(x))
    # Guard against transcendental-wrapping kernels (tanh, sigmoid, etc.)
    # whose RECIPROCAL sits inside a larger expression. Those must route
    # to their dedicated renderers further down the chain rather than
    # silently miscompiling as c/x.
    if (len(src_params) == 1 and has_recip and has_mul and all_float
            and not has_where and not has_cmplt
            and not any(op_counts.get(n, 0) for n in ("EXP2", "LOG2", "SIN", "SQRT"))):
        recip_uops = [u for u in uops if u.op is Ops.RECIPROCAL and _has_load_src(u)]
        # Find float const used as MUL operand alongside the RECIPROCAL
        num_const = None
        for u in uops:
            if u.op is Ops.MUL and any(s.op is Ops.RECIPROCAL for s in u.src):
                for s in u.src:
                    if s.op is Ops.CONST and isinstance(s.arg, float):
                        num_const = float(s.arg)
                        break
                if num_const is not None:
                    break
        if recip_uops and num_const is not None:
            import struct
            denom_arg = src_params[0]
            const_bits = struct.unpack("<i", struct.pack("<f", num_const))[0]
            num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
            addrs_per_tile = 3  # const, denom, out
            all_instrs, data_plan, outputs = [], [], []
            for tile_idx in range(num_tiles):
                base = tile_idx * addrs_per_tile
                offset = tile_idx * _TILE_ELEMS
                count = min(_TILE_ELEMS, out_size - offset)
                data_plan.append({"type": "VMEM", "addr": base,
                                  "layout": "broadcast_const", "value": const_bits,
                                  "count": count, "dtype": "int32"})
                data_plan.append({"type": "VMEM", "addr": base + 1, "param": denom_arg,
                                  "offset": offset, "count": count, "dtype": "int32"})
                out_vmem = base + 2
                all_instrs += [
                    _load(0, base),
                    _load(1, base + 1),
                    _vpu(2, 1, FRECIP_OP),
                    _vpu(3, 0, FMUL_OP, 2),
                    _store(out_vmem, 3),
                ]
                outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
            all_instrs.append(_halt())
            return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
                    "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg}

    # --- ABS: 2 params, WHERE+CMPLT+CMPNE+MUL pattern → SUB(0,x), MAX(x, neg) ---
    # For float tensors use FSUB/FMAX; bits for 0.0 and 0 are identical so broadcast const reuses 0.
    # Reject kernels that also carry RECIPROCAL or data-path ADD — those
    # are larger expressions (e.g. softsign = x / (1 + |x|)) whose abs
    # subtree is only part of the computation.
    if (len(src_params) == 1 and has_where and has_cmplt and has_cmpne and has_mul
            and not has_idiv and not has_mod
            and op_counts.get("RECIPROCAL", 0) == 0
            and data_alu.get("ADD", 0) == 0):
        src_arg = src_params[0]
        src_is_float = "float" in str(params[src_arg].dtype)
        sub_op = _VPU_OPS["FSUB"] if src_is_float else SUB_OP
        max_op = _VPU_OPS["FMAX"] if src_is_float else MAX_OP
        num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
        addrs_per_tile = 3  # src, zeros, out
        all_instrs, data_plan, outputs = [], [], []
        for tile_idx in range(num_tiles):
            base = tile_idx * addrs_per_tile
            offset = tile_idx * _TILE_ELEMS
            count = min(_TILE_ELEMS, out_size - offset)
            data_plan.append({"type": "VMEM", "addr": base, "param": src_arg,
                              "offset": offset, "count": count, "dtype": "int32"})
            data_plan.append({"type": "VMEM", "addr": base + 1,
                              "layout": "broadcast_const", "value": 0, "count": count, "dtype": "int32"})
            out_vmem = base + 2
            all_instrs += [
                _load(0, base),        # v0 = src
                _load(1, base + 1),    # v1 = zeros
                _vpu(2, 1, sub_op, 0), # v2 = 0 - src = -src
                _vpu(3, 0, max_op, 2), # v3 = max(src, -src) = abs(src)
                _store(out_vmem, 3),
            ]
            outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
        all_instrs.append(_halt())
        return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
                "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg}

    # --- Float MIN decomposition: MUL(-1) + MAX pattern → emit FCMPLT + SELECT ---
    # For float minimum(a, b), tinygrad emits -max(-a, -b) using MUL by -1.0 (no XOR).
    # Lower as: cond = a < b; result = cond ? a : b
    FCMPLT_OP = _VPU_OPS["FCMPLT"]
    if (len(src_params) == 2 and has_max and has_mul and all_float
            and not has_where and data_alu.get("XOR", 0) == 0 and not has_cmplt and not has_idiv):
        # Find MUL(x, -1.0) UOps — trace to the params being negated
        neg_params: list[int] = []
        for u in uops:
            if u.op is Ops.MUL and _has_load_src(u):
                for s in u.src:
                    if s.op is Ops.CONST and isinstance(s.arg, float) and s.arg == -1.0:
                        other = next((o for o in u.src if o is not s), None)
                        if other is not None:
                            p = _find_unique_param_arg(other)
                            if p is not None and p not in neg_params:
                                neg_params.append(p)
                        break
        if len(neg_params) == 2:
            lhs_arg, rhs_arg = neg_params[0], neg_params[1]
            num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
            addrs_per_tile = 3
            all_instrs, data_plan, outputs = [], [], []
            for tile_idx in range(num_tiles):
                base = tile_idx * addrs_per_tile
                offset = tile_idx * _TILE_ELEMS
                count = min(_TILE_ELEMS, out_size - offset)
                data_plan.append({"type": "VMEM", "addr": base, "param": lhs_arg,
                                  "offset": offset, "count": count, "dtype": "int32"})
                data_plan.append({"type": "VMEM", "addr": base + 1, "param": rhs_arg,
                                  "offset": offset, "count": count, "dtype": "int32"})
                out_vmem = base + 2
                all_instrs += [
                    _load(0, base),                # v0 = a
                    _load(1, base + 1),            # v1 = b
                    _vpu(2, 0, FCMPLT_OP, 1),      # v2 = (a < b) ? 1 : 0
                    _select(3, 2, 0, 1),           # v3 = v2 ? a : b
                    _store(out_vmem, 3),
                ]
                outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
            all_instrs.append(_halt())
            return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
                    "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg}

    # --- Float MIN scalar const: minimum(x, c) = -max(-x, -c) → FCMPLT+SELECT with broadcast const ---
    # Reject chained max/min patterns (e.g. maximum(x,a).minimum(b)) where a
    # MAX has another MAX-derived value as an input — those need the multi-step
    # clip lowering, not a single-op min_const kernel.
    _has_chained_max = any(u.op is Ops.MAX and _has_load_src(u)
                           and any(_uop_contains(s, Ops.MAX) for s in u.src)
                           for u in uops)
    # Reject kernels that also carry transcendentals / reciprocal. Softplus
    # (log(1+exp(x))) decomposes to MAX+MUL+EXP2+LOG2 with no WHERE/CMPLT;
    # without this guard it silently false-matches as minimum(x, c).
    _has_trans_or_recip = any(op_counts.get(n, 0) for n in
                              ("EXP2", "LOG2", "SIN", "SQRT", "RECIPROCAL"))
    if (len(src_params) == 1 and has_max and has_mul and all_float
            and not has_where and data_alu.get("XOR", 0) == 0 and not has_cmplt and not has_idiv
            and not _has_chained_max and not _has_trans_or_recip):
        # Find MAX UOp and extract the constant (it's stored as -c)
        max_uops = [u for u in uops if u.op is Ops.MAX and _has_load_src(u)]
        if max_uops:
            max_uop = max_uops[0]
            const_children = [s for s in max_uop.src if s.op is Ops.CONST
                              and isinstance(s.arg, float)]
            if len(const_children) == 1:
                # Recover original const: stored as -c, actual c = -stored
                import struct
                actual_c = -float(const_children[0].arg)
                # Convert to int32 bit representation for broadcast
                const_bits = int(np.frombuffer(np.float32(actual_c).tobytes(), dtype=np.int32)[0])
                src_arg = src_params[0]
                num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
                addrs_per_tile = 3  # src, const, out
                all_instrs, data_plan, outputs = [], [], []
                for tile_idx in range(num_tiles):
                    base = tile_idx * addrs_per_tile
                    offset = tile_idx * _TILE_ELEMS
                    count = min(_TILE_ELEMS, out_size - offset)
                    data_plan.append({"type": "VMEM", "addr": base, "param": src_arg,
                                      "offset": offset, "count": count, "dtype": "int32"})
                    data_plan.append({"type": "VMEM", "addr": base + 1,
                                      "layout": "broadcast_const", "value": const_bits,
                                      "count": count, "dtype": "int32"})
                    out_vmem = base + 2
                    all_instrs += [
                        _load(0, base),                # v0 = x
                        _load(1, base + 1),            # v1 = const (broadcast)
                        _vpu(2, 0, FCMPLT_OP, 1),      # v2 = (x < c) ? 1 : 0
                        _select(3, 2, 0, 1),           # v3 = v2 ? x : c
                        _store(out_vmem, 3),
                    ]
                    outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
                all_instrs.append(_halt())
                return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
                        "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg}

    # --- MIN decomposition: XOR+MAX pattern → emit VPU MIN ---
    has_xor = data_alu.get("XOR", 0) > 0
    if has_xor and has_max and not has_where and not has_cmplt and not has_idiv:
        if len(src_params) == 2:
            # Tensor-tensor MIN
            # Find operand params from the MAX UOp sources
            max_uops = [u for u in uops if u.op is Ops.MAX and _has_load_src(u)]
            if max_uops:
                # The MAX operates on XOR(x,-1) and XOR(y,-1), trace through to params
                lhs_arg = rhs_arg = None
                for u in uops:
                    if u.op is Ops.XOR and _has_load_src(u):
                        p = _find_unique_param_arg(u)
                        if p is not None:
                            if lhs_arg is None: lhs_arg = p
                            elif lhs_arg != p: rhs_arg = p
                if lhs_arg is not None and rhs_arg is not None:
                    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
                    addrs_per_tile = 3
                    all_instrs, data_plan, outputs = [], [], []
                    for tile_idx in range(num_tiles):
                        base = tile_idx * addrs_per_tile
                        offset = tile_idx * _TILE_ELEMS
                        count = min(_TILE_ELEMS, out_size - offset)
                        data_plan.append({"type": "VMEM", "addr": base, "param": lhs_arg,
                                          "offset": offset, "count": count, "dtype": "int32"})
                        data_plan.append({"type": "VMEM", "addr": base + 1, "param": rhs_arg,
                                          "offset": offset, "count": count, "dtype": "int32"})
                        out_vmem = base + 2
                        all_instrs += [_load(0, base), _load(1, base + 1),
                                       _vpu(2, 0, MIN_OP, 1), _store(out_vmem, 2)]
                        outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
                    all_instrs.append(_halt())
                    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
                            "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg}
        # Scalar-const MIN stays on old path (XOR encoding makes const recovery fragile)

    # --- FUSED ADD+RELU: 3 params, ADD+WHERE+CMPLT → ADD then RELU ---
    has_add = data_alu.get("ADD", 0) > 0
    if (len(src_params) == 2 and has_where and has_cmplt and has_add and not has_idiv
            and op_counts.get("WHERE", 0) == op_counts.get("STORE", 0)):
        # Trace ADD operand order
        add_uops_data = [u for u in uops if u.op is Ops.ADD and _has_load_src(u)]
        if add_uops_data:
            lhs_arg = _find_unique_param_arg(add_uops_data[0].src[0])
            rhs_arg = _find_unique_param_arg(add_uops_data[0].src[1])
            if lhs_arg is not None and rhs_arg is not None:
                num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
                addrs_per_tile = 3  # lhs, rhs, out
                all_instrs, data_plan, outputs = [], [], []
                RELU_OP = 2  # VPU_RELU opcode
                for tile_idx in range(num_tiles):
                    base = tile_idx * addrs_per_tile
                    offset = tile_idx * _TILE_ELEMS
                    count = min(_TILE_ELEMS, out_size - offset)
                    data_plan.append({"type": "VMEM", "addr": base, "param": lhs_arg,
                                      "offset": offset, "count": count, "dtype": "int32"})
                    data_plan.append({"type": "VMEM", "addr": base + 1, "param": rhs_arg,
                                      "offset": offset, "count": count, "dtype": "int32"})
                    out_vmem = base + 2
                    all_instrs += [
                        _load(0, base), _load(1, base + 1),
                        _vpu(2, 0, ADD_OP, 1),   # v2 = x + y
                        _vpu(3, 2, RELU_OP),      # v3 = relu(v2)
                        _store(out_vmem, 3),
                    ]
                    outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
                all_instrs.append(_halt())
                return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
                        "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg}

    # --- CLIP: 2 params, WHERE+CMPLT (double count vs STORE), constants → MIN(x, hi), MAX(result, lo) ---
    store_count = op_counts.get("STORE", 0)
    if (len(src_params) == 1 and has_where and has_cmplt and not has_mul and not has_idiv
            and op_counts.get("WHERE", 0) > store_count):
        # Find clip constants: CONST values used by CMPLT (the comparisons in clip)
        cmplt_uops = [u for u in uops if u.op is Ops.CMPLT]
        clip_consts = sorted(set(s.arg for u in cmplt_uops for s in u.src
                                 if s.op is Ops.CONST and isinstance(s.arg, int)))
        if len(clip_consts) >= 2:
            lo_const = min(clip_consts)
            hi_const = max(clip_consts)
            src_arg = src_params[0]
            num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
            addrs_per_tile = 4  # src, hi, lo, out
            all_instrs, data_plan, outputs = [], [], []
            for tile_idx in range(num_tiles):
                base = tile_idx * addrs_per_tile
                offset = tile_idx * _TILE_ELEMS
                count = min(_TILE_ELEMS, out_size - offset)
                data_plan.append({"type": "VMEM", "addr": base, "param": src_arg,
                                  "offset": offset, "count": count, "dtype": "int32"})
                data_plan.append({"type": "VMEM", "addr": base + 1,
                                  "layout": "broadcast_const", "value": hi_const, "count": count, "dtype": "int32"})
                data_plan.append({"type": "VMEM", "addr": base + 2,
                                  "layout": "broadcast_const", "value": lo_const, "count": count, "dtype": "int32"})
                out_vmem = base + 3
                all_instrs += [
                    _load(0, base),         # v0 = src
                    _load(1, base + 1),     # v1 = hi
                    _load(2, base + 2),     # v2 = lo
                    _vpu(3, 0, MIN_OP, 1),  # v3 = min(src, hi)
                    _vpu(4, 3, MAX_OP, 2),  # v4 = max(min(src, hi), lo)
                    _store(out_vmem, 4),
                ]
                outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
            all_instrs.append(_halt())
            return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
                    "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg}

    # --- MOD: 3 params, IDIV or MOD present → DIV(x,y), MUL(q,y), SUB(x, product) ---
    if len(src_params) == 2 and (has_mod or has_idiv) and has_mul:
        # Find which param is lhs (dividend) and rhs (divisor)
        # MOD UOp if present, else look at IDIV
        if has_mod:
            mod_uop = next(u for u in uops if u.op is Ops.MOD)
            lhs_arg = _find_unique_param_arg(mod_uop.src[0])
            rhs_arg = _find_unique_param_arg(mod_uop.src[1])
        else:
            idiv_uop = next(u for u in uops if u.op is Ops.IDIV)
            lhs_arg = _find_unique_param_arg(idiv_uop.src[0])
            rhs_arg = _find_unique_param_arg(idiv_uop.src[1])
        if lhs_arg is None or rhs_arg is None:
            return None

        num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
        addrs_per_tile = 3  # lhs, rhs, out
        all_instrs, data_plan, outputs = [], [], []
        for tile_idx in range(num_tiles):
            base = tile_idx * addrs_per_tile
            offset = tile_idx * _TILE_ELEMS
            count = min(_TILE_ELEMS, out_size - offset)
            data_plan.append({"type": "VMEM", "addr": base, "param": lhs_arg,
                              "offset": offset, "count": count, "dtype": "int32"})
            data_plan.append({"type": "VMEM", "addr": base + 1, "param": rhs_arg,
                              "offset": offset, "count": count, "dtype": "int32"})
            out_vmem = base + 2
            all_instrs += [
                _load(0, base),         # v0 = x (dividend)
                _load(1, base + 1),     # v1 = y (divisor)
                _vpu(2, 0, DIV_OP, 1),  # v2 = x / y
                _vpu(3, 2, MUL_OP, 1),  # v3 = (x/y) * y
                _vpu(4, 0, SUB_OP, 3),  # v4 = x - (x/y)*y = x % y
                _store(out_vmem, 4),
            ]
            outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
        all_instrs.append(_halt())
        return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
                "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg}

    # --- CMPEQ: 2-3 params, chained CMPNE → CMPNE(x, y/const), CMPNE(result, True) ---
    # This is NOT(CMPNE(x, y)) = CMPEQ. Detect by checking if any CMPNE has a CMPNE source.
    if has_cmpne and not has_where and not has_mul and not has_idiv:
        cmpne_uops = [u for u in uops if u.op is Ops.CMPNE]
        has_chained_cmpne = any(s.op is Ops.CMPNE for u in cmpne_uops for s in u.src)
        if has_chained_cmpne:
            if len(src_params) == 2:
                # Tensor-tensor CMPEQ
                first_cmpne = next(u for u in cmpne_uops if all(s.op is not Ops.CMPNE for s in u.src))
                lhs_arg = _find_unique_param_arg(first_cmpne.src[0])
                rhs_arg = _find_unique_param_arg(first_cmpne.src[1])
                if lhs_arg is None or rhs_arg is None:
                    return None
                num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
                addrs_per_tile = 3  # lhs, rhs, out
                all_instrs, data_plan, outputs = [], [], []
                for tile_idx in range(num_tiles):
                    base = tile_idx * addrs_per_tile
                    offset = tile_idx * _TILE_ELEMS
                    count = min(_TILE_ELEMS, out_size - offset)
                    data_plan.append({"type": "VMEM", "addr": base, "param": lhs_arg,
                                      "offset": offset, "count": count, "dtype": "int32"})
                    data_plan.append({"type": "VMEM", "addr": base + 1, "param": rhs_arg,
                                      "offset": offset, "count": count, "dtype": "int32"})
                    out_vmem = base + 2
                    all_instrs += [
                        _load(0, base),
                        _load(1, base + 1),
                        _vpu(2, 0, _VPU_OPS["CMPEQ"], 1),
                        _store(out_vmem, 2),
                    ]
                    outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
                all_instrs.append(_halt())
                return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
                        "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg, "bool_out": True}
            elif len(src_params) == 1:
                # Scalar-const CMPEQ: find the const from the first CMPNE
                first_cmpne = next(u for u in cmpne_uops if all(s.op is not Ops.CMPNE for s in u.src))
                const_src = next((s for s in first_cmpne.src if s.op is Ops.CONST), None)
                if const_src is None:
                    return None
                # For float operands, pack float bits so bit-level CMPEQ matches IEEE754 equality.
                src_arg_tmp = src_params[0]
                if "float" in str(params[src_arg_tmp].dtype):
                    import struct
                    const_val = struct.unpack("<i", struct.pack("<f", float(const_src.arg)))[0]
                else:
                    const_val = int(const_src.arg)
                src_arg = src_params[0]
                num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
                addrs_per_tile = 3  # src, const, out
                all_instrs, data_plan, outputs = [], [], []
                for tile_idx in range(num_tiles):
                    base = tile_idx * addrs_per_tile
                    offset = tile_idx * _TILE_ELEMS
                    count = min(_TILE_ELEMS, out_size - offset)
                    data_plan.append({"type": "VMEM", "addr": base, "param": src_arg,
                                      "offset": offset, "count": count, "dtype": "int32"})
                    data_plan.append({"type": "VMEM", "addr": base + 1,
                                      "layout": "broadcast_const", "value": const_val, "count": count, "dtype": "int32"})
                    out_vmem = base + 2
                    all_instrs += [
                        _load(0, base),
                        _load(1, base + 1),
                        _vpu(2, 0, _VPU_OPS["CMPEQ"], 1),
                        _store(out_vmem, 2),
                    ]
                    outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
                all_instrs.append(_halt())
                return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
                        "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg, "bool_out": True}

    return None


def _render_where_sxu_program(uops: list[UOp]) -> dict | None:
    """Render a WHERE (ternary select) kernel as SXU_PROGRAM.

    WHERE(cond, lhs, rhs) = cond*lhs + (1-cond)*rhs via 4 VPU instructions per tile.
    Expects 4 params: out, cond (bool), lhs (int32), rhs (int32).
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("WHERE", 0) == 0:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) not in (2, 3, 4):
        return None

    # Find output param
    out_params = set()
    for s in uops:
        if s.op is Ops.STORE:
            p = _find_unique_param_arg(s.src[0])
            if p is not None: out_params.add(p)
    if len(out_params) != 1:
        return None
    out_arg = next(iter(out_params))
    out_size = params[out_arg].dtype.size
    input_args = sorted(k for k in params if k != out_arg)

    # All inputs must be same size or size 1 (broadcast)
    if not all(params[a].dtype.size in {1, out_size} for a in input_args):
        return None

    # Identify cond (bool dtype), lhs, rhs from WHERE UOp sources.
    where_uop = next(u for u in uops if u.op is Ops.WHERE)
    cond_arg = _find_unique_param_arg(where_uop.src[0])
    lhs_arg = _find_unique_param_arg(where_uop.src[1])
    rhs_arg = _find_unique_param_arg(where_uop.src[2])
    lhs_const = None
    rhs_const = None
    # 2-param (out+cond) / 3-param (out+cond+tensor) shapes allow CONST for the
    # missing lhs/rhs. 4-param keeps original strict tensor requirement.
    if len(params) in (2, 3):
        # Reject kernels with any data-path ALU (abs = MUL + WHERE, etc.).
        if sum(_data_alu_ops(uops).values()) > 0:
            return None
        # Reject kernels that also carry transcendentals / reciprocals —
        # these are not simple WHERE kernels (e.g. softplus decomposes to
        # WHERE+EXP+LOG). Without this, the outer WHERE would silently
        # render as `cond ? lhs : rhs` dropping the math branch.
        if any(op_counts.get(n, 0) for n in ("EXP2", "LOG2", "SIN", "SQRT", "RECIPROCAL")):
            return None
        # Extract CONST sides that are consistent across every WHERE.
        where_lhs_const = None
        where_rhs_const = None
        lhs_kind = rhs_kind = None  # "const" or "param"
        for w in uops:
            if w.op is not Ops.WHERE:
                continue
            for side, src in (("lhs", w.src[1]), ("rhs", w.src[2])):
                if src.op is Ops.CONST and not isinstance(src.arg, bool):
                    if side == "lhs":
                        if lhs_kind == "param": return None
                        lhs_kind = "const"
                        if where_lhs_const is None: where_lhs_const = src.arg
                        elif where_lhs_const != src.arg: return None
                    else:
                        if rhs_kind == "param": return None
                        rhs_kind = "const"
                        if where_rhs_const is None: where_rhs_const = src.arg
                        elif where_rhs_const != src.arg: return None
                else:
                    # The side must resolve to a tensor param.
                    p = _find_unique_param_arg(src)
                    if p is None:
                        return None
                    if side == "lhs":
                        if lhs_kind == "const": return None
                        lhs_kind = "param"
                        if lhs_arg is None: lhs_arg = p
                        elif lhs_arg != p: return None
                    else:
                        if rhs_kind == "const": return None
                        rhs_kind = "param"
                        if rhs_arg is None: rhs_arg = p
                        elif rhs_arg != p: return None
        lhs_const = where_lhs_const if lhs_kind == "const" else None
        rhs_const = where_rhs_const if rhs_kind == "const" else None
        if cond_arg is None or cond_arg not in input_args:
            return None
        # Tensor inputs used must match input_args exactly.
        tensor_inputs = sorted({a for a in (cond_arg, lhs_arg, rhs_arg) if a is not None})
        if tensor_inputs != sorted(input_args):
            return None
    else:
        if cond_arg is None or lhs_arg is None or rhs_arg is None:
            return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    addrs_per_tile = 4  # cond, lhs, rhs, out

    all_instrs: list[str] = []
    data_plan: list[dict] = []
    outputs: list[dict] = []
    uses_scalar_broadcast = False

    def _const_bits(c):
        if isinstance(c, bool): return int(c)
        if isinstance(c, float): return int(np.frombuffer(np.float32(c).tobytes(), dtype=np.int32)[0])
        return int(c)

    for tile_idx in range(num_tiles):
        base = tile_idx * addrs_per_tile
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)

        data_plan.append({"type": "VMEM", "addr": base, "param": cond_arg,
                          "offset": offset, "count": count, "dtype": "int32", "bool": True})
        if lhs_arg is not None:
            data_plan.append({"type": "VMEM", "addr": base + 1, "param": lhs_arg,
                              "offset": offset, "count": count, "dtype": "int32"})
        else:
            data_plan.append({"type": "VMEM", "addr": base + 1, "layout": "broadcast_const",
                              "value": _const_bits(lhs_const), "count": count, "dtype": "int32"})
        if rhs_arg is not None:
            data_plan.append({"type": "VMEM", "addr": base + 2, "param": rhs_arg,
                              "offset": offset, "count": count, "dtype": "int32"})
        else:
            data_plan.append({"type": "VMEM", "addr": base + 2, "layout": "broadcast_const",
                              "value": _const_bits(rhs_const), "count": count, "dtype": "int32"})

        out_vmem = base + 3
        all_instrs += [
            _load(0, base),         # v0 = cond
            _load(1, base + 1),     # v1 = lhs (true values)
            _load(2, base + 2),     # v2 = rhs (false values)
            _select(3, 0, 1, 2),    # v3 = (cond!=0) ? lhs : rhs
            _store(out_vmem, 3),
        ]
        outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})

    all_instrs.append(_halt())

    return {
        "op": "SXU_PROGRAM",
        "primitive": "SELECT",
        "instructions": all_instrs,
        "data_plan": data_plan,
        "outputs": outputs,
        "num_output_tiles": num_tiles,
        "out": out_arg,
    }


def _render_rowbc_sxu_program(uops: list[UOp]) -> dict | None:
    """Render row-broadcast binary ops as SXU_PROGRAM.

    Handles patterns of the form (nrows x ncols) OP (ncols,), such as GEMM
    output plus row bias. Each row becomes one VMEM tile result.
    """
    op_counts = Counter(u.op.name for u in uops)
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 3:
        return None

    has_bool_logic = op_counts.get("AND", 0) > 0 or op_counts.get("OR", 0) > 0 or op_counts.get("XOR", 0) > 0
    has_fused_cmp = op_counts.get("CMPLT", 0) > 0 and op_counts.get("WHERE", 0) > 0
    if not (op_counts.get("GROUP", 0) == 1 and op_counts.get("RANGE", 0) == 1 and
            op_counts.get("STORE", 0) == 4 and op_counts.get("LOAD", 0) == 8 and
            not has_bool_logic and not has_fused_cmp):
        return None

    if op_counts.get("CMPNE", 0):
        op_name = "CMPNE"
    elif op_counts.get("CMPLT", 0):
        op_name = "CMPLT"
    elif op_counts.get("MAX", 0):
        op_name = "MAX"
    elif op_counts.get("MUL", 0) > 1:
        op_name = "MUL"
    elif op_counts.get("ADD", 0):
        op_name = "ADD"
    else:
        return None

    out_arg = 0
    out_size = params[out_arg].dtype.size
    input_args = sorted(arg for arg in params if arg != out_arg)
    if out_size <= 0 or len(input_args) != 2:
        return None

    full_args = [arg for arg in input_args if params[arg].dtype.size == out_size]
    row_args = [arg for arg in input_args
                if 0 < params[arg].dtype.size < out_size
                and out_size % params[arg].dtype.size == 0
                and params[arg].dtype.size <= _TILE_ELEMS]
    if len(full_args) != 1 or len(row_args) != 1:
        return None

    lhs_arg = full_args[0]
    rhs_arg = row_args[0]
    ncols = params[rhs_arg].dtype.size
    nrows = out_size // ncols
    rhs_addr = 0

    data_plan: list[dict] = [{
        "type": "VMEM", "addr": rhs_addr, "param": rhs_arg,
        "offset": 0, "count": ncols, "dtype": "int32",
    }]
    instructions: list[str] = []
    outputs: list[dict] = []
    # Remap integer VPU ops to float variants when operating on float tensors.
    is_float = any("float" in str(params[p].dtype) for p in (out_arg, lhs_arg, rhs_arg))
    if is_float:
        _FLOAT_REMAP = {"ADD": "FADD", "SUB": "FSUB", "MUL": "FMUL",
                        "MAX": "FMAX", "MIN": "FMIN", "CMPLT": "FCMPLT"}
        if op_name in _FLOAT_REMAP:
            op_name = _FLOAT_REMAP[op_name]
    vpu_op = _VPU_OPS[op_name]
    rows_per_tile = _ROWS
    num_chunks = (nrows + rows_per_tile - 1) // rows_per_tile
    out_base = 1 + num_chunks

    for chunk_idx in range(num_chunks):
        row = chunk_idx * rows_per_tile
        rows_this_chunk = min(rows_per_tile, nrows - row)
        lhs_addr = 1 + chunk_idx
        out_addr = out_base + chunk_idx
        offset = row * ncols
        chunk_count = rows_this_chunk * ncols
        data_plan.append({
            "type": "VMEM", "addr": lhs_addr, "param": lhs_arg,
            "offset": offset, "count": chunk_count, "dtype": "int32",
        })
        instructions += [
            _load(0, lhs_addr),
            _load(1, rhs_addr),
            _broadcast_row(2, 1, 0),
            _vpu(3, 0, vpu_op, 2),
            _store(out_addr, 3),
        ]
        outputs.append({
            "addr": out_addr, "param": out_arg,
            "offset": offset, "count": chunk_count,
        })

    instructions.append(_halt())
    return {
        "op": "SXU_PROGRAM",
        "primitive": "BROADCAST_ROW",
        "instructions": instructions,
        "data_plan": data_plan,
        "outputs": outputs,
        "num_output_tiles": num_chunks,
        "out": out_arg,
        "bool_out": vpu_op in _VPU_BOOL_OPS,
    }


def _render_colbc_sxu_program(uops: list[UOp]) -> dict | None:
    """Render column-broadcast binary ops as SXU_PROGRAM for single-tile 2D kernels."""
    op_counts = Counter(u.op.name for u in uops)
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 3:
        return None

    out_arg = 0
    out_size = params[out_arg].dtype.size
    input_args = sorted(arg for arg in params if arg != out_arg)
    if out_size <= 0 or len(input_args) != 2 or out_size > _TILE_ELEMS:
        return None

    full_args = [arg for arg in input_args if params[arg].dtype.size == out_size]
    col_args = [arg for arg in input_args
                if 0 < params[arg].dtype.size < out_size
                and params[arg].dtype.size <= _ROWS
                and _classify_structured_broadcast_axis(uops, arg) == "col"]
    if len(full_args) != 1 or len(col_args) != 1:
        return None

    lhs_arg = full_args[0]
    rhs_arg = col_args[0]
    nrows = params[rhs_arg].dtype.size
    # A size-1 broadcast operand is a scalar broadcast, not a column broadcast.
    # Let the elementwise renderer emit BROADCAST_SCALAR for it.
    if nrows <= 1 or out_size % nrows != 0:
        return None
    ncols = out_size // nrows
    if ncols > _COLS:
        return None

    # Detect SUB lowered as ADD(a, MUL(b, -1)). op_counts alone would pick
    # MUL and silently emit the wrong kernel for col-broadcast subtract.
    neg_mul = next((u for u in uops if u.op is Ops.MUL and _has_load_src(u)
                    and any(s.op is Ops.CONST and s.arg == -1 for s in u.src)), None)
    neg_add = None
    if neg_mul is not None:
        neg_add = next((u for u in uops if u.op is Ops.ADD and _has_load_src(u)
                        and any(s is neg_mul for s in u.src)), None)

    col_is_lhs = False
    if neg_add is not None:
        vpu_name = "SUB"
        unneg_src = next(s for s in neg_add.src if s is not neg_mul)
        neg_src = next((s for s in neg_mul.src if s.op is not Ops.CONST), None)
        unneg_param = _find_unique_param_arg(unneg_src)
        neg_param = _find_unique_param_arg(neg_src) if neg_src is not None else None
        if unneg_param == rhs_arg and neg_param == lhs_arg:
            col_is_lhs = True
        elif unneg_param == lhs_arg and neg_param == rhs_arg:
            col_is_lhs = False
        else:
            return None
    else:
        non_comm_ops = {"CMPLT": Ops.CMPLT, "CMPNE": Ops.CMPNE, "SUB": Ops.SUB}
        vpu_name = None
        for name in ("CMPLT", "CMPNE", "CMPEQ", "MAX", "MIN", "SUB", "MUL", "ADD"):
            if op_counts.get(name, 0):
                vpu_name = name
                break
        if vpu_name is None:
            return None
        if vpu_name in non_comm_ops:
            op_uop = next((u for u in uops if u.op is non_comm_ops[vpu_name] and _has_load_src(u)), None)
            if op_uop is None:
                return None
            lhs_param = _find_unique_param_arg(op_uop.src[0])
            rhs_param = _find_unique_param_arg(op_uop.src[1])
            if lhs_param == rhs_arg and rhs_param == lhs_arg:
                col_is_lhs = True
            elif lhs_param == lhs_arg and rhs_param == rhs_arg:
                col_is_lhs = False
            else:
                return None

    # Remap integer VPU ops to float variants when operating on float tensors.
    is_float = any("float" in str(params[p].dtype)
                   for p in (out_arg, lhs_arg, rhs_arg))
    if is_float:
        _FLOAT_REMAP = {"ADD": "FADD", "SUB": "FSUB", "MUL": "FMUL",
                        "MAX": "FMAX", "MIN": "FMIN", "CMPLT": "FCMPLT"}
        if vpu_name in _FLOAT_REMAP:
            vpu_name = _FLOAT_REMAP[vpu_name]
    vpu_op = _VPU_OPS[vpu_name]
    data_plan: list[dict] = [
        {"type": "VMEM", "addr": 0, "param": rhs_arg, "offset": 0, "count": nrows, "dtype": "int32",
         "mode": "MATRIX_TILE", "matrix_nrows": nrows, "matrix_ncols": 1,
         "row_base": 0, "col_base": 0, "tile_rows": nrows, "tile_cols": 1},
        {"type": "VMEM", "addr": 1, "param": lhs_arg, "offset": 0, "count": out_size, "dtype": "int32"},
    ]
    va, vb = (2, 0) if col_is_lhs else (0, 2)
    instructions = [
        _load(0, 1),
        _load(1, 0),
        _broadcast_col(2, 1, 0),
        _vpu(3, va, vpu_op, vb),
        _store(2, 3),
        _halt(),
    ]
    outputs = [{"addr": 2, "param": out_arg, "offset": 0, "count": out_size}]
    return {
        "op": "SXU_PROGRAM",
        "primitive": "BROADCAST_COL",
        "instructions": instructions,
        "data_plan": data_plan,
        "outputs": outputs,
        "num_output_tiles": 1,
        "out": out_arg,
        "bool_out": vpu_op in _VPU_BOOL_OPS,
    }


def _render_colbc_where_sxu_program(uops: list[UOp]) -> dict | None:
    """Render WHERE(full < col_broadcast, full, full * const) as SXU_PROGRAM."""
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("WHERE", 0) == 0 or op_counts.get("CMPLT", 0) == 0 or op_counts.get("MUL", 0) == 0:
        return None

    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 3:
        return None
    out_arg = 0
    out_size = params[out_arg].dtype.size
    if out_size <= 0 or out_size > _TILE_ELEMS:
        return None

    input_args = sorted(arg for arg in params if arg != out_arg)
    full_args = [arg for arg in input_args if params[arg].dtype.size == out_size]
    col_args = [arg for arg in input_args
                if 0 < params[arg].dtype.size < out_size
                and params[arg].dtype.size <= _ROWS
                and _classify_structured_broadcast_axis(uops, arg) == "col"]
    if len(full_args) != 1 or len(col_args) != 1:
        return None
    full_arg, col_arg = full_args[0], col_args[0]

    where_uops = [u for u in uops if u.op is Ops.WHERE]
    if not where_uops:
        return None

    mul_consts: set[int] = set()
    for where_uop in where_uops:
        if len(where_uop.src) != 3:
            return None
        cond_uop, true_uop, false_uop = where_uop.src
        if cond_uop.op is not Ops.CMPLT or true_uop.op is not Ops.LOAD or false_uop.op is not Ops.MUL:
            return None
        if _find_unique_param_arg(true_uop) != full_arg:
            return None
        mul_loads = [src for src in false_uop.src if src.op is Ops.LOAD]
        mul_const_nodes = [src for src in false_uop.src if src.op is Ops.CONST and isinstance(src.arg, int)]
        if len(mul_loads) != 1 or len(mul_const_nodes) != 1 or _find_unique_param_arg(mul_loads[0]) != full_arg:
            return None
        mul_consts.add(int(mul_const_nodes[0].arg))

        cmplt_full = [src for src in cond_uop.src if src.op is Ops.LOAD and _find_unique_param_arg(src) == full_arg]
        cmplt_col = [src for src in cond_uop.src if src.op is Ops.LOAD and _find_unique_param_arg(src) == col_arg]
        if len(cmplt_full) != 1 or len(cmplt_col) != 1:
            return None

    if len(mul_consts) != 1:
        return None
    mul_const = next(iter(mul_consts))

    data_plan: list[dict] = [
        {"type": "VMEM", "addr": 0, "param": col_arg, "offset": 0, "count": params[col_arg].dtype.size, "dtype": "int32",
         "mode": "MATRIX_TILE", "matrix_nrows": params[col_arg].dtype.size, "matrix_ncols": 1,
         "row_base": 0, "col_base": 0, "tile_rows": params[col_arg].dtype.size, "tile_cols": 1},
        {"type": "VMEM", "addr": 1, "param": full_arg, "offset": 0, "count": out_size, "dtype": "int32"},
        {"type": "VMEM", "addr": 2, "layout": "broadcast_const", "value": mul_const, "count": out_size, "dtype": "int32"},
    ]
    instructions = [
        _load(0, 1),
        _load(1, 0),
        _broadcast_col(2, 1, 0),
        _vpu(3, 0, _VPU_OPS["CMPLT"], 2),
        _load(4, 2),
        _vpu(5, 0, _VPU_OPS["MUL"], 4),
        _select(6, 3, 0, 5),
        _store(3, 6),
        _halt(),
    ]
    outputs = [{"addr": 3, "param": out_arg, "offset": 0, "count": out_size}]
    return {
        "op": "SXU_PROGRAM",
        "primitive": "BROADCAST_COL_SELECT",
        "instructions": instructions,
        "data_plan": data_plan,
        "outputs": outputs,
        "num_output_tiles": 1,
        "out": out_arg,
    }


def _render_min_const_sxu_program(uops: list[UOp]) -> dict | None:
    """Render minimum(x, const) through native VPU MIN in SXU_PROGRAM."""
    op_counts = Counter(u.op.name for u in uops)
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or op_counts.get("STORE", 0) < 1 or op_counts.get("XOR", 0) == 0 or op_counts.get("MAX", 0) == 0:
        return None

    # Reject chained max/min patterns (e.g. max(0).min(5)) where a MAX consumes
    # another MAX's result. Those need the multi-step clip lowering, not a
    # single-op VPU_MIN that drops the outer maximum.
    if any(u.op is Ops.MAX and _has_load_src(u)
           and any(_uop_contains(s, Ops.MAX) for s in u.src)
           for u in uops):
        return None

    min_const = _find_min_scalar_const(uops)
    out_size = params[0].dtype.size
    src_size = params[1].dtype.size
    if min_const is None or out_size != src_size or out_size <= 0:
        return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan: list[dict] = []
    instructions: list[str] = []
    outputs: list[dict] = []
    const_addr = 0
    data_plan.append({
        "type": "VMEM", "addr": const_addr,
        "layout": "broadcast_const", "value": min_const,
        "count": _TILE_ELEMS, "dtype": "int32",
    })
    for tile_idx in range(num_tiles):
        lhs_addr = 1 + tile_idx
        out_addr = 1 + num_tiles + tile_idx
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)
        data_plan.append({
            "type": "VMEM", "addr": lhs_addr, "param": 1,
            "offset": offset, "count": count, "dtype": "int32",
        })
        instructions += [
            _load(0, lhs_addr),
            _load(1, const_addr),
            _vpu(2, 0, _VPU_OPS["MIN"], 1),
            _store(out_addr, 2),
        ]
        outputs.append({
            "addr": out_addr, "param": 0,
            "offset": offset, "count": count,
        })
    instructions.append(_halt())
    return {
        "op": "SXU_PROGRAM",
        "instructions": instructions,
        "data_plan": data_plan,
        "outputs": outputs,
        "num_output_tiles": num_tiles,
        "out": 0,
    }


def _find_alu_const(data_alu_uops: list[UOp], alu_op) -> int | None:
    """Find the scalar constant used as an operand of the given data-path ALU op."""
    for u in data_alu_uops:
        if u.op is alu_op:
            for src in u.src:
                if src.op is Ops.CONST and not isinstance(src.arg, bool):
                    return src.arg
                if src.op is Ops.CONST and isinstance(src.arg, bool):
                    return int(src.arg)
    return None

def _render_trunc_sxu_program(uops: list[UOp]) -> dict | None:
    """Render float32 trunc() as SXU_PROGRAM via F2I+I2F round-trip.

    Replaces the legacy HOST_UNARY TRUNC path. Pattern: single TRUNC UOp on float src.
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("TRUNC", 0) < 1:
        return None
    _allowed = {"TRUNC", "CONST", "INDEX", "LOAD", "STORE", "PARAM", "SINK",
                "GROUP", "END", "RANGE", "VECTORIZE", "GEP", "MUL", "ADD", "CAST"}
    if any(c > 0 and n not in _allowed for n, c in op_counts.items()):
        return None
    for u in uops:
        if u.op in (Ops.MUL, Ops.ADD) and _has_load_src(u):
            return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    src_size = params[src_arg].dtype.size
    if out_size != src_size or out_size <= 0:
        return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    all_instrs, data_plan, outputs = [], [], []
    F2I_OP = _VPU_OPS["F2I"]
    I2F_OP = _VPU_OPS["I2F"]
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)
        base = tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": base, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        out_vmem = base + 1
        all_instrs += [
            _load(0, base),
            _vpu(1, 0, F2I_OP),
            _vpu(2, 1, I2F_OP),
            _store(out_vmem, 2),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_unary_transcendental_sxu_program(
        uops: list[UOp], uop_name: str, emit_vpu) -> dict | None:
    """Shared shape for EXP2/LOG2/SIN single-op renderers.

    Pattern: one instance of the named UOp on a float source, no other
    compute ops in the data path. Multi-tile kernels emit one VPU dispatch
    per tile; the TranscUnit handles per-lane walk, SXU stalls on vpu.isDone.
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get(uop_name, 0) < 1:
        return None
    _allowed = {uop_name, "CONST", "INDEX", "LOAD", "STORE", "PARAM", "SINK",
                "GROUP", "END", "RANGE", "VECTORIZE", "GEP", "MUL", "ADD", "CAST"}
    if any(c > 0 and n not in _allowed for n, c in op_counts.items()):
        return None
    for u in uops:
        if u.op in (Ops.MUL, Ops.ADD) and _has_load_src(u):
            return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    src_size = params[src_arg].dtype.size
    if out_size != src_size or out_size <= 0:
        return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    all_instrs, data_plan, outputs = [], [], []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)
        base = tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": base, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        out_vmem = base + 1
        all_instrs += [
            _load(0, base),
            emit_vpu(1, 0),
            _store(out_vmem, 1),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_exp2_sxu_program(uops: list[UOp]) -> dict | None:
    return _render_unary_transcendental_sxu_program(uops, "EXP2", _vpu_exp2)


def _render_log2_sxu_program(uops: list[UOp]) -> dict | None:
    return _render_unary_transcendental_sxu_program(uops, "LOG2", _vpu_log2)


def _render_scaled_log2_sxu_program(uops: list[UOp]) -> dict | None:
    """Render k · log2(x) — Tensor.log() lowers as LOG2(x) · ln(2).

    Pattern: MUL(LOG2(x), const). Emits per tile:
      LOAD v0 = broadcast const
      LOAD v1 = x
      VPU  v2 = LOG2(v1)
      VPU  v3 = FMUL(v2, v0)
      STORE   = v3
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("LOG2", 0) < 1:
        return None
    if op_counts.get("EXP2", 0) > 0 or op_counts.get("SIN", 0) > 0 or op_counts.get("SQRT", 0) > 0:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None

    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None
    val = stores[0].src[1]
    while val.op in (Ops.VECTORIZE, Ops.CAST, Ops.GEP):
        val = val.src[0] if len(val.src) > 0 else val
    if val.op is not Ops.MUL:
        return None
    scale_const = next((s for s in val.src if s.op is Ops.CONST and not isinstance(s.arg, bool)), None)
    log2_node  = next((s for s in val.src if s is not scale_const), None)
    if scale_const is None or log2_node is None:
        return None
    while log2_node.op in (Ops.CAST, Ops.GEP):
        log2_node = log2_node.src[0]
    if log2_node.op is not Ops.LOG2:
        return None
    input_src = log2_node.src[0]
    while input_src.op in (Ops.CAST, Ops.GEP, Ops.VECTORIZE):
        input_src = input_src.src[0]
    # Require the LOG2 input to terminate at a LOAD. Compound expressions
    # (e.g. LOG2(ADD(x, 5)) for Tensor(x+5).log()) would otherwise be
    # silently rendered as log(x).
    if input_src.op is not Ops.LOAD:
        return None

    scale_bits = int(np.frombuffer(np.float32(float(scale_const.arg)).tobytes(), dtype=np.int32)[0])
    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan = [{"type": "VMEM", "addr": 0, "layout": "broadcast_const",
                  "value": scale_bits, "count": _TILE_ELEMS, "dtype": "int32"}]
    all_instrs = [_load(0, 0)]
    outputs = []
    FMUL_OP = _VPU_OPS["FMUL"]
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = 1 + tile_idx * 2
        out_vmem = 2 + tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(1, in_vmem),
            _vpu_log2(2, 1),
            _vpu(3, 2, FMUL_OP, 0),
            _store(out_vmem, 3),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_sin_sxu_program(uops: list[UOp]) -> dict | None:
    return _render_unary_transcendental_sxu_program(uops, "SIN", _vpu_sin)


def _render_scaled_sin_sxu_program(uops: list[UOp]) -> dict | None:
    """Render sin(s·x + t) — Tensor.cos() lowers to sin(-x + π/2).

    Matches SIN of either a MUL(x, const) or an ADD(const, MUL(x, const)).
    Emits per tile:
      LOAD v0 = broadcast scale
      LOAD v1 = broadcast shift (only if present)
      LOAD v2 = x
      VPU  v3 = FMUL(v2, v0)
      VPU  v4 = FADD(v3, v1)     # skipped when shift is 0
      VPU  v5 = SIN(v4 or v3)
      STORE   = v5
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("SIN", 0) < 1:
        return None
    if op_counts.get("EXP2", 0) > 0 or op_counts.get("LOG2", 0) > 0 or op_counts.get("SQRT", 0) > 0:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None

    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None
    val = stores[0].src[1]
    while val.op in (Ops.VECTORIZE, Ops.CAST, Ops.GEP):
        val = val.src[0] if len(val.src) > 0 else val
    if val.op is not Ops.SIN:
        return None
    inner = val.src[0]
    while inner.op in (Ops.CAST, Ops.GEP):
        inner = inner.src[0]

    # shape options:
    #   MUL(x, scale_const)
    #   ADD(shift_const, MUL(x, scale_const))
    #   ADD(shift_const, LOAD x)              (pure shift)
    shift_const = None
    if inner.op is Ops.ADD:
        shift_const = next((s for s in inner.src if s.op is Ops.CONST and not isinstance(s.arg, bool)), None)
        other = next((s for s in inner.src if s is not shift_const), None)
        if shift_const is None or other is None:
            return None
        while other.op in (Ops.CAST, Ops.GEP):
            other = other.src[0]
        inner = other

    scale_const = None
    if inner.op is Ops.MUL:
        scale_const = next((s for s in inner.src if s.op is Ops.CONST and not isinstance(s.arg, bool)), None)
        load_src = next((s for s in inner.src if s is not scale_const), None)
        if scale_const is None or load_src is None:
            return None
        # Require the non-const factor to terminate at a plain LOAD.
        tail = load_src
        while tail.op in (Ops.CAST, Ops.GEP, Ops.VECTORIZE):
            tail = tail.src[0]
        if tail.op is not Ops.LOAD:
            return None
    else:
        # no MUL — inner should be load-directly (pure-shift case)
        tail = inner
        while tail.op in (Ops.CAST, Ops.GEP, Ops.VECTORIZE):
            tail = tail.src[0]
        if tail.op is not Ops.LOAD:
            return None

    scale_bits = (int(np.frombuffer(np.float32(float(scale_const.arg)).tobytes(), dtype=np.int32)[0])
                  if scale_const is not None else None)
    shift_bits = (int(np.frombuffer(np.float32(float(shift_const.arg)).tobytes(), dtype=np.int32)[0])
                  if shift_const is not None else None)

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan = []
    next_const_addr = 0
    scale_addr = shift_addr = None
    if scale_bits is not None:
        scale_addr = next_const_addr
        data_plan.append({"type": "VMEM", "addr": scale_addr, "layout": "broadcast_const",
                          "value": scale_bits, "count": _TILE_ELEMS, "dtype": "int32"})
        next_const_addr += 1
    if shift_bits is not None:
        shift_addr = next_const_addr
        data_plan.append({"type": "VMEM", "addr": shift_addr, "layout": "broadcast_const",
                          "value": shift_bits, "count": _TILE_ELEMS, "dtype": "int32"})
        next_const_addr += 1

    all_instrs = []
    if scale_bits is not None:
        all_instrs.append(_load(0, scale_addr))
    if shift_bits is not None:
        all_instrs.append(_load(1, shift_addr))

    FMUL_OP, FADD_OP = _VPU_OPS["FMUL"], _VPU_OPS["FADD"]
    outputs = []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = next_const_addr + tile_idx * 2
        out_vmem = in_vmem + 1
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs.append(_load(2, in_vmem))
        cur = 2
        if scale_bits is not None:
            all_instrs.append(_vpu(3, cur, FMUL_OP, 0))
            cur = 3
        if shift_bits is not None:
            all_instrs.append(_vpu(4, cur, FADD_OP, 1))
            cur = 4
        all_instrs.append(_vpu_sin(5, cur))
        all_instrs.append(_store(out_vmem, 5))
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_scaled_exp2_sxu_program(uops: list[UOp]) -> dict | None:
    """Render exp2(x * k) or exp2(x + k) — Tensor.exp() / exp2-with-preamble.

    Pattern: one EXP2 UOp whose input flows through a single MUL or ADD
    with a CONST. Emits:
      LOAD v0 = scalar const
      LOAD v1 = x
      VPU  v2 = FMUL/FADD(v1, v0)
      VPU  v3 = EXP2(v2)
      STORE  v3
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("EXP2", 0) < 1:
        return None
    if op_counts.get("LOG2", 0) > 0 or op_counts.get("SIN", 0) > 0 or op_counts.get("SQRT", 0) > 0:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None

    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None
    val = stores[0].src[1]
    while val.op in (Ops.VECTORIZE, Ops.CAST, Ops.GEP):
        val = val.src[0] if len(val.src) > 0 else val
    if val.op is not Ops.EXP2:
        return None
    inner = val.src[0]
    while inner.op in (Ops.CAST, Ops.GEP):
        inner = inner.src[0]
    if inner.op not in (Ops.MUL, Ops.ADD):
        return None
    const_src = next((s for s in inner.src if s.op is Ops.CONST and not isinstance(s.arg, bool)), None)
    load_src = next((s for s in inner.src if s is not const_src), None)
    if const_src is None or load_src is None:
        return None
    # Tighten: the non-const operand must terminate at a plain LOAD.
    # Otherwise expressions like EXP2(MUL(x+k, c)) would silently drop
    # the inner ADD and render as EXP2(x*c).
    load_tail = load_src
    while load_tail.op in (Ops.CAST, Ops.GEP, Ops.VECTORIZE):
        load_tail = load_tail.src[0]
    if load_tail.op is not Ops.LOAD:
        return None
    const_bits = int(np.frombuffer(np.float32(float(const_src.arg)).tobytes(), dtype=np.int32)[0])
    mul_opcode = _VPU_OPS["FMUL" if inner.op is Ops.MUL else "FADD"]

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan = [{
        "type": "VMEM", "addr": 0, "layout": "broadcast_const",
        "value": const_bits, "count": _TILE_ELEMS, "dtype": "int32",
    }]
    all_instrs = [_load(0, 0)]
    outputs = []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = 1 + tile_idx * 2
        out_vmem = 2 + tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(1, in_vmem),
            _vpu(2, 1, mul_opcode, 0),
            _vpu_exp2(3, 2),
            _store(out_vmem, 3),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_tanh_sxu_program(uops: list[UOp]) -> dict | None:
    """Render tanh(x) = 2*sigmoid(2x) - 1.

    The tinygrad decomposition chains an outer +(-1) and *2 around the
    sigmoid. The simplest way to match the chain is to run sigmoid's
    FMUL+EXP2+FADD+FRECIP microprogram, then rescale: t = 2*s - 1.
    Full kernel per tile:
      LOAD v0 = broadcast (2 * -1/ln2) = -2.885390 (fold 2* inside exp arg)
      LOAD v1 = broadcast 1.0
      LOAD v2 = broadcast 2.0
      LOAD v3 = broadcast -1.0
      LOAD v4 = x
      VPU  v5 = FMUL(v4, v0)       # 2x*-1/ln2
      VPU  v6 = EXP2(v5)
      VPU  v7 = FADD(v6, v1)        # 1 + exp(-2x)
      VPU  v8 = FRECIP(v7)           # sigmoid(2x)
      VPU  v9 = FMUL(v8, v2)         # 2*sigmoid(2x)
      VPU  v10 = FADD(v9, v3)         # 2*sigmoid(2x) - 1 = tanh(x)
      STORE   = v10
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("RECIPROCAL", 0) < 1 or op_counts.get("EXP2", 0) < 1:
        return None
    if op_counts.get("LOG2", 0) > 0 or op_counts.get("SIN", 0) > 0 or op_counts.get("SQRT", 0) > 0:
        return None
    # tanh chain wraps the sigmoid shape with 2 extra muls + 1 extra add.
    # Distinguish from sigmoid: tanh has >1 ADD and >1 MUL in data path.
    adds = op_counts.get("ADD", 0)
    muls = op_counts.get("MUL", 0)
    if adds < 2 or muls < 2:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None

    # Walk the store value to confirm: ADD(MUL(CONST(2), RECIPROCAL(...)), CONST(-1))
    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None
    val = stores[0].src[1]
    while val.op in (Ops.VECTORIZE, Ops.CAST, Ops.GEP):
        val = val.src[0] if len(val.src) > 0 else val
    if val.op is not Ops.ADD:
        return None
    outer_const = next((s for s in val.src
                        if (s.op is Ops.CONST) or
                        (s.op in (Ops.CAST, Ops.GEP) and s.src and s.src[0].op is Ops.CONST)), None)
    inner = next((s for s in val.src if s is not outer_const), None)
    if outer_const is None or inner is None:
        return None
    while outer_const.op in (Ops.CAST, Ops.GEP):
        outer_const = outer_const.src[0]
    if float(outer_const.arg) != -1.0:
        return None
    while inner.op in (Ops.CAST, Ops.GEP):
        inner = inner.src[0]
    if inner.op is not Ops.MUL:
        return None
    scale_const = next((s for s in inner.src if s.op is Ops.CONST), None)
    sigmoid_node = next((s for s in inner.src if s is not scale_const), None)
    if scale_const is None or sigmoid_node is None or float(scale_const.arg) != 2.0:
        return None
    while sigmoid_node.op in (Ops.CAST, Ops.GEP):
        sigmoid_node = sigmoid_node.src[0]
    if sigmoid_node.op is not Ops.RECIPROCAL:
        return None
    add_in = sigmoid_node.src[0]
    while add_in.op in (Ops.CAST, Ops.GEP):
        add_in = add_in.src[0]
    if add_in.op is not Ops.ADD:
        return None
    one_c = next((s for s in add_in.src
                  if (s.op is Ops.CONST) or
                  (s.op in (Ops.CAST, Ops.GEP) and s.src and s.src[0].op is Ops.CONST)), None)
    exp_src = next((s for s in add_in.src if s is not one_c), None)
    if one_c is None or exp_src is None:
        return None
    while one_c.op in (Ops.CAST, Ops.GEP):
        one_c = one_c.src[0]
    if float(one_c.arg) != 1.0:
        return None
    while exp_src.op in (Ops.CAST, Ops.GEP):
        exp_src = exp_src.src[0]
    if exp_src.op is not Ops.EXP2:
        return None
    exp_in = exp_src.src[0]
    while exp_in.op in (Ops.CAST, Ops.GEP):
        exp_in = exp_in.src[0]
    if exp_in.op is not Ops.MUL:
        return None
    inner_scale = next((s for s in exp_in.src if s.op is Ops.CONST), None)
    input_src   = next((s for s in exp_in.src if s is not inner_scale), None)
    if inner_scale is None or input_src is None or not _has_load_src(input_src):
        return None

    # tinygrad's tanh decomposition pre-scales by (2·-1/ln2) before EXP2,
    # so inner_scale.arg already holds the full -2.885 multiplier. Use
    # it directly — the earlier "* 2" fold was double-scaling.
    combined_scale = float(inner_scale.arg)
    scale_bits   = int(np.frombuffer(np.float32(combined_scale).tobytes(), dtype=np.int32)[0])
    one_bits     = int(np.frombuffer(np.float32(1.0).tobytes(),  dtype=np.int32)[0])
    two_bits     = int(np.frombuffer(np.float32(2.0).tobytes(),  dtype=np.int32)[0])
    negone_bits  = int(np.frombuffer(np.float32(-1.0).tobytes(), dtype=np.int32)[0])

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan = [
        {"type": "VMEM", "addr": 0, "layout": "broadcast_const", "value": scale_bits,
         "count": _TILE_ELEMS, "dtype": "int32"},
        {"type": "VMEM", "addr": 1, "layout": "broadcast_const", "value": one_bits,
         "count": _TILE_ELEMS, "dtype": "int32"},
        {"type": "VMEM", "addr": 2, "layout": "broadcast_const", "value": two_bits,
         "count": _TILE_ELEMS, "dtype": "int32"},
        {"type": "VMEM", "addr": 3, "layout": "broadcast_const", "value": negone_bits,
         "count": _TILE_ELEMS, "dtype": "int32"},
    ]
    all_instrs = [_load(0, 0), _load(1, 1), _load(2, 2), _load(3, 3)]
    FMUL_OP   = _VPU_OPS["FMUL"]
    FADD_OP   = _VPU_OPS["FADD"]
    FRECIP_OP = _VPU_OPS["FRECIP"]
    outputs = []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = 4 + tile_idx * 2
        out_vmem = 5 + tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(4, in_vmem),
            _vpu(5, 4, FMUL_OP, 0),
            _vpu_exp2(6, 5),
            _vpu(7, 6, FADD_OP, 1),
            _vpu(8, 7, FRECIP_OP),
            _vpu(9, 8, FMUL_OP, 2),
            _vpu(10, 9, FADD_OP, 3),
            _store(out_vmem, 10),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_softsign_sxu_program(uops: list[UOp]) -> dict | None:
    """Render softsign(x) = x / (1 + |x|).

    Pattern signature:
      * single src PARAM
      * no transcendentals (EXP2/LOG2/SIN/SQRT)
      * WHERE+CMPLT+CMPNE+MUL from |x| decomposition
      * RECIPROCAL, data-path ADD present
      * one outer MUL combining LOAD(x) and RECIPROCAL(...)

    Per tile emits:
      LOAD v0 = x
      LOAD v1 = zeros
      LOAD v2 = broadcast 1.0
      FSUB v3 = v1 - v0   (= -x)
      FMAX v4 = v0 max v3 (= |x|)
      FADD v5 = v4 + v2   (= 1 + |x|)
      FRECIP v6 = 1 / v5
      FMUL v7 = v0 * v6   (= softsign)
      STORE v7
    """
    op_counts = Counter(u.op.name for u in uops)
    if any(op_counts.get(n, 0) for n in ("EXP2", "LOG2", "SIN", "SQRT")):
        return None
    if op_counts.get("RECIPROCAL", 0) < 1:
        return None
    if op_counts.get("WHERE", 0) < 1:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None
    # Verify the STORE is MUL(LOAD(x), RECIPROCAL(ADD(1, abs-shape))) on the
    # shared PARAM.
    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None
    val = stores[0].src[1]
    while val.op in (Ops.VECTORIZE, Ops.CAST, Ops.GEP):
        val = val.src[0] if len(val.src) > 0 else val
    if val.op is not Ops.MUL:
        return None
    a, b = val.src[0], val.src[1]
    recip = next((s for s in (a, b) if _chase(s).op is Ops.RECIPROCAL), None)
    direct_x = next((s for s in (a, b) if s is not recip), None)
    if recip is None or direct_x is None or not _has_load_src(direct_x):
        return None
    recip = _chase(recip)
    add_node = recip.src[0]
    while add_node.op in (Ops.CAST, Ops.GEP):
        add_node = add_node.src[0]
    if add_node.op is not Ops.ADD:
        return None
    one_src = next((s for s in add_node.src
                    if s.op is Ops.CONST or
                    (s.op in (Ops.CAST, Ops.VECTORIZE, Ops.GEP) and
                     s.src and s.src[0].op is Ops.CONST)), None)
    if one_src is None:
        return None
    cst = one_src
    while cst.op in (Ops.CAST, Ops.VECTORIZE, Ops.GEP):
        cst = cst.src[0]
    if cst.op is not Ops.CONST or float(cst.arg) != 1.0:
        return None

    one_bits = int(np.frombuffer(np.float32(1.0).tobytes(), dtype=np.int32)[0])
    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan = [{"type": "VMEM", "addr": 0, "layout": "broadcast_const",
                  "value": 0, "count": _TILE_ELEMS, "dtype": "int32"},
                 {"type": "VMEM", "addr": 1, "layout": "broadcast_const",
                  "value": one_bits, "count": _TILE_ELEMS, "dtype": "int32"}]
    all_instrs = [_load(0, 0), _load(1, 1)]
    FSUB, FMAX, FADD = _VPU_OPS["FSUB"], _VPU_OPS["FMAX"], _VPU_OPS["FADD"]
    FRECIP, FMUL = _VPU_OPS["FRECIP"], _VPU_OPS["FMUL"]
    outputs = []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = 2 + tile_idx * 2
        out_vmem = 3 + tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(2, in_vmem),               # v2 = x
            _vpu(3, 0, FSUB, 2),             # v3 = 0 - x = -x
            _vpu(4, 2, FMAX, 3),             # v4 = max(x, -x) = |x|
            _vpu(5, 4, FADD, 1),             # v5 = |x| + 1
            _vpu(6, 5, FRECIP),              # v6 = 1/(1+|x|)
            _vpu(7, 2, FMUL, 6),             # v7 = x * v6
            _store(out_vmem, 7),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_leaky_relu_sxu_program(uops: list[UOp]) -> dict | None:
    """Render leaky_relu(x, alpha) = max(alpha*x, x) for alpha in (0, 1).

    tinygrad lowers `(x < 0).where(x*alpha, x)`. For 0 < alpha < 1 this
    equals `max(alpha*x, x)`. Per tile:
      LOAD v0 = x
      LOAD v1 = broadcast alpha
      FMUL v2 = alpha*x
      FMAX v3 = max(v2, v0)
      STORE v3
    """
    op_counts = Counter(u.op.name for u in uops)
    if any(op_counts.get(n, 0) for n in ("EXP2", "LOG2", "SIN", "SQRT", "RECIPROCAL")):
        return None
    if op_counts.get("WHERE", 0) < 1 or op_counts.get("CMPLT", 0) < 1:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None
    # Find the scalar slope: a data-path MUL with a float CONST.
    alpha = None
    for u in uops:
        if u.op is Ops.MUL and _has_load_src(u):
            cst = next((s for s in u.src if s.op is Ops.CONST and isinstance(s.arg, float)), None)
            if cst is not None:
                if alpha is None:
                    alpha = float(cst.arg)
                elif alpha != float(cst.arg):
                    return None
    if alpha is None or not (0.0 < alpha < 1.0):
        return None
    # Reject kernels with more than one CONST used as WHERE lhs/rhs (that
    # would indicate clip-shape, not leaky_relu).
    where_consts = []
    for w in uops:
        if w.op is not Ops.WHERE: continue
        for s in (w.src[1], w.src[2]):
            if s.op is Ops.CONST and isinstance(s.arg, float):
                where_consts.append(s.arg)
    # leaky_relu's WHERE has no CONST bodies (alpha*x and x are both
    # expressions). If we see CONSTs in WHERE bodies, bail.
    if where_consts:
        return None

    alpha_bits = int(np.frombuffer(np.float32(alpha).tobytes(), dtype=np.int32)[0])
    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan = [{"type": "VMEM", "addr": 0, "layout": "broadcast_const",
                  "value": alpha_bits, "count": _TILE_ELEMS, "dtype": "int32"}]
    all_instrs = [_load(0, 0)]
    FMUL_OP, FMAX_OP = _VPU_OPS["FMUL"], _VPU_OPS["FMAX"]
    outputs = []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = 1 + tile_idx * 2
        out_vmem = 2 + tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(1, in_vmem),
            _vpu(2, 1, FMUL_OP, 0),       # v2 = alpha * x
            _vpu(3, 2, FMAX_OP, 1),       # v3 = max(v2, x)
            _store(out_vmem, 3),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_clamp_single_bound_sxu_program(uops: list[UOp]) -> dict | None:
    """Render Tensor.clamp(min_=c) = max(x, c) / clamp(max_=c) = min(x, c).

    Single-bound clamp decomposes to WHERE(CMPLT(x, c), c, x) or
    WHERE(CMPLT(c, x), c, x). One WHERE, one CMPLT, one float CONST,
    one PARAM, all float. Emits per tile:
      LOAD v0 = x
      LOAD v1 = broadcast c
      FMAX/FMIN v2 = max-or-min(v0, v1)
      STORE v2
    """
    op_counts = Counter(u.op.name for u in uops)
    if any(op_counts.get(n, 0) for n in ("EXP2", "LOG2", "SIN", "SQRT", "RECIPROCAL")):
        return None
    if op_counts.get("WHERE", 0) < 1 or op_counts.get("CMPLT", 0) < 1:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None
    # Require exactly one data-path float CONST (the bound). No data-path MULs.
    data_alu = _data_alu_ops(uops)
    if data_alu.get("MUL", 0) > 0:
        return None
    # Extract bound + detect which side of WHERE it's on.
    where_uop = next((u for u in uops if u.op is Ops.WHERE), None)
    if where_uop is None:
        return None
    cond = where_uop.src[0]
    while cond.op in (Ops.CAST, Ops.GEP, Ops.VECTORIZE):
        cond = cond.src[0]
    if cond.op is not Ops.CMPLT:
        return None
    lhs_body = where_uop.src[1]
    rhs_body = where_uop.src[2]
    def _is_float_const(u):
        while u.op in (Ops.CAST, Ops.GEP, Ops.VECTORIZE):
            u = u.src[0]
        return u.op is Ops.CONST and not isinstance(u.arg, bool) and isinstance(u.arg, float)
    # Exactly one of the WHERE bodies must be a float CONST, the other
    # must be the raw LOAD. 0.0 is handled by the RELU path.
    if _is_float_const(lhs_body) == _is_float_const(rhs_body):
        return None
    bound_side = lhs_body if _is_float_const(lhs_body) else rhs_body
    raw_side   = rhs_body if _is_float_const(lhs_body) else lhs_body
    while bound_side.op in (Ops.CAST, Ops.GEP, Ops.VECTORIZE):
        bound_side = bound_side.src[0]
    if bound_side.op is not Ops.CONST:
        return None
    bound = float(bound_side.arg)
    if bound == 0.0:
        # RELU handles this in the elementwise renderer.
        return None
    # Raw side must terminate at a LOAD.
    tail = raw_side
    while tail.op in (Ops.CAST, Ops.GEP, Ops.VECTORIZE):
        tail = tail.src[0]
    if tail.op is not Ops.LOAD:
        return None
    # Decide FMAX vs FMIN: CMPLT(x, bound) with WHERE selecting bound on
    # true means clamp-min (x < bound → replace with bound → max(x, bound)).
    # CMPLT(bound, x) with WHERE selecting bound on true means clamp-max
    # (bound < x → replace with bound → min(x, bound)).
    cmplt_a = cond.src[0]
    while cmplt_a.op in (Ops.CAST, Ops.GEP, Ops.VECTORIZE):
        cmplt_a = cmplt_a.src[0]
    # When CMPLT's first operand is the LOAD (x), it's clamp-min → FMAX.
    use_fmax = cmplt_a.op is Ops.LOAD
    # If WHERE's lhs_body is the CONST, the "true" branch picks the bound;
    # the Python decomposition `(x < c).where(c, x)` has lhs_body=bound.
    if not _is_float_const(lhs_body):
        # WHERE selects bound on FALSE — opposite of the decomposition above.
        use_fmax = not use_fmax

    bound_bits = int(np.frombuffer(np.float32(bound).tobytes(), dtype=np.int32)[0])
    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan = [{"type": "VMEM", "addr": 0, "layout": "broadcast_const",
                  "value": bound_bits, "count": _TILE_ELEMS, "dtype": "int32"}]
    all_instrs = [_load(0, 0)]
    op = _VPU_OPS["FMAX" if use_fmax else "FMIN"]
    outputs = []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = 1 + tile_idx * 2
        out_vmem = 2 + tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(1, in_vmem),
            _vpu(2, 1, op, 0),
            _store(out_vmem, 2),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_clip_sxu_program(uops: list[UOp]) -> dict | None:
    """Render Tensor.clip(a, b) = max(lo, min(hi, x)) for float tensors.

    tinygrad decomposes clip as `(x < lo).where(lo, x)` then
    `(ret > hi).where(hi, ret)`. The kernel has 2 WHEREs and 2 CMPLTs
    per output element plus two float CONSTs (the bounds). Emits per
    tile:
      LOAD v0 = x
      LOAD v1 = broadcast lo
      LOAD v2 = broadcast hi
      FMIN v3 = min(x, hi)
      FMAX v4 = max(v3, lo)
      STORE v4
    """
    op_counts = Counter(u.op.name for u in uops)
    if any(op_counts.get(n, 0) for n in ("EXP2", "LOG2", "SIN", "SQRT", "RECIPROCAL")):
        return None
    if op_counts.get("WHERE", 0) < 2 or op_counts.get("CMPLT", 0) < 2:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None
    # Require exactly 2 data-path CONSTs used in WHERE bodies (the bounds).
    where_uops = [u for u in uops if u.op is Ops.WHERE]
    bound_consts: list[float] = []
    for w in where_uops:
        for s in (w.src[1], w.src[2]):
            if s.op is Ops.CONST and isinstance(s.arg, float):
                bound_consts.append(float(s.arg))
    unique_bounds = sorted(set(bound_consts))
    if len(unique_bounds) != 2:
        return None
    lo, hi = unique_bounds[0], unique_bounds[1]

    lo_bits = int(np.frombuffer(np.float32(lo).tobytes(), dtype=np.int32)[0])
    hi_bits = int(np.frombuffer(np.float32(hi).tobytes(), dtype=np.int32)[0])

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan = [
        {"type": "VMEM", "addr": 0, "layout": "broadcast_const",
         "value": lo_bits, "count": _TILE_ELEMS, "dtype": "int32"},
        {"type": "VMEM", "addr": 1, "layout": "broadcast_const",
         "value": hi_bits, "count": _TILE_ELEMS, "dtype": "int32"},
    ]
    all_instrs = [_load(0, 0), _load(1, 1)]
    FMIN_OP, FMAX_OP = _VPU_OPS["FMIN"], _VPU_OPS["FMAX"]
    outputs = []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = 2 + tile_idx * 2
        out_vmem = 3 + tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(2, in_vmem),              # v2 = x
            _vpu(3, 2, FMIN_OP, 1),         # v3 = min(x, hi)
            _vpu(4, 3, FMAX_OP, 0),         # v4 = max(v3, lo)
            _store(out_vmem, 4),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_swish_sxu_program(uops: list[UOp]) -> dict | None:
    """Render swish(x) / silu(x) = x * sigmoid(x).

    Pattern: MUL(LOAD(x), RECIPROCAL(ADD(CONST(1.0), EXP2(MUL(LOAD(x), CONST(-1/ln2))))))
    where both LOAD(x) references share the same PARAM.

    Per tile emits:
      LOAD v0 = broadcast -1/ln2
      LOAD v1 = broadcast 1.0
      LOAD v2 = x
      FMUL v3 = v2 * v0  (x * -1/ln2)
      EXP2 v4 = exp2(v3)
      FADD v5 = v4 + v1  (1 + exp2)
      FRECIP v6 = 1/v5    (sigmoid)
      FMUL v7 = v2 * v6   (x * sigmoid)
      STORE v7
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("RECIPROCAL", 0) < 1 or op_counts.get("EXP2", 0) < 1:
        return None
    if op_counts.get("LOG2", 0) > 0 or op_counts.get("SIN", 0) > 0 or op_counts.get("SQRT", 0) > 0:
        return None
    if op_counts.get("MUL", 0) < 2:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None

    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None
    val = stores[0].src[1]
    while val.op in (Ops.VECTORIZE, Ops.CAST, Ops.GEP):
        val = val.src[0] if len(val.src) > 0 else val
    # Outer MUL: one src is the input x (as a LOAD-tail), other is RECIPROCAL.
    if val.op is not Ops.MUL:
        return None
    a, b = val.src[0], val.src[1]
    recip = next((s for s in (a, b) if _chase(s).op is Ops.RECIPROCAL), None)
    direct_x = next((s for s in (a, b) if s is not recip), None)
    if recip is None or direct_x is None:
        return None
    recip = _chase(recip)
    if not _has_load_src(direct_x):
        return None
    # Dig into RECIPROCAL -> ADD(1, EXP2(MUL(x, c)))
    add_node = recip.src[0]
    while add_node.op in (Ops.CAST, Ops.GEP):
        add_node = add_node.src[0]
    if add_node.op is not Ops.ADD:
        return None
    one_src = next((s for s in add_node.src
                    if s.op is Ops.CONST or
                    (s.op in (Ops.CAST, Ops.VECTORIZE, Ops.GEP) and
                     s.src and s.src[0].op is Ops.CONST)), None)
    exp_src = next((s for s in add_node.src if s is not one_src), None)
    if one_src is None or exp_src is None:
        return None
    cst = one_src
    while cst.op in (Ops.CAST, Ops.VECTORIZE, Ops.GEP):
        cst = cst.src[0]
    if cst.op is not Ops.CONST or float(cst.arg) != 1.0:
        return None
    while exp_src.op in (Ops.CAST, Ops.GEP):
        exp_src = exp_src.src[0]
    if exp_src.op is not Ops.EXP2:
        return None
    mul_node = exp_src.src[0]
    while mul_node.op in (Ops.CAST, Ops.GEP):
        mul_node = mul_node.src[0]
    if mul_node.op is not Ops.MUL:
        return None
    scale_const = next((s for s in mul_node.src if s.op is Ops.CONST), None)
    input_src   = next((s for s in mul_node.src if s is not scale_const), None)
    if scale_const is None or input_src is None or not _has_load_src(input_src):
        return None

    scale_bits = int(np.frombuffer(np.float32(float(scale_const.arg)).tobytes(), dtype=np.int32)[0])
    one_bits   = int(np.frombuffer(np.float32(1.0).tobytes(), dtype=np.int32)[0])

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan = [
        {"type": "VMEM", "addr": 0, "layout": "broadcast_const",
         "value": scale_bits, "count": _TILE_ELEMS, "dtype": "int32"},
        {"type": "VMEM", "addr": 1, "layout": "broadcast_const",
         "value": one_bits,   "count": _TILE_ELEMS, "dtype": "int32"},
    ]
    all_instrs = [_load(0, 0), _load(1, 1)]
    FMUL_OP  = _VPU_OPS["FMUL"]
    FADD_OP  = _VPU_OPS["FADD"]
    FRECIP_OP = _VPU_OPS["FRECIP"]
    outputs = []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = 2 + tile_idx * 2
        out_vmem = 3 + tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(2, in_vmem),              # v2 = x
            _vpu(3, 2, FMUL_OP, 0),         # v3 = x * -1/ln2
            _vpu_exp2(4, 3),                # v4 = exp2(x * -1/ln2)
            _vpu(5, 4, FADD_OP, 1),         # v5 = 1 + exp2
            _vpu(6, 5, FRECIP_OP),          # v6 = sigmoid(x)
            _vpu(7, 2, FMUL_OP, 6),         # v7 = x * sigmoid(x)
            _store(out_vmem, 7),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_sigmoid_sxu_program(uops: list[UOp]) -> dict | None:
    """Render sigmoid(x) = reciprocal(1 + exp2(x * -1/ln2)).

    Pattern: one RECIPROCAL whose input is an ADD of CONST(1.0) and an
    EXP2 of a scalar-const-scaled input. Emits per tile:
      LOAD v0 = broadcast -1/ln2
      LOAD v1 = broadcast 1.0
      LOAD v2 = x
      VPU  v3 = FMUL(v2, v0)
      VPU  v4 = EXP2(v3)
      VPU  v5 = FADD(v4, v1)
      VPU  v6 = FRECIP(v5)
      STORE   = v6
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("RECIPROCAL", 0) < 1 or op_counts.get("EXP2", 0) < 1:
        return None
    if op_counts.get("LOG2", 0) > 0 or op_counts.get("SIN", 0) > 0 or op_counts.get("SQRT", 0) > 0:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None

    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None
    val = stores[0].src[1]
    while val.op in (Ops.VECTORIZE, Ops.CAST, Ops.GEP):
        val = val.src[0] if len(val.src) > 0 else val
    if val.op is not Ops.RECIPROCAL:
        return None
    add_node = val.src[0]
    while add_node.op in (Ops.CAST, Ops.GEP):
        add_node = add_node.src[0]
    if add_node.op is not Ops.ADD:
        return None
    const_src = next((s for s in add_node.src
                      if s.op is Ops.CONST or
                      (s.op in (Ops.CAST, Ops.VECTORIZE, Ops.GEP) and
                       s.src and s.src[0].op is Ops.CONST)), None)
    exp_src = next((s for s in add_node.src if s is not const_src), None)
    if const_src is None or exp_src is None:
        return None
    # Unwrap to find CONST
    cst = const_src
    while cst.op in (Ops.CAST, Ops.VECTORIZE, Ops.GEP):
        cst = cst.src[0]
    if cst.op is not Ops.CONST or float(cst.arg) != 1.0:
        return None
    while exp_src.op in (Ops.CAST, Ops.GEP):
        exp_src = exp_src.src[0]
    if exp_src.op is not Ops.EXP2:
        return None
    mul_node = exp_src.src[0]
    while mul_node.op in (Ops.CAST, Ops.GEP):
        mul_node = mul_node.src[0]
    if mul_node.op is not Ops.MUL:
        return None
    scale_const = next((s for s in mul_node.src if s.op is Ops.CONST), None)
    input_src   = next((s for s in mul_node.src if s is not scale_const), None)
    if scale_const is None or input_src is None or not _has_load_src(input_src):
        return None

    scale_bits = int(np.frombuffer(np.float32(float(scale_const.arg)).tobytes(), dtype=np.int32)[0])
    one_bits   = int(np.frombuffer(np.float32(1.0).tobytes(), dtype=np.int32)[0])

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan = [
        {"type": "VMEM", "addr": 0, "layout": "broadcast_const",
         "value": scale_bits, "count": _TILE_ELEMS, "dtype": "int32"},
        {"type": "VMEM", "addr": 1, "layout": "broadcast_const",
         "value": one_bits,   "count": _TILE_ELEMS, "dtype": "int32"},
    ]
    all_instrs = [_load(0, 0), _load(1, 1)]
    FMUL_OP  = _VPU_OPS["FMUL"]
    FADD_OP  = _VPU_OPS["FADD"]
    FRECIP_OP = _VPU_OPS["FRECIP"]
    outputs = []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = 2 + tile_idx * 2
        out_vmem = 3 + tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(2, in_vmem),
            _vpu(3, 2, FMUL_OP, 0),
            _vpu_exp2(4, 3),
            _vpu(5, 4, FADD_OP, 1),
            _vpu(6, 5, FRECIP_OP),
            _store(out_vmem, 6),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_self_cube_sxu_program(uops: list[UOp]) -> dict | None:
    """Render x · x · x (= x**3).

    tinygrad lowers Tensor(x)**3 as MUL(LOAD(x), MUL(LOAD(x), LOAD(x)))
    — three LOAD nodes but all into the same PARAM. Detect the three-
    way self-MUL tree and emit a two-step FMUL sequence per tile.
    """
    op_counts = Counter(u.op.name for u in uops)
    allowed = {"MUL", "CONST", "INDEX", "LOAD", "STORE", "PARAM", "SINK",
               "END", "RANGE", "VECTORIZE", "GEP", "CAST"}
    if any(c > 0 and n not in allowed for n, c in op_counts.items()):
        return None
    if op_counts.get("MUL", 0) < 2:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None

    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None
    val = stores[0].src[1]
    while val.op in (Ops.VECTORIZE, Ops.CAST, Ops.GEP):
        val = val.src[0] if len(val.src) > 0 else val
    if val.op is not Ops.MUL:
        return None
    # Outer MUL: one src is a MUL (inner square), the other a LOAD(x).
    a, b = val.src[0], val.src[1]
    inner = next((s for s in (a, b) if _chase(s).op is Ops.MUL), None)
    outer_load = next((s for s in (a, b) if _chase(s).op is not Ops.MUL), None)
    if inner is None or outer_load is None:
        return None
    inner_chased = _chase(inner)
    if inner_chased.op is not Ops.MUL:
        return None
    # Inner MUL must be self-square on the same PARAM.
    if inner_chased.src[0] is not inner_chased.src[1]:
        return None
    # Verify all three LOAD tails point to the same PARAM.
    if not _has_load_src(outer_load) or not _has_load_src(inner_chased.src[0]):
        return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    FMUL_OP = _VPU_OPS["FMUL"]
    all_instrs, data_plan, outputs = [], [], []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = tile_idx * 2
        out_vmem = in_vmem + 1
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(0, in_vmem),              # v0 := x
            _vpu(1, 0, FMUL_OP, 0),         # v1 := x * x
            _vpu(2, 0, FMUL_OP, 1),         # v2 := x * v1 = x^3
            _store(out_vmem, 2),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _chase(u: "UOp") -> "UOp":
    """Follow VECTORIZE/CAST/GEP wrappers down to the interesting node."""
    while u.op in (Ops.VECTORIZE, Ops.CAST, Ops.GEP):
        if len(u.src) == 0:
            return u
        u = u.src[0]
    return u


def _render_self_square_sxu_program(uops: list[UOp]) -> dict | None:
    """Render x · x (= x**2, Tensor.square).

    tinygrad's elementwise renderer expects two PARAMs for a binary op;
    the self-multiply pattern has a single PARAM fed into both MUL
    operands, so elementwise rejects it. This renderer detects the
    STORE(..., MUL(INDEX(p), INDEX(p))) shape and emits
    LOAD x → FMUL(v, v) → STORE per tile.
    """
    op_counts = Counter(u.op.name for u in uops)
    allowed = {"MUL", "CONST", "INDEX", "LOAD", "STORE", "PARAM", "SINK",
               "END", "RANGE", "VECTORIZE", "GEP", "CAST"}
    if any(c > 0 and n not in allowed for n, c in op_counts.items()):
        return None
    if op_counts.get("MUL", 0) < 1:
        return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None

    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None
    val = stores[0].src[1]
    while val.op in (Ops.VECTORIZE, Ops.CAST, Ops.GEP):
        val = val.src[0] if len(val.src) > 0 else val
    if val.op is not Ops.MUL:
        return None
    s_left, s_right = val.src[0], val.src[1]
    if s_left is not s_right:
        return None
    # Must chase through CAST/GEP and terminate at a LOAD on src_arg — if
    # the left factor is itself a MUL we are looking at x**4 or higher,
    # which needs a different renderer.
    cur = s_left
    while cur.op in (Ops.CAST, Ops.GEP, Ops.VECTORIZE):
        cur = cur.src[0]
    if cur.op is not Ops.LOAD:
        return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    FMUL_OP = _VPU_OPS["FMUL"]
    all_instrs, data_plan, outputs = [], [], []
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = tile_idx * 2
        out_vmem = in_vmem + 1
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(0, in_vmem),
            _vpu(1, 0, FMUL_OP, 0),
            _store(out_vmem, 1),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_rsqrt_sxu_program(uops: list[UOp]) -> dict | None:
    """Render RECIPROCAL(SQRT(x)) as Exp2(-0.5 * Log2(x)).

    Tensor.rsqrt() lowers as RECIPROCAL(SQRT(x)); a direct microprogram
    skips one extra division by rolling the sign of the log2 scale.
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("SQRT", 0) < 1 or op_counts.get("RECIPROCAL", 0) < 1:
        return None
    _allowed = {"SQRT", "RECIPROCAL", "CONST", "INDEX", "LOAD", "STORE",
                "PARAM", "SINK", "GROUP", "END", "RANGE", "VECTORIZE",
                "GEP", "MUL", "ADD", "CAST"}
    if any(c > 0 and n not in _allowed for n, c in op_counts.items()):
        return None
    for u in uops:
        if u.op in (Ops.MUL, Ops.ADD) and _has_load_src(u):
            return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    if out_size != params[src_arg].dtype.size or out_size <= 0:
        return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    neg_half_bits = int(np.frombuffer(np.float32(-0.5).tobytes(), dtype=np.int32)[0])
    data_plan = [{
        "type": "VMEM", "addr": 0, "layout": "broadcast_const",
        "value": neg_half_bits, "count": _TILE_ELEMS, "dtype": "int32",
    }]
    all_instrs = [_load(0, 0)]
    outputs = []
    FMUL_OP = _VPU_OPS["FMUL"]
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = 1 + tile_idx * 2
        out_vmem = 2 + tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(1, in_vmem),                  # v1 := x
            _vpu_log2(2, 1),                    # v2 := log2(x)
            _vpu(3, 2, FMUL_OP, 0),             # v3 := -0.5 * log2(x)
            _vpu_exp2(4, 3),                    # v4 := exp2(v3) = 1/sqrt(x)
            _store(out_vmem, 4),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_sqrt_sxu_program(uops: list[UOp]) -> dict | None:
    """Render Ops.SQRT as Exp2(0.5 * Log2(x)) using LOG2+FMUL+EXP2.

    SQRT isn't a direct VPU opcode; this microprogram decomposes it into
    the existing transcendentals. Accuracy compounds the degree-2/5
    Taylor errors in LOG2 and EXP2 — exact at powers of two.
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("SQRT", 0) < 1:
        return None
    _allowed = {"SQRT", "CONST", "INDEX", "LOAD", "STORE", "PARAM", "SINK",
                "GROUP", "END", "RANGE", "VECTORIZE", "GEP", "MUL", "ADD", "CAST"}
    if any(c > 0 and n not in _allowed for n, c in op_counts.items()):
        return None
    for u in uops:
        if u.op in (Ops.MUL, Ops.ADD) and _has_load_src(u):
            return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    src_size = params[src_arg].dtype.size
    if out_size != src_size or out_size <= 0:
        return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    half_bits = int(np.frombuffer(np.float32(0.5).tobytes(), dtype=np.int32)[0])
    # VMEM layout: addr 0 = broadcast-0.5 const tile, then per-tile slots
    # (2 per tile: input, output) starting at addr 1.
    data_plan = [{
        "type": "VMEM", "addr": 0, "layout": "broadcast_const",
        "value": half_bits, "count": _TILE_ELEMS, "dtype": "int32",
    }]
    all_instrs = [_load(0, 0)]   # v0 := [0.5, 0.5, …]
    outputs = []
    FMUL_OP = _VPU_OPS["FMUL"]
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count  = min(_TILE_ELEMS, out_size - offset)
        in_vmem  = 1 + tile_idx * 2
        out_vmem = 2 + tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": in_vmem, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        all_instrs += [
            _load(1, in_vmem),                  # v1 := x
            _vpu_log2(2, 1),                    # v2 := log2(x)
            _vpu(3, 2, FMUL_OP, 0),             # v3 := 0.5 * log2(x)
            _vpu_exp2(4, 3),                    # v4 := exp2(v3) = sqrt(x)
            _store(out_vmem, 4),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_reciprocal_sxu_program(uops: list[UOp]) -> dict | None:
    """Render plain float32 reciprocal (1/x) as SXU_PROGRAM using VPU_FRECIP.

    Replaces the legacy HOST_UNARY RECIPROCAL path. Pattern: single RECIPROCAL UOp
    on a float source, no other compute ops in the data path.
    """
    op_counts = Counter(u.op.name for u in uops)
    if op_counts.get("RECIPROCAL", 0) < 1:
        return None
    # Reject kernels that mix RECIPROCAL with transcendentals (tanh,
    # sigmoid, scaled-exp2) — those belong to their dedicated composite
    # renderers, not the plain 1/x path.
    if any(op_counts.get(n, 0) for n in ("EXP2", "LOG2", "SIN", "SQRT")):
        return None
    _allowed = {"RECIPROCAL", "CONST", "INDEX", "LOAD", "STORE", "PARAM", "SINK",
                "GROUP", "END", "RANGE", "VECTORIZE", "GEP", "MUL", "ADD", "CAST"}
    if any(c > 0 and n not in _allowed for n, c in op_counts.items()):
        return None
    for u in uops:
        if u.op in (Ops.MUL, Ops.ADD) and _has_load_src(u):
            return None
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or 0 not in params:
        return None
    src_arg = next((k for k in params if k != 0), None)
    if src_arg is None:
        return None
    if not ("float" in str(params[0].dtype) and "float" in str(params[src_arg].dtype)):
        return None
    out_size = params[0].dtype.size
    src_size = params[src_arg].dtype.size
    if out_size != src_size or out_size <= 0:
        return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    all_instrs, data_plan, outputs = [], [], []
    FRECIP_OP = _VPU_OPS["FRECIP"]
    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)
        base = tile_idx * 2
        data_plan.append({"type": "VMEM", "addr": base, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        out_vmem = base + 1
        all_instrs += [
            _load(0, base),
            _vpu(1, 0, FRECIP_OP),
            _store(out_vmem, 1),
        ]
        outputs.append({"addr": out_vmem, "param": 0, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": 0}


def _render_scalar_const_divmod_sxu_program(uops: list[UOp]) -> dict | None:
    """Render scalar-const int32 IDIV/MOD as SXU_PROGRAM.

    Replaces the legacy VPU_BINARY (IDIV) and VPU_PROGRAM (MOD) paths:
    - IDIV: broadcast divisor, dispatch VPU_DIV.
    - MOD: DIV, MUL, SUB sequence (x - (x//c)*c).
    """
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if 0 not in params:
        return None
    pattern = _classify_divmod_pattern(uops)
    if pattern is None or pattern[0] not in ("IDIV", "MOD"):
        return None
    kind, rhs_const = pattern

    out_arg = 0
    out_size = params[out_arg].dtype.size
    src_args = [a for a in sorted(params) if a != out_arg]
    if out_size <= 0 or len(src_args) not in (1, 2):
        return None
    # Float operands are not supported — IDIV/MOD are integer opcodes.
    if any("float" in str(params[a].dtype) for a in [out_arg] + src_args):
        return None
    # Scalar-const case needs rhs_const; tensor-tensor case needs two size-matching src params.
    if len(src_args) == 1 and rhs_const is None:
        return None
    if len(src_args) == 2:
        for a in src_args:
            sz = params[a].dtype.size
            if sz not in {1, out_size}:
                return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    addrs_per_tile = 3  # lhs, rhs, out
    all_instrs, data_plan, outputs = [], [], []
    DIV_OP = _VPU_OPS["DIV"]
    MUL_OP = _VPU_OPS["MUL"]
    SUB_OP = _VPU_OPS["SUB"]
    for tile_idx in range(num_tiles):
        base = tile_idx * addrs_per_tile
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)
        if len(src_args) == 1:
            src_arg = src_args[0]
            src_size = params[src_arg].dtype.size
            lhs_entry = {"type": "VMEM", "addr": base, "param": src_arg,
                         "offset": 0 if src_size == 1 else offset,
                         "count": 1 if src_size == 1 else count, "dtype": "int32"}
            data_plan.append(lhs_entry)
            data_plan.append({"type": "VMEM", "addr": base + 1,
                              "layout": "broadcast_const", "value": int(rhs_const),
                              "count": count, "dtype": "int32"})
        else:
            lhs_arg, rhs_arg = src_args
            data_plan.append({"type": "VMEM", "addr": base, "param": lhs_arg,
                              "offset": offset, "count": count, "dtype": "int32"})
            data_plan.append({"type": "VMEM", "addr": base + 1, "param": rhs_arg,
                              "offset": offset, "count": count, "dtype": "int32"})
        out_vmem = base + 2
        if kind == "IDIV":
            all_instrs += [
                _load(0, base),
                _load(1, base + 1),
                _vpu(2, 0, DIV_OP, 1),
                _store(out_vmem, 2),
            ]
        else:  # MOD: x - (x // y) * y
            all_instrs += [
                _load(0, base),
                _load(1, base + 1),
                _vpu(2, 0, DIV_OP, 1),   # v2 = x // y
                _vpu(3, 2, MUL_OP, 1),   # v3 = v2 * y
                _vpu(4, 0, SUB_OP, 3),   # v4 = x - v3
                _store(out_vmem, 4),
            ]
        outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
    all_instrs.append(_halt())
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg}


def _render_const_fill_sxu_program(uops: list[UOp]) -> dict | None:
    """Render a pure STORE-CONST kernel (Tensor.zeros/ones/full) as broadcast store."""
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 1:
        return None
    out_arg = next(iter(params))
    out_size = params[out_arg].dtype.size
    if out_size <= 0:
        return None
    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None
    # Every STORE must write the same CONST (directly or through CAST).
    values = set()
    for s in stores:
        v = s.src[1]
        while v.op in (Ops.CAST, Ops.VECTORIZE):
            v = v.src[0] if len(v.src) > 0 else v
        if v.op is not Ops.CONST:
            return None
        values.add(v.arg)
    if len(values) != 1:
        return None
    const_val = next(iter(values))
    if isinstance(const_val, bool):
        const_bits = int(const_val)
    elif isinstance(const_val, float):
        const_bits = int(np.frombuffer(np.float32(const_val).tobytes(), dtype=np.int32)[0])
    else:
        const_bits = int(const_val)
    # Reject anything besides STORE/CONST/PARAM/INDEX/CAST/VECTORIZE/GROUP/SINK/END/RANGE.
    allowed = {"STORE", "CONST", "PARAM", "INDEX", "CAST", "VECTORIZE", "GROUP", "SINK", "END", "RANGE"}
    op_counts = Counter(u.op.name for u in uops)
    if any(c > 0 and n not in allowed for n, c in op_counts.items()):
        return None

    is_bool_out = (isinstance(params[out_arg].dtype, PtrDType)
                   and params[out_arg].dtype.base.itemsize == 1)
    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    data_plan = [{
        "type": "VMEM", "addr": 0, "layout": "broadcast_const",
        "value": const_bits, "count": _TILE_ELEMS, "dtype": "int32",
    }]
    instructions = [_load(0, 0)]
    outputs = []
    for t in range(num_tiles):
        offset = t * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)
        out_addr = 1 + t
        instructions.append(_store(out_addr, 0))
        outputs.append({"addr": out_addr, "param": out_arg, "offset": offset, "count": count})
    instructions.append(_halt())
    return {
        "op": "SXU_PROGRAM", "primitive": "CONST_FILL",
        "instructions": instructions, "data_plan": data_plan,
        "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg,
        "bool_out": is_bool_out,
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


def _render_chained_const_sxu_program(uops: list[UOp]) -> dict | None:
    """Render N chained scalar-const binary ops on one tensor input.

    Handles ((x op1 c1) op2 c2 ... opN cN) patterns emitted per element by
    tinygrad. Walks the ALU chain ending at the STORE value, collecting each
    step as (op, const, src_is_lhs), then emits inner→outer VPU ops with
    broadcast-const tiles. Only the final op may be a comparison.
    """
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2:
        return None
    out_params = set()
    for s in uops:
        if s.op is Ops.STORE:
            p = _find_unique_param_arg(s.src[0])
            if p is not None: out_params.add(p)
    if len(out_params) != 1:
        return None
    out_arg = next(iter(out_params))
    src_params = [k for k in params if k != out_arg]
    if len(src_params) != 1:
        return None
    src_arg = src_params[0]
    out_size = params[out_arg].dtype.size
    src_size = params[src_arg].dtype.size
    if out_size <= 0 or out_size != src_size:
        return None

    data_alu = [u for u in uops if u.op in _ALU_OPS and _has_load_src(u)]
    if not data_alu:
        return None
    # Reject patterns that other renderers own.
    op_counts = Counter(u.op.name for u in uops)
    if any(op_counts.get(n, 0) for n in ("WHERE", "MOD", "RECIPROCAL", "TRUNC", "SELECT", "WMMA", "MULACC")):
        return None

    # Pick one representative STORE and walk its value expression to extract the
    # per-element chain (ordered innermost -> outermost). Each link must be an
    # ALU UOp with one CONST operand and one prior-step operand.
    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None
    val = stores[0].src[1]
    # Unwrap VECTORIZE(elem0, elem1, ...) — per-element chains should be identical.
    if val.op is Ops.VECTORIZE:
        val = val.src[0]
    # Unwrap CAST (dtype conversion wrappers around the chain).
    while val.op is Ops.CAST:
        val = val.src[0]

    chain: list[tuple] = []  # each: (op_enum, const_arg, src_is_lhs)
    cur = val
    while cur.op in _ALU_OPS:
        const_src = next((s for s in cur.src if s.op is Ops.CONST and not isinstance(s.arg, bool)), None)
        other_src = next((s for s in cur.src if s is not const_src), None)
        if const_src is None or other_src is None:
            return None
        src_is_lhs = cur.src[0] is other_src
        chain.append((cur.op, const_src.arg, src_is_lhs))
        # Peel CAST/GEP wrappers between ALU steps so float chains (which
        # insert CAST to/from int32 bits) continue to match.
        cur = other_src
        while cur.op in (Ops.CAST, Ops.GEP):
            cur = cur.src[0]
    # cur must ultimately reach a LOAD on src_arg
    if not _has_load_src(cur):
        return None
    if len(chain) < 2:
        return None
    chain.reverse()  # now innermost-first

    # Same chain shape must repeat per element — data_alu size = N_chain * N_elem,
    # and op-count multiples of chain length.
    expected_total = len(chain) * len(data_alu) // len(chain)  # trivially true
    # Verify each chain op appears len(data_alu)/len(chain) times.
    per_elem = len(data_alu) // len(chain)
    if per_elem == 0 or len(data_alu) != per_elem * len(chain):
        return None
    chain_op_counts = Counter(op for op, _, _ in chain)
    alu_counts = Counter(u.op for u in data_alu)
    if any(alu_counts.get(op, 0) != cnt * per_elem for op, cnt in chain_op_counts.items()):
        return None
    if sum(alu_counts.values()) != per_elem * len(chain):
        return None

    is_float = "float" in str(params[out_arg].dtype) or "float" in str(params[src_arg].dtype)
    float_remap = {"ADD": "FADD", "MUL": "FMUL", "SUB": "FSUB", "MAX": "FMAX", "CMPLT": "FCMPLT"}
    resolved = []  # list of (vpu_op_int, const_bits, src_is_lhs, name)
    def _bits(c):
        if is_float and isinstance(c, float):
            return int(np.frombuffer(np.float32(c).tobytes(), dtype=np.int32)[0])
        return int(c)
    for op_enum, const_arg, src_is_lhs in chain:
        name = _ALU_OPS[op_enum]
        if is_float:
            name = float_remap.get(name, name)
        if name not in _VPU_OPS:
            return None
        resolved.append((_VPU_OPS[name], _bits(const_arg), src_is_lhs, name))
    # Only the outermost step may be a comparison (bool result).
    for _, _, _, n in resolved[:-1]:
        if n in {"CMPLT", "CMPNE", "CMPEQ", "FCMPLT"}:
            return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    addrs_per_tile = 1 + len(chain) + 1  # src + N consts + out
    all_instrs, data_plan, outputs = [], [], []
    for tile_idx in range(num_tiles):
        base = tile_idx * addrs_per_tile
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)
        data_plan.append({"type": "VMEM", "addr": base, "param": src_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        for i, (_, bits, _, _) in enumerate(resolved):
            data_plan.append({"type": "VMEM", "addr": base + 1 + i, "layout": "broadcast_const",
                              "value": bits, "count": count, "dtype": "int32"})
        out_vmem = base + 1 + len(chain)
        # VREGs: 0=src, 1..N=consts, N+1..2N = step results.
        all_instrs.append(_load(0, base))
        for i in range(len(chain)):
            all_instrs.append(_load(1 + i, base + 1 + i))
        prev_vreg = 0
        for i, (op_int, _, src_is_lhs, _) in enumerate(resolved):
            dst_vreg = 1 + len(chain) + i
            const_vreg = 1 + i
            va, vb = (prev_vreg, const_vreg) if src_is_lhs else (const_vreg, prev_vreg)
            all_instrs.append(_vpu(dst_vreg, va, op_int, vb))
            prev_vreg = dst_vreg
        all_instrs.append(_store(out_vmem, prev_vreg))
        outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})
    all_instrs.append(_halt())
    bool_out = resolved[-1][3] in {"CMPLT", "CMPNE", "CMPEQ", "FCMPLT"}
    return {"op": "SXU_PROGRAM", "instructions": all_instrs, "data_plan": data_plan,
            "outputs": outputs, "num_output_tiles": num_tiles, "out": out_arg,
            "bool_out": bool_out}


def _render_elementwise_sxu_program(uops: list[UOp]) -> dict | None:
    """Render an elementwise kernel as an SXU_PROGRAM.

    Handles: tensor-tensor binary, scalar-const binary (x+c, x*c, NEG, NOT),
    unary (RELU), bool-typed ops (AND/OR/XOR/NOT). Each tile chunk gets:
    LOAD inputs → VPU ops → STORE output.
    """
    _ALU_MAP = _ALU_OPS

    op_counts = Counter(u.op.name for u in uops)
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores or not params:
        return None
    # Transcendental UOps are handled by the dedicated transcendental
    # renderers above. If one appears in the kernel, don't silently render
    # just the adjacent ALU ops (which would drop the transcendental and
    # produce wrong numeric results).
    if any(op_counts.get(n, 0) for n in ("EXP2", "LOG2", "SIN", "SQRT")):
        return None

    # Find output param
    out_params = set()
    for s in stores:
        p = _find_unique_param_arg(s.src[0])
        if p is not None: out_params.add(p)
    if len(out_params) != 1:
        return None
    out_arg = next(iter(out_params))
    out_size = params[out_arg].dtype.size
    src_params = sorted(k for k in params if k != out_arg)

    # WHERE kernels: only handle simple RELU (WHERE+CMPLT with 1 src, no other ALU).
    # Use data-path ALU counts to avoid false positives from index arithmetic.
    has_where = op_counts.get("WHERE", 0) > 0
    store_count = op_counts.get("STORE", 0)
    data_alu_counts = _data_alu_ops(uops)
    # WHERE count may equal STORE count (GROUP lowering) or STORE * vector_width
    # (VECTORIZE lowering, e.g. float). Allow both patterns.
    vec_count = op_counts.get("VECTORIZE", 0)
    where_ok = (op_counts.get("WHERE", 0) == store_count
                or (vec_count > 0 and op_counts.get("WHERE", 0) == op_counts.get("CMPLT", 0)))
    # RELU uses one compare per output element. Kernels with more than one
    # compare per element (e.g. clip / hardtanh = max(lo, min(hi, x)) has
    # two) share the WHERE+CMPLT shape but are not RELU.
    cmplt_total = op_counts.get("CMPLT", 0) + op_counts.get("FCMPLT", 0)
    is_relu_candidate = (has_where and len(params) == 2 and len(src_params) == 1
                         and where_ok
                         and cmplt_total <= max(1, out_size)
                         and not any(data_alu_counts.get(n, 0) > 0 for n in ["ADD", "MUL", "MAX"]))
    if has_where and not is_relu_candidate:
        return None
    # Check param sizes — allow size-1 scalar broadcast and single-tile
    # structured row/column broadcast.
    src_sizes = {k: params[k].dtype.size for k in src_params}
    broadcast_params = {k for k, sz in src_sizes.items() if sz == 1 and out_size > 1}
    structured_broadcasts = {
        k: _classify_structured_broadcast_axis(uops, k)
        for k, sz in src_sizes.items()
        if k not in broadcast_params and 0 < sz < out_size and out_size <= _TILE_ELEMS and sz <= _ROWS
    }
    structured_broadcasts = {k: axis for k, axis in structured_broadcasts.items() if axis is not None}
    regular_sizes = [sz for k, sz in src_sizes.items() if k not in broadcast_params and k not in structured_broadcasts]
    if len(set(regular_sizes)) > 1 or (regular_sizes and regular_sizes[0] != out_size):
        return None
    if any(k not in broadcast_params and k not in structured_broadcasts and sz != out_size for k, sz in src_sizes.items()):
        return None

    # Only count ALU UOps in the data path (have LOAD in source tree), not index arithmetic
    alu_uops = [u for u in uops if u.op in _ALU_MAP and _has_load_src(u)]
    alu_op_types = len(set(_ALU_MAP[u.op] for u in alu_uops))

    # Detect bool dtype on params
    has_bool_in = any(isinstance(p.dtype, PtrDType) and p.dtype.base.itemsize == 1
                      for k, p in params.items() if k in src_params)
    has_bool_out = isinstance(params[out_arg].dtype, PtrDType) and params[out_arg].dtype.base.itemsize == 1

    # Detect SUB pattern: MUL(x, -1) + ADD → emit VPU SUB
    # Works for both tensor-tensor (2 params) and scalar-const reverse sub (1 param)
    is_neg_add = (alu_op_types == 2
                  and set(_ALU_MAP[u.op] for u in alu_uops) == {"MUL", "ADD"}
                  and any(u.op is Ops.CONST and u.arg == -1 for u in uops))

    # Only handle single-ALU-op kernels (not multi-op patterns like abs=MUL+MAX)
    if alu_op_types > 1 and not is_neg_add:
        return None
    # Don't handle compound patterns like CMPEQ (= NOT(CMPNE(x,y))) where data ALU ops chain
    # Exception: is_neg_add (SUB = ADD(MUL(x,-1), y)) is an expected MUL→ADD chain
    data_alu_set = set(alu_uops)
    if not is_neg_add and any(s in data_alu_set for u in data_alu_set for s in u.src):
        return None

    # RELU = WHERE(CMPLT(x, 0), 0, x). Tighten: require the CMPLT's CONST
    # operand to be 0 so clamp(max=c) for c != 0 doesn't false-match as
    # RELU (it also has WHERE+CMPLT+CONST but the CONST is the bound, not 0).
    cmplt_zero_ok = any(u.op is Ops.CMPLT and any(
        s.op is Ops.CONST and not isinstance(s.arg, bool) and float(s.arg) == 0.0
        for s in u.src) for u in uops)
    is_relu = is_relu_candidate and op_counts.get("CMPLT", 0) > 0 and cmplt_zero_ok

    # --- Determine VPU op, operand sources, and inputs_per_tile ---
    const_val = None  # set if one operand is a scalar constant
    is_bool_out_flag = has_bool_out
    primitive_tags: set[str] = set()

    if is_relu and len(src_params) == 1:
        tile_vpu_op = 2  # VPU_RELU
        inputs_per_tile = 1
    elif len(src_params) == 2:
        # Tensor-tensor binary
        if is_neg_add:
            vpu_name = "SUB"
        else:
            vpu_name = None
            for u in alu_uops:
                name = _ALU_MAP.get(u.op)
                if name and name not in {"MUL"}:
                    vpu_name = name
                    break
            if vpu_name is None:
                vpu_name = "MUL" if op_counts.get("MUL", 0) > 0 else None
            if vpu_name is None:
                return None

        tile_vpu_op = _VPU_OPS[vpu_name]
        is_bool_out_flag = is_bool_out_flag or vpu_name in {"CMPLT", "CMPNE", "CMPEQ"}

        # Determine operand order
        if is_neg_add:
            # Find data-path ADD that has a data-path MUL as source
            data_adds = [u for u in alu_uops if u.op is Ops.ADD]
            data_muls = {u for u in alu_uops if u.op is Ops.MUL}
            add_uop = next(u for u in data_adds if any(s in data_muls for s in u.src))
            lhs_param = _find_unique_param_arg(add_uop.src[0])
            mul_uop = next(s for s in add_uop.src if s in data_muls)
            rhs_param = _find_unique_param_arg(mul_uop)
            if rhs_param is None:
                for s in mul_uop.src:
                    if s.op is Ops.LOAD:
                        rhs_param = _find_unique_param_arg(s)
                        break
        else:
            alu_uop = next(u for u in alu_uops if _ALU_MAP.get(u.op) == vpu_name)
            lhs_param = _find_unique_param_arg(alu_uop.src[0])
            rhs_param = _find_unique_param_arg(alu_uop.src[1])

        if lhs_param is None or rhs_param is None:
            return None
        # Remap integer VPU ops to float variants when operating on float32 tensors
        is_float = any("float" in str(params[p].dtype) for p in [out_arg, lhs_param, rhs_param])
        if is_float:
            _FLOAT_REMAP = {"ADD": "FADD", "MUL": "FMUL", "SUB": "FSUB", "MAX": "FMAX", "CMPLT": "FCMPLT"}
            if vpu_name in _FLOAT_REMAP:
                vpu_name = _FLOAT_REMAP[vpu_name]
                tile_vpu_op = _VPU_OPS[vpu_name]
                is_bool_out_flag = is_bool_out_flag or vpu_name == "FCMPLT"
        inputs_per_tile = 2
        src_params = [lhs_param, rhs_param]
    elif len(src_params) == 1:
        # 1 source param: either unary RELU (handled above), or scalar-const binary
        if is_neg_add:
            # Scalar-const reverse sub: const - x = ADD(MUL(x, -1), const)
            vpu_name = "SUB"
            alu_op_enum = Ops.ADD
            # The constant is the non-(-1) CONST source of the ADD
            add_uops = [u for u in alu_uops if u.op is Ops.ADD]
            const_val = None
            for u in add_uops:
                for s in u.src:
                    if s.op is Ops.CONST and s.arg != -1 and not isinstance(s.arg, bool):
                        const_val = s.arg
                        break
                if const_val is not None: break
            if const_val is None:
                return None
        else:
            # Find the ALU op from data-path UOps only
            vpu_name = None
            alu_op_enum = None
            for op_enum, name in _ALU_MAP.items():
                if any(u.op is op_enum for u in alu_uops):
                    alu_op_enum = op_enum
                    vpu_name = name
                    break
            if vpu_name is None:
                return None
            # Find the constant value from the UOp graph
            const_val = _find_alu_const(alu_uops, alu_op_enum)
            if const_val is None:
                return None

        # Determine operand order: is src the lhs or rhs?
        if is_neg_add:
            # Reverse sub: const - x → SUB(const, x), const is lhs
            src_is_lhs = False
        else:
            alu_uop = next(u for u in alu_uops if u.op is alu_op_enum)
            src_is_lhs = _find_unique_param_arg(alu_uop.src[0]) is not None

        # Remap integer VPU ops to float variants when operating on float32 tensors
        is_float = any("float" in str(params[p].dtype) for p in [out_arg] + src_params)
        if is_float:
            _FLOAT_REMAP = {"ADD": "FADD", "MUL": "FMUL", "SUB": "FSUB", "MAX": "FMAX", "CMPLT": "FCMPLT"}
            vpu_name = _FLOAT_REMAP.get(vpu_name, vpu_name)
        tile_vpu_op = _VPU_OPS[vpu_name]
        is_bool_out_flag = is_bool_out_flag or vpu_name in {"CMPLT", "CMPNE", "CMPEQ", "FCMPLT"}
        inputs_per_tile = 2  # src tile + const broadcast tile
        # For float scalar const, bitcast to int32 representation
        if is_float and const_val is not None and not isinstance(const_val, bool):
            const_val = int(np.frombuffer(np.float32(const_val).tobytes(), dtype=np.int32)[0])
        if src_is_lhs:
            src_params = [src_params[0], None]  # None = const slot
        else:
            src_params = [None, src_params[0]]  # const is lhs
    else:
        return None

    # Build full program: repeat per tile chunk
    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    addrs_per_tile = inputs_per_tile + 1  # inputs + output

    all_instrs: list[str] = []
    data_plan: list[dict] = []
    outputs: list[dict] = []

    for tile_idx in range(num_tiles):
        base = tile_idx * addrs_per_tile
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)

        for inp_idx in range(inputs_per_tile):
            src_arg = src_params[inp_idx] if inputs_per_tile > 1 else src_params[0]
            if src_arg is None:
                # Broadcast constant tile
                data_plan.append({
                    "type": "VMEM", "addr": base + inp_idx,
                    "layout": "broadcast_const", "value": const_val,
                    "count": count, "dtype": "int32",
                })
            elif src_arg in broadcast_params:
                # Scalar broadcast: load the scalar into lane [0,0], then
                # expand it in hardware via BROADCAST_SCALAR.
                entry = {"type": "VMEM", "addr": base + inp_idx,
                         "param": src_arg, "offset": 0, "count": 1, "dtype": "int32",
                         "broadcast": False}
                data_plan.append(entry)
                primitive_tags.add("BROADCAST_SCALAR")
            elif src_arg in structured_broadcasts:
                axis = structured_broadcasts[src_arg]
                if axis == "row":
                    data_plan.append({
                        "type": "VMEM", "addr": base + inp_idx,
                        "param": src_arg, "offset": 0, "count": src_sizes[src_arg], "dtype": "int32",
                    })
                    primitive_tags.add("BROADCAST_ROW")
                elif axis == "col":
                    data_plan.append({
                        "type": "VMEM", "addr": base + inp_idx,
                        "param": src_arg, "offset": 0, "count": src_sizes[src_arg], "dtype": "int32",
                        "mode": "MATRIX_TILE", "matrix_nrows": src_sizes[src_arg], "matrix_ncols": 1,
                        "row_base": 0, "col_base": 0, "tile_rows": src_sizes[src_arg], "tile_cols": 1,
                    })
                    primitive_tags.add("BROADCAST_COL")
                else:
                    return None
            else:
                entry = {"type": "VMEM", "addr": base + inp_idx,
                         "param": src_arg, "offset": offset, "count": count, "dtype": "int32"}
                if has_bool_in and isinstance(params[src_arg].dtype, PtrDType) and params[src_arg].dtype.base.itemsize == 1:
                    entry["bool"] = True
                data_plan.append(entry)

        out_vmem = base + inputs_per_tile
        if inputs_per_tile == 1:
            all_instrs += [_load(0, base), _vpu(1, 0, tile_vpu_op), _store(out_vmem, 1)]
        else:
            tile_instrs = [_load(0, base), _load(1, base + 1)]
            if src_params[0] in broadcast_params:
                tile_instrs.append(_broadcast_scalar(0, 0, 0, 0))
            elif src_params[0] in structured_broadcasts:
                tile_instrs.append(_broadcast_row(0, 0, 0) if structured_broadcasts[src_params[0]] == "row" else _broadcast_col(0, 0, 0))
            if src_params[1] in broadcast_params:
                tile_instrs.append(_broadcast_scalar(1, 1, 0, 0))
            elif src_params[1] in structured_broadcasts:
                tile_instrs.append(_broadcast_row(1, 1, 0) if structured_broadcasts[src_params[1]] == "row" else _broadcast_col(1, 1, 0))
            tile_instrs += [_vpu(2, 0, tile_vpu_op, 1), _store(out_vmem, 2)]
            all_instrs += tile_instrs

        outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})

    all_instrs.append(_halt())

    return {
        "op": "SXU_PROGRAM",
        **({"primitive": next(iter(sorted(primitive_tags)))} if len(primitive_tags) == 1 else {}),
        "instructions": all_instrs,
        "data_plan": data_plan,
        "outputs": outputs,
        "num_output_tiles": num_tiles,
        "out": out_arg,
        "bool_out": is_bool_out_flag,
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


def _classify_structured_broadcast_axis(uops: list[UOp], param_arg: int) -> str | None:
    """Infer row/column broadcast orientation for a short 2D operand."""
    param_indices = [u.src[1] for u in uops
                     if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and u.src[0].arg == param_arg]
    if not param_indices:
        return None
    if all(idx.op is Ops.RANGE for idx in param_indices):
        return "col"
    if all(idx.op is Ops.CONST for idx in param_indices):
        vals = [int(idx.arg) for idx in param_indices]
        uniq = sorted(set(vals))
        if uniq and len(vals) % len(uniq) == 0:
            chunk = len(vals) // len(uniq)
            if chunk > 1 and vals == [v for u in uniq for v in [u] * chunk]:
                return "col"
            if chunk > 1 and vals == uniq * chunk:
                return "row"
        # chunk==1 (all-unique param indices) is ambiguous from val ordering
        # alone. Correlate each bias CONST with the store addresses that
        # consume it: bias[k] used at addrs satisfying addr % bias_size == k
        # is a row broadcast; addr // stride == k is a col broadcast.
        bias_size = len(uniq) if uniq else 0
        idx_uops = [u for u in uops if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM
                    and u.src[0].arg == param_arg and u.src[1].op is Ops.CONST]
        stores = [s for s in uops if s.op is Ops.STORE and s.src[0].op is Ops.INDEX
                  and s.src[0].src[1].op is Ops.CONST]
        def _depends(u, target, seen=None):
            if seen is None: seen = set()
            if id(u) in seen: return False
            seen.add(id(u))
            if u is target: return True
            return any(_depends(s, target, seen) for s in u.src)
        pairs: list[tuple[int, int]] = []  # (bias_idx, output_addr)
        for idx_uop in idx_uops:
            k = int(idx_uop.src[1].arg)
            for s in stores:
                if _depends(s.src[1], idx_uop):
                    pairs.append((k, int(s.src[0].src[1].arg)))
        if bias_size > 0 and pairs:
            row_ok = all(a % bias_size == k for k, a in pairs)
            all_addrs = [a for _, a in pairs]
            col_stride_candidates = {a // k for k, a in pairs if k != 0 and a >= k and a % k == 0}
            col_stride_candidates |= {(max(all_addrs) + 1) // bias_size} if all_addrs else set()
            col_ok = any(stride > 0 and all(a // stride == k for k, a in pairs) for stride in col_stride_candidates)
            if row_ok and not col_ok:
                return "row"
            if col_ok and not row_ok:
                return "col"
        return "row"

    store = next((u for u in uops if u.op is Ops.STORE and u.src[0].op is Ops.INDEX), None)
    if store is None:
        return None
    out_idx = store.src[0].src[1]
    row_range = col_range = None
    if out_idx.op is Ops.ADD:
        for src in out_idx.src:
            if src.op is Ops.MUL:
                row_range = next((s for s in src.src if s.op is Ops.RANGE), None)
            elif src.op is Ops.RANGE:
                col_range = src
    if row_range is None or col_range is None:
        return None

    uses_row = any(idx is row_range for idx in param_indices)
    uses_col = any(idx is col_range for idx in param_indices)
    if uses_row and not uses_col:
        return "col"
    if uses_col and not uses_row:
        return "row"
    return None


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


def _find_min_scalar_const(uops:list[UOp]) -> int | None:
    """Extract the scalar constant from minimum(x, c) = ~max(~x, ~c) decomposition.
    Structure: XOR(MAX(XOR(x, -1), CONST(~c)), -1). The CONST child of MAX is ~c."""
    for u in uops:
        if u.op is Ops.MAX:
            xor_srcs = [s for s in u.src if s.op is Ops.XOR]
            const_srcs = [s for s in u.src if s.op is Ops.CONST and not isinstance(s.arg, bool)]
            if len(xor_srcs) >= 1 and len(const_srcs) == 1:
                return ~int(const_srcs[0].arg)  # ~(~c) = c
    return None


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



def _const_arg(u:UOp) -> int | bool | None:
    return u.arg if u.op is Ops.CONST else None


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

def _set_pred_if_zero(vs: int) -> str:
    # SXU_SET_PRED_IF_ZERO opcode = 20; pred := (vs[0][0] == 0).
    return f"2 20 0 0 {vs} 0 0 0 0 0"

def _skip_if_pred() -> str:
    # SXU_SKIP_IF_PRED opcode = 21; if pred, skip the next instruction.
    return f"2 21 0 0 0 0 0 0 0 0"

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
