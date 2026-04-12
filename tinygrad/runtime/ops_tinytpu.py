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
             "SUM_REDUCE_TILE": 32, "MAX_REDUCE_TILE": 33, "MIN_REDUCE_TILE": 34}
_VPU_BOOL_OPS = {_VPU_OPS["CMPLT"], _VPU_OPS["CMPNE"], _VPU_OPS["CMPEQ"]}
_SXU_OPS = {"LOAD_VREG": 0, "STORE_VREG": 1, "DISPATCH_VPU": 2, "DISPATCH_XLU_BROADCAST": 3, "DISPATCH_MXU": 4, "WAIT_MXU": 5, "LOAD_MXU_RESULT": 6, "HALT": 7, "DISPATCH_SELECT": 8, "BROADCAST_SCALAR": 9, "BROADCAST_ROW": 10, "BROADCAST_COL": 11}

_ALU_OPS = {Ops.ADD: "ADD", Ops.MUL: "MUL", Ops.SUB: "SUB", Ops.MAX: "MAX",
            Ops.CMPLT: "CMPLT", Ops.CMPNE: "CMPNE", Ops.CMPEQ: "CMPEQ",
            Ops.AND: "AND", Ops.OR: "OR", Ops.XOR: "XOR",
            Ops.SHL: "SHL", Ops.SHR: "SHR", Ops.IDIV: "DIV"}

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

def _detect_reduce_op(op_counts: Counter) -> str | None:
    """Detect SUM/MAX/MIN from UOp op counts."""
    nloads = op_counts.get("LOAD", 0)
    if op_counts.get("ADD", 0) > nloads - 1 and op_counts.get("MAX", 0) == 0:
        return "SUM"
    if op_counts.get("MAX", 0) >= nloads - 1 and op_counts.get("XOR", 0) == 0:
        return "MAX"
    if op_counts.get("MAX", 0) >= nloads - 1 and op_counts.get("XOR", 0) > 0:
        return "MIN"
    return None

def _render_legacy_descriptor(uops: list[UOp]) -> dict | None:
    """Slim fallback renderer for patterns not yet handled by SXU_PROGRAM.

    Handles: HOST_UNARY, GEMM4x4, and remaining scalar-const VPU_BINARY /
    VPU_PROGRAM patterns. Row/column reductions now go through the
    SXU_PROGRAM renderers and no longer fall back to host numpy.
    """
    op_counts = Counter(u.op.name for u in uops)
    params = [u for u in uops if u.op is Ops.PARAM]
    param_sizes: dict[int, int] = {}
    for p in params:
        if not isinstance(p.dtype, PtrDType):
            return None
        param_sizes[p.arg] = p.dtype.size

    # --- HOST_UNARY: TRUNC or RECIPROCAL on float ---
    if (len(params) == 2 and (op_counts.get("TRUNC", 0) > 0 or op_counts.get("RECIPROCAL", 0) > 0)
            and "float" in str(params[0].dtype) and "float" in str(params[1].dtype)):
        host_op = "TRUNC" if op_counts.get("TRUNC", 0) > 0 else "RECIPROCAL"
        src_size = param_sizes.get(1, 0)
        if src_size > 0:
            return {"op": "HOST_UNARY", "host_op": host_op, "dtype": "float32",
                    "out": 0, "src": 1, "num_elems": src_size}

# --- GEMM fallback: 3 params with MULACC or scalar MUL+RANGE pattern ---
    has_mulacc = any(u.op is Ops.MULACC for u in uops)
    has_store = op_counts.get("STORE", 0) > 0
    is_gemm = has_mulacc or (len(params) == 3 and op_counts.get("MUL", 0) > 0
                              and op_counts.get("RANGE", 0) > 0 and has_store)
    if is_gemm and len(param_sizes) == 3 and op_counts.get("GROUP", 0) == 0:
        tiling = _infer_tiling(param_sizes.get(0), param_sizes.get(1), param_sizes.get(2, 0))
        if tiling is not None:
            num_vecs, num_k_tiles, num_weight_tiles = tiling
            return {"op": "GEMM4x4", "out": 0, "act": 1, "weight": 2,
                    "num_vecs": num_vecs, "num_k_tiles": num_k_tiles,
                    "num_weight_tiles": num_weight_tiles}

    # --- Remaining complex patterns: delegate to analyze_tinytpu_uops ---
    diag = analyze_tinytpu_uops(uops)
    if diag["supported"]:
        _KIND_SCHEMA = {
            "vpu_binary":      ("VPU_BINARY",       ["vpu_op", "out_arg:out", "lhs_arg:lhs", "lhs_const", "lhs_broadcast", "rhs_arg:rhs", "rhs_const", "rhs_broadcast", "num_elems", "bool_out", "bool_in"]),
            "vpu_program":     ("VPU_PROGRAM",       ["out_arg:out", "num_elems", "inputs", "steps", "output_reg"]),
        }
        if diag["kind"] in _KIND_SCHEMA:
            op_name, keys = _KIND_SCHEMA[diag["kind"]]
            desc: dict = {"op": op_name}
            for key in keys:
                src, dst = key.split(":") if ":" in key else (key, key)
                desc[dst] = diag.get(src, False)
            return desc
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
        if (legacy := _render_legacy_descriptor(uops)) is not None:
            return _dump_lowering(json.dumps(legacy))
        op_counts = dict(sorted(Counter(u.op.name for u in uops).items()))
        return _dump_lowering(json.dumps({
            "op": "UNSUPPORTED",
            "reason": "no SXU_PROGRAM or legacy renderer matched",
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
        if (where_desc := _render_where_sxu_program(uops)) is not None:
            return where_desc
        if (min_const_desc := _render_min_const_sxu_program(uops)) is not None:
            return min_const_desc
        if (multi_desc := _render_multistep_sxu_program(uops)) is not None:
            return multi_desc
        if (rowbc_desc := _render_rowbc_sxu_program(uops)) is not None:
            return rowbc_desc
        if (colred_desc := _render_colreduce_sxu_program(uops)) is not None:
            return colred_desc
        if (rowred_desc := _render_rowreduce_sxu_program(uops)) is not None:
            return rowred_desc
        if (copy_desc := _render_copy_sxu_program(uops)) is not None:
            return copy_desc
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

    # Generate SXU instructions
    instructions = _generate_gemm_sxu_instructions(
        num_vecs, num_k_tiles, num_weight_tiles,
        has_bias=has_bias, bias_vmem_base=bias_vmem_base,
        has_relu=has_relu,
    )

    # Output VMEM addresses
    out_vmem_base = num_weight_tiles if has_bias else 0
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

    # Detect reduction type from UOp tree
    has_add = op_counts.get("ADD", 0) > 0
    has_max = op_counts.get("MAX", 0) > 0
    has_xor = op_counts.get("XOR", 0) > 0
    has_store = op_counts.get("STORE", 0) > 0
    if not has_store:
        return None

    # INT32 identity bounds used as padding so tile-reduce produces correct
    # results on partial last tiles.
    _INT32_MIN = -(1 << 31)
    _INT32_MAX = (1 << 31) - 1
    pad_value = 0
    if has_add and not has_max:
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
                      _VPU_OPS["MIN_REDUCE_TILE"]: "min"}

    # Scalar reduction
    num_tiles = (src_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    all_instrs: list[str] = []
    data_plan: list[dict] = []

    for tile_idx in range(num_tiles):
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, src_size - offset)
        vmem_addr = tile_idx
        entry = {"type": "VMEM", "addr": vmem_addr, "param": src_arg,
                 "offset": offset, "count": count, "dtype": "int32"}
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
        out_vmem = num_tiles
        all_instrs.append(_store(out_vmem, acc_vreg))
    else:
        out_vmem = 1
        all_instrs.append(_store(out_vmem, 1))

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
    if op_counts.get("MUL", 0) != 0: return None  # col-reduce has no stride multiply
    reduce_op = _detect_reduce_op(op_counts)
    if reduce_op is None: return None
    ncols = out_size
    nrows = src_size // ncols

    _INT32_MIN = -(1 << 31)
    _INT32_MAX = (1 << 31) - 1
    _REDUCE_VPU = {
        "SUM": (_VPU_OPS["SUM_REDUCE_COL"], _VPU_OPS["ADD"], 0),
        "MAX": (_VPU_OPS["MAX_REDUCE_COL"], _VPU_OPS["MAX"], _INT32_MIN),
        "MIN": (_VPU_OPS["MIN_REDUCE_COL"], _VPU_OPS["MIN"], _INT32_MAX),
    }
    vpu_op, combine_op, pad_value = _REDUCE_VPU[reduce_op]

    out_arg, src_arg = 0, 1
    num_row_tiles = (nrows + _ROWS - 1) // _ROWS
    num_col_tiles = (ncols + _COLS - 1) // _COLS
    data_plan: list[dict] = []
    all_instrs: list[str] = []
    outputs: list[dict] = []

    src_addr = 0
    vreg = 0
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
    # Row-reduce has a stride-multiply in the index expression.
    if op_counts.get("MUL", 0) != 1: return None
    reduce_op = _detect_reduce_op(op_counts)
    if reduce_op is None: return None
    nrows = out_size
    ncols = src_size // nrows
    if ncols < 2: return None
    # Tinygrad may fully unroll or keep a RANGE loop; match both.
    nloads = op_counts.get("LOAD", 0)
    if nloads != ncols and nloads != 1: return None
    _INT32_MIN = -(1 << 31)
    _INT32_MAX = (1 << 31) - 1
    _REDUCE_VPU = {
        "SUM": (_VPU_OPS["SUM_REDUCE"], _VPU_OPS["ADD"], 0),
        "MAX": (_VPU_OPS["MAX_REDUCE"], _VPU_OPS["MAX"], _INT32_MIN),
        "MIN": (_VPU_OPS["MIN_REDUCE"], _VPU_OPS["MIN"], _INT32_MAX),
    }
    vpu_op, combine_op, pad_value = _REDUCE_VPU[reduce_op]

    out_arg, src_arg = 0, 1
    num_row_tiles = (nrows + _ROWS - 1) // _ROWS
    num_col_tiles = (ncols + _COLS - 1) // _COLS
    data_plan: list[dict] = []
    all_instrs: list[str] = []
    outputs: list[dict] = []

    src_addr = 0
    vreg = 0
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
    if out_size != src_size or out_size <= 0:
        return None
    # Require matching element dtype — dtype conversions (bool<->int32 casts)
    # need the VPU_BINARY widening path.
    out_base = params[out_arg].dtype.base.itemsize
    src_base = params[src_arg].dtype.base.itemsize
    if out_base != src_base:
        return None
    # Require identity index mapping: each STORE must pair with a LOAD that
    # reads from the same index expression. This rules out permute/transpose.
    def _index_of(addr_uop):
        # INDEX UOp: src[0] = PARAM, src[1] = offset (flat index expression).
        if addr_uop.op is Ops.INDEX:
            return addr_uop.src[1]
        return None
    for store in stores:
        out_addr = store.src[0]
        val = store.src[1]
        if val.op is not Ops.LOAD:
            return None
        in_addr = val.src[0]
        out_idx = _index_of(out_addr)
        in_idx = _index_of(in_addr)
        if out_idx is None or in_idx is None or out_idx is not in_idx:
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
            "offset": offset, "count": count, "dtype": "int32",
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

    # --- ABS: 2 params, WHERE+CMPLT+CMPNE+MUL pattern → SUB(0,x), MAX(x, neg) ---
    if (len(src_params) == 1 and has_where and has_cmplt and has_cmpne and has_mul
            and not has_idiv and not has_mod):
        src_arg = src_params[0]
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
                _vpu(2, 1, SUB_OP, 0), # v2 = 0 - src = -src
                _vpu(3, 0, MAX_OP, 2), # v3 = max(src, -src) = abs(src)
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
    if len(params) != 4:
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
    if len(input_args) != 3:
        return None

    # All inputs must be same size or size 1 (broadcast)
    if not all(params[a].dtype.size in {1, out_size} for a in input_args):
        return None

    # Identify cond (bool dtype), lhs, rhs from WHERE UOp sources
    where_uop = next(u for u in uops if u.op is Ops.WHERE)
    cond_arg = _find_unique_param_arg(where_uop.src[0])
    lhs_arg = _find_unique_param_arg(where_uop.src[1])
    rhs_arg = _find_unique_param_arg(where_uop.src[2])
    if cond_arg is None or lhs_arg is None or rhs_arg is None:
        return None

    num_tiles = (out_size + _TILE_ELEMS - 1) // _TILE_ELEMS
    addrs_per_tile = 4  # cond, lhs, rhs, out

    all_instrs: list[str] = []
    data_plan: list[dict] = []
    outputs: list[dict] = []
    uses_scalar_broadcast = False

    for tile_idx in range(num_tiles):
        base = tile_idx * addrs_per_tile
        offset = tile_idx * _TILE_ELEMS
        count = min(_TILE_ELEMS, out_size - offset)

        data_plan.append({"type": "VMEM", "addr": base, "param": cond_arg,
                          "offset": offset, "count": count, "dtype": "int32", "bool": True})
        data_plan.append({"type": "VMEM", "addr": base + 1, "param": lhs_arg,
                          "offset": offset, "count": count, "dtype": "int32"})
        data_plan.append({"type": "VMEM", "addr": base + 2, "param": rhs_arg,
                          "offset": offset, "count": count, "dtype": "int32"})

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


def _render_min_const_sxu_program(uops: list[UOp]) -> dict | None:
    """Render minimum(x, const) through native VPU MIN in SXU_PROGRAM."""
    op_counts = Counter(u.op.name for u in uops)
    params = {u.arg: u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)}
    if len(params) != 2 or op_counts.get("STORE", 0) < 1 or op_counts.get("XOR", 0) == 0 or op_counts.get("MAX", 0) == 0:
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
    is_relu_candidate = (has_where and len(params) == 2 and len(src_params) == 1
                         and op_counts.get("WHERE", 0) == store_count
                         and not any(data_alu_counts.get(n, 0) > 0 for n in ["ADD", "MUL", "MAX"]))
    if has_where and not is_relu_candidate:
        return None
    # Check param sizes — allow size-1 broadcast (scalar → tile)
    src_sizes = {k: params[k].dtype.size for k in src_params}
    broadcast_params = {k for k, sz in src_sizes.items() if sz == 1 and out_size > 1}
    non_bc_sizes = [sz for k, sz in src_sizes.items() if k not in broadcast_params]
    if len(set(non_bc_sizes)) > 1 or (non_bc_sizes and non_bc_sizes[0] != out_size):
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

    is_relu = is_relu_candidate and op_counts.get("CMPLT", 0) > 0

    # --- Determine VPU op, operand sources, and inputs_per_tile ---
    const_val = None  # set if one operand is a scalar constant
    is_bool_out_flag = has_bool_out
    uses_scalar_broadcast = False

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

        tile_vpu_op = _VPU_OPS[vpu_name]
        is_bool_out_flag = is_bool_out_flag or vpu_name in {"CMPLT", "CMPNE", "CMPEQ"}
        inputs_per_tile = 2  # src tile + const broadcast tile
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
                uses_scalar_broadcast = True
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
            if src_params[1] in broadcast_params:
                tile_instrs.append(_broadcast_scalar(1, 1, 0, 0))
            tile_instrs += [_vpu(2, 0, tile_vpu_op, 1), _store(out_vmem, 2)]
            all_instrs += tile_instrs

        outputs.append({"addr": out_vmem, "param": out_arg, "offset": offset, "count": count})

    all_instrs.append(_halt())

    return {
        "op": "SXU_PROGRAM",
        **({"primitive": "BROADCAST_SCALAR"} if uses_scalar_broadcast else {}),
        "instructions": all_instrs,
        "data_plan": data_plan,
        "outputs": outputs,
        "num_output_tiles": num_tiles,
        "out": out_arg,
        "bool_out": is_bool_out_flag,
    }


def _generate_gemm_sxu_instructions(num_vecs: int, num_k_tiles: int, num_weight_tiles: int,
                                     *, has_bias: bool = False, bias_vmem_base: int = 0,
                                     has_relu: bool = False) -> list[str]:
    """Generate SXU instruction strings for a GEMM kernel.

    These are the same instructions that _build_full_gemm_bundle generates,
    but without any data records -- just the SXU program lines.
    """
    out_vmem_base = num_weight_tiles if has_bias else 0
    prog_lines: list[str] = []

    for row in range(num_vecs):
        for tile_idx in range(num_weight_tiles):
            # MXU dispatches for K-tile accumulation
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

def _mxu(wbase: int, abase: int, tiles: int) -> str:
    return f"2 4 0 0 0 0 0 {wbase} {abase} {tiles}"

def _wait_mxu() -> str: return "2 5 0 0 0 0 0 0 0 0"
def _load_mxu_result(vd: int) -> str: return f"2 6 0 {vd} 0 0 0 0 0 0"
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
_SUPPORTED_OPS = {"GEMM4x4", "SXU_PROGRAM", "VPU_BINARY", "VPU_PROGRAM", "HOST_BINARY", "HOST_UNARY"}

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

    def _exec_vpu_binary(self, bufs):
        prog = self.prog
        out_buf = bufs[prog["out"]]
        num_elems = int(prog["num_elems"])
        bool_inputs = prog.get("bool_in", False) or prog.get("bool_out", False)
        def _read_operand(key, const_key):
            if prog.get(const_key) is not None:
                return np.full(num_elems, int(prog[const_key]), dtype="<i4")
            raw = np.frombuffer(bytes(bufs[prog[key]]), dtype=np.bool_ if bool_inputs else "<i4")
            return raw.astype(np.int32) if bool_inputs else raw
        lhs_i32 = _read_operand("lhs", "lhs_const")
        rhs_i32 = _read_operand("rhs", "rhs_const")
        lhs_bc = bool(prog.get("lhs_broadcast", False)) and prog.get("lhs_const") is None
        rhs_bc = bool(prog.get("rhs_broadcast", False)) and prog.get("rhs_const") is None
        is_bool = int(prog["vpu_op"]) in _VPU_BOOL_OPS or prog.get("bool_out", False)
        vpu_op = int(prog["vpu_op"])
        return self._run_tiled_vpu(out_buf, num_elems,
            lambda s, e, n: _build_vpu_binary_bundle(
                lhs_i32[:1] if lhs_bc else lhs_i32[s:e],
                rhs_i32[:1] if rhs_bc else rhs_i32[s:e],
                n, vpu_op, lhs_broadcast=lhs_bc, rhs_broadcast=rhs_bc),
            out_dtype=np.dtype(np.bool_) if is_bool else np.dtype("<i4"))



    def _exec_vpu_program(self, bufs):
        prog = self.prog
        out_buf = bufs[prog["out"]]
        num_elems = int(prog["num_elems"])
        if len(out_buf) < num_elems * _BYTES_PER_ELEM:
            raise RuntimeError(f"TinyTPU output buffer too small for VPU program elements={num_elems}")
        out_offset = 0
        for chunk_start in range(0, num_elems, _TILE_ELEMS):
            chunk_end = min(chunk_start + _TILE_ELEMS, num_elems)
            chunk_size = chunk_end - chunk_start
            input_tiles: list[np.ndarray] = []
            input_broadcasts: list[bool] = []
            for spec in prog["inputs"]:
                if "const" in spec:
                    input_tiles.append(np.full(chunk_size, int(spec["const"]), dtype=np.int32))
                    input_broadcasts.append(False)
                    continue
                is_bool = bool(spec.get("bool", False))
                broadcast = bool(spec.get("broadcast", False))
                raw = np.frombuffer(bytes(bufs[int(spec["arg"])]), dtype=np.bool_ if is_bool else "<i4")
                if raw.size == 1 and chunk_size > 1 and broadcast:
                    chunk = raw[:1].astype(np.int32) if is_bool else raw[:1]
                else:
                    chunk = raw[chunk_start:chunk_end].astype(np.int32) if is_bool else raw[chunk_start:chunk_end]
                    if raw.size == 1 and chunk_size > 1:
                        scalar = int(raw[0])
                        chunk = np.full(chunk_size, scalar, dtype=np.int32)
                if chunk.size != chunk_size:
                    if not (broadcast and chunk.size == 1):
                        raise RuntimeError(f"TinyTPU VPU program input expected {chunk_size} elements, got {chunk.size}")
                input_tiles.append(np.asarray(chunk, dtype=np.int32))
                input_broadcasts.append(broadcast)
            stdout = self._run(_build_vpu_program_bundle(input_tiles, chunk_size, prog["steps"], int(prog["output_reg"]),
                                                                input_broadcasts=input_broadcasts))
            result = _parse_vmem_output(stdout)
            if result is None:
                raise RuntimeError(f"TinyTPU sim produced no vmem_result\nstdout: {stdout}")
            chunk_out = np.array(result[:chunk_size], dtype="<i4")
            out_buf[out_offset : out_offset + len(chunk_out) * _BYTES_PER_ELEM] = chunk_out.tobytes()
            out_offset += len(chunk_out) * _BYTES_PER_ELEM
        return 1e-3

    def _exec_host_binary(self, bufs):
        prog = self.prog
        out_buf = bufs[prog["out"]]
        num_elems = int(prog["num_elems"])
        lhs_i32 = np.full(num_elems, int(prog["lhs_const"]), dtype=np.int32) if prog.get("lhs_const") is not None else np.frombuffer(bytes(bufs[prog["lhs"]]), dtype="<i4")
        rhs_i32 = np.full(num_elems, int(prog["rhs_const"]), dtype=np.int32) if prog.get("rhs_const") is not None else np.frombuffer(bytes(bufs[prog["rhs"]]), dtype="<i4")
        if lhs_i32.size == 1 and num_elems > 1:
            lhs_i32 = np.full(num_elems, int(lhs_i32[0]), dtype=np.int32)
        if rhs_i32.size == 1 and num_elems > 1:
            rhs_i32 = np.full(num_elems, int(rhs_i32[0]), dtype=np.int32)
        if lhs_i32.size != num_elems or rhs_i32.size != num_elems:
            raise RuntimeError(f"TinyTPU host binary op expected {num_elems} elements, got lhs={lhs_i32.size} rhs={rhs_i32.size}")
        if np.any(rhs_i32 == 0):
            raise ZeroDivisionError("TinyTPU host division fallback received divisor 0")
        q = np.trunc(lhs_i32.astype(np.float64) / rhs_i32.astype(np.float64)).astype(np.int32)
        if prog["host_op"] == "IDIV":
            out_i32 = q
        elif prog["host_op"] == "MOD":
            out_i32 = lhs_i32 - q * rhs_i32
        else:
            raise RuntimeError(f"unknown TinyTPU host binary op {prog['host_op']}")
        out_buf[: len(out_i32) * _BYTES_PER_ELEM] = np.asarray(out_i32, dtype="<i4").tobytes()
        return 1e-3

    def _exec_host_unary(self, bufs):
        prog = self.prog
        out_buf = bufs[prog["out"]]
        num_elems = int(prog["num_elems"])
        if prog.get("dtype") != "float32":
            raise RuntimeError(f"unsupported TinyTPU host unary dtype {prog.get('dtype')}")
        src_f32 = np.frombuffer(bytes(bufs[prog["src"]]), dtype="<f4")
        if src_f32.size != num_elems:
            raise RuntimeError(f"TinyTPU host unary op expected {num_elems} elements, got src={src_f32.size}")
        if prog["host_op"] == "TRUNC":
            out_f32 = np.trunc(src_f32).astype(np.float32)
        elif prog["host_op"] == "RECIPROCAL":
            out_f32 = np.reciprocal(src_f32.astype(np.float32))
        else:
            raise RuntimeError(f"unknown TinyTPU host unary op {prog['host_op']}")
        out_buf[: len(out_f32) * 4] = np.asarray(out_f32, dtype="<f4").tobytes()
        return 1e-3



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

    def _exec_gemm4x4(self, bufs):
        prog = self.prog
        out_buf    = bufs[prog["out"]]
        act_buf    = bufs[prog["act"]]
        weight_buf = bufs[prog["weight"]]

        # Decode int32 data from the raw bytearrays
        act_i32    = np.frombuffer(bytes(act_buf),    dtype="<i4")
        weight_i32 = np.frombuffer(bytes(weight_buf), dtype="<i4")
        num_vecs   = int(prog.get("num_vecs", max(1, act_i32.size // _ROWS)))
        num_k_tiles = int(prog.get("num_k_tiles", max(1, act_i32.size // max(1, num_vecs * _ROWS))))
        num_weight_tiles = int(prog.get("num_weight_tiles", max(1, weight_i32.size // (_ROWS * _COLS))))
        out_cols = num_weight_tiles * _COLS
        k_cols = num_k_tiles * _ROWS

        if act_i32.size != num_vecs * k_cols:
            raise RuntimeError(f"TinyTPU activation buffer size {act_i32.size} does not match shape=({num_vecs}, {k_cols})")
        if weight_i32.size != num_k_tiles * num_weight_tiles * _ROWS * _COLS:
            raise RuntimeError(f"TinyTPU weight buffer size {weight_i32.size} does not match tiling=({num_k_tiles}, {num_weight_tiles})")
        if len(out_buf) < num_vecs * out_cols * _BYTES_PER_ELEM:
            raise RuntimeError(f"TinyTPU output buffer too small for shape=({num_vecs}, {out_cols})")

        _require_int8_range("weight", weight_i32)
        _require_int8_range("activation", act_i32)

        # Downcast to int8 (hardware operand type)
        weight_matrix = weight_i32.reshape(k_cols, out_cols).astype(np.int8)
        act_rows = act_i32.reshape(num_vecs, k_cols).astype(np.int8)

        # Parse epilogue
        epilogue = prog.get("epilogue", [])
        hw_bias: np.ndarray | None = None
        hw_relu = False
        for step in epilogue:
            if step["op"] == "ADD":
                hw_bias = np.frombuffer(bytes(bufs[int(step["arg"])]), dtype="<i4")
            elif step["op"] == "RELU":
                hw_relu = True

        # Build one bundle for the entire GEMM (all rows × tiles)
        bundle = _build_full_gemm_bundle(act_rows, weight_matrix,
                                         num_vecs, num_k_tiles, num_weight_tiles,
                                         bias_i32=hw_bias, relu=hw_relu)
        stdout = self._run(bundle)
        vmem_results = _parse_multi_vmem_output(stdout)

        expected_tiles = num_vecs * num_weight_tiles
        if len(vmem_results) != expected_tiles:
            raise RuntimeError(
                f"TinyTPU full GEMM expected {expected_tiles} vmem_result lines, got {len(vmem_results)}\n"
                f"stdout: {stdout[:500]}")

        # Assemble output from VMEM tiles (row-major: row0_tile0, row0_tile1, ..., row1_tile0, ...)
        out_i32 = np.empty(num_vecs * out_cols, dtype="<i4")
        for row in range(num_vecs):
            for tile_idx in range(num_weight_tiles):
                tile_data = vmem_results[row * num_weight_tiles + tile_idx]
                col_base = row * out_cols + tile_idx * _COLS
                out_i32[col_base : col_base + _COLS] = tile_data[:_COLS]

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
