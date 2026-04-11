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
_TILE_ELEMS = _ROWS * _COLS   # 16 elements per VMEM tile
_VPU_OPS = {"ADD": 0, "MUL": 1, "MAX": 3, "CMPLT": 5, "CMPNE": 6, "SUB": 7, "CMPEQ": 8, "MAX_REDUCE": 9, "SHL": 10, "SHR": 11, "MIN": 12, "MIN_REDUCE": 13, "DIV": 14, "AND": 15, "OR": 16, "XOR": 17}
_VPU_BOOL_OPS = {_VPU_OPS["CMPLT"], _VPU_OPS["CMPNE"], _VPU_OPS["CMPEQ"]}
_SXU_OPS = {"LOAD_VREG": 0, "STORE_VREG": 1, "DISPATCH_VPU": 2, "DISPATCH_XLU_BROADCAST": 3, "DISPATCH_MXU": 4, "WAIT_MXU": 5, "HALT": 6}

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
            if diag["kind"] == "gemm":
                return _dump_lowering(json.dumps({"op": "GEMM4x4",
                                                  "out": diag["out_arg"],
                                                  "act": diag["act_arg"],
                                                  "weight": diag["weight_arg"],
                                                  "num_vecs": diag["num_vecs"],
                                                  "num_k_tiles": diag["num_k_tiles"],
                                                  "num_weight_tiles": diag["num_weight_tiles"]}))
            if diag["kind"] == "vpu_binary":
                return _dump_lowering(json.dumps({"op": "VPU_BINARY",
                                                  "vpu_op": diag["vpu_op"],
                                                  "out": diag["out_arg"],
                                                  "lhs": diag["lhs_arg"],
                                                  "lhs_const": diag["lhs_const"],
                                                  "lhs_broadcast": diag.get("lhs_broadcast", False),
                                                  "rhs": diag["rhs_arg"],
                                                  "rhs_const": diag["rhs_const"],
                                                  "rhs_broadcast": diag.get("rhs_broadcast", False),
                                                  "num_elems": diag["num_elems"],
                                                  "bool_out": diag.get("bool_out", False),
                                                  "bool_in": diag.get("bool_in", False)}))
            if diag["kind"] == "vpu_unary":
                return _dump_lowering(json.dumps({"op": "VPU_UNARY",
                                                  "vpu_op": diag["vpu_op"],
                                                  "out": diag["out_arg"],
                                                  "src": diag["src_arg"],
                                                  "num_elems": diag["num_elems"],
                                                  "out_elems": diag["out_elems"]}))
            if diag["kind"] == "vpu_where":
                return _dump_lowering(json.dumps({"op": "VPU_WHERE",
                                                  "out": diag["out_arg"],
                                                  "cond": diag["cond_arg"],
                                                  "lhs": diag["lhs_arg"],
                                                  "rhs": diag["rhs_arg"],
                                                  "num_elems": diag["num_elems"]}))
            if diag["kind"] == "vpu_rowsum":
                return _dump_lowering(json.dumps({"op": "VPU_ROWSUM",
                                                  "out": diag["out_arg"],
                                                  "src": diag["src_arg"],
                                                  "num_rows": diag["num_rows"],
                                                  "num_cols": diag["num_cols"],
                                                  "vpu_op": diag["vpu_op"]}))
            if diag["kind"] == "vpu_program":
                return _dump_lowering(json.dumps({"op": "VPU_PROGRAM",
                                                  "out": diag["out_arg"],
                                                  "num_elems": diag["num_elems"],
                                                  "inputs": diag["inputs"],
                                                  "steps": diag["steps"],
                                                  "output_reg": diag["output_reg"]}))
            if diag["kind"] == "host_binary":
                return _dump_lowering(json.dumps({"op": "HOST_BINARY",
                                                  "host_op": diag["host_op"],
                                                  "out": diag["out_arg"],
                                                  "lhs": diag["lhs_arg"],
                                                  "lhs_const": diag["lhs_const"],
                                                  "rhs": diag["rhs_arg"],
                                                  "rhs_const": diag["rhs_const"],
                                                  "num_elems": diag["num_elems"]}))
            if diag["kind"] == "host_unary":
                return _dump_lowering(json.dumps({"op": "HOST_UNARY",
                                                  "host_op": diag["host_op"],
                                                  "dtype": diag["host_dtype"],
                                                  "out": diag["out_arg"],
                                                  "src": diag["src_arg"],
                                                  "num_elems": diag["num_elems"]}))
        return _dump_lowering(json.dumps({
            "op": "UNSUPPORTED",
            "reason": diag["reason"],
            "missing_instructions": diag["missing_instructions"],
            "notes": diag["notes"],
            "op_counts": diag["op_counts"],
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

def analyze_tinytpu_uops(uops:list[UOp]) -> dict:
    params = [u for u in uops if u.op is Ops.PARAM]
    op_counts = Counter(u.op.name for u in uops)
    has_mulacc = any(u.op is Ops.MULACC for u in uops)
    has_mul = any(u.op is Ops.MUL for u in uops)
    has_range = any(u.op is Ops.RANGE for u in uops)
    has_store = any(u.op is Ops.STORE for u in uops)
    is_gemm = has_mulacc or (len(params) == 3 and has_mul and has_range and has_store)

    diag = {
        "supported": False,
        "kind": None,
        "reason": "",
        "missing_instructions": [],
        "notes": [],
        "op_counts": dict(sorted(op_counts.items())),
        "out_arg": None, "act_arg": None, "weight_arg": None,
        "src_arg": None,
        "lhs_arg": None, "lhs_const": None, "rhs_arg": None,
        "rhs_const": None,
        "num_elems": None,
        "out_elems": None,
        "vpu_op": None,
        "inputs": None,
        "steps": None,
        "output_reg": None,
        "host_op": None,
        "host_dtype": None,
        "lhs_broadcast": False,
        "rhs_broadcast": False,
        "num_vecs": None,
        "num_k_tiles": None,
        "num_weight_tiles": None,
    }

    param_sizes: dict[int, int] = {}
    for p in params:
        if not isinstance(p.dtype, PtrDType):
            diag["reason"] = "non-ptr param"
            diag["notes"].append("TinyTPU kernels currently expect pointer-backed buffers only.")
            return diag
        param_sizes[p.arg] = p.dtype.size

    if len(param_sizes) == 1 and has_range and has_store and op_counts.get("GROUP", 0) == 1 and op_counts.get("MUL", 0) > 0:
        diag["reason"] = "zero-sized gemm"
        diag["notes"].append("Zero-sized GEMM buffers are not lowered through the current TinyTPU path.")
        diag["missing_instructions"] = ["SXU_DISPATCH_VPU", "SXU_LOAD_VREG", "SXU_STORE_VREG"]
        return diag

    if is_gemm and (len(param_sizes) == 1 or (param_sizes and all(sz == 0 for sz in param_sizes.values()))):
        diag["reason"] = "zero-sized gemm"
        diag["notes"].append("Zero-sized GEMM buffers are not lowered through the current TinyTPU path.")
        diag["missing_instructions"] = ["SXU_DISPATCH_VPU", "SXU_LOAD_VREG", "SXU_STORE_VREG"]
        return diag

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
    _is_rowwise = (len(params) == 2
                   and op_counts.get("RANGE", 0) == 1
                   and op_counts.get("MUL", 0) == 1
                   and op_counts.get("LOAD", 0) == 4
                   and op_counts.get("STORE", 0) == 1
                   and param_sizes.get(1) is not None
                   and param_sizes.get(0) is not None
                   and 1 < param_sizes.get(0, 0)
                   and param_sizes.get(1) == param_sizes.get(0, 0) * _COLS)
    if _is_rowwise:
        nrows = param_sizes[0]
        # Discriminate by the reduction operator in the loop body:
        # sum:  ADD > 3 (3 for addressing + 3 for summing), no MAX
        # max:  MAX == _COLS-1 (tree of comparisons), no XOR
        # min:  MAX == _COLS-1 and XOR > 0 (tinygrad encodes min as XOR+MAX)
        if op_counts.get("ADD", 0) > _COLS - 1 and op_counts.get("MAX", 0) == 0:
            row_vpu_op = 4  # SUM_REDUCE
            row_reason = f"supported row-wise sum {nrows}x{_COLS}"
        elif op_counts.get("MAX", 0) >= _COLS - 1 and op_counts.get("XOR", 0) == 0:
            row_vpu_op = _VPU_OPS["MAX_REDUCE"]
            row_reason = f"supported row-wise max {nrows}x{_COLS}"
        elif op_counts.get("MAX", 0) >= _COLS - 1 and op_counts.get("XOR", 0) > 0:
            row_vpu_op = _VPU_OPS["MIN_REDUCE"]
            row_reason = f"supported row-wise min {nrows}x{_COLS}"
        else:
            row_vpu_op = None
            row_reason = None
        if row_vpu_op is not None:
            diag.update({
                "supported": True,
                "kind": "vpu_rowsum",
                "reason": row_reason,
                "out_arg": 0,
                "src_arg": 1,
                "num_rows": nrows,
                "num_cols": _COLS,
                "vpu_op": row_vpu_op,
            })
            return diag
    if len(params) == 2 and op_counts.get("STORE", 0) == 1 and op_counts.get("ADD", 0) > 0 and param_sizes.get(0) == 1:
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        num_adds = op_counts.get("ADD", 0)
        num_loads = op_counts.get("LOAD", 0)
        is_sum_tree = (src_size is not None and src_size > 0
                       and num_adds == src_size - 1 and num_loads == src_size)
        if is_sum_tree:
            diag.update({
                "supported": True,
                "kind": "vpu_unary",
                "reason": "supported vpu sum_reduce",
                "out_arg": 0,
                "src_arg": 1,
                "num_elems": src_size,
                "out_elems": out_size,
                "vpu_op": 4,
            })
            return diag
        diag["reason"] = f"unsupported vpu sum_reduce sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU SUM_REDUCE lowering handles int32 sum reduction to scalar.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 2 and op_counts.get("STORE", 0) == 1 and op_counts.get("MAX", 0) > 0
          and op_counts.get("XOR", 0) == 0 and param_sizes.get(0) == 1):
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        num_maxes = op_counts.get("MAX", 0)
        num_loads = op_counts.get("LOAD", 0)
        is_max_tree = (src_size is not None and src_size > 0
                       and num_maxes == src_size - 1 and num_loads == src_size)
        if is_max_tree:
            diag.update({
                "supported": True,
                "kind": "vpu_unary",
                "reason": "supported vpu max_reduce",
                "out_arg": 0,
                "src_arg": 1,
                "num_elems": src_size,
                "out_elems": out_size,
                "vpu_op": _VPU_OPS["MAX_REDUCE"],
            })
            return diag
        diag["reason"] = f"unsupported vpu max_reduce sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU MAX_REDUCE lowering handles int32 max reduction to scalar.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 2 and op_counts.get("STORE", 0) == 1 and op_counts.get("MAX", 0) > 0
          and op_counts.get("XOR", 0) > 0 and param_sizes.get(0) == 1):
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        num_maxes = op_counts.get("MAX", 0)
        num_loads = op_counts.get("LOAD", 0)
        is_min_tree = (src_size is not None and src_size > 0
                       and num_maxes == src_size - 1 and num_loads == src_size)
        if is_min_tree:
            diag.update({
                "supported": True,
                "kind": "vpu_unary",
                "reason": "supported vpu min_reduce",
                "out_arg": 0,
                "src_arg": 1,
                "num_elems": src_size,
                "out_elems": out_size,
                "vpu_op": _VPU_OPS["MIN_REDUCE"],
            })
            return diag
        diag["reason"] = f"unsupported vpu min_reduce sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU MIN_REDUCE lowering handles int32 min reduction to scalar.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 2 and op_counts.get("XOR", 0) > 0 and op_counts.get("MAX", 0) > 0
          and not in_is_bool and op_counts.get("STORE", 0) >= 1):
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        min_const = _find_min_scalar_const(uops)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < src_size and min_const is not None:
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu min const",
                "out_arg": 0,
                "lhs_arg": 1,
                "lhs_const": None,
                "rhs_arg": None,
                "rhs_const": min_const,
                "num_elems": src_size,
                "vpu_op": binary_vpu_ops["MIN"],
            })
            return diag
        diag["reason"] = f"unsupported vpu min const sizes {dict(sorted(param_sizes.items()))}"
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 2 and op_counts.get("CMPLT", 0) > 0 and op_counts.get("WHERE", 0) > 0
          and not _has_complex_op
          and op_counts.get("WHERE", 0) == op_counts.get("CMPLT", 0)
          and op_counts.get("WHERE", 0) <= op_counts.get("STORE", 0)):
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < src_size:
            diag.update({
                "supported": True,
                "kind": "vpu_unary",
                "reason": "supported vpu relu",
                "out_arg": 0,
                "src_arg": 1,
                "num_elems": src_size,
                "out_elems": out_size,
                "vpu_op": 2,
            })
            return diag
        diag["reason"] = f"unsupported vpu relu sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU RELU lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif len(params) in {2, 3} and divmod_pattern is not None and divmod_pattern[0] == "IDIV":
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
    elif len(params) == 4 and op_counts.get("WHERE", 0) > 0:
        out_size = param_sizes.get(0)
        input_args = [arg for arg in sorted(param_sizes) if arg != 0]
        if out_size is not None and len(input_args) == 3 and 0 < out_size and all(param_sizes[arg] in {1, out_size} for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_where",
                "reason": "supported vpu where",
                "out_arg": 0,
                "cond_arg": input_args[0],
                "lhs_arg": input_args[1],
                "rhs_arg": input_args[2],
                "num_elems": out_size,
            })
            return diag
        diag["reason"] = f"unsupported vpu where sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU WHERE lowering handles int32 VMEM tiles.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 2 and op_counts.get("TRUNC", 0) > 0 and "float" in str(params[0].dtype) and "float" in str(params[1].dtype)):
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < src_size:
            diag.update({
                "supported": True,
                "kind": "host_unary",
                "reason": "supported host trunc fallback",
                "host_op": "TRUNC",
                "host_dtype": "float32",
                "out_arg": 0,
                "src_arg": 1,
                "num_elems": src_size,
            })
            return diag
    elif (len(params) == 2 and op_counts.get("RECIPROCAL", 0) > 0 and "float" in str(params[0].dtype) and "float" in str(params[1].dtype)):
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < src_size:
            diag.update({
                "supported": True,
                "kind": "host_unary",
                "reason": "supported host reciprocal fallback",
                "host_op": "RECIPROCAL",
                "host_dtype": "float32",
                "out_arg": 0,
                "src_arg": 1,
                "num_elems": src_size,
            })
            return diag
    elif len(params) == 3 and is_gemm and has_store:
        sizes = sorted(param_sizes.values())
        candidate_weights = [arg for arg, sz in param_sizes.items() if sz >= 16 and sz % 16 == 0]
        for weight_arg in candidate_weights:
            weight_size = param_sizes[weight_arg]
            non_weight = {arg: sz for arg, sz in param_sizes.items() if arg != weight_arg}
            out_arg = 0
            if out_arg not in non_weight:
                continue
            act_arg = next((arg for arg in non_weight if arg != out_arg), None)
            if act_arg is None:
                continue
            out_size = non_weight.get(out_arg)
            act_size = non_weight.get(act_arg)
            tiling = _infer_tiling(out_size, act_size, weight_size)
            if tiling is not None:
                num_vecs, num_k_tiles, inferred_n_tiles = tiling
                diag.update({
                    "supported": True,
                    "kind": "gemm",
                    "reason": "supported gemm4x4",
                    "out_arg": out_arg,
                    "act_arg": act_arg,
                    "weight_arg": weight_arg,
                    "num_vecs": num_vecs,
                    "num_k_tiles": num_k_tiles,
                    "num_weight_tiles": inferred_n_tiles,
                })
                return diag
        diag["reason"] = f"unexpected param sizes {sizes}"
        diag["notes"].append("Current TinyTPU backend only handles int32 matmul cases whose flattened buffers can be factored into MxK, KxN, and MxN with K and N tiled in groups of 4.")
        if len(candidate_weights) == 1:
            weight_arg = candidate_weights[0]
            non_weight = {arg: sz for arg, sz in param_sizes.items() if arg != weight_arg}
            out_size = non_weight.get(0)
            act_arg = next((arg for arg in non_weight if arg != 0), None)
            act_size = non_weight.get(act_arg) if act_arg is not None else None
            diag["notes"].append(_tiling_failure_note(out_size, act_size, param_sizes[weight_arg]))
        else:
            out_size = param_sizes.get(0)
            remaining = {arg: sz for arg, sz in param_sizes.items() if arg != 0}
            if len(remaining) == 2:
                by_size = sorted(remaining.items(), key=lambda item: item[1])
                act_size = by_size[0][1]
                weight_size = by_size[1][1]
                diag["notes"].append(_tiling_failure_note(out_size, act_size, weight_size))
            diag["notes"].append("No buffer looked like a valid 4x4-tiled weight matrix.")
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
    lines.append(f"2 {_SXU_OPS['DISPATCH_MXU']} 0 0 0 0 0 0 1 1")
    # Record 2: SXU_WAIT_MXU (opcode=4)
    lines.append(f"2 {_SXU_OPS['WAIT_MXU']} 0 0 0 0 0 0 0 0")
    # Record 2: SXU_HALT (opcode=5)
    lines.append(f"2 {_SXU_OPS['HALT']} 0 0 0 0 0 0 0 0")
    # Record 3: OUTPUT_MXU = 1
    lines.append("3 1")
    # Record 4: END
    lines.append("4")
    return "\n".join(lines) + "\n"


def _build_vpu_binary_bundle(lhs_i32: np.ndarray, rhs_i32: np.ndarray, num_elems: int, vpu_op: int,
                             lhs_broadcast: bool = False, rhs_broadcast: bool = False) -> str:
    def tile(vals: np.ndarray) -> list[int]:
        padded = np.zeros(_ROWS * _COLS, dtype=np.int32)
        padded[:num_elems] = vals[:num_elems]
        return [int(x) for x in padded]

    lines: list[str] = []
    lines.append("5 0 " + " ".join(str(x) for x in tile(lhs_i32)))
    lines.append("5 1 " + " ".join(str(x) for x in tile(rhs_i32)))
    # LOAD VMEM[0]->v0, optional XLU broadcast, LOAD VMEM[1]->v1,
    # optional XLU broadcast, VPU op v0/v1->v2, STORE v2->VMEM[2]
    lines.append(f"2 {_SXU_OPS['LOAD_VREG']} 0 0 0 0 0 0 0 0")
    if lhs_broadcast:
        lines.append(f"2 {_SXU_OPS['DISPATCH_XLU_BROADCAST']} 0 0 0 0 0 0 0 0")
    lines.append(f"2 {_SXU_OPS['LOAD_VREG']} 1 1 0 0 0 0 0 0")
    if rhs_broadcast:
        lines.append(f"2 {_SXU_OPS['DISPATCH_XLU_BROADCAST']} 0 1 1 0 0 0 0 0")
    lines.append(f"2 {_SXU_OPS['DISPATCH_VPU']} 0 2 0 {vpu_op} 1 0 0 0")
    lines.append(f"2 {_SXU_OPS['STORE_VREG']} 2 0 2 0 0 0 0 0")
    lines.append(f"2 {_SXU_OPS['HALT']} 0 0 0 0 0 0 0 0")
    lines.append("6 2")
    lines.append("4")
    return "\n".join(lines) + "\n"


def _build_vpu_unary_bundle(src_i32: np.ndarray, num_elems: int, vpu_op: int) -> str:
    padded = np.zeros(_ROWS * _COLS, dtype=np.int32)
    padded[:num_elems] = src_i32[:num_elems]

    lines: list[str] = []
    lines.append("5 0 " + " ".join(str(int(x)) for x in padded))
    # LOAD VMEM[0]->v0, VPU unary op v0->v1, STORE v1->VMEM[2]
    lines.append(f"2 {_SXU_OPS['LOAD_VREG']} 0 0 0 0 0 0 0 0")
    lines.append(f"2 {_SXU_OPS['DISPATCH_VPU']} 0 1 0 {vpu_op} 0 0 0 0")
    lines.append(f"2 {_SXU_OPS['STORE_VREG']} 2 0 1 0 0 0 0 0")
    lines.append(f"2 {_SXU_OPS['HALT']} 0 0 0 0 0 0 0 0")
    lines.append("6 2")
    lines.append("4")
    return "\n".join(lines) + "\n"


def _build_vpu_where_bundle(cond_i32: np.ndarray, lhs_i32: np.ndarray, rhs_i32: np.ndarray, num_elems: int) -> str:
    """WHERE(cond, lhs, rhs) = cond*lhs + (1-cond)*rhs via multi-instruction VPU bundle."""
    def tile(vals: np.ndarray) -> list[int]:
        padded = np.zeros(_ROWS * _COLS, dtype=np.int32)
        padded[:num_elems] = vals[:num_elems]
        return [int(x) for x in padded]

    ones = [1] * _ROWS * _COLS
    lines: list[str] = []
    # VMEM[0]=cond, VMEM[1]=lhs, VMEM[2]=rhs, VMEM[3]=ones
    lines.append("5 0 " + " ".join(str(x) for x in tile(cond_i32)))
    lines.append("5 1 " + " ".join(str(x) for x in tile(lhs_i32)))
    lines.append("5 2 " + " ".join(str(x) for x in tile(rhs_i32)))
    lines.append("5 3 " + " ".join(str(x) for x in ones))
    # LOAD cond->v0, lhs->v1, rhs->v2, ones->v3
    lines.append(f"2 {_SXU_OPS['LOAD_VREG']} 0 0 0 0 0 0 0 0")   # LOAD VMEM[0]->v0
    lines.append(f"2 {_SXU_OPS['LOAD_VREG']} 1 1 0 0 0 0 0 0")   # LOAD VMEM[1]->v1
    lines.append(f"2 {_SXU_OPS['LOAD_VREG']} 2 2 0 0 0 0 0 0")   # LOAD VMEM[2]->v2
    lines.append(f"2 {_SXU_OPS['LOAD_VREG']} 3 3 0 0 0 0 0 0")   # LOAD VMEM[3]->v3
    # v4 = cond * lhs (MUL v0, v1 -> v4)
    lines.append(f"2 {_SXU_OPS['DISPATCH_VPU']} 0 4 0 {_VPU_OPS['MUL']} 1 0 0 0")
    # v5 = 1 - cond (SUB v3, v0 -> v5)
    lines.append(f"2 {_SXU_OPS['DISPATCH_VPU']} 0 5 3 {_VPU_OPS['SUB']} 0 0 0 0")
    # v6 = (1-cond) * rhs (MUL v5, v2 -> v6)
    lines.append(f"2 {_SXU_OPS['DISPATCH_VPU']} 0 6 5 {_VPU_OPS['MUL']} 2 0 0 0")
    # v7 = cond*lhs + (1-cond)*rhs (ADD v4, v6 -> v7)
    lines.append(f"2 {_SXU_OPS['DISPATCH_VPU']} 0 7 4 {_VPU_OPS['ADD']} 6 0 0 0")
    # STORE v7 -> VMEM[4]
    lines.append(f"2 {_SXU_OPS['STORE_VREG']} 4 0 7 0 0 0 0 0")
    lines.append(f"2 {_SXU_OPS['HALT']} 0 0 0 0 0 0 0 0")   # HALT
    lines.append("6 4")   # output VMEM[4]
    lines.append("4")     # END
    return "\n".join(lines) + "\n"


def _build_vpu_program_bundle(inputs: list[np.ndarray], num_elems: int, steps: list[dict], output_reg: int,
                              input_broadcasts: list[bool] | None = None) -> str:
    def tile(vals: np.ndarray) -> list[int]:
        padded = np.zeros(_TILE_ELEMS, dtype=np.int32)
        padded[:num_elems] = vals[:num_elems]
        return [int(x) for x in padded]

    lines: list[str] = []
    if input_broadcasts is None:
        input_broadcasts = [False] * len(inputs)
    for idx, vals in enumerate(inputs):
        lines.append("5 " + str(idx) + " " + " ".join(str(x) for x in tile(vals)))
    for idx in range(len(inputs)):
        lines.append(f"2 {_SXU_OPS['LOAD_VREG']} {idx} {idx} 0 0 0 0 0 0")
        if input_broadcasts[idx]:
            lines.append(f"2 {_SXU_OPS['DISPATCH_XLU_BROADCAST']} 0 {idx} {idx} 0 0 0 0 0")
    for step in steps:
        lhs = int(step["lhs"])
        dst = int(step["dst"])
        rhs = int(step.get("rhs", 0))
        rhs_en = 1 if "rhs" in step else 0
        lines.append(f"2 {_SXU_OPS['DISPATCH_VPU']} 0 {dst} {lhs} {int(step['op'])} {rhs} {rhs_en} 0 0")
    lines.append(f"2 {_SXU_OPS['STORE_VREG']} {len(inputs)} 0 {output_reg} 0 0 0 0 0")
    lines.append(f"2 {_SXU_OPS['HALT']} 0 0 0 0 0 0 0 0")
    lines.append(f"6 {len(inputs)}")
    lines.append("4")
    return "\n".join(lines) + "\n"


def _parse_sim_output(stdout: str) -> list[int] | None:
    """Extract mxu_result from BSV sim stdout.  Returns None if not found."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("mxu_result "):
            vals = line.split()[1:]
            if len(vals) != _COLS:
                raise ValueError(f"mxu_result expects {_COLS} values, got {len(vals)}")
            try:
                return [int(x) for x in vals]
            except ValueError as exc:
                bad = next((x for x in vals if not x.lstrip("-").isdigit()), vals[0])
                raise ValueError(f"invalid mxu_result integer {bad!r}") from exc
    return None


def _parse_vmem_output(stdout: str) -> list[int] | None:
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("vmem_result "):
            vals = line.split()[1:]
            if len(vals) != _ROWS * _COLS:
                raise ValueError(f"vmem_result expects {_ROWS * _COLS} values, got {len(vals)}")
            try:
                return [int(x) for x in vals]
            except ValueError as exc:
                bad = next((x for x in vals if not x.lstrip("-").isdigit()), vals[0])
                raise ValueError(f"invalid vmem_result integer {bad!r}") from exc
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

    result = _parse_sim_output(proc.stdout)
    if result is None:
        raise RuntimeError(
            f"TinyTPU sim produced no mxu_result\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return result


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
        if prog.get("op") not in {"GEMM4x4", "VPU_BINARY", "VPU_UNARY", "VPU_WHERE", "VPU_PROGRAM", "VPU_ROWSUM", "HOST_BINARY", "HOST_UNARY"}:
            raise NotImplementedError(_unsupported_message(prog))

        if prog.get("op") == "VPU_BINARY":
            out_buf = bufs[prog["out"]]
            num_elems = int(prog["num_elems"])
            bool_inputs = prog.get("bool_in", False) or prog.get("bool_out", False)
            if prog.get("lhs_const") is None:
                lhs_raw = np.frombuffer(bytes(bufs[prog["lhs"]]), dtype=np.bool_ if bool_inputs else "<i4")
                lhs_i32 = lhs_raw.astype(np.int32) if bool_inputs else lhs_raw
            else:
                lhs_i32 = np.full(num_elems, int(prog["lhs_const"]), dtype="<i4")
            if prog.get("rhs_const") is None:
                rhs_raw = np.frombuffer(bytes(bufs[prog["rhs"]]), dtype=np.bool_ if bool_inputs else "<i4")
                rhs_i32 = rhs_raw.astype(np.int32) if bool_inputs else rhs_raw
            else:
                rhs_i32 = np.full(num_elems, int(prog["rhs_const"]), dtype="<i4")
            lhs_broadcast = bool(prog.get("lhs_broadcast", False)) and prog.get("lhs_const") is None
            rhs_broadcast = bool(prog.get("rhs_broadcast", False)) and prog.get("rhs_const") is None
            if lhs_i32.size not in {1, num_elems} or rhs_i32.size not in {1, num_elems}:
                raise RuntimeError(f"TinyTPU VPU binary op expected {num_elems} elements, got lhs={lhs_i32.size} rhs={rhs_i32.size}")
            is_bool = int(prog["vpu_op"]) in _VPU_BOOL_OPS or prog.get("bool_out", False)
            out_elem_bytes = 1 if is_bool else _BYTES_PER_ELEM
            if len(out_buf) < num_elems * out_elem_bytes:
                raise RuntimeError(f"TinyTPU output buffer too small for VPU binary op elements={num_elems}")
            sim = _sim_path()
            vpu_op = int(prog["vpu_op"])
            out_offset = 0
            for chunk_start in range(0, num_elems, _TILE_ELEMS):
                chunk_end = min(chunk_start + _TILE_ELEMS, num_elems)
                chunk_size = chunk_end - chunk_start
                lhs_chunk = lhs_i32[:1] if lhs_broadcast else lhs_i32[chunk_start:chunk_end]
                rhs_chunk = rhs_i32[:1] if rhs_broadcast else rhs_i32[chunk_start:chunk_end]
                stdout = _run_bundle(sim, _build_vpu_binary_bundle(lhs_chunk, rhs_chunk, chunk_size, vpu_op,
                                                                   lhs_broadcast=lhs_broadcast, rhs_broadcast=rhs_broadcast))
                result = _parse_vmem_output(stdout)
                if result is None:
                    raise RuntimeError(f"TinyTPU sim produced no vmem_result\nstdout: {stdout}")
                if is_bool:
                    chunk_out = np.array(result[:chunk_size], dtype=np.bool_)
                    out_buf[out_offset : out_offset + len(chunk_out)] = chunk_out.tobytes()
                    out_offset += len(chunk_out)
                else:
                    chunk_out = np.array(result[:chunk_size], dtype="<i4")
                    out_buf[out_offset : out_offset + len(chunk_out) * _BYTES_PER_ELEM] = chunk_out.tobytes()
                    out_offset += len(chunk_out) * _BYTES_PER_ELEM
            return 1e-3

        if prog.get("op") == "VPU_WHERE":
            out_buf = bufs[prog["out"]]
            num_elems = int(prog["num_elems"])
            cond_raw = np.frombuffer(bytes(bufs[prog["cond"]]), dtype=np.bool_)
            lhs_i32 = np.frombuffer(bytes(bufs[prog["lhs"]]), dtype="<i4")
            rhs_i32 = np.frombuffer(bytes(bufs[prog["rhs"]]), dtype="<i4")
            cond_i32 = cond_raw[:num_elems].astype(np.int32)
            if len(out_buf) < num_elems * _BYTES_PER_ELEM:
                raise RuntimeError(f"TinyTPU output buffer too small for VPU WHERE elements={num_elems}")
            sim = _sim_path()
            out_offset = 0
            for chunk_start in range(0, num_elems, _TILE_ELEMS):
                chunk_end = min(chunk_start + _TILE_ELEMS, num_elems)
                chunk_size = chunk_end - chunk_start
                stdout = _run_bundle(sim, _build_vpu_where_bundle(
                    cond_i32[chunk_start:chunk_end], lhs_i32[chunk_start:chunk_end],
                    rhs_i32[chunk_start:chunk_end], chunk_size))
                result = _parse_vmem_output(stdout)
                if result is None:
                    raise RuntimeError(f"TinyTPU sim produced no vmem_result\nstdout: {stdout}")
                chunk_out = np.array(result[:chunk_size], dtype="<i4")
                out_buf[out_offset : out_offset + len(chunk_out) * _BYTES_PER_ELEM] = chunk_out.tobytes()
                out_offset += len(chunk_out) * _BYTES_PER_ELEM
            return 1e-3

        if prog.get("op") == "VPU_PROGRAM":
            out_buf = bufs[prog["out"]]
            num_elems = int(prog["num_elems"])
            if len(out_buf) < num_elems * _BYTES_PER_ELEM:
                raise RuntimeError(f"TinyTPU output buffer too small for VPU program elements={num_elems}")
            sim = _sim_path()
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
                stdout = _run_bundle(sim, _build_vpu_program_bundle(input_tiles, chunk_size, prog["steps"], int(prog["output_reg"]),
                                                                    input_broadcasts=input_broadcasts))
                result = _parse_vmem_output(stdout)
                if result is None:
                    raise RuntimeError(f"TinyTPU sim produced no vmem_result\nstdout: {stdout}")
                chunk_out = np.array(result[:chunk_size], dtype="<i4")
                out_buf[out_offset : out_offset + len(chunk_out) * _BYTES_PER_ELEM] = chunk_out.tobytes()
                out_offset += len(chunk_out) * _BYTES_PER_ELEM
            return 1e-3

        if prog.get("op") == "HOST_BINARY":
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

        if prog.get("op") == "HOST_UNARY":
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

        if prog.get("op") == "VPU_UNARY":
            out_buf = bufs[prog["out"]]
            src_i32 = np.frombuffer(bytes(bufs[prog["src"]]), dtype="<i4")
            num_elems = int(prog["num_elems"])
            out_elems = int(prog["out_elems"])
            if src_i32.size != num_elems:
                raise RuntimeError(f"TinyTPU VPU unary op expected {num_elems} elements, got src={src_i32.size}")
            if len(out_buf) < out_elems * _BYTES_PER_ELEM:
                raise RuntimeError(f"TinyTPU output buffer too small for VPU unary op elements={out_elems}")
            sim = _sim_path()
            vpu_op = int(prog["vpu_op"])
            is_sum_reduce = vpu_op == 4 and out_elems == 1
            is_max_reduce = vpu_op == _VPU_OPS["MAX_REDUCE"] and out_elems == 1
            is_min_reduce = vpu_op == _VPU_OPS["MIN_REDUCE"] and out_elems == 1
            if is_sum_reduce:
                # Sum reduction: chunk into tiles, sum each via VPU_SUM_REDUCE,
                # then accumulate partial sums on the host.
                total = np.int32(0)
                for chunk_start in range(0, num_elems, _TILE_ELEMS):
                    chunk_end = min(chunk_start + _TILE_ELEMS, num_elems)
                    chunk_size = chunk_end - chunk_start
                    src_chunk = src_i32[chunk_start:chunk_end]
                    # Pad chunk to 4 elements minimum for VPU_SUM_REDUCE row
                    padded = np.zeros(_TILE_ELEMS, dtype=np.int32)
                    padded[:chunk_size] = src_chunk
                    stdout = _run_bundle(sim, _build_vpu_unary_bundle(padded, _TILE_ELEMS, vpu_op))
                    result = _parse_vmem_output(stdout)
                    if result is None:
                        raise RuntimeError(f"TinyTPU sim produced no vmem_result\nstdout: {stdout}")
                    # VPU_SUM_REDUCE broadcasts row sums; sum the 4 row sums
                    row_sums = [result[r * _COLS] for r in range(_ROWS)]
                    total += np.int32(sum(row_sums))
                out_i32 = np.array([total], dtype="<i4")
                out_buf[: _BYTES_PER_ELEM] = out_i32.tobytes()
            elif is_max_reduce:
                # Max reduction: chunk into tiles, max each via VPU_MAX_REDUCE,
                # then take the running max across tiles on the host.
                import sys
                running_max = np.int32(-2**31)  # INT32_MIN
                for chunk_start in range(0, num_elems, _TILE_ELEMS):
                    chunk_end = min(chunk_start + _TILE_ELEMS, num_elems)
                    chunk_size = chunk_end - chunk_start
                    src_chunk = src_i32[chunk_start:chunk_end]
                    padded = np.full(_TILE_ELEMS, -2**31, dtype=np.int32)
                    padded[:chunk_size] = src_chunk
                    stdout = _run_bundle(sim, _build_vpu_unary_bundle(padded, _TILE_ELEMS, vpu_op))
                    result = _parse_vmem_output(stdout)
                    if result is None:
                        raise RuntimeError(f"TinyTPU sim produced no vmem_result\nstdout: {stdout}")
                    # VPU_MAX_REDUCE broadcasts row maxes; take max of all row maxes
                    row_maxes = [np.int32(result[r * _COLS]) for r in range(_ROWS)]
                    tile_max = max(row_maxes)
                    running_max = max(running_max, tile_max)
                out_i32 = np.array([running_max], dtype="<i4")
                out_buf[: _BYTES_PER_ELEM] = out_i32.tobytes()
            elif is_min_reduce:
                running_min = np.int32(2**31 - 1)  # INT32_MAX
                for chunk_start in range(0, num_elems, _TILE_ELEMS):
                    chunk_end = min(chunk_start + _TILE_ELEMS, num_elems)
                    chunk_size = chunk_end - chunk_start
                    src_chunk = src_i32[chunk_start:chunk_end]
                    padded = np.full(_TILE_ELEMS, 2**31 - 1, dtype=np.int32)
                    padded[:chunk_size] = src_chunk
                    stdout = _run_bundle(sim, _build_vpu_unary_bundle(padded, _TILE_ELEMS, vpu_op))
                    result = _parse_vmem_output(stdout)
                    if result is None:
                        raise RuntimeError(f"TinyTPU sim produced no vmem_result\nstdout: {stdout}")
                    row_mins = [np.int32(result[r * _COLS]) for r in range(_ROWS)]
                    tile_min = min(row_mins)
                    running_min = min(running_min, tile_min)
                out_i32 = np.array([running_min], dtype="<i4")
                out_buf[: _BYTES_PER_ELEM] = out_i32.tobytes()
            else:
                out_offset = 0
                for chunk_start in range(0, num_elems, _TILE_ELEMS):
                    chunk_end = min(chunk_start + _TILE_ELEMS, num_elems)
                    chunk_size = chunk_end - chunk_start
                    out_chunk_size = min(out_elems - (chunk_start if out_elems == num_elems else 0), chunk_size)
                    src_chunk = src_i32[chunk_start:chunk_end]
                    stdout = _run_bundle(sim, _build_vpu_unary_bundle(src_chunk, chunk_size, vpu_op))
                    result = _parse_vmem_output(stdout)
                    if result is None:
                        raise RuntimeError(f"TinyTPU sim produced no vmem_result\nstdout: {stdout}")
                    chunk_out = np.array(result[:out_chunk_size], dtype="<i4")
                    out_buf[out_offset : out_offset + len(chunk_out) * _BYTES_PER_ELEM] = chunk_out.tobytes()
                    out_offset += len(chunk_out) * _BYTES_PER_ELEM
            return 1e-3

        if prog.get("op") == "VPU_ROWSUM":
            # Row-wise sum: for each row of a (nrows × ncols) tile, sum all
            # columns using VPU_SUM_REDUCE (which broadcasts the row sum to all
            # lane positions). Extract position 0 of each row from vmem_result.
            out_buf  = bufs[prog["out"]]
            src_i32  = np.frombuffer(bytes(bufs[prog["src"]]), dtype="<i4")
            num_rows = int(prog["num_rows"])
            num_cols = int(prog["num_cols"])
            if src_i32.size < num_rows * num_cols:
                raise RuntimeError(
                    f"TinyTPU VPU_ROWSUM expected at least {num_rows*num_cols} elements, "
                    f"got {src_i32.size}")
            src_i32 = src_i32[:num_rows * num_cols]  # trim tinygrad buffer padding
            sim = _sim_path()
            row_vpu_op = int(prog.get("vpu_op", 4))  # default 4=SUM_REDUCE
            # Run VPU in _ROWS-row tiles; extract lane-0 per row across all tiles.
            all_row_results: list[int] = []
            for tile_start in range(0, num_rows, _ROWS):
                tile_end = min(tile_start + _ROWS, num_rows)
                tile_nrows = tile_end - tile_start
                tile_padded = np.zeros(_TILE_ELEMS, dtype=np.int32)
                tile_padded[:tile_nrows * num_cols] = src_i32[tile_start * num_cols:tile_end * num_cols]
                tile_stdout = _run_bundle(sim, _build_vpu_unary_bundle(tile_padded, _TILE_ELEMS, row_vpu_op))
                tile_result = _parse_vmem_output(tile_stdout)
                if tile_result is None:
                    raise RuntimeError(
                        f"TinyTPU VPU_ROWSUM tile {tile_start}: no vmem_result\nstdout: {tile_stdout}")
                all_row_results.extend(tile_result[r * _COLS] for r in range(tile_nrows))
            row_sums = np.array(all_row_results, dtype=np.int32)
            out_buf[:num_rows * _BYTES_PER_ELEM] = row_sums.tobytes()
            return 1e-3

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
        weight_matrix = weight_i32.reshape(k_cols, out_cols)
        sim = _sim_path()
        out_i32 = np.empty(num_vecs * out_cols, dtype="<i4")
        act_rows = act_i32.reshape(num_vecs, k_cols).astype(np.int8)
        for i, act_row in enumerate(act_rows):
            row_base = i * out_cols
            for tile_idx in range(num_weight_tiles):
                col_base = row_base + tile_idx * _COLS
                acc = np.zeros(_COLS, dtype=np.int32)
                for k_idx in range(num_k_tiles):
                    act_i8 = act_row[k_idx * _ROWS : (k_idx + 1) * _ROWS]
                    weight_i8 = weight_matrix[k_idx * _ROWS : (k_idx + 1) * _ROWS,
                                              tile_idx * _COLS : (tile_idx + 1) * _COLS].astype(np.int8)
                    acc += np.array(_run_gemm_vec(sim, weight_i8, act_i8), dtype=np.int32)
                out_i32[col_base : col_base + _COLS] = acc
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
