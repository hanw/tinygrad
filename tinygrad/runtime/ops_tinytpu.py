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
_VPU_OPS = {"ADD": 0, "MUL": 1, "MAX": 3, "CMPLT": 5, "CMPNE": 6, "SUB": 7, "CMPEQ": 8}
_VPU_BOOL_OPS = {_VPU_OPS["CMPLT"], _VPU_OPS["CMPNE"], _VPU_OPS["CMPEQ"]}

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
                                                  "rhs": diag["rhs_arg"],
                                                  "rhs_const": diag["rhs_const"],
                                                  "num_elems": diag["num_elems"]}))
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
    is_gemm = has_mulacc or (has_mul and has_range)

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

    if is_gemm and (len(param_sizes) == 1 or (param_sizes and all(sz == 0 for sz in param_sizes.values()))):
        diag["reason"] = "zero-sized gemm"
        diag["notes"].append("Zero-sized GEMM buffers are not lowered through the current TinyTPU path.")
        diag["missing_instructions"] = ["SXU_DISPATCH_VPU", "SXU_LOAD_VREG", "SXU_STORE_VREG"]
        return diag

    binary_vpu_ops = _VPU_OPS
    matched_single_binary_ops = [name for name in binary_vpu_ops if op_counts.get(name, 0) in {1, 4}]
    matched_grouped_binary_ops = [("CMPNE" if op_counts.get("CMPNE", 0) else "CMPLT" if op_counts.get("CMPLT", 0) else "MAX" if op_counts.get("MAX", 0) else "MUL" if op_counts.get("MUL", 0) > 1 else "ADD")] if any(op_counts.get(name, 0) for name in binary_vpu_ops) else []
    out_is_bool = any(p.arg == 0 and "bool" in str(p.dtype) for p in params)
    scalar_const_binary_ops = [name for name in binary_vpu_ops
                               if op_counts.get(name, 0) in {1, 4} and _find_scalar_const_binary(uops, name) is not None]
    if out_is_bool:
        scalar_const_binary_ops = [name for name in scalar_const_binary_ops if name in {"CMPLT", "CMPNE"}]
    else:
        scalar_const_binary_ops = [name for name in scalar_const_binary_ops if name in {"ADD", "MUL", "MAX", "SUB"}]
    scalar_const = _find_scalar_const_binary(uops, scalar_const_binary_ops[0]) if len(scalar_const_binary_ops) == 1 else None
    reverse_sub_const = _find_reverse_sub_const(uops)
    eq_scalar_const = _find_eq_scalar_const(uops)
    is_eq_from_cmpne = _has_eq_from_cmpne(uops)
    # tinygrad may leave pointer reads as INDEX nodes for a fully upcast 16-lane
    # tile, while smaller tiles materialize explicit LOAD UOps.
    is_single_binary = len(params) == 3 and len(matched_single_binary_ops) == 1 and op_counts.get("LOAD", 0) in {0, 2} and op_counts.get("STORE", 0) == 1
    is_grouped_binary = len(params) == 3 and len(matched_grouped_binary_ops) == 1 and op_counts.get("STORE", 0) == 4 and op_counts.get("GROUP", 0) == 1
    if len(params) == 2 and op_counts.get("ADD", 0) == 3 and op_counts.get("LOAD", 0) == 4 and op_counts.get("STORE", 0) == 1:
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        if out_size == 1 and src_size == 4:
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
        diag["notes"].append("Current TinyTPU VPU SUM_REDUCE lowering handles a 4-element int32 row reduced to one scalar.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif len(params) == 2 and op_counts.get("CMPLT", 0) > 0 and op_counts.get("WHERE", 0) > 0:
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
    elif (len(params) == 2 and reverse_sub_const is not None and op_counts.get("STORE", 0) in {1, 4} and
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
                "num_elems": src_size,
                "vpu_op": binary_vpu_ops["SUB"],
            })
            return diag
        diag["reason"] = f"unsupported vpu reverse sub const sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU reverse SUB constant lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 2 and len(scalar_const_binary_ops) == 1 and scalar_const is not None and op_counts.get("STORE", 0) in {1, 4} and
          (op_counts.get("STORE", 0) == 4 or op_counts.get("LOAD", 0) == 1)):
        op_name = scalar_const_binary_ops[0]
        out_size = param_sizes.get(0)
        src_size = param_sizes.get(1)
        if out_size is not None and src_size is not None and out_size == src_size and 0 < src_size:
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": f"supported vpu {op_name.lower()} const",
                "out_arg": 0,
                "lhs_arg": 1,
                "lhs_const": None,
                "rhs_arg": None,
                "rhs_const": scalar_const,
                "num_elems": src_size,
                "vpu_op": binary_vpu_ops[op_name],
            })
            return diag
        diag["reason"] = f"unsupported vpu {op_name.lower()} const sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append(f"Current TinyTPU VPU {op_name} constant lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif (len(params) == 2 and eq_scalar_const is not None and op_counts.get("STORE", 0) in {1, 4} and
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
        if out_size is not None and len(input_args) == 2 and 0 < out_size and all(param_sizes[arg] == out_size for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu cmpeq",
                "out_arg": 0,
                "lhs_arg": input_args[0],
                "lhs_const": None,
                "rhs_arg": input_args[1],
                "rhs_const": None,
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
        if out_size is not None and len(input_args) == 2 and 0 < out_size and all(param_sizes[arg] == out_size for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": "supported vpu sub",
                "out_arg": 0,
                "lhs_arg": input_args[0],
                "lhs_const": None,
                "rhs_arg": input_args[1],
                "rhs_const": None,
                "num_elems": out_size,
                "vpu_op": binary_vpu_ops["SUB"],
            })
            return diag
        diag["reason"] = f"unsupported vpu sub sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append("Current TinyTPU VPU SUB lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif is_single_binary or is_grouped_binary:
        op_name = (matched_single_binary_ops if is_single_binary else matched_grouped_binary_ops)[0]
        out_size = param_sizes.get(0)
        input_args = [arg for arg in sorted(param_sizes) if arg != 0]
        if out_size is not None and len(input_args) == 2 and 0 < out_size and all(param_sizes[arg] == out_size for arg in input_args):
            diag.update({
                "supported": True,
                "kind": "vpu_binary",
                "reason": f"supported vpu {op_name.lower()}",
                "out_arg": 0,
                "lhs_arg": input_args[0],
                "lhs_const": None,
                "rhs_arg": input_args[1],
                "rhs_const": None,
                "num_elems": out_size,
                "vpu_op": binary_vpu_ops[op_name],
            })
            return diag
        diag["reason"] = f"unsupported vpu {op_name.lower()} sizes {dict(sorted(param_sizes.items()))}"
        diag["notes"].append(f"Current TinyTPU VPU {op_name} lowering handles one int32 VMEM tile with 1..16 elements.")
        diag["missing_instructions"] = ["SXU_LOAD_VREG", "SXU_DISPATCH_VPU", "SXU_STORE_VREG"]
    elif len(params) == 4 and op_counts.get("WHERE", 0) > 0 and not is_gemm:
        out_size = param_sizes.get(0)
        input_args = [arg for arg in sorted(param_sizes) if arg != 0]
        if out_size is not None and len(input_args) == 3 and 0 < out_size and all(param_sizes[arg] == out_size for arg in input_args):
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


def _find_scalar_const_binary(uops:list[UOp], op_name:str) -> int | None:
    for u in uops:
        if u.op.name != op_name:
            continue
        consts = [s for s in u.src if s.op is Ops.CONST]
        non_consts = [s for s in u.src if s.op is not Ops.CONST]
        if len(consts) == 1 and len(non_consts) == 1:
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


def _build_vpu_binary_bundle(lhs_i32: np.ndarray, rhs_i32: np.ndarray, num_elems: int, vpu_op: int) -> str:
    def tile(vals: np.ndarray) -> list[int]:
        padded = np.zeros(_ROWS * _COLS, dtype=np.int32)
        padded[:num_elems] = vals[:num_elems]
        return [int(x) for x in padded]

    lines: list[str] = []
    lines.append("5 0 " + " ".join(str(x) for x in tile(lhs_i32)))
    lines.append("5 1 " + " ".join(str(x) for x in tile(rhs_i32)))
    # LOAD VMEM[0]->v0, LOAD VMEM[1]->v1, VPU op v0/v1->v2, STORE v2->VMEM[2]
    lines.append("2 0 0 0 0 0 0 0 0 0")
    lines.append("2 0 1 1 0 0 0 0 0 0")
    lines.append(f"2 2 0 2 0 {vpu_op} 1 0 0 0")
    lines.append("2 1 2 0 2 0 0 0 0 0")
    lines.append("2 5 0 0 0 0 0 0 0 0")
    lines.append("6 2")
    lines.append("4")
    return "\n".join(lines) + "\n"


def _build_vpu_unary_bundle(src_i32: np.ndarray, num_elems: int, vpu_op: int) -> str:
    padded = np.zeros(_ROWS * _COLS, dtype=np.int32)
    padded[:num_elems] = src_i32[:num_elems]

    lines: list[str] = []
    lines.append("5 0 " + " ".join(str(int(x)) for x in padded))
    # LOAD VMEM[0]->v0, VPU unary op v0->v1, STORE v1->VMEM[2]
    lines.append("2 0 0 0 0 0 0 0 0 0")
    lines.append(f"2 2 0 1 0 {vpu_op} 0 0 0 0")
    lines.append("2 1 2 0 1 0 0 0 0 0")
    lines.append("2 5 0 0 0 0 0 0 0 0")
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
    lines.append("2 0 0 0 0 0 0 0 0 0")   # LOAD VMEM[0]->v0
    lines.append("2 0 1 1 0 0 0 0 0 0")   # LOAD VMEM[1]->v1
    lines.append("2 0 2 2 0 0 0 0 0 0")   # LOAD VMEM[2]->v2
    lines.append("2 0 3 3 0 0 0 0 0 0")   # LOAD VMEM[3]->v3
    # v4 = cond * lhs (MUL v0, v1 -> v4)
    lines.append(f"2 2 0 4 0 {_VPU_OPS['MUL']} 1 0 0 0")
    # v5 = 1 - cond (SUB v3, v0 -> v5)
    lines.append(f"2 2 0 5 3 {_VPU_OPS['SUB']} 0 0 0 0")
    # v6 = (1-cond) * rhs (MUL v5, v2 -> v6)
    lines.append(f"2 2 0 6 5 {_VPU_OPS['MUL']} 2 0 0 0")
    # v7 = cond*lhs + (1-cond)*rhs (ADD v4, v6 -> v7)
    lines.append(f"2 2 0 7 4 {_VPU_OPS['ADD']} 6 0 0 0")
    # STORE v7 -> VMEM[4]
    lines.append("2 1 4 0 7 0 0 0 0 0")
    lines.append("2 5 0 0 0 0 0 0 0 0")   # HALT
    lines.append("6 4")   # output VMEM[4]
    lines.append("4")     # END
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
        if prog.get("op") not in {"GEMM4x4", "VPU_BINARY", "VPU_UNARY", "VPU_WHERE"}:
            raise NotImplementedError(_unsupported_message(prog))

        if prog.get("op") == "VPU_BINARY":
            out_buf = bufs[prog["out"]]
            num_elems = int(prog["num_elems"])
            if prog.get("lhs_const") is None:
                lhs_i32 = np.frombuffer(bytes(bufs[prog["lhs"]]), dtype="<i4")
            else:
                lhs_i32 = np.full(num_elems, int(prog["lhs_const"]), dtype="<i4")
            if prog.get("rhs_const") is None:
                rhs_i32 = np.frombuffer(bytes(bufs[prog["rhs"]]), dtype="<i4")
            else:
                rhs_i32 = np.full(num_elems, int(prog["rhs_const"]), dtype="<i4")
            if lhs_i32.size != num_elems or rhs_i32.size != num_elems:
                raise RuntimeError(f"TinyTPU VPU binary op expected {num_elems} elements, got lhs={lhs_i32.size} rhs={rhs_i32.size}")
            is_bool = int(prog["vpu_op"]) in _VPU_BOOL_OPS
            out_elem_bytes = 1 if is_bool else _BYTES_PER_ELEM
            if len(out_buf) < num_elems * out_elem_bytes:
                raise RuntimeError(f"TinyTPU output buffer too small for VPU binary op elements={num_elems}")
            sim = _sim_path()
            vpu_op = int(prog["vpu_op"])
            out_offset = 0
            for chunk_start in range(0, num_elems, _TILE_ELEMS):
                chunk_end = min(chunk_start + _TILE_ELEMS, num_elems)
                chunk_size = chunk_end - chunk_start
                lhs_chunk = lhs_i32[chunk_start:chunk_end]
                rhs_chunk = rhs_i32[chunk_start:chunk_end]
                stdout = _run_bundle(sim, _build_vpu_binary_bundle(lhs_chunk, rhs_chunk, chunk_size, vpu_op))
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
