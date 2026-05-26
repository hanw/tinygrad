"""TinyTPU GEMM lowering.

Relocated verbatim from ops_tinytpu.py: the WMMA-driven GEMM lowering path
(``lower_gemm``), the non-WMMA matmul fallback (``lower_gemm_fallback``), and
the GEMM-only helpers they call (tiling inference, SXU instruction generation,
epilogue extraction).

``classify(uops)`` returns ``KernelClass.GEMM`` when a WMMA UOp is present;
``render()`` dispatches that to ``lower_gemm``. The non-WMMA matmul fallback is
reached after the structural recognizers via ``lower_gemm_fallback``.
"""
from __future__ import annotations
import math
from collections import Counter
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import PtrDType
from tinygrad.renderer.tinytpu.common import (
  _ROWS, _COLS, _VPU, _find_unique_param_arg,
  _psum_clear, _mxu_psum_acc, _wait_mxu, _psum_read_row, _mxu,
  _load_mxu_result, _vpu, _load, _store, _halt, _mxu_vpu_epilogue)


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


def _generate_gemm_sxu_instructions(num_vecs: int, num_k_tiles: int, num_weight_tiles: int,
                                     *, has_bias: bool = False, bias_vmem_base: int = 0,
                                     has_relu: bool = False,
                                     use_psum: bool = False,
                                     fuse_mxu_vpu_epilogue: bool = False) -> list[str]:
    """Generate SXU instruction strings for a GEMM kernel.

    When use_psum=True and num_k_tiles>1, accumulate K-tiles in the
    PSUM bucket bank instead of reading each partial into a vreg and
    chaining VPU_ADDs. This eliminates num_k_tiles-1 VPU_ADDs and
    num_k_tiles LOAD_MXU_RESULT instructions per output tile.

    When fuse_mxu_vpu_epilogue=True, replace the legacy chain of
    _mxu + _wait_mxu + _load_mxu_result + _load(bias) + _vpu(ADD) + _store
    with a single _mxu_vpu_epilogue dispatch (op 46) per output tile.
    Caller must guarantee num_k_tiles == 1 and not has_relu — op 46
    does not multi-K-accumulate and does not fuse ReLU.
    """
    out_vmem_base = num_weight_tiles if has_bias else 0
    prog_lines: list[str] = []

    if fuse_mxu_vpu_epilogue:
        # CODA-style single-bundle residual epilogue. One op-46 dispatch
        # per output tile; the Controller drains the GEMM, adds the
        # tile-shape src2 lane-wise, and writes the result directly to
        # VMEM[out_addr].
        assert has_bias and num_k_tiles == 1 and not has_relu, \
            "fuse_mxu_vpu_epilogue requires has_bias + num_k_tiles=1 + not has_relu"
        src2_vreg = 0
        for row in range(num_vecs):
            for tile_idx in range(num_weight_tiles):
                wmem_addr = tile_idx
                amem_addr = row
                out_addr  = out_vmem_base + row * num_weight_tiles + tile_idx
                prog_lines.append(_load(src2_vreg, bias_vmem_base + tile_idx))
                prog_lines.append(_mxu_vpu_epilogue(
                    wmem_addr, amem_addr, 1, src2_vreg=src2_vreg,
                    vpu_op=_VPU["ADD"], dst=out_addr, vmem_dst=True))
        prog_lines.append(_halt())
        return prog_lines

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
                    prog_lines.append(_vpu(acc, 0, _VPU["ADD"], 1))
                    cur = acc
                    for k in range(2, num_k_tiles):
                        nxt = cur + 1
                        prog_lines.append(_vpu(nxt, cur, _VPU["ADD"], k))
                        cur = nxt

            # Bias epilogue
            if has_bias:
                bias_vreg = cur + 1
                prog_lines.append(_load(bias_vreg, bias_vmem_base + tile_idx))
                result_vreg = bias_vreg + 1
                prog_lines.append(_vpu(result_vreg, cur, _VPU["ADD"], bias_vreg))
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


def lower_gemm(uops: list[UOp]) -> dict | None:
    """Lower a WMMA GEMM kernel into an SXU_PROGRAM descriptor.

    This is the WMMA branch relocated verbatim from ``_render_sxu_program``.
    Returns ``None`` when the WMMA kernel does not factor into a supported
    tiling / epilogue shape, so ``render()`` can fall through to the
    structural recognizers exactly as before.
    """
    wmmas = [u for u in uops if u.op is Ops.WMMA]
    if not wmmas:
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

    # CODA-style residual fusion: collapse the GEMM + tile-shape add
    # into a single op-46 dispatch per output tile. Requires the
    # epilogue to be exactly a FULL-shape bias add (no relu) and the
    # GEMM to fit in one K-tile (op-46 doesn't multi-K-accumulate).
    fuse_residual = (has_bias and bias_mode == "FULL"
                     and num_k_tiles == 1 and not has_relu)

    # Generate SXU instructions
    instructions = _generate_gemm_sxu_instructions(
        num_vecs, num_k_tiles, num_weight_tiles,
        has_bias=has_bias, bias_vmem_base=bias_vmem_base,
        has_relu=has_relu,
        use_psum=use_psum,
        fuse_mxu_vpu_epilogue=fuse_residual,
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


def lower_gemm_fallback(uops: list[UOp]) -> dict | None:
    """Render MULACC or scalar MUL+RANGE GEMMs (no WMMA UOp) as SXU_PROGRAM.

    Same structure as the WMMA SXU path but triggered by the non-WMMA lowering
    pattern. No epilogue support (bias/relu) — that still requires the WMMA
    UOp path in ``lower_gemm``.
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
    # The compute signature of a non-WMMA matmul is acc += a*b, where the MUL
    # multiplies two loaded values. Address-arithmetic MULs (RANGE * stride
    # or CONST * stride) must not qualify — otherwise reductions and other
    # kernels with stride MULs get silently lowered as a GEMM and produce
    # garbage.
    has_load_mul = any(u.op is Ops.MUL and len(u.src) == 2
                       and all(s.op is Ops.LOAD for s in u.src)
                       for u in uops)
    is_gemm = has_mulacc or (len(params) == 3 and has_load_mul
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
