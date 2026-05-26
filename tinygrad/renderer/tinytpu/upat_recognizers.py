"""UPat-based recognizers for fused MXU epilogue patterns.

This module migrates the post-WMMA epilogue detection from hand-written
op_name Counter walks to tinygrad's declarative PatternMatcher framework
(tinygrad/uop/ops.py:1157,1299).

Two recognizers live here:

  - recognize_residual_epilogue: matches the (D + R) shape, one ADD per
    output lane combining a GEP of the WMMA result with a LOAD from a
    parameter buffer at the matching index. Equivalent in semantics to
    today's _extract_wmma_epilogue FULL-mode branch; intended as a drop-in
    replacement once the existing tests prove it's faithful.

  - recognize_rope_epilogue: matches the (D*C + swap(D)*S) shape, used by
    fused RoPE epilogues. Each output lane's STORE source is an ADD of
    two MULs, each multiplying a different GEP of the same WMMA by a
    LOAD from a parameter buffer. The "no-swap" leg's GEP index equals
    the output index; the "swap" leg's GEP index is out_idx XOR 1.

Both recognizers walk the lowered kernel's STORE set, run the matcher
per store, and stitch the per-lane captures into a kernel-level
descriptor.

The PatternMatcher framework gives us:
  - Op-keyed dispatch tables (O(1) early reject by root op).
  - Name-binding consistency (naming "wmma" in two subpatterns forces
    them to point at the same UOp instance — exactly what we need to
    verify both MULs in a RoPE lane share the same WMMA accumulator).
  - Commutative-src permutation via the `src=list` form, so the matcher
    handles (D*C + swap(D)*S) and the reverse without two patterns.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher


# ---------------------------------------------------------------------------
# Residual: STORE(INDEX(P_out, out_idx), ADD(GEP(WMMA, i), LOAD(INDEX(P_r, j))))
# ---------------------------------------------------------------------------

_residual_store_pat = UPat(Ops.STORE, src=(
    UPat(Ops.INDEX, src=(
        UPat(Ops.PARAM, name="p_out"),
        UPat(Ops.CONST, name="out_idx_const"),
    )),
    UPat(Ops.ADD, src=[
        UPat(Ops.GEP, src=(UPat(Ops.WMMA, name="wmma"),), name="gep"),
        UPat(Ops.LOAD, src=(
            UPat(Ops.INDEX, src=(
                UPat(Ops.PARAM, name="p_r"),
                UPat(Ops.CONST, name="r_idx_const"),
            )),
        ), name="load_r"),
    ]),
), name="store")


# Wrap the UPat in a PatternMatcher so we get the framework's compiled
# match path + early-reject table. The rewrite_fn returns the bound
# capture dict; rewrite() returns None when the pattern doesn't match.
_residual_matcher = PatternMatcher([
    (_residual_store_pat, lambda **kw: kw),
])


@dataclass(frozen=True)
class ResidualEpilogue:
    """Output of recognize_residual_epilogue: identifies the single residual
    buffer and the WMMA whose result it adds to."""
    wmma: UOp
    residual_param: UOp


def recognize_residual_epilogue(uops: list[UOp]) -> ResidualEpilogue | None:
    """Walk all STORE uops; if every one is shaped STORE(INDEX(P_out, i),
    ADD(GEP(W, i), LOAD(INDEX(P_r, i)))) with a consistent W and P_r,
    return ResidualEpilogue(W, P_r); else None.

    Lane-index consistency check: the GEP index, the LOAD's INDEX const,
    and the STORE's INDEX const must all be equal — that's what makes
    it elementwise residual rather than some general reshuffle.
    """
    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None

    wmma_uop: UOp | None = None
    residual_param: UOp | None = None
    seen_out_indices: set[int] = set()

    for store in stores:
        captures = _residual_matcher.rewrite(store)
        if captures is None:
            return None
        out_idx = captures["out_idx_const"].arg
        gep_idx = captures["gep"].arg[0]
        r_idx = captures["r_idx_const"].arg
        if out_idx != gep_idx or out_idx != r_idx:
            return None
        if wmma_uop is None:
            wmma_uop = captures["wmma"]
            residual_param = captures["p_r"]
        elif captures["wmma"] is not wmma_uop or captures["p_r"] is not residual_param:
            return None
        seen_out_indices.add(out_idx)

    # Ensure full lane coverage (no missing/duplicate output indices).
    if seen_out_indices != set(range(len(stores))):
        return None

    assert wmma_uop is not None and residual_param is not None
    return ResidualEpilogue(wmma=wmma_uop, residual_param=residual_param)


# ---------------------------------------------------------------------------
# RoPE: STORE(INDEX(P_out, i),
#             ADD(MUL(GEP(WMMA, ia), LOAD(INDEX(P_a, ja))),
#                 MUL(GEP(WMMA, ib), LOAD(INDEX(P_b, jb)))))
#
# Naming "wmma" in both GEPs enforces they reference the same WMMA
# instance — the framework's name-binding does the cross-leg consistency
# check for us.
# ---------------------------------------------------------------------------

_rope_store_pat = UPat(Ops.STORE, src=(
    UPat(Ops.INDEX, src=(
        UPat(Ops.PARAM, name="p_out"),
        UPat(Ops.CONST, name="out_idx_const"),
    )),
    UPat(Ops.ADD, src=[
        UPat(Ops.MUL, src=(
            UPat(Ops.GEP, src=(UPat(Ops.WMMA, name="wmma"),), name="gep_a"),
            UPat(Ops.LOAD, src=(
                UPat(Ops.INDEX, src=(
                    UPat(Ops.PARAM, name="p_a"),
                    UPat(Ops.CONST, name="idx_a_const"),
                )),
            )),
        )),
        UPat(Ops.MUL, src=(
            UPat(Ops.GEP, src=(UPat(Ops.WMMA, name="wmma"),), name="gep_b"),
            UPat(Ops.LOAD, src=(
                UPat(Ops.INDEX, src=(
                    UPat(Ops.PARAM, name="p_b"),
                    UPat(Ops.CONST, name="idx_b_const"),
                )),
            )),
        )),
    ]),
), name="store")


_rope_matcher = PatternMatcher([
    (_rope_store_pat, lambda **kw: kw),
])


@dataclass(frozen=True)
class RopeEpilogue:
    """Output of recognize_rope_epilogue: identifies the WMMA and the two
    parameter buffers carrying the per-lane coefficients. The convention
    is `c_param` holds the multipliers on the no-swap legs (out[i] uses
    c_param at the same index i) and `s_param` holds the multipliers on
    the swap legs (out[i] uses s_param at index i XOR 1).
    """
    wmma: UOp
    c_param: UOp
    s_param: UOp


def recognize_rope_epilogue(uops: list[UOp]) -> RopeEpilogue | None:
    """Walk all STOREs; if every one matches the rope shape with consistent
    WMMA, C param, and S param — and the GEP / LOAD index relations are
    a clean pair-swap — return RopeEpilogue, else None.

    Per-lane validation:
      The two MULs land in unspecified order in the captures (ADD is
      commutative — `src=list` enumerates permutations). For each match
      we identify the no-swap leg (GEP index == out_idx) and the swap
      leg (GEP index == out_idx XOR 1). LOAD indices on each leg must
      equal the corresponding GEP index.
    """
    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores:
        return None

    wmma_uop: UOp | None = None
    c_param: UOp | None = None
    s_param: UOp | None = None
    seen_out_indices: set[int] = set()

    for store in stores:
        captures = _rope_matcher.rewrite(store)
        if captures is None:
            return None
        out_idx = captures["out_idx_const"].arg
        gep_a_idx = captures["gep_a"].arg[0]
        gep_b_idx = captures["gep_b"].arg[0]
        idx_a = captures["idx_a_const"].arg
        idx_b = captures["idx_b_const"].arg
        swap_idx = out_idx ^ 1

        # Determine which leg is no-swap (cos / C) and which is swap (sin / S).
        if gep_a_idx == out_idx and gep_b_idx == swap_idx:
            store_c_param, store_s_param = captures["p_a"], captures["p_b"]
            c_idx, s_idx = idx_a, idx_b
        elif gep_b_idx == out_idx and gep_a_idx == swap_idx:
            store_c_param, store_s_param = captures["p_b"], captures["p_a"]
            c_idx, s_idx = idx_b, idx_a
        else:
            return None

        # The LOAD index on each leg must match its GEP index — that's
        # what makes it an elementwise rotate rather than some general
        # cross-lane gather.
        if c_idx != out_idx or s_idx != swap_idx:
            return None

        if wmma_uop is None:
            wmma_uop = captures["wmma"]
            c_param, s_param = store_c_param, store_s_param
        elif (captures["wmma"] is not wmma_uop
              or store_c_param is not c_param
              or store_s_param is not s_param):
            return None
        seen_out_indices.add(out_idx)

    if seen_out_indices != set(range(len(stores))):
        return None

    assert wmma_uop is not None and c_param is not None and s_param is not None
    return RopeEpilogue(wmma=wmma_uop, c_param=c_param, s_param=s_param)
