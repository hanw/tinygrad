from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
from tinygrad.device import Compiler
from tinygrad.renderer.tinytpu.common import _vmem, _wmem, _amem, _output_vmem, _end, _bundle
from tinygrad.runtime.support.tinytpu import ROWS, COLS, TILE_ELEMS

SUPPORTED_TINYTPU_OPS = {"SXU_PROGRAM"}

class TinyTPUCompiler(Compiler):
  """JSON descriptor compiler.

  TinyTPU rendering produces a typed descriptor rather than source text. The
  compiler boundary validates that descriptor and caches its canonical JSON.
  """
  def compile(self, src: str) -> bytes:
    return TinyTPUKernel.from_json(src).to_json().encode()

def unsupported_message(desc: dict) -> str:
  parts = [f"TinyTPU: unsupported op '{desc.get('op')}' (reason: {desc.get('reason', 'n/a')})"]
  missing = desc.get("missing_instructions") or []
  notes = desc.get("notes") or []
  op_counts = desc.get("op_counts") or {}
  if missing: parts.append("missing instructions: " + ", ".join(str(x) for x in missing))
  if notes: parts.append("notes: " + " | ".join(str(x) for x in notes))
  if op_counts: parts.append("op_counts: " + ", ".join(f"{k}={v}" for k, v in sorted(op_counts.items())))
  return "; ".join(parts)

def _require_int8_range(name: str, values: np.ndarray) -> None:
  if values.size == 0: return
  min_val, max_val = int(values.min()), int(values.max())
  if min_val < -128 or max_val > 127:
    raise ValueError(f"TinyTPU {name} values must fit in signed int8, got range [{min_val}, {max_val}]")

@dataclass(frozen=True)
class TinyTPUKernel:
  """Compiler/runtime artifact consumed by TinyTPUProgram."""
  desc: dict

  @classmethod
  def from_json(cls, raw: str | bytes) -> TinyTPUKernel:
    desc = json.loads(raw)
    if not isinstance(desc, dict): raise ValueError("TinyTPU kernel descriptor must be a JSON object")
    return cls(desc)

  def to_json(self) -> str:
    return json.dumps(self.desc, sort_keys=True, separators=(",", ":"))

  @property
  def op(self) -> str | None:
    return self.desc.get("op")

  def require_supported(self) -> None:
    if self.op not in SUPPORTED_TINYTPU_OPS: raise NotImplementedError(unsupported_message(self.desc))

  def build_bundle(self, bufs: tuple[bytearray, ...]) -> str:
    if self.op != "SXU_PROGRAM": raise NotImplementedError(unsupported_message(self.desc))
    desc = self.desc
    data_lines: list[str] = []
    for entry in desc["data_plan"]:
      mem_type = entry["type"]
      if entry.get("layout") == "broadcast_const":
        val = int(entry["value"])
        data_lines.append(_vmem(int(entry["addr"]), [val] * TILE_ELEMS))
        continue
      buf_data = bufs[int(entry["param"])]

      if mem_type == "WMEM":
        weight_i32 = np.frombuffer(bytes(buf_data), dtype="<i4")
        _require_int8_range("weight", weight_i32)
        nk, nwt = entry["num_k_tiles"], entry["num_weight_tiles"]
        weight_matrix = weight_i32.astype(np.int8).reshape(nk * ROWS, nwt * COLS)
        for k in range(nk):
          for t in range(nwt):
            w_tile = weight_matrix[k * ROWS : (k + 1) * ROWS, t * COLS : (t + 1) * COLS]
            data_lines.append(_wmem(k * nwt + t, [int(x) for x in w_tile.flatten()]))

      elif mem_type == "AMEM":
        act_i32 = np.frombuffer(bytes(buf_data), dtype="<i4")
        _require_int8_range("activation", act_i32)
        nv, nk = entry["num_vecs"], entry["num_k_tiles"]
        act_rows = act_i32.astype(np.int8).reshape(nv, nk * ROWS)
        for row in range(nv):
          for k in range(nk):
            a_tile = act_rows[row, k * ROWS : (k + 1) * ROWS]
            data_lines.append(_amem(row * nk + k, [int(x) for x in a_tile]))

      elif mem_type == "VMEM":
        self._append_vmem_data(data_lines, entry, buf_data)
      else:
        raise ValueError(f"unknown TinyTPU data_plan memory type {mem_type!r}")

    output_lines = [_output_vmem(int(o["addr"])) for o in desc["outputs"]] + [_end()]
    return _bundle(*(data_lines + desc["instructions"] + output_lines))

  def write_outputs(self, bufs: tuple[bytearray, ...], vmem_results: list[list[int]]) -> None:
    desc = self.desc
    expected = int(desc["num_output_tiles"])
    if len(vmem_results) != expected:
      raise RuntimeError(f"SXU_PROGRAM expected {expected} vmem tiles, got {len(vmem_results)}")

    out_buf = bufs[int(desc["out"])]
    out_dtype = np.dtype(np.bool_) if desc.get("bool_out", False) else np.dtype("<i4")
    reduce_mode = desc.get("reduce")
    out_offset = 0
    for idx, out_entry in enumerate(desc["outputs"]):
      count = int(out_entry["count"])
      tile_data = vmem_results[idx]
      if reduce_mode and count == 1:
        chunk_out = np.array([tile_data[0]], dtype=out_dtype)
      elif out_entry.get("extract") == "row_heads":
        chunk_out = np.array([tile_data[r * COLS] for r in range(count)], dtype=out_dtype)
      else:
        chunk_out = np.array(tile_data[:count], dtype=out_dtype)
      out_buf[out_offset : out_offset + len(chunk_out) * out_dtype.itemsize] = chunk_out.tobytes()
      out_offset += len(chunk_out) * out_dtype.itemsize

  @staticmethod
  def _append_vmem_data(data_lines: list[str], entry: dict, buf_data: bytearray) -> None:
    is_bool = entry.get("bool", False)
    raw = np.frombuffer(bytes(buf_data), dtype=np.bool_ if is_bool else "<i4")
    if is_bool: raw = raw.astype(np.int32)
    addr = int(entry["addr"])
    if entry.get("broadcast", False):
      val = int(raw[0]) if len(raw) > 0 else 0
      data_lines.append(_vmem(addr, [val] * TILE_ELEMS))
      return

    mode = entry.get("mode", "TILE")
    if mode == "ROW_BROADCAST":
      nwt = entry.get("num_weight_tiles", 1)
      for t in range(nwt):
        tile = [0] * TILE_ELEMS
        for i in range(COLS): tile[i] = int(raw[t * COLS + i])
        data_lines.append(_vmem(addr + t, tile))
    elif mode == "PAD_FILL":
      tile = [int(entry.get("pad_value", 0))] * TILE_ELEMS
      for dst, src in entry["pad_map"]:
        if src is not None and 0 <= src < len(raw): tile[dst] = int(raw[src])
      data_lines.append(_vmem(addr, tile))
    elif mode == "MATRIX_TILE":
      tile = [int(entry.get("pad_value", 0))] * TILE_ELEMS
      nrows_mat, ncols_mat = int(entry["matrix_nrows"]), int(entry["matrix_ncols"])
      row_base, col_base = int(entry.get("row_base", 0)), int(entry.get("col_base", 0))
      tile_rows, tile_cols = int(entry.get("tile_rows", ROWS)), int(entry.get("tile_cols", COLS))
      for r in range(tile_rows):
        mr = row_base + r
        if mr >= nrows_mat: break
        for c in range(tile_cols):
          mc = col_base + c
          if mc >= ncols_mat: break
          tile[r * COLS + c] = int(raw[mr * ncols_mat + mc])
      data_lines.append(_vmem(addr, tile))
    else:
      offset, count = int(entry.get("offset", 0)), int(entry.get("count", TILE_ELEMS))
      tile = [int(entry.get("pad_value", 0))] * TILE_ELEMS
      chunk = raw[offset : offset + count]
      for i in range(min(count, len(chunk))): tile[i] = int(chunk[i])
      data_lines.append(_vmem(addr, tile))
