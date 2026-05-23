from __future__ import annotations
import os, subprocess, tempfile
import numpy as np
from tinygrad.renderer.tinytpu.common import (
  _vmem, _wmem, _amem, _load, _store, _vpu, _mxu, _wait_mxu, _load_mxu_result,
  _halt, _output_vmem, _end, _bundle, _broadcast)

ROWS = 4
COLS = 4
TILE_ELEMS = ROWS * COLS

def build_vpu_binary_bundle(lhs_i32: np.ndarray, rhs_i32: np.ndarray, num_elems: int, vpu_op: int,
                            lhs_broadcast: bool = False, rhs_broadcast: bool = False) -> str:
  """VMEM[0]=lhs, VMEM[1]=rhs, VPU v2=OP(v0,v1), OUTPUT_VMEM VMEM[2]."""
  def tile(vals: np.ndarray) -> list[int]:
    padded = np.zeros(TILE_ELEMS, dtype=np.int32)
    padded[:num_elems] = vals[:num_elems]
    return [int(x) for x in padded]

  lines = [
    _vmem(0, tile(lhs_i32)),
    _vmem(1, tile(rhs_i32)),
    _load(0, 0),
  ]
  if lhs_broadcast: lines.append(_broadcast(0))
  lines.append(_load(1, 1))
  if rhs_broadcast: lines.append(_broadcast(1))
  lines += [_vpu(2, 0, vpu_op, 1), _store(2, 2), _halt(), _output_vmem(2), _end()]
  return _bundle(*lines)

def build_vpu_unary_bundle(src_i32: np.ndarray, num_elems: int, vpu_op: int) -> str:
  """VMEM[0]=src, VPU v1=OP(v0), OUTPUT_VMEM VMEM[2]."""
  padded = np.zeros(TILE_ELEMS, dtype=np.int32)
  padded[:num_elems] = src_i32[:num_elems]
  return _bundle(
    _vmem(0, [int(x) for x in padded]), _load(0, 0), _vpu(1, 0, vpu_op),
    _store(2, 1), _halt(), _output_vmem(2), _end(),
  )

def build_vpu_program_bundle(inputs: list[np.ndarray], num_elems: int, steps: list[dict], output_reg: int,
                             input_broadcasts: list[bool] | None = None) -> str:
  """Multi-step VPU program: VMEM[0..N-1]=inputs, execute steps, VMEM[N]=output_reg."""
  def tile(vals: np.ndarray) -> list[int]:
    padded = np.zeros(TILE_ELEMS, dtype=np.int32)
    padded[:num_elems] = vals[:num_elems]
    return [int(x) for x in padded]

  if input_broadcasts is None: input_broadcasts = [False] * len(inputs)
  lines: list[str] = []
  for idx, vals in enumerate(inputs): lines.append(_vmem(idx, tile(vals)))
  for idx in range(len(inputs)):
    lines.append(_load(idx, idx))
    if input_broadcasts[idx]: lines.append(_broadcast(idx))
  for step in steps:
    lines.append(_vpu(int(step["dst"]), int(step["lhs"]), int(step["op"]), int(step.get("rhs", 0))))
  lines += [_store(len(inputs), output_reg), _halt(), _output_vmem(len(inputs)), _end()]
  return _bundle(*lines)

def build_full_gemm_bundle(act_rows_i8: np.ndarray, weight_matrix_i8: np.ndarray,
                           num_vecs: int, num_k_tiles: int, num_weight_tiles: int,
                           vpu_ops: dict[str, int], bias_i32: np.ndarray | None = None, relu: bool = False) -> str:
  """Build a single SXU program that computes all rows x tiles of a GEMM + epilogue."""
  data_lines: list[str] = []
  for k in range(num_k_tiles):
    for t in range(num_weight_tiles):
      w_tile = weight_matrix_i8[k * ROWS : (k + 1) * ROWS, t * COLS : (t + 1) * COLS]
      data_lines.append(_wmem(k * num_weight_tiles + t, [int(x) for x in w_tile.flatten()]))

  for row in range(num_vecs):
    for k in range(num_k_tiles):
      a_tile = act_rows_i8[row, k * ROWS : (k + 1) * ROWS]
      data_lines.append(_amem(row * num_k_tiles + k, [int(x) for x in a_tile]))

  bias_vmem_base = 0
  if bias_i32 is not None:
    for t in range(num_weight_tiles):
      bias_tile = [0] * TILE_ELEMS
      for i in range(COLS): bias_tile[i] = int(bias_i32[t * COLS + i])
      data_lines.append(_vmem(bias_vmem_base + t, bias_tile))

  out_vmem_base = num_weight_tiles if bias_i32 is not None else 0
  prog_lines: list[str] = []
  for row in range(num_vecs):
    for tile_idx in range(num_weight_tiles):
      for k in range(num_k_tiles):
        prog_lines.append(_mxu(k * num_weight_tiles + tile_idx, row * num_k_tiles + k, 1))
        prog_lines.append(_wait_mxu())
        prog_lines.append(_load_mxu_result(k))

      if num_k_tiles == 1:
        cur = 0
      else:
        acc = num_k_tiles
        prog_lines.append(_vpu(acc, 0, vpu_ops["ADD"], 1))
        cur = acc
        for k in range(2, num_k_tiles):
          nxt = cur + 1
          prog_lines.append(_vpu(nxt, cur, vpu_ops["ADD"], k))
          cur = nxt

      if bias_i32 is not None:
        bias_vreg = cur + 1
        prog_lines.append(_load(bias_vreg, bias_vmem_base + tile_idx))
        result_vreg = bias_vreg + 1
        prog_lines.append(_vpu(result_vreg, cur, vpu_ops["ADD"], bias_vreg))
        cur = result_vreg

      if relu:
        nxt = cur + 1
        prog_lines.append(_vpu(nxt, cur, 2))
        cur = nxt

      prog_lines.append(_store(out_vmem_base + row * num_weight_tiles + tile_idx, cur))

  output_lines = [_output_vmem(out_vmem_base + row * num_weight_tiles + tile_idx)
                  for row in range(num_vecs) for tile_idx in range(num_weight_tiles)]
  return _bundle(*(data_lines + prog_lines + [_halt()] + output_lines + [_end()]))

def parse_result_line(line: str, prefix: str, expected_count: int) -> list[int]:
  """Parse a single sim output line like 'mxu_result v0 v1 ...' or 'vmem_result v0 v1 ...'."""
  vals = line.split()[1:]
  if len(vals) != expected_count: raise ValueError(f"{prefix} expects {expected_count} values, got {len(vals)}")
  try:
    return [int(x) for x in vals]
  except ValueError as exc:
    bad = next((x for x in vals if not x.lstrip("-").isdigit()), vals[0])
    raise ValueError(f"invalid {prefix} integer {bad!r}") from exc

def parse_sim_output(stdout: str) -> list[int] | None:
  """Extract mxu_result from BSV sim stdout. Returns None if not found."""
  for line in stdout.splitlines():
    if line.strip().startswith("mxu_result "): return parse_result_line(line.strip(), "mxu_result", COLS)
  return None

def parse_vmem_output(stdout: str) -> list[int] | None:
  """Extract first vmem_result from BSV sim stdout. Returns None if not found."""
  for line in stdout.splitlines():
    if line.strip().startswith("vmem_result "): return parse_result_line(line.strip(), "vmem_result", TILE_ELEMS)
  return None

def parse_asram_output(stdout: str) -> list[int] | None:
  """Extract first asram_result from BSV sim stdout. Returns None if not found."""
  for line in stdout.splitlines():
    if line.strip().startswith("asram_result "): return parse_result_line(line.strip(), "asram_result", COLS)
  return None

def parse_multi_vmem_output(stdout: str) -> list[list[int]]:
  """Extract all vmem_result lines from BSV sim stdout."""
  return [parse_result_line(line.strip(), "vmem_result", TILE_ELEMS)
          for line in stdout.splitlines() if line.strip().startswith("vmem_result ")]

def run_bundle(sim: str, bundle_text: str) -> str:
  with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write(bundle_text)
    bundle_path = f.name

  try:
    env = {**os.environ, "TINYTPU_BUNDLE": bundle_path}
    proc = subprocess.run([sim], env=env, capture_output=True, text=True, timeout=30)
  finally:
    os.unlink(bundle_path)

  if proc.returncode != 0:
    raise RuntimeError(f"TinyTPU sim exited {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}")
  for line in proc.stdout.splitlines():
    line = line.strip()
    if line.startswith("FAIL:") or line.startswith("ERROR:"):
      raise RuntimeError(f"TinyTPU simulator reported failure: {line}\nstdout: {proc.stdout}\nstderr: {proc.stderr}")
  if "status ok" not in {line.strip() for line in proc.stdout.splitlines()}:
    raise RuntimeError(f"TinyTPU simulator did not report `status ok`\nstdout: {proc.stdout}\nstderr: {proc.stderr}")
  return proc.stdout
