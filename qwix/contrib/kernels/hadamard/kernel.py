# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pallas/Mosaic TPU kernels and heuristics for Hadamard rotation + quantize.

Executes optional Rademacher sign flips, Hadamard (or Block-Hadamard) MXU
rotation, `1 / sqrt(n)` normalization, and optional row-wise symmetric
quantization (`int8`, `float8_e4m3fn`, `float8_e5m2`) inside a single VMEM
pipeline without materializing the rotated floating-point activation in HBM.

Four kernel layouts share one entry point (`hadamard_pallas_2d`):

1. **Two-stage Sylvester Hadamard** (`block_size is None`, `k` a power of two
   `>= 256`, `block_n` unset): uses `H_k = (H_{k/128} (x) I_128) (I_{k/128} (x)
   H_128)`, the same matrix as the single-panel layout. The within-block
   `H_128` runs on the MXU and the cross-block `H_{k/128}` runs as a radix-2
   butterfly over whole 128-lane slabs on the VPU, i.e. `2 * 128` MXU flops plus
   `log2(k / 128)` additions per element instead of `2 * k` MXU flops.
2. **Single-panel Kronecker Hadamard** (`block_size is None`, `bn == k`):
   holds the `(k, k)` Hadamard matrix stationary in VMEM (`(0, 0)` index map)
   across all row blocks (`k <= 4096` in `bfloat16` fits on 64 MiB TPUs). Used
   for Paley-I orders (e.g. `k = 3200`) or when `block_n = k` is passed.
3. **Multi-panel Kronecker Hadamard** (`block_size is None`, `bn < k`):
   streams `(k, bn)` panels when `(k, k)` exceeds the per-core VMEM budget
   (e.g. Paley-I `k = 4480` in `float32` on 64 MiB VMEM, or 16 MiB VMEM chips).
   When quantizing, caches rotated chunks in VMEM and computes the exact
   full-row absmax scale on the final panel.
4. **Block-Hadamard** (`block_size = b`): holds `(b, b)` stationary in VMEM,
   rotates all `k // b` diagonal blocks in VMEM, and quantizes across the full
   row `k`.

Row counts `m` that are not multiples of `bm` are zero-padded outside
`pl.kernel` and sliced back on exit, since RHT and row-wise quantization act
independently on each row.
"""

import dataclasses
import functools
import math
from typing import Literal, overload

import jax
from jax import custom_batching
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from qwix.contrib.kernels.hadamard import reference

_CORE_AXIS_NAME = "core"

# TPU VMEM lane width; block sizes and rotated dimensions are multiples of 128.
BLOCK_ALIGNMENT = 128

# Per-row scales have logical shape `(m, 1)`, which would scatter four bytes per
# 512-byte `(8, 128)` VMEM row if written directly. Broadcast to `(m, 128)` in
# VMEM and slice `[:, :1]` after `pl.kernel`.
SX_LANES = 128

DEFAULT_VMEM_LIMIT_BYTES = 64 * 1024 * 1024
VMEM_BUDGET_FRACTION = 0.85

# Rows per step of the two-stage cross-block butterfly: one `(8, 128)` float32
# vreg per 128-lane slab, so `k // 128 <= 32` slabs stay in vregs.
_BUTTERFLY_ROWS = 8


def hadamard_pallas_supported(
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> bool:
  """Returns whether the active or AOT-targeted device supports Pallas TPU RHT.

  Queries `pltpu.get_tpu_info()`, which inspects
  `jax.sharding.use_abstract_mesh` first (for CPU-hosted AOT compilation
  targeting TPU) and falls back to the default runtime device.

  Args:
    dtype: Unquantized input activation dtype (`bfloat16` or `float32`).
  """
  try:
    info = pltpu.get_tpu_info()
  except ValueError:
    return False
  dt = jnp.dtype(dtype)
  return info.is_matmul_supported(dt, dt)


def default_vmem_limit_bytes() -> int:
  """Returns the target TensorCore's physical VMEM capacity in bytes."""
  try:
    return int(pltpu.get_tpu_info().vmem_capacity_bytes)
  except ValueError:
    return DEFAULT_VMEM_LIMIT_BYTES


def divisors_aligned(
    dim: int, alignment: int = BLOCK_ALIGNMENT
) -> tuple[int, ...]:
  """Returns `alignment`-multiple divisors of `dim` in descending order."""
  if dim < alignment or dim % alignment != 0:
    raise ValueError(f"dim ({dim}) must be a positive multiple of {alignment}.")
  q = dim // alignment
  divs = [d * alignment for d in range(q, 0, -1) if q % d == 0]
  return tuple(divs)


@dataclasses.dataclass(frozen=True)
class Config:
  """Tiling configuration for `hadamard_pallas_2d`.

  Attributes:
    block_m: Row tile size along flattened non-rotated dimensions (`m`). Must be
      a positive multiple of `BLOCK_ALIGNMENT` (128). `m` does not need to be a
      multiple of `block_m` (rows are zero-padded and sliced).
    block_n: Column panel size along `k` for the dense `(k, k)` kernels when
      `block_size is None`. Must divide `k` and be a multiple of
      `BLOCK_ALIGNMENT`. `None` selects the kernel automatically: the two-stage
      kernel when `k` is a power of two `>= 256`, otherwise the single-panel
      kernel (`block_n = k`). Pass `block_n = k` to force the dense single-panel
      kernel. Must be `None` in `block_size` mode.
  """

  block_m: int
  block_n: int | None = None

  def __post_init__(self):
    if self.block_m < BLOCK_ALIGNMENT or self.block_m % BLOCK_ALIGNMENT != 0:
      raise ValueError(
          f"block_m must be a positive multiple of {BLOCK_ALIGNMENT}, got"
          f" {self.block_m}."
      )
    if self.block_n is not None and (
        self.block_n < BLOCK_ALIGNMENT or self.block_n % BLOCK_ALIGNMENT != 0
    ):
      raise ValueError(
          f"block_n must be a positive multiple of {BLOCK_ALIGNMENT}, got"
          f" {self.block_n}."
      )


def validate_pallas_config(
    k: int,
    *,
    block_size: int | None,
    config: Config | None,
) -> None:
  """Validates rotated dimension alignment and `Config` compatibility."""
  rot_dim = k if block_size is None else block_size
  if rot_dim % BLOCK_ALIGNMENT != 0:
    raise ValueError(
        f"The Pallas RHT kernel requires rotated dimension ({rot_dim}) to be a"
        f" multiple of {BLOCK_ALIGNMENT}. Pad the contraction axis or pass"
        " implementation='xla'."
    )
  if config is not None:
    if block_size is not None and config.block_n is not None:
      raise ValueError(
          "config.block_n must be None when block_size is specified, got"
          f" block_n={config.block_n}."
      )
    if (
        block_size is None
        and config.block_n is not None
        and k % config.block_n != 0
    ):
      raise ValueError(
          f"Rotated dimension ({k}) must be divisible by config.block_n"
          f" ({config.block_n})."
      )


Layout = Literal["block", "two_stage", "single_panel", "multi_panel"]


def _choose_layout(
    k: int, *, block_size: int | None, block_n: int | None
) -> Layout:
  """Chooses the kernel layout for a rotation of size `k`.

  * `"block"`: block-diagonal Hadamard, when `block_size` is given.
  * `"two_stage"`: `(H_{k/128} (x) I_128) (I_{k/128} (x) H_128)`, when
    `block_n` is unset and `k` is a power of two `>= 256`. Other Hadamard
    orders (e.g. Paley-I `3200 = 200 * 16`) have no factor of order `k / 128`.
  * `"single_panel"`: the dense `(k, k)` Hadamard matrix held in VMEM, when
    `block_n` is unset (other Hadamard orders) or equals `k`.
  * `"multi_panel"`: the dense matrix streamed in `(k, block_n)` panels.

  Args:
    k: Size of the rotated dimension.
    block_size: Optional diagonal Hadamard block size.
    block_n: Optional column panel size of the dense layouts.

  Returns:
    The layout `hadamard_pallas_2d` runs.
  """
  if block_size is not None:
    return "block"
  if block_n is None and k >= 2 * BLOCK_ALIGNMENT and k & (k - 1) == 0:
    return "two_stage"
  if block_n is None or block_n == k:
    return "single_panel"
  return "multi_panel"


def hadamard_vmem_bytes(
    bm: int,
    k: int,
    *,
    bn: int | None = None,
    block_size: int | None = None,
    lhs_itemsize: int = 2,
    quantized: bool = False,
) -> int:
  """Estimates scoped VMEM footprint in bytes for a given tile configuration."""
  layout = _choose_layout(k, block_size=block_size, block_n=bn)
  signs_bytes = k * 4
  x_bytes = 2 * bm * k * lhs_itemsize
  out_itemsize = 1 if quantized else lhs_itemsize
  out_bytes = 2 * bm * k * out_itemsize + (
      2 * bm * SX_LANES * 4 if quantized else 0
  )

  if layout == "block":
    assert block_size is not None
    h_bytes = block_size * block_size * lhs_itemsize
    scratch_bytes = (
        (bm * k * lhs_itemsize + 2 * bm * k * 4)
        if quantized
        else (bm * block_size * 4)
    )
    return x_bytes + h_bytes + signs_bytes + out_bytes + scratch_bytes

  if layout == "two_stage":
    # `(128, 128)` H, the `(bm, k)` float32 accumulator scratch, and the same
    # float32 temporaries as the single-panel epilogue.
    h_bytes = BLOCK_ALIGNMENT * BLOCK_ALIGNMENT * lhs_itemsize
    acc_bytes = bm * k * 4
    f32_tmp = (2 if quantized else 1) * bm * k * 4
    return x_bytes + h_bytes + signs_bytes + out_bytes + acc_bytes + f32_tmp

  if layout == "single_panel":
    h_bytes = k * k * lhs_itemsize
    f32_tmp = (2 if quantized else 1) * bm * k * 4
    return x_bytes + h_bytes + signs_bytes + out_bytes + f32_tmp

  # Multi-panel: `h` is double-buffered `(k, bn)` and when quantized
  # `rot_cache` stores `(k // bn, bm, bn)` in `lhs.dtype`.
  assert bn is not None
  h_bytes = 2 * k * bn * lhs_itemsize
  if quantized:
    cache_bytes = bm * k * lhs_itemsize
    f32_tmp = 2 * bm * bn * 4
    return x_bytes + h_bytes + signs_bytes + out_bytes + cache_bytes + f32_tmp
  out_panel_bytes = 2 * bm * bn * lhs_itemsize
  f32_tmp = bm * bn * 4
  return x_bytes + h_bytes + signs_bytes + out_panel_bytes + f32_tmp


def get_heuristics_config(
    m: int,
    k: int,
    *,
    block_size: int | None = None,
    lhs_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    quantized: bool = False,
) -> Config:
  """Selects a `(block_m, block_n)` tiling that fits the target VMEM budget.

  `block_m` prefers tiles that divide `m`: other tiles pad `m` outside the
  kernel, which costs a pad and a slice copy plus the padded rows. For the
  dense layouts `block_n` prefers a single `(k, k)` panel. The two-stage and
  block layouts return `block_n=None`.

  Args:
    m: Number of rows (flattened non-rotated dimensions).
    k: Size of the rotated dimension.
    block_size: Optional diagonal Hadamard block size.
    lhs_dtype: Unquantized input dtype.
    quantized: Whether the kernel also quantizes rows.

  Returns:
    A `Config` whose estimated VMEM footprint fits the budget, or the smallest
    tiling if none does.
  """
  budget = int(default_vmem_limit_bytes() * VMEM_BUDGET_FRACTION)
  lhs_itemsize = jnp.dtype(lhs_dtype).itemsize

  max_bm = max(BLOCK_ALIGNMENT, 1 << (max(m, 1) - 1).bit_length())
  sizes = tuple(c for c in (1024, 512, 256, 128) if c <= max_bm) or (
      BLOCK_ALIGNMENT,
  )
  bm_candidates = tuple(c for c in sizes if m % c == 0) + tuple(
      c for c in sizes if m % c
  )

  def fits(bm: int, bn: int | None) -> bool:
    used = hadamard_vmem_bytes(
        bm,
        k,
        bn=bn,
        block_size=block_size,
        lhs_itemsize=lhs_itemsize,
        quantized=quantized,
    )
    return used <= budget

  if _choose_layout(k, block_size=block_size, block_n=None) in (
      "block",
      "two_stage",
  ):
    bm = next((c for c in bm_candidates if fits(c, None)), BLOCK_ALIGNMENT)
    return Config(block_m=bm, block_n=None)

  bn_candidates = divisors_aligned(k, BLOCK_ALIGNMENT)
  for bn in bn_candidates:
    for bm in bm_candidates:
      if fits(bm, bn):
        return Config(block_m=bm, block_n=bn)
  return Config(block_m=BLOCK_ALIGNMENT, block_n=bn_candidates[-1])


def _matmul_f32(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
  """MXU matmul accumulating in `float32`, preserving full `float32` inputs."""
  precision = (
      jax.lax.Precision.HIGHEST
      if lhs.dtype == jnp.float32
      else jax.lax.Precision.DEFAULT
  )
  return jnp.matmul(
      lhs,
      rhs,
      precision=precision,
      preferred_element_type=jnp.float32,
  )


def _split_signs(
    refs: tuple[jax.Ref, ...], has_signs: bool
) -> tuple[jax.Ref | None, tuple[jax.Ref, ...]]:
  """Splits the optional `signs` ref off the front of `refs`."""
  return (refs[0], refs[1:]) if has_signs else (None, refs)


def _rotate(
    x_vmem: jax.Ref,
    h: jax.Array,
    s_vmem: jax.Ref | None,
    *,
    inv_sqrt_n: float,
    inverse: bool,
    cols: slice | pl.Slice = slice(None),
) -> jax.Array:
  """Returns `(x * s) @ h / sqrt(n)` on `cols`; `inverse` applies `s` last."""
  x = x_vmem[:, cols]
  if s_vmem is not None and not inverse:
    x = (x.astype(jnp.float32) * s_vmem[cols]).astype(x.dtype)
  y = _matmul_f32(x, h) * inv_sqrt_n
  if s_vmem is not None and inverse:
    y = y * s_vmem[cols]
  return y.astype(x.dtype)


def _hadamard_single_panel_kernel(
    x_hbm: jax.Ref,
    h_hbm: jax.Ref,
    *refs: jax.Ref,
    bm: int,
    k: int,
    inv_sqrt_n: float,
    has_signs: bool,
    inverse: bool,
    qdtype: jnp.dtype | None,
):
  """Pallas kernel when `block_size is None` and `bn == k`."""
  s_hbm, refs = _split_signs(refs, has_signs)
  rotate = functools.partial(_rotate, inv_sqrt_n=inv_sqrt_n, inverse=inverse)
  m_pad, _ = x_hbm.shape
  grid = (m_pad // bm,)
  x_spec = pl.BlockSpec((bm, k), lambda a: (a, 0))
  h_spec = pl.BlockSpec((k, k), lambda a: (0, 0))
  in_specs = [x_spec, h_spec]
  in_refs = [x_hbm, h_hbm]
  if s_hbm is not None:
    in_specs.append(pl.BlockSpec((k,), lambda a: (0,)))
    in_refs.append(s_hbm)

  if qdtype is None:
    (out_hbm,) = refs
    out_spec = pl.BlockSpec((bm, k), lambda a: (a, 0))

    def body_unquantized(step: pltpu.PipelineStep, x_vmem, h_vmem, *vmem_refs):
      del step  # Unused.
      s_vmem, (o_vmem,) = _split_signs(vmem_refs, has_signs)
      o_vmem[...] = rotate(x_vmem, h_vmem[...], s_vmem)

    pltpu.emit_pipeline(
        body_unquantized,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        core_axis_name=_CORE_AXIS_NAME,
        dimension_semantics=(pltpu.PARALLEL,),
        _explicit_indices=True,
    )(*in_refs, out_hbm)
  else:
    xq_hbm, sx_hbm = refs
    xq_spec = pl.BlockSpec((bm, k), lambda a: (a, 0))
    sx_spec = pl.BlockSpec((bm, SX_LANES), lambda a: (a, 0))

    def body_quantized(step: pltpu.PipelineStep, x_vmem, h_vmem, *vmem_refs):
      del step  # Unused.
      s_vmem, (xq_vmem, sx_vmem) = _split_signs(vmem_refs, has_signs)
      y = rotate(x_vmem, h_vmem[...], s_vmem)
      xq_t, sx_f32 = reference.quantize_rows_2d(y, qdtype)
      xq_vmem[...] = xq_t
      sx_vmem[...] = jnp.broadcast_to(sx_f32, sx_vmem.shape)

    pltpu.emit_pipeline(
        body_quantized,
        grid=grid,
        in_specs=in_specs,
        out_specs=[xq_spec, sx_spec],
        core_axis_name=_CORE_AXIS_NAME,
        dimension_semantics=(pltpu.PARALLEL,),
        _explicit_indices=True,
    )(*in_refs, xq_hbm, sx_hbm)


def _hadamard_multi_panel_kernel(
    x_hbm: jax.Ref,
    h_hbm: jax.Ref,
    *refs: jax.Ref,
    bm: int,
    bn: int,
    k: int,
    inv_sqrt_n: float,
    has_signs: bool,
    inverse: bool,
    qdtype: jnp.dtype | None,
):
  """Pallas kernel when `block_size is None` and `bn < k`."""
  s_hbm, refs = _split_signs(refs, has_signs)
  rotate = functools.partial(_rotate, inv_sqrt_n=inv_sqrt_n, inverse=inverse)
  m_pad, _ = x_hbm.shape
  grid_n = k // bn
  grid = (m_pad // bm, grid_n)

  x_spec = pl.BlockSpec((bm, k), lambda a, b: (a, 0))
  h_spec = pl.BlockSpec((k, bn), lambda a, b: (0, b))
  in_specs = [x_spec, h_spec]
  in_refs = [x_hbm, h_hbm]
  if s_hbm is not None:
    if inverse:
      in_specs.append(pl.BlockSpec((bn,), lambda a, b: (b,)))
    else:
      in_specs.append(pl.BlockSpec((k,), lambda a, b: (0,)))
    in_refs.append(s_hbm)

  if qdtype is None:
    (out_hbm,) = refs
    out_spec = pl.BlockSpec((bm, bn), lambda a, b: (a, b))

    def body_unquantized(step: pltpu.PipelineStep, x_vmem, h_vmem, *vmem_refs):
      del step  # Unused.
      s_vmem, (o_vmem,) = _split_signs(vmem_refs, has_signs)
      o_vmem[...] = rotate(x_vmem, h_vmem[...], s_vmem)

    pltpu.emit_pipeline(
        body_unquantized,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        core_axis_name=_CORE_AXIS_NAME,
        dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
        _explicit_indices=True,
    )(*in_refs, out_hbm)
  else:
    xq_hbm, sx_hbm, rot_cache = refs
    xq_spec = pl.BlockSpec((bm, k), lambda a, b: (a, 0))
    sx_spec = pl.BlockSpec((bm, SX_LANES), lambda a, b: (a, 0))

    def body_quantized(step: pltpu.PipelineStep, x_vmem, h_vmem, *vmem_refs):
      s_vmem, (xq_vmem, sx_vmem) = _split_signs(vmem_refs, has_signs)
      nind = step.index[1]
      rot_cache[nind] = rotate(x_vmem, h_vmem[...], s_vmem)

      @pl.when(nind == grid_n - 1)
      def _quantize_full_row():
        row_absmax = jnp.zeros((bm, 1), dtype=jnp.float32)
        for j in range(grid_n):
          chunk_f32 = rot_cache[j].astype(jnp.float32)
          row_absmax = jnp.maximum(
              row_absmax, jnp.max(jnp.abs(chunk_f32), axis=-1, keepdims=True)
          )
        scale_f32 = reference.scale_from_absmax_f32(
            row_absmax, orig_dtype=x_vmem.dtype, qdtype=qdtype
        )
        sx_vmem[...] = jnp.broadcast_to(scale_f32, sx_vmem.shape)
        for j in range(grid_n):
          xq_vmem[:, pl.Slice(j * bn, bn)] = reference.quantize_with_scale_2d(
              rot_cache[j], scale_f32, qdtype
          )

    pltpu.emit_pipeline(
        body_quantized,
        grid=grid,
        in_specs=in_specs,
        out_specs=[xq_spec, sx_spec],
        core_axis_name=_CORE_AXIS_NAME,
        dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
        _explicit_indices=True,
    )(*in_refs, xq_hbm, sx_hbm)


def _hadamard_block_kernel(
    x_hbm: jax.Ref,
    h_hbm: jax.Ref,
    *refs: jax.Ref,
    bm: int,
    k: int,
    block_size: int,
    inv_sqrt_n: float,
    has_signs: bool,
    inverse: bool,
    qdtype: jnp.dtype | None,
):
  """Pallas kernel for block-diagonal Hadamard (`block_size = b`)."""
  s_hbm, refs = _split_signs(refs, has_signs)
  m_pad, _ = x_hbm.shape
  grid = (m_pad // bm,)
  num_blocks = k // block_size

  x_spec = pl.BlockSpec((bm, k), lambda a: (a, 0))
  h_spec = pl.BlockSpec((block_size, block_size), lambda a: (0, 0))
  in_specs = [x_spec, h_spec]
  in_refs = [x_hbm, h_hbm]
  if s_hbm is not None:
    in_specs.append(pl.BlockSpec((k,), lambda a: (0,)))
    in_refs.append(s_hbm)

  def _rotate_blocks_into(
      x_vmem: jax.Ref,
      h_vmem: jax.Ref,
      s_vmem: jax.Ref | None,
      dst_vmem: jax.Ref,
  ):
    h_mat = h_vmem[...]
    for blk in range(num_blocks):
      cols = pl.Slice(blk * block_size, block_size)
      dst_vmem[:, cols] = _rotate(
          x_vmem,
          h_mat,
          s_vmem,
          inv_sqrt_n=inv_sqrt_n,
          inverse=inverse,
          cols=cols,
      )

  if qdtype is None:
    (out_hbm,) = refs
    out_spec = pl.BlockSpec((bm, k), lambda a: (a, 0))

    def body_unquantized(step: pltpu.PipelineStep, x_vmem, h_vmem, *vmem_refs):
      del step  # Unused.
      s_vmem, (o_vmem,) = _split_signs(vmem_refs, has_signs)
      _rotate_blocks_into(x_vmem, h_vmem, s_vmem, o_vmem)

    pltpu.emit_pipeline(
        body_unquantized,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        core_axis_name=_CORE_AXIS_NAME,
        dimension_semantics=(pltpu.PARALLEL,),
        _explicit_indices=True,
    )(*in_refs, out_hbm)
  else:
    xq_hbm, sx_hbm, y_scratch = refs
    xq_spec = pl.BlockSpec((bm, k), lambda a: (a, 0))
    sx_spec = pl.BlockSpec((bm, SX_LANES), lambda a: (a, 0))

    def body_quantized(step: pltpu.PipelineStep, x_vmem, h_vmem, *vmem_refs):
      del step  # Unused.
      s_vmem, (xq_vmem, sx_vmem) = _split_signs(vmem_refs, has_signs)
      _rotate_blocks_into(x_vmem, h_vmem, s_vmem, y_scratch)
      xq_t, sx_f32 = reference.quantize_rows_2d(y_scratch[...], qdtype)
      xq_vmem[...] = xq_t
      sx_vmem[...] = jnp.broadcast_to(sx_f32, sx_vmem.shape)

    pltpu.emit_pipeline(
        body_quantized,
        grid=grid,
        in_specs=in_specs,
        out_specs=[xq_spec, sx_spec],
        core_axis_name=_CORE_AXIS_NAME,
        dimension_semantics=(pltpu.PARALLEL,),
        _explicit_indices=True,
    )(*in_refs, xq_hbm, sx_hbm)


def _cross_block_butterfly(acc_vmem: jax.Ref, *, bm: int, num_blocks: int):
  """Applies `H_{num_blocks} (x) I_128` in place to a `(bm, k)` float32 ref.

  Runs a radix-2 Sylvester (natural-order) Walsh-Hadamard butterfly across the
  `num_blocks` 128-lane slabs, `_BUTTERFLY_ROWS` rows at a time so each slab is
  one vreg. Butterfly partners are whole slabs, so no lane shuffles are needed.

  Args:
    acc_vmem: `(bm, num_blocks * 128)` float32 VMEM ref, updated in place.
    bm: Number of rows in `acc_vmem`.
    num_blocks: Number of 128-lane slabs (a power of two).
  """

  def _rows(i, carry):
    r = pl.multiple_of(i * _BUTTERFLY_ROWS, _BUTTERFLY_ROWS)
    rows = pl.ds(r, _BUTTERFLY_ROWS)
    cols = [
        pl.ds(g * BLOCK_ALIGNMENT, BLOCK_ALIGNMENT) for g in range(num_blocks)
    ]
    slabs = [acc_vmem[rows, c] for c in cols]
    half = 1
    while half < num_blocks:
      for start in range(0, num_blocks, 2 * half):
        for j in range(start, start + half):
          a, b = slabs[j], slabs[j + half]
          slabs[j], slabs[j + half] = a + b, a - b
      half *= 2
    for c, slab in zip(cols, slabs):
      acc_vmem[rows, c] = slab
    return carry

  jax.lax.fori_loop(0, bm // _BUTTERFLY_ROWS, _rows, 0)


def _hadamard_two_stage_kernel(
    x_hbm: jax.Ref,
    h_hbm: jax.Ref,
    *refs: jax.Ref,
    bm: int,
    k: int,
    inv_sqrt_n: float,
    has_signs: bool,
    inverse: bool,
    qdtype: jnp.dtype | None,
):
  """Pallas kernel for `block_size is None` and a power-of-two `k`.

  Computes `x @ H_k` as `(x @ (I_{k/128} (x) H_128)) @ (H_{k/128} (x) I_128)`:
  the first stage is one `(bm, 128) @ (128, 128)` MXU matmul per 128-lane block
  into a float32 scratch, the second an in-vreg butterfly across blocks
  (`_cross_block_butterfly`). Sylvester matrices satisfy
  `H_{k/128} (x) H_128 = H_k`, so this is the same matrix as the single-panel
  kernel's `(k, k)` operand, and `H_k` is symmetric, so the inverse is the
  same two stages followed by the sign flip.

  Args:
    x_hbm: The `(m, k)` input HBM ref.
    h_hbm: The `H_128` HBM ref.
    *refs: The `signs` HBM ref if `has_signs`, the output HBM ref(s), and the
      `(bm, k)` float32 accumulator scratch.
    bm: Row tile size.
    k: Size of the rotated dimension (a power of two).
    inv_sqrt_n: `1 / sqrt(k)`.
    has_signs: Whether a `signs` ref is passed.
    inverse: Whether to apply the inverse rotation.
    qdtype: Optional quantized dtype; `None` returns the rotation.
  """
  s_hbm, refs = _split_signs(refs, has_signs)
  m_pad, _ = x_hbm.shape
  grid = (m_pad // bm,)
  num_blocks = k // BLOCK_ALIGNMENT

  x_spec = pl.BlockSpec((bm, k), lambda a: (a, 0))
  h_spec = pl.BlockSpec((BLOCK_ALIGNMENT, BLOCK_ALIGNMENT), lambda a: (0, 0))
  in_specs = [x_spec, h_spec]
  in_refs = [x_hbm, h_hbm]
  if s_hbm is not None:
    in_specs.append(pl.BlockSpec((k,), lambda a: (0,)))
    in_refs.append(s_hbm)

  def _rotate_tile(
      x_vmem: jax.Ref,
      h_vmem: jax.Ref,
      s_vmem: jax.Ref | None,
      acc_vmem: jax.Ref,
  ) -> jax.Array:
    h_mat = h_vmem[...]
    for blk in range(num_blocks):
      slc = pl.ds(blk * BLOCK_ALIGNMENT, BLOCK_ALIGNMENT)
      x_blk = x_vmem[:, slc]
      if s_vmem is not None and not inverse:
        x_blk = (x_blk.astype(jnp.float32) * s_vmem[slc]).astype(x_blk.dtype)
      acc_vmem[:, slc] = _matmul_f32(x_blk, h_mat)
    _cross_block_butterfly(acc_vmem, bm=bm, num_blocks=num_blocks)
    y_f32 = acc_vmem[...] * inv_sqrt_n
    if s_vmem is not None and inverse:
      y_f32 = y_f32 * s_vmem[...]
    return y_f32.astype(x_vmem.dtype)

  if qdtype is None:
    out_hbm, acc_scratch = refs
    out_spec = pl.BlockSpec((bm, k), lambda a: (a, 0))

    def body_unquantized(step: pltpu.PipelineStep, x_vmem, h_vmem, *vmem_refs):
      del step  # Unused.
      s_vmem, (o_vmem,) = _split_signs(vmem_refs, has_signs)
      o_vmem[...] = _rotate_tile(x_vmem, h_vmem, s_vmem, acc_scratch)

    pltpu.emit_pipeline(
        body_unquantized,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        core_axis_name=_CORE_AXIS_NAME,
        dimension_semantics=(pltpu.PARALLEL,),
        _explicit_indices=True,
    )(*in_refs, out_hbm)
  else:
    xq_hbm, sx_hbm, acc_scratch = refs
    xq_spec = pl.BlockSpec((bm, k), lambda a: (a, 0))
    sx_spec = pl.BlockSpec((bm, SX_LANES), lambda a: (a, 0))

    def body_quantized(step: pltpu.PipelineStep, x_vmem, h_vmem, *vmem_refs):
      del step  # Unused.
      s_vmem, (xq_vmem, sx_vmem) = _split_signs(vmem_refs, has_signs)
      y = _rotate_tile(x_vmem, h_vmem, s_vmem, acc_scratch)
      xq_t, sx_f32 = reference.quantize_rows_2d(y, qdtype)
      xq_vmem[...] = xq_t
      sx_vmem[...] = jnp.broadcast_to(sx_f32, sx_vmem.shape)

    pltpu.emit_pipeline(
        body_quantized,
        grid=grid,
        in_specs=in_specs,
        out_specs=[xq_spec, sx_spec],
        core_axis_name=_CORE_AXIS_NAME,
        dimension_semantics=(pltpu.PARALLEL,),
        _explicit_indices=True,
    )(*in_refs, xq_hbm, sx_hbm)


@overload
def hadamard_pallas_2d(
    x: jax.Array,
    *,
    signs: jax.Array | None = ...,
    block_size: int | None = ...,
    inverse: bool = ...,
    qtype: Literal[None] = ...,
    config: Config | None = ...,
) -> jax.Array:
  ...


@overload
def hadamard_pallas_2d(
    x: jax.Array,
    *,
    signs: jax.Array | None = ...,
    block_size: int | None = ...,
    inverse: bool = ...,
    qtype: jax.typing.DTypeLike,
    config: Config | None = ...,
) -> tuple[jax.Array, jax.Array]:
  ...


def hadamard_pallas_2d(
    x: jax.Array,
    *,
    signs: jax.Array | None = None,
    block_size: int | None = None,
    inverse: bool = False,
    qtype: jax.typing.DTypeLike | None = None,
    config: Config | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """2-D Pallas entry point: rotates `(m, k)` and optionally quantizes rows.

  Supports `jax.jit` and `jax.vmap` (collapsing unbatched-`signs` `vmap` axes
  into `m` so small per-example batches still execute as a single dense MXU
  pass).

  Args:
    x: 2-D array of shape `(m, k)` in `bfloat16` or `float32`.
    signs: Optional 1-D sign vector of shape `(k,)`.
    block_size: Optional diagonal Hadamard block size, a multiple of 128.
    inverse: If True, applies the inverse rotation (requires `qtype is None`).
    qtype: Optional symmetric quantization dtype (`int8`, `float8_e4m3fn`,
      `float8_e5m2`).
    config: Optional explicit tiling `Config`.

  Returns:
    Rotated `jax.Array` of shape `(m, k)` in `x.dtype` when `qtype is None`, or
    `(xq, scale)` with `xq` of shape `(m, k)` in `qtype` and `scale` of shape
    `(m, 1)` in `x.dtype`.
  """
  if x.ndim != 2:
    raise ValueError(f"hadamard_pallas_2d expects a 2-D array, got {x.shape}.")
  if inverse and qtype is not None:
    raise ValueError("inverse=True cannot be combined with quantization.")
  qdtype = None if qtype is None else reference.normalize_qtype(qtype)

  _, k = x.shape
  rot_dim = k if block_size is None else block_size
  validate_pallas_config(k, block_size=block_size, config=config)

  def _run_2d(
      x_in: jax.Array, s_in: jax.Array | None
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    m_in, k_in = x_in.shape
    cfg = config or get_heuristics_config(
        m_in,
        k_in,
        block_size=block_size,
        lhs_dtype=x_in.dtype,
        quantized=(qdtype is not None),
    )
    bm = cfg.block_m
    layout = _choose_layout(k_in, block_size=block_size, block_n=cfg.block_n)

    m_pad = ((max(m_in, 1) + bm - 1) // bm) * bm
    if m_pad != m_in:
      x_padded = jnp.pad(x_in, ((0, m_pad - m_in), (0, 0)))
    else:
      x_padded = x_in

    h_np = reference.hadamard_matrix(
        BLOCK_ALIGNMENT if layout == "two_stage" else rot_dim
    )
    if inverse:
      h_np = h_np.T
    h = jnp.asarray(h_np, dtype=x_in.dtype)

    in_args: list[jax.Array] = [x_padded, h]
    has_signs = s_in is not None
    if s_in is not None:
      in_args.append(s_in.astype(jnp.float32))

    # `jax.shard_map(check_vma=True)` requires each output to declare the mesh
    # axes it varies over: here, those of `x` and `signs` combined.
    if s_in is not None:
      out_mat = jax.eval_shape(
          lambda a, s: a[:, :1] + s[:1].astype(a.dtype), x_padded, s_in
      ).manual_axis_type
    else:
      out_mat = jax.typeof(x_padded).manual_axis_type
    if qdtype is None:
      out_type: (
          jax.ShapeDtypeStruct
          | tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]
      ) = jax.ShapeDtypeStruct(
          (m_pad, k_in), x_in.dtype, manual_axis_type=out_mat
      )
    else:
      out_type = (
          jax.ShapeDtypeStruct((m_pad, k_in), qdtype, manual_axis_type=out_mat),
          jax.ShapeDtypeStruct(
              (m_pad, SX_LANES), jnp.float32, manual_axis_type=out_mat
          ),
      )

    vmem_limit = default_vmem_limit_bytes()
    scratch_types = ()
    kernel_args = {
        "bm": bm,
        "k": k_in,
        "inv_sqrt_n": 1.0 / math.sqrt(rot_dim),
        "has_signs": has_signs,
        "inverse": inverse,
        "qdtype": qdtype,
    }
    if layout == "block":
      assert block_size is not None
      if qdtype is not None:
        scratch_types = (pltpu.VMEM((bm, k_in), x_in.dtype),)
      kern = functools.partial(
          _hadamard_block_kernel, block_size=block_size, **kernel_args
      )
    elif layout == "two_stage":
      scratch_types = (pltpu.VMEM((bm, k_in), jnp.float32),)
      kern = functools.partial(_hadamard_two_stage_kernel, **kernel_args)
    elif layout == "single_panel":
      kern = functools.partial(_hadamard_single_panel_kernel, **kernel_args)
    else:
      bn = cfg.block_n
      assert bn is not None
      if qdtype is not None:
        scratch_types = (pltpu.VMEM((k_in // bn, bm, bn), x_in.dtype),)
      kern = functools.partial(
          _hadamard_multi_panel_kernel, bn=bn, **kernel_args
      )

    outputs = pl.kernel(
        kern,
        out_type=out_type,
        mesh=pltpu.TensorCoreMesh(axis_name=_CORE_AXIS_NAME),
        scratch_types=scratch_types,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit),
    )(*in_args)

    if qdtype is None:
      return outputs[:m_in]
    xq_pad, sx_wide = outputs
    return xq_pad[:m_in], sx_wide[:m_in, :1].astype(x_in.dtype)

  out_batched = True if qdtype is None else (True, True)

  def _fold_vmap_axis(call, axis_size, x_arg, *args):
    """Calls `call` with the leading vmap axis of `x_arg` folded into rows."""
    _, m_sub, k_sub = x_arg.shape
    out = call(x_arg.reshape(axis_size * m_sub, k_sub), *args)
    unfold = lambda o: o.reshape(axis_size, m_sub, o.shape[-1])
    return jax.tree.map(unfold, out), out_batched

  # JAX only calls a `def_vmap` rule when at least one input is batched.
  if signs is None:

    @custom_batching.custom_vmap
    def _call_no_signs(x_arg: jax.Array):
      return _run_2d(x_arg, None)

    @_call_no_signs.def_vmap
    def _vmap_no_signs(axis_size, in_batched, x_arg):
      del in_batched  # `x_arg` is batched.
      return _fold_vmap_axis(_call_no_signs, axis_size, x_arg)

    return _call_no_signs(x)

  @custom_batching.custom_vmap
  def _call_with_signs(x_arg: jax.Array, s_arg: jax.Array):
    return _run_2d(x_arg, s_arg)

  @_call_with_signs.def_vmap
  def _vmap_with_signs(axis_size, in_batched, x_arg, s_arg):
    x_b, s_b = in_batched
    if not s_b:
      return _fold_vmap_axis(_call_with_signs, axis_size, x_arg, s_arg)
    # Each example has its own signs, i.e. its own rotation: run them in turn.
    if not x_b:
      x_arg = jnp.broadcast_to(x_arg, (axis_size,) + x_arg.shape)
    outs = jax.lax.map(lambda a: _call_with_signs(*a), (x_arg, s_arg))
    return outs, out_batched

  return _call_with_signs(x, signs)
