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

"""Pallas/Mosaic TPU kernel for the fused FP8 weight gradient (dRHS).

Computes `drhs = lhs^T @ dout` for the fused FP8 matmul, where `lhs` is the
`(m, k)` FP8 activation with one float32 scale per row, i.e. the residual the
forward pass saved. The contraction runs over `m`, so the row scale cannot be
factored out of it: the kernel folds it into the cotangent, quantizes the
scaled cotangent block to FP8 with one scale per column, runs the contraction
on the FP8 MXU and applies the column scale to the result. The quantized
cotangent never leaves VMEM.

The numerics are those of `reference.fused_fp8_drhs_reference`, except that
the column scale is taken per m block rather than over all of `m`: `m` is the
contracting dimension and does not fit in one block at training shapes, so
each `(bm, bn)` cotangent block is quantized against its own column absmax
and the rescaled partial product is accumulated in float32.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from qwix.contrib.kernels.fused_fp8_matmul import forward_kernel
from qwix.contrib.kernels.fused_fp8_matmul import reference

_CORE_AXIS_NAME = "core"

# An 8-bit memref tiles VMEM as `(32, 128)`, so a block edge along the minor
# (lane) axis of an FP8 operand has to be a whole number of 128-lane tiles or
# Mosaic rejects the DMA. `bk` is the lane axis of the activation block.
_FP8_LANE_TILE = 128

# One float32 `(sublane, lane)` VMEM tile: the smallest useful allocation,
# used for a scratch buffer the chosen tiling never reads.
_DEAD_SCRATCH_SHAPE = (8, 128)

# Tilings measured on TPU7x, with a bfloat16 cotangent, for the training shapes
# this kernel was built for, in order of preference. (4096, 1024, 1024) wins
# outright on (k, n) = (3072, 4096), (4096, 3072) and (3072, 2048), and
# (4096, 1792, 512) on (1792, 3072), where k admits no bk = 1024. `bm` is the
# axis that pays: the grid has m innermost, so the float32 output tile is
# read-modify-written once per m block; with the cotangent handed in
# pre-quantized, (32768, 3072, 4096) measured 877 / 646 / 535 / 486 us at
# bm = 512 / 1024 / 2048 / 4096 at constant HBM traffic. bm = 8192 is absent on
# purpose: it forces bn down to 512 and measured 14% slower.
_MEASURED_TILES = (
    (4096, 1024, 1024),
    (4096, 1792, 512),
    (2048, 1024, 1024),
    (2048, 1792, 512),
    (1024, 1024, 2048),
    (1024, 1024, 1536),
    (1024, 512, 3072),
)
# Limits for shapes the table does not cover, from the same measurements. A
# `(bk, bn)` float32 output tile is live several times over inside the kernel
# (double-buffered output block, the dot result, the rescaled partial, the
# accumulator), which caps `bk * bn`: 1024 x 2048 compiles, 1536 x 2048 does
# not. The `(bm, bn)` cotangent block runs out of VMEM independently of that
# cap: (1024, 1024, 2048) compiles, (1024, 512, 4096) does not. Below roughly
# 1M elements the output tile is under-filled and grid overhead dominates:
# (512, 2048) at bm = 1024 measured 911 us against 691 us for (1024, 2048).
_MAX_TILE_ELEMS = 2_200_000
_MAX_G_BLOCK_ELEMS = 3_500_000
_MIN_TILE_ELEMS = 1_000_000
# The largest cotangent block in the measured table. The table and the caps
# above were measured with a bfloat16 cotangent; a float32 cotangent block
# takes twice the VMEM, and (4096, 1024, 1024) with one overflows VMEM by
# 7 MiB on (32768, 3072, 4096) while (2048, 1024, 1024) fits. So block sizes
# are compared in bfloat16-equivalent elements, `bm * bn * itemsize / 2`.
_MAX_MEASURED_G_BLOCK_ELEMS = 4096 * 1024


def fit_drhs_tiles(
    m: int, k: int, n: int, *, cotangent_itemsize: int
) -> tuple[int, int, int]:
  """Picks `(bm, bk, bn)` for the dRHS kernel.

  The first entry of `_MEASURED_TILES` that divides the shape and whose
  cotangent block is no larger than the largest one measured wins. For other
  shapes the tile is derived: the largest `bm` first (one m block when `m`
  itself fits, which also quantizes each cotangent column against a single
  scale), then the widest `bn` whose cotangent block fits, then the largest
  `bk` under the output-tile cap, moving to a narrower `bn` when the widest
  one leaves room only for an under-filled tile.

  Args:
    m: The contracting dimension.
    k: The row dimension of `drhs`.
    n: The column dimension of `drhs`.
    cotangent_itemsize: Bytes per element of the cotangent: 2 for `bfloat16`, 4
      for `float32`. Block sizes are compared in bfloat16-equivalent elements,
      see `_MAX_MEASURED_G_BLOCK_ELEMS`.

  Returns:
    `(bm, bk, bn)`, each dividing its dimension, `bk` a multiple of 128.
  """
  # Cotangent block size in bfloat16-equivalent elements.
  g_elems = lambda bm, bn: bm * bn * cotangent_itemsize // 2

  for bm, bk, bn in _MEASURED_TILES:
    if m % bm == 0 and k % bk == 0 and n % bn == 0:
      if g_elems(bm, bn) <= _MAX_MEASURED_G_BLOCK_ELEMS:
        return bm, bk, bn

  bms = forward_kernel.tile_candidates(m, (4096, 2048, 1024, 512, 256, 128))
  if m <= 4096 and m % 128 == 0 and m not in bms:
    bms = (m,) + bms
  bns = forward_kernel.tile_candidates(n, (n, 2048, 1536, 1024, 512, 256))
  bks = forward_kernel.tile_candidates(k, (k, 2048, 1536, 1024, 512, 256, 128))
  bks = tuple(b for b in bks if b % _FP8_LANE_TILE == 0) or (_FP8_LANE_TILE,)
  for bm in bms:
    for bn in bns:
      if g_elems(bm, bn) > _MAX_G_BLOCK_ELEMS:
        continue
      for bk in bks:
        if bk * bn > _MAX_TILE_ELEMS:
          continue
        if bk * bn < _MIN_TILE_ELEMS and (bk, bn) != (k, n):
          break  # Under-filled; a narrower `bn` admits a larger `bk`.
        return bm, bk, bn
  return bms[-1], bks[-1], bns[-1]


def get_drhs_heuristics_config(
    m: int, k: int, n: int, *, lhs_dtype: jax.typing.DTypeLike = jnp.bfloat16
) -> forward_kernel.Config:
  """Returns the tiling the measured heuristics pick for the dRHS kernel.

  Each m block of the cotangent is quantized against its own column scales, so
  `block_m == m` gives one scale per column and needs no accumulator. Measured
  on TPU7x.

  Args:
    m: The contracting dimension (rows of the activation and cotangent).
    k: The row dimension of `drhs`.
    n: The column dimension of `drhs`.
    lhs_dtype: The activation dtype, which is also the cotangent's. A `float32`
      cotangent block takes twice the VMEM of a `bfloat16` one.
  """
  bm, bk, bn = fit_drhs_tiles(
      m, k, n, cotangent_itemsize=jnp.dtype(lhs_dtype).itemsize
  )
  return forward_kernel.Config(block_m=bm, block_k=bk, block_n=bn)


def _quantize_g(g_vmem: jax.Ref, sx_vmem: jax.Ref):
  """Folds the activation row scales into the cotangent block and quantizes it.

  The multiply stays in the cotangent's own dtype: promoting the `(bm, bn)`
  block to float32 made 8 MB temporaries at `bm = 512, bn = 4096` in an
  earlier version and most large tilings failed to compile.

  Args:
    g_vmem: The `(bm, bn)` cotangent block.
    sx_vmem: The `(bm, 1)` float32 activation row scales.

  Returns:
    `(gq, sg)`: the FP8 block and its `(bn,)` float32 column scales.
  """
  g = g_vmem[...]
  return reference.quantize_cols_fp8(g * sx_vmem[...].astype(g.dtype))


def _partial_product(xq_vmem: jax.Ref, sx_vmem: jax.Ref, g_vmem: jax.Ref):
  """Returns one m block's rescaled contribution to the `(bk, bn)` output.

  Args:
    xq_vmem: The `(bm, bk)` FP8 activation block.
    sx_vmem: The `(bm, 1)` float32 activation row scales.
    g_vmem: The `(bm, bn)` cotangent block.

  Returns:
    The `(bk, bn)` float32 partial product, already rescaled by the block's
    column scales.
  """
  gq, sg = _quantize_g(g_vmem, sx_vmem)
  # Contract the leading dim of both blocks: the MXU takes the FP8 activation
  # block transposed.
  xg = jax.lax.dot_general(
      xq_vmem[...],
      gq,
      dimension_numbers=(((0,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  return xg * sg  # `(bn,)` broadcasts along the minor axis for free.


def _drhs_kernel(
    xq_hbm: jax.Ref,
    sx_hbm: jax.Ref,
    g_hbm: jax.Ref,
    o_hbm: jax.Ref,
    accum_vmem: jax.Ref,
    *,
    bm: int,
    bk: int,
    bn: int,
):
  """Pipelined dRHS kernel body. See `fused_fp8_drhs_pallas`.

  Args:
    xq_hbm: `(m, k)` FP8 activation.
    sx_hbm: `(m, 1)` float32 activation row scales.
    g_hbm: `(m, n)` cotangent.
    o_hbm: `(k, n)` weight gradient.
    accum_vmem: `(bk, bn)` float32 accumulator, read only when `bm < m`.
    bm: Block size along `m`, the contracting dimension.
    bk: Block size along `k`.
    bn: Block size along `n`.
  """
  m, k = xq_hbm.shape
  _, n = g_hbm.shape
  # `m` innermost: the output block index does not depend on it, so the
  # pipeline keeps one output block resident across the whole reduction.
  grid = (pl.cdiv(k, bk), pl.cdiv(n, bn), pl.cdiv(m, bm))
  single_m = grid[2] == 1

  xq_spec = pl.BlockSpec((bm, bk), lambda ki, ni, mi: (mi, ki))
  sx_spec = pl.BlockSpec((bm, 1), lambda ki, ni, mi: (mi, 0))
  g_spec = pl.BlockSpec((bm, bn), lambda ki, ni, mi: (mi, ni))
  o_spec = pl.BlockSpec((bk, bn), lambda ki, ni, mi: (ki, ni))

  def body_single_m(xq_vmem, sx_vmem, g_vmem, o_vmem):
    """Direct write when `bm == m`: no accumulator needed."""
    o_vmem[...] = _partial_product(xq_vmem, sx_vmem, g_vmem).astype(
        o_vmem.dtype
    )

  def body_multi_m(xq_vmem, sx_vmem, g_vmem, o_vmem):
    """Pipelined reduction over m blocks when `bm < m`."""
    m_index = pl.program_id(2)

    @pl.when(m_index == 0)
    def _init():
      accum_vmem[...] = jnp.zeros_like(accum_vmem)

    # Each m block is quantized against its own column absmax, so the partial
    # product comes back from `_partial_product` already rescaled.
    accum_vmem[...] += _partial_product(xq_vmem, sx_vmem, g_vmem)

    @pl.when(m_index == pl.num_programs(2) - 1)
    def _write():
      o_vmem[...] = accum_vmem[...].astype(o_vmem.dtype)

  pltpu.emit_pipeline(
      body_single_m if single_m else body_multi_m,
      grid=grid,
      in_specs=[xq_spec, sx_spec, g_spec],
      out_specs=o_spec,
      core_axis_name=_CORE_AXIS_NAME,
      dimension_semantics=(
          pltpu.PARALLEL,
          pltpu.PARALLEL,
          pltpu.PARALLEL if single_m else pltpu.ARBITRARY,
      ),
  )(xq_hbm, sx_hbm, g_hbm, o_hbm)


def fused_fp8_drhs_pallas(
    lhs_q: jax.Array,
    lhs_scale: jax.Array,
    dout: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    out_dtype: jnp.dtype | None = None,
) -> jax.Array:
  """Fused FP8 weight gradient: `lhs_q^T @ quantize_cols(dout * lhs_scale)`.

  Args:
    lhs_q: Activations of shape `(m, k)` already quantized to `float8_e4m3fn`.
    lhs_scale: Per-row activation scales of shape `(m, 1)`, `float32`.
    dout: Cotangent of shape `(m, n)`, `bfloat16` or `float32`.
    bm: Block size along `m`, the contracting dimension. Must divide `m`. Each m
      block of the cotangent is quantized against its own column scales; with
      `bm == m` there is one scale per column and no accumulator.
    bk: Block size along `k`. Must divide `k` and be a multiple of 128.
    bn: Block size along `n`. Must divide `n`.
    out_dtype: Output dtype. Defaults to `dout.dtype`.

  Returns:
    `drhs` of shape `(k, n)`.
  """
  m, k = lhs_q.shape
  m_g, n = dout.shape
  if m_g != m:
    raise ValueError(f"Contracting dims differ: lhs has m={m}, dout has {m_g}.")
  if lhs_scale.shape != (m, 1):
    raise ValueError(
        f"lhs_scale must have shape ({m}, 1), got {lhs_scale.shape}."
    )
  if lhs_q.dtype != reference.FP8_DTYPE:
    raise ValueError(f"lhs_q must be {reference.FP8_DTYPE}, got {lhs_q.dtype}.")
  if m % bm:
    raise ValueError(f"m ({m}) must be divisible by bm ({bm}).")
  if k % bk:
    raise ValueError(f"k ({k}) must be divisible by bk ({bk}).")
  if n % bn:
    raise ValueError(f"n ({n}) must be divisible by bn ({bn}).")
  if bk % _FP8_LANE_TILE:
    raise ValueError(
        f"bk ({bk}) must be a multiple of {_FP8_LANE_TILE}: it is the lane axis"
        " of the FP8 activation block."
    )

  # The accumulator is dead when `bm == m`; allocate one tile then rather
  # than dropping it, so the kernel signature stays fixed.
  accum_shape = (bk, bn) if m > bm else _DEAD_SCRATCH_SHAPE
  kernel = functools.partial(_drhs_kernel, bm=bm, bk=bk, bn=bn)
  return pl.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((k, n), out_dtype or dout.dtype),
      mesh=pltpu.create_tensorcore_mesh(axis_name=_CORE_AXIS_NAME),
      scratch_types=[pltpu.VMEM(accum_shape, jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=forward_kernel.VMEM_LIMIT_BYTES
      ),
  )(lhs_q, lhs_scale, dout)
