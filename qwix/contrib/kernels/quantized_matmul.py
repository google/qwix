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
"""Implements a quantized matmul kernel."""

import dataclasses
from typing import Any

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from qwix._src.core import qarray

# TODO(chapmanjames): Update flag to False for production use.
INTERPRET: bool = True


@dataclasses.dataclass
class QuantizedMatmulConfig:
  bm: int = 128
  bk: int = 128
  bn: int = 128
  dtype: jnp.dtype = jnp.float32


def can_use_qmm(x, sx, y, sy, *, bm, bk, bn):
  """Returns whether the quantized matmul can be used."""
  mdim, kdim = x.shape
  _, ndim = y.shape
  k_tiles = sx.shape[1]

  if mdim % bm != 0 or ndim % bn != 0 or kdim % bk != 0:
    # Block size must divide matrix size.
    return False
  grid = (mdim // bm, ndim // bn, kdim // bk)

  # k information
  k_tile_size = kdim // k_tiles
  if k_tile_size != bk:
    # Block size must match the tile size for the reduction axis.
    return False
  if sx.shape[1] != sy.shape[0]:
    # Number of tiles must match for the scales.
    return False

  if sx.shape[0] != grid[0] and sx.shape[0] != 1:
    # Scale size must match grid size or be 1.
    return False

  if sy.shape[1] != grid[1] and sy.shape[1] != 1:
    # Scale size must match grid size or be 1.
    return False

  return True


def can_use_qmm_in_dot_general(
    lhs: qarray.QArray | jax.Array,
    rhs: qarray.QArray | jax.Array,
    dimension_numbers: Any,
    *,
    config: QuantizedMatmulConfig,
):
  """Returns whether the quantized matmul can be used in dot_general."""
  # Check the qarrays.
  if not isinstance(lhs, qarray.QArray):
    return False
  if not isinstance(rhs, qarray.QArray):
    return False
  if lhs.zero_point is not None or rhs.zero_point is not None:
    return False

  # Check the dimension numbers.
  if not (
      len(dimension_numbers) == 2
      and len(dimension_numbers[0]) == 2
      and len(dimension_numbers[1]) == 2
      and tuple(dimension_numbers[0][0]) == (1,)
      and tuple(dimension_numbers[0][1]) == (0,)
      and len(dimension_numbers[1][0]) == 0
      and len(dimension_numbers[1][1]) == 0
  ):
    return False

  if not can_use_qmm(
      lhs.qvalue,
      lhs.scale,
      rhs.qvalue,
      rhs.scale,
      bm=config.bm,
      bk=config.bk,
      bn=config.bn,
  ):
    return False

  return True


def quantized_matmul_kernel(x_ref, sx_ref, y_ref, sy_ref, o_ref):
  @pl.when(pl.program_id(2) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)

  o_ref[...] += (
      jnp.matmul(x_ref[...], y_ref[...], preferred_element_type=o_ref.dtype)
      * sx_ref[...]
      * sy_ref[...]
  )


def q_matmul(x, sx, y, sy, *, bm=128, bk=128, bn=128, dtype=jnp.float32):
  """Computes a quantized matmul with support for subchannel quantization.

  This kernel does not cover all cases. In particular, it requires that
  the block sizes match the tile sizes, and that the scale sizes match the grid
  size or be 1.

  Args:
    x: The left-hand side matrix.
    sx: The left-hand side scales.
    y: The right-hand side matrix.
    sy: The right-hand side scales.
    bm: The block size for the m dimension.
    bk: The block size for the k dimension.
    bn: The block size for the n dimension.
    dtype: The data type of the output.

  Returns:
    The quantized matmul.
  """
  mdim, kdim = x.shape
  _, ndim = y.shape
  k_tiles = sx.shape[1]

  # Block specs for x and y.
  assert mdim % bm == 0, f'Block size must divide matrix size,  {mdim=} {bm=}'
  assert ndim % bn == 0, f'Block size must divide matrix size,  {ndim=} {bn=}'
  assert kdim % bk == 0, f'Block size must divide matrix size,  {kdim=} {bk=}'
  grid = (mdim // bm, ndim // bn, kdim // bk)
  x_blockspec = pl.BlockSpec((bm, bk), lambda a, b, c: (a, c))
  y_blockspec = pl.BlockSpec((bk, bn), lambda a, b, c: (c, b))

  # k information
  k_tile_size = kdim // k_tiles
  assert k_tile_size == bk, (
      'Block size must match the tile size for the reduction axis'
      f' {k_tile_size=} {bk=}'
  )
  assert sx.shape[1] == sy.shape[0], 'Number of tiles must match for the scales'

  # m information
  if sx.shape[0] == 1:
    sx_blockspec = pl.BlockSpec((1, 1), lambda a, b, c: (0, c))
  else:
    assert (
        sx.shape[0] == grid[0]
    ), f'Scale size must match grid size,  {sx.shape[0]=} {grid[0]=}'
    sx_blockspec = pl.BlockSpec((1, 1), lambda a, b, c: (a, c))

  # n information
  if sy.shape[1] == 1:
    sy_blockspec = pl.BlockSpec((1, 1), lambda a, b, c: (c, 0))
  else:
    assert (
        sy.shape[1] == grid[1]
    ), f'Scale size must match grid size,  {sy.shape[1]=} {grid[1]=}'
    sy_blockspec = pl.BlockSpec((1, 1), lambda a, b, c: (c, b))

  return pl.pallas_call(
      quantized_matmul_kernel,
      out_shape=jax.ShapeDtypeStruct((mdim, ndim), dtype),
      grid=grid,
      in_specs=(x_blockspec, sx_blockspec, y_blockspec, sy_blockspec),
      out_specs=pl.BlockSpec((bm, bn), lambda a, b, c: (a, b)),
      interpret=INTERPRET,
  )(x, sx, y, sy).astype(dtype)
