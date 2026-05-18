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
"""Implements a quantized matmul kernel.

Info:
This is being actively developed and features will be added over time.

Terminology:
- tile: refers to the sub-tensor that a scale corresponds to.
- block: refers to the sub-tensor that pallas loads into vmem.

Future work:
- Multi-device kernels
- Additional dtype support
- Testing and benchmarking on all TPU generations (focused on TPUv5 for now)
- Integration into the main qwix library
"""

import dataclasses
import functools
from typing import Any

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from qwix._src.core import qarray

_CORE_AXIS_NAME = 'core'


@dataclasses.dataclass
class QuantizedMatmulConfig:
  """Configuration for the quantized matmul kernel determined by TPUv5 sweep."""

  bm: int = 256
  bk: int = 512
  bn: int = 1024
  dtype: jnp.dtype = jnp.float32


def can_use_qmm(x, sx, y, sy, *, bm, bk, bn):
  """Returns whether the quantized matmul can be used."""
  mdim, kdim = x.shape
  _, ndim = y.shape

  if mdim % bm != 0 or ndim % bn != 0 or kdim % bk != 0:
    # Block size must divide matrix size.
    return False

  # Check the scales reduction axis
  if sx.shape[1] != sy.shape[0]:
    # Number of tiles must match for the scales on the reduction axis.
    return False

  # Scale tile sizes are sufficiently large.
  sm, sk, sn = sx.shape
  if mdim // sm < 128 or kdim // sk < 128 or ndim // sn < 128:
    return False

  # TODO(chapmanjames): Improve this for bfloat16 scales.
  if sx.dtype != jnp.float32 or sy.dtype != jnp.float32:
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


def quantized_matmul_kernel(
    x_hbm: jax.Ref,
    sx_hbm: jax.Ref,
    y_hbm: jax.Ref,
    sy_hbm: jax.Ref,
    o_hbm: jax.Ref,
    accum_vmem: jax.Ref,
    *,
    bm: int,
    bk: int,
    bn: int,
):
  """Quantized matmul kernel.

  Args:
    x_hbm: Reference to the LHS input array in hbm (m, k)
    sx_hbm: Reference to the LHS scale array in hbm (sm_global, sk_global)
    y_hbm: Reference to the RHS input array in hbm (k, n)
    sy_hbm: Reference to the RHS scale array in hbm (sk_global, sn_global)
    o_hbm: Reference to the output array in hbm (m, n)
    accum_vmem: Reference to the accumulation buffer in vmem (bm, bn)
    bm: blockspec for the m dimension
    bk: blockspec for the k dimension
    bn: blockspec for the n dimension
  """

  sm_global, sk_global, *_ = sx_hbm.shape
  sn_global = sy_hbm.shape[1]

  # Grid
  m, k = x_hbm.shape
  _, n = y_hbm.shape
  grid = (pl.cdiv(m, bm), pl.cdiv(n, bn), pl.cdiv(k, bk))

  # Scale shapes within the kernel
  sm = pl.cdiv(sm_global, grid[0])
  sk = pl.cdiv(sk_global, grid[2])
  sn = pl.cdiv(sn_global, grid[1])

  # Blockspecs for the kernel
  x_spec = pl.BlockSpec((bm, bk), lambda a, b, c: (a, c))
  sx_spec = pl.BlockSpec((sm, sk, 1, 1), lambda a, b, c: (a, c, 0, 0))
  y_spec = pl.BlockSpec((bk, bn), lambda a, b, c: (c, b))
  sy_spec = pl.BlockSpec((sk, sn, 1, 1), lambda a, b, c: (c, b, 0, 0))
  o_spec = pl.BlockSpec((bm, bn), lambda a, b, c: (a, b))

  # Tile sizes corresponding to scale entries
  m_tile_size = pl.cdiv(bm, sm)
  k_tile_size = pl.cdiv(bk, sk)
  n_tile_size = pl.cdiv(bn, sn)

  # Kernel body
  def quantized_matmul_body(
      x_vmem: jax.Ref,
      sx_vmem: jax.Ref,
      y_vmem: jax.Ref,
      sy_vmem: jax.Ref,
      o_vmem: jax.Ref,
  ):
    kind = pl.program_id(2)

    # Initialize accumulation buffer
    @pl.when(kind == 0)
    def _init():
      accum_vmem[...] = jnp.zeros_like(accum_vmem)

    for mloop in range(sm):
      data_m_slc = pl.Slice(mloop * m_tile_size, m_tile_size)

      for nloop in range(sn):
        data_n_slc = pl.Slice(nloop * n_tile_size, n_tile_size)

        # Loop over subchannel axis
        for kloop in range(sk):
          data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)

          # Load lhs and rhs
          x = x_vmem[data_m_slc, data_k_slc]
          y = y_vmem[data_k_slc, data_n_slc]

          # Low Precision Matmul
          xy = jnp.matmul(x, y, preferred_element_type=reduction_dtype)

          # Access the scales we need
          sx = sx_vmem[mloop, kloop, :, :]
          sy = sy_vmem[kloop, nloop, :, :]

          # Dequantize
          xys = (xy * sx) * sy

          # Accumulate results
          accum_vmem[data_m_slc, data_n_slc] += xys

    # Write results to output buffer.
    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _write():
      o_vmem[...] = accum_vmem[...].astype(o_vmem.dtype)

  # Set up accumulation buffer
  reduction_dtype = jnp.int32

  # Call the kernel.
  pltpu.emit_pipeline(
      quantized_matmul_body,
      grid=grid,
      in_specs=[x_spec, sx_spec, y_spec, sy_spec],
      out_specs=o_spec,
      core_axis_name=_CORE_AXIS_NAME,
      dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
  )(x_hbm, sx_hbm, y_hbm, sy_hbm, o_hbm)


def quantized_matmul(
    x: jax.Array,
    sx: jax.Array,
    y: jax.Array,
    sy: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    accum_dtype=jnp.float32,
    dtype,
):
  """Computes a quantized matmul using Pallas.

  This function implements a quantized matmul kernel using Pallas. It supports
  2d qvalue arrays and 2d scale arrays. If the scales are 1d, please ensure that
  you expend them along the appropriate axis to match the required 2d shapes.
  e.g. if sx is (sm,) then convert it to (sm, 1). If sx is (sk,) then convert it
  to (1, sk).

  This currently only support float32 scales.

  Args:
    x: The LHS input array (m, k)
    sx: The LHS scale array (sm, sk)
    y: The RHS input array (k, n)
    sy: The RHS scale array (sk, sn)
    bm: blockspec for the m dimension
    bk: blockspec for the k dimension
    bn: blockspec for the n dimension
    accum_dtype: The dtype of the accumulation buffer
    dtype: The dtype of the output array

  Returns:
    The result of the quantized matmul (m, n)
  """

  m, k = x.shape
  _, n = y.shape

  # Check blockspecs divide matrix dimensions.
  if m % bm != 0:
    raise ValueError(f'Block size must divide matrix size,  {m=} {bm=}')
  if n % bn != 0:
    raise ValueError(f'Block size must divide matrix size,  {n=} {bn=}')
  if k % bk != 0:
    raise ValueError(f'Block size must divide matrix size,  {k=} {bk=}')

  # Check matrix reduction axis
  if x.shape[1] != y.shape[0]:
    raise ValueError(
        f'Matrix shapes must match for the reduction axis, {x.shape[1]=}'
        f' {y.shape[0]=}'
    )

  # Check scale reduction axis
  if sx.shape[1] != sy.shape[0]:
    raise ValueError(
        f'Scale shapes must match for the reduction axis, {sx.shape[1]=}'
        f' {sy.shape[0]=}'
    )

  # Pre-process the scales
  min_size = 128
  sx = jnp.expand_dims(sx, axis=(-1, -2))
  sy = jnp.expand_dims(sy, axis=(-1, -2))

  # Tile sizes corresponding to scale entries
  # e.g. shape of the submatrix in x or y that each scale entry covers.
  m_tile_size = m // sx.shape[0]
  k_tile_size = k // sx.shape[1]
  n_tile_size = n // sy.shape[1]

  # Expand scale shapes to make scale indexing easier
  if m_tile_size > min_size:
    sx = jnp.repeat(sx, repeats=m_tile_size // min_size, axis=0)
  if k_tile_size > min_size:
    sx = jnp.repeat(sx, repeats=k_tile_size // min_size, axis=1)
    sy = jnp.repeat(sy, repeats=k_tile_size // min_size, axis=0)
  if n_tile_size > min_size:
    sy = jnp.repeat(sy, repeats=n_tile_size // min_size, axis=1)

  # Create the tensor core mesh
  tc_mesh = pltpu.create_tensorcore_mesh(axis_name=_CORE_AXIS_NAME)

  # Create the output type
  out_type = jax.core.ShapedArray((x.shape[0], y.shape[1]), dtype)

  # Create the kernel with kwargs
  kernel = functools.partial(
      quantized_matmul_kernel,
      bm=bm,
      bk=bk,
      bn=bn,
  )

  if sx.dtype != jnp.float32 or sy.dtype != jnp.float32:
    raise ValueError(f'Scales must be float32, {sx.dtype=} {sy.dtype=}')

  # Call the kernel
  return pl.kernel(
      kernel,
      out_type=out_type,
      mesh=tc_mesh,
      scratch_types=[pltpu.VMEM((bm, bn), accum_dtype)],
  )(x, sx, y, sy)
