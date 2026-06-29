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
"""Fused Hadamard Quantization.

This kernel fuses together the Hadamard transform and the quantization step.

Currently supports doing a Hadamard transform on the final axis and quantizes
with 2d scales. Does not use random keys. For more features, refer to the jax
implementation in qwix. Supports int8 quantization.
"""

import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from qwix.contrib import hadamard_rot

_CORE_AXIS_NAME = "core"


def quantize_a_tile(x, dtype=jnp.int8):
  """Quantizes a tile of LHS values."""
  s = jnp.max(jnp.abs(x), keepdims=True, axis=(-2, -1)) / (
      jnp.iinfo(dtype).max + 0.5
  )
  x = jnp.round(x / s).astype(dtype)
  return x, s


def fused_hadamard_quantize_kernel(
    x_hbm: jax.Ref,
    had_mat_hbm: jax.Ref,
    xq_hbm: jax.Ref,
    s_hbm: jax.Ref,
    *,
    bm: int,
    bn: int,
    sm_global: int,
    sn_global: int,
):
  """Fused Hadamard Quantize Kernel.

  Args:
    x_hbm: Reference to the input array in hbm (m, k)
    had_mat_hbm: Reference to the Hadamard matrix in hbm (had_size, had_size)
    xq_hbm: Reference to the output array in hbm (m, k)
    s_hbm: Reference to the output scale array in hbm (sm, sn, 1, 1)
    bm: blockspec for the m dimension
    bn: blockspec for the n dimension
    sm_global: Scale shape for the m dimension
    sn_global: Scale shape for the n dimension
  """
  # Grid
  m, n = x_hbm.shape
  grid = (pl.cdiv(m, bm), pl.cdiv(n, bn))

  # Scale shapes within the kernel
  sm = pl.cdiv(sm_global, grid[0])
  sn = pl.cdiv(sn_global, grid[1])

  # Blockspecs for the kernel
  x_spec = pl.BlockSpec((bm, bn), lambda a, b: (a, b))
  had_mat_spec = pl.BlockSpec(had_mat_hbm.shape, lambda a, b: (0, 0))
  xq_spec = pl.BlockSpec((bm, bn), lambda a, b: (a, b))
  s_spec = pl.BlockSpec((sm, sn, 1, 1), lambda a, b: (a, b, 0, 0))

  # Tile sizes corresponding to scale entries
  m_tile_size = pl.cdiv(bm, sm)
  n_tile_size = pl.cdiv(bn, sn)

  # Kernel body
  def kernel_body(
      x_vmem: jax.Ref,
      had_mat_vmem: jax.Ref,
      xq_vmem: jax.Ref,
      s_vmem: jax.Ref,
  ):
    # Initialize accumulation buffer
    had_mat = had_mat_vmem[...]

    for nloop in range(sn):
      data_n_slc = pl.Slice(nloop * n_tile_size, n_tile_size)
      x = x_vmem[:, data_n_slc]
      xrot = jnp.matmul(x, had_mat)

      # Quantize
      xq, sx = quantize_a_tile(xrot.reshape(-1, m_tile_size, n_tile_size))

      # Write results
      xq_vmem[:, data_n_slc] = xq.reshape(-1, n_tile_size)
      s_vmem[:, nloop] = sx.reshape(-1, 1, 1)

  # Call the kernel.
  pltpu.emit_pipeline(
      kernel_body,
      grid=grid,
      in_specs=[x_spec, had_mat_spec],
      out_specs=[xq_spec, s_spec],
      core_axis_name=_CORE_AXIS_NAME,
      dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
  )(x_hbm, had_mat_hbm, xq_hbm, s_hbm)


def fused_hadamard_quantize(
    x: jax.Array,
    *,
    bm: int,
    bn: int,
    sm: int,
    sn: int,
    dtype: jax.typing.DTypeLike = jnp.int8,
    hadamard_power: int,
):
  """Fused Hadamard Quantization.

  Args:
    x: Input array (m, k)
    bm: blockspec for the m dimension
    bn: blockspec for the n dimension
    sm: Scale shape for the m dimension
    sn: Scale shape for the n dimension
    dtype: Output quantized dtype
    hadamard_power: Power of 2 for the Hadamard matrix

  Returns:
    A tuple of the quantized array and the scales.
  """
  # Create the tensor core mesh
  tc_mesh = pltpu.create_tensorcore_mesh(axis_name=_CORE_AXIS_NAME)

  # Create the output type
  xq_out_type = jax.core.ShapedArray(x.shape, dtype)
  s_out_type = jax.core.ShapedArray((sm, sn, 1, 1), x.dtype)

  had_size = int(2**hadamard_power)
  had_mat = hadamard_rot._create_hadamard_matrix(  # pylint: disable=protected-access
      hadamard_power,
      None,
      row_sign_flip=False,
      col_sign_flip=False,
      dtype=jnp.int8,
  )[
      0
  ]
  n_tile_size = x.shape[1] // sn
  if dtype != jnp.int8:
    raise ValueError("Only int8 quantization is supported.")
  if had_size != n_tile_size:
    raise ValueError(
        "Hadamard matrix size must match the tile size of the unquantized"
        " matrix."
    )

  # Create the kernel with kwargs
  kernel = functools.partial(
      fused_hadamard_quantize_kernel,
      bm=bm,
      bn=bn,
      sm_global=sm,
      sn_global=sn,
  )

  # Call the kernel
  xq, s = pl.kernel(
      kernel,
      out_type=[xq_out_type, s_out_type],
      mesh=tc_mesh,
  )(x, had_mat)
  return xq, s.reshape(sm, sn)
