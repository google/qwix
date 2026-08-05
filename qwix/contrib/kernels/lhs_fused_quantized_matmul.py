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
"""LHS fused quantized matmul implementation using Pallas.

This kernel fuses together the quantization of the LHS and the multiplication
with the RHS. This provides a performance improvement by overlapping the
quantization with the matmul. It caches the quantized LHS in vmem to avoid
recomputing it multiple times.

Info:
This is being actively developed and features will be added over time.

This currently only supports absmax quantization to int8.

Future work:
- Multi-device kernels
- Additional dtype support
- Testing and benchmarking on all TPU generations (focused on TPUv5 for now)
- Integration into the main qwix library
- Stochastic rounding
"""

import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

_CORE_AXIS_NAME = "core"


def quantize_a_tile(x: jax.Array, *, qtype=jnp.int8, method="absmax"):
  """Quantizes a tile of LHS values."""
  # TODO(chapmanjames): Update to support stochastic rounding.
  max_val = jnp.iinfo(qtype).max + 0.49
  if method == "absmax":
    s = jnp.max(jnp.abs(x), axis=(-1, -2), keepdims=True) / max_val
    s_inv = jax.lax.reciprocal(s)
    xq = jnp.round(x * s_inv).astype(qtype)
  else:
    raise ValueError(f"Unsupported quantization method: {method}")
  return xq, s


def lhs_fused_qmm_kernel(
    x_hbm: jax.Ref,
    y_hbm: jax.Ref,
    sy_hbm: jax.Ref,
    o_hbm: jax.Ref,
    accum_vmem: jax.Ref,
    *,
    bm: int,
    bk: int,
    bn: int,
    sm_global: int,
):
  """Fused Quantized Matmul kernel.

  Args:
    x_hbm: Reference to the LHS input array in hbm (m, k)
    y_hbm: Reference to the RHS input array in hbm (k, n)
    sy_hbm: Reference to the RHS scale array in hbm (sk_global, sn_global)
    o_hbm: Reference to the output array in hbm (bm, bn)
    accum_vmem: Scratch reference to the accumulation buffer in vmem (bm, bn)
    bm: blockspec for the m dimension
    bk: blockspec for the k dimension
    bn: blockspec for the n dimension
    sm_global: Scale shape for the m dimension
  """
  sk_global, sn_global = sy_hbm.shape[:2]

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
  y_spec = pl.BlockSpec((bk, bn), lambda a, b, c: (c, b))
  sy_spec = pl.BlockSpec((sk, sn, 1, 1), lambda a, b, c: (c, b, 0, 0))
  o_spec = pl.BlockSpec((bm, bn), lambda a, b, c: (a, b))

  # Tile sizes corresponding to scale entries
  m_tile_size = pl.cdiv(bm, sm)
  k_tile_size = pl.cdiv(bk, sk)
  n_tile_size = pl.cdiv(bn, sn)

  def kernel_body(
      step,
      x_vmem: jax.Ref,
      y_vmem: jax.Ref,
      sy_vmem: jax.Ref,
      o_vmem: jax.Ref,
  ):
    kind = step.index[2]

    @pl.when(kind == 0)
    def _init():
      accum_vmem[...] = jnp.zeros_like(accum_vmem)

    for mloop in range(sm):
      data_m_slc = pl.Slice(mloop * m_tile_size, m_tile_size)
      xq_list, sx_list = [], []

      for nloop in range(sn):
        data_n_slc = pl.Slice(nloop * n_tile_size, n_tile_size)

        # Loop over subchannel axis
        for kloop in range(sk):
          data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)

          # Quantize/Load x
          if nloop == 0:
            # Quantize x
            x = x_vmem[data_m_slc, data_k_slc]
            xq, sx = quantize_a_tile(x)
            xq_list.append(xq)
            sx_list.append(sx)
          else:
            # Load xq and sx
            xq, sx = xq_list[kloop], sx_list[kloop]

          # Load y
          y = y_vmem[data_k_slc, data_n_slc]

          # Low Precision Matmul
          xy = jnp.matmul(xq, y, preferred_element_type=reduction_dtype)

          # Access the sy scales we need
          sy = sy_vmem[kloop, nloop, :, :]

          # Dequantize
          xys = (xy * sx) * sy

          # Accumulate results
          accum_vmem[data_m_slc, data_n_slc] += xys

    # Write results to output buffer.
    @pl.when(kind == grid[2] - 1)
    def _write():
      o_vmem[...] = accum_vmem[...].astype(o_hbm.dtype)

  reduction_dtype = jnp.int32

  # Call the kernel.
  pltpu.emit_pipeline(
      kernel_body,
      grid=grid,
      in_specs=[x_spec, y_spec, sy_spec],
      out_specs=o_spec,
      core_axis_name=_CORE_AXIS_NAME,
      dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY, pltpu.ARBITRARY),
      _explicit_indices=True,
  )(x_hbm, y_hbm, sy_hbm, o_hbm)


def lhs_fused_quantized_matmul(
    x: jax.Array,
    y: jax.Array,
    sy: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    sm: int,
    accum_dtype=jnp.float32,
):
  """LHS Fused Quantized Matmul.

  This function implements a fused quantized matmul operation. The left hand
  side quantization is fused with the matmul operation to improve
  performance.

  Args:
    x: LHS input array (m, k) (not quantized)
    y: RHS input array (k, n) (pre-quantized)
    sy: RHS scale array (sk, sn)
    bm: blockspec for the m dimension
    bk: blockspec for the k dimension
    bn: blockspec for the n dimension
    sm: Scale shape for the m dimension
    accum_dtype: The dtype of the accumulation buffer

  Returns:
    The output array (m, n)
  """

  sy = jnp.expand_dims(sy, axis=(-1, -2))
  dtype = x.dtype

  # Create the tensor core mesh
  tc_mesh = pltpu.create_tensorcore_mesh(axis_name=_CORE_AXIS_NAME)

  # Create the output type
  out_type = jax.core.ShapedArray((x.shape[0], y.shape[1]), dtype)

  # Create the kernel with kwargs
  kernel = functools.partial(
      lhs_fused_qmm_kernel,
      bm=bm,
      bk=bk,
      bn=bn,
      sm_global=sm,
  )

  accum_buffer = pltpu.VMEM((bm, bn), accum_dtype)

  # Call the kernel
  return pl.kernel(
      kernel,
      out_type=out_type,  # pyrefly: ignore
      mesh=tc_mesh,  # pyrefly: ignore
      scratch_types=[accum_buffer],  # pyrefly: ignore
  )(x, y, sy)
