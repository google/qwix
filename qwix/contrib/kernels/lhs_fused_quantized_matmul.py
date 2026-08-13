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


def quantize_a_tile(x, axis, dtype=jnp.int8):
  """Quantizes a tile of LHS values."""
  s = jnp.max(jnp.abs(x), axis=axis, keepdims=True) / (
      jnp.iinfo(dtype).max + 0.5
  )
  x = jnp.rint(x / s).astype(dtype)
  return x, s


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
    quantize_tile_fn=quantize_a_tile,
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
    quantize_tile_fn: Function to quantize a tile of LHS values.
  """
  sk_global, sn_global = sy_hbm.shape
  qtype = y_hbm.dtype
  reduction_dtype = jnp.int32

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
  sy_spec = pl.BlockSpec(
      (sk, sn_global), lambda a, b, c: (c, 0), memory_space=pltpu.SMEM
  )
  o_spec = pl.BlockSpec((bm, bn), lambda a, b, c: (a, b))

  # Tile sizes corresponding to scale entries
  m_tile_size = pl.cdiv(bm, sm)
  k_tile_size = pl.cdiv(bk, sk)
  n_tile_size = pl.cdiv(bn, sn)

  # Check if sm == bm (e.g. lhs should use 1d sub-channel quantization)
  quantize_tile_fn_axis = 1 if sm == bm else None
  m_subtile_iter_size = min(128, bm) if sm == bm else m_tile_size
  m_sub_iters = bm // m_subtile_iter_size
  assert bm % m_subtile_iter_size == 0

  def kernel_body_single_k(
      x_vmem: jax.Ref, y_vmem: jax.Ref, sy_vmem: jax.Ref, o_vmem: jax.Ref
  ):
    """Quantize then matmul when bk == k."""
    nind = pl.program_id(1)

    # Quantize
    all_xq, all_sx = dict(), dict()
    for mloop in range(m_sub_iters):
      data_m_slc = pl.Slice(mloop * m_subtile_iter_size, m_subtile_iter_size)
      for kloop in range(sk):
        data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)
        x = x_vmem[data_m_slc, data_k_slc]
        xq, sx = quantize_tile_fn(x, quantize_tile_fn_axis, qtype)
        all_xq[(mloop, kloop)] = xq
        all_sx[(mloop, kloop)] = sx

    # Quantized Matmul
    for mloop in range(m_sub_iters):
      data_m_slc = pl.Slice(mloop * m_subtile_iter_size, m_subtile_iter_size)
      for nloop in range(sn):
        data_n_slc = pl.Slice(nloop * n_tile_size, n_tile_size)
        res = jnp.zeros(
            (m_subtile_iter_size, n_tile_size), dtype=accum_vmem.dtype
        )
        for kloop in range(sk):
          data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)

          xq = all_xq[(mloop, kloop)]
          sx = all_sx[(mloop, kloop)]
          y = y_vmem[data_k_slc, data_n_slc]
          xy = jnp.matmul(xq, y, preferred_element_type=reduction_dtype)
          sy = sy_vmem[kloop, nind * sn + nloop]
          xys = xy * (sx * sy)
          res += xys

        # Write results to output buffer.
        o_vmem[data_m_slc, data_n_slc] = res.astype(o_vmem.dtype)

  def kernel_body_multiple_k(
      x_vmem: jax.Ref,
      y_vmem: jax.Ref,
      sy_vmem: jax.Ref,
      o_vmem: jax.Ref,
  ):
    """Quantize then matmul when bk != k."""
    nind, kind = pl.program_id(1), pl.program_id(2)

    @pl.when(kind == 0)
    def _init():
      # Hadamard and quantize
      all_xq, all_sx = dict(), dict()
      for mloop in range(m_sub_iters):
        data_m_slc = pl.Slice(mloop * m_subtile_iter_size, m_subtile_iter_size)
        for kloop in range(sk):
          data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)
          x = x_vmem[data_m_slc, data_k_slc]
          xq, sx = quantize_tile_fn(x, quantize_tile_fn_axis, qtype)
          all_xq[(mloop, kloop)] = xq
          all_sx[(mloop, kloop)] = sx

      # Quantized Matmul
      res = None
      for mloop in range(m_sub_iters):
        data_m_slc = pl.Slice(mloop * m_subtile_iter_size, m_subtile_iter_size)
        for nloop in range(sn):
          data_n_slc = pl.Slice(nloop * n_tile_size, n_tile_size)
          for kloop in range(sk):
            data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)

            xq = all_xq[(mloop, kloop)]
            sx = all_sx[(mloop, kloop)]
            y = y_vmem[data_k_slc, data_n_slc]
            xy = jnp.matmul(xq, y, preferred_element_type=reduction_dtype)
            sy = sy_vmem[kloop, nind * sn + nloop]
            xys = xy * (sx * sy)
            if kloop == 0:
              res = xys
            else:
              res += xys
          accum_vmem[data_m_slc, data_n_slc] = res

    @pl.when((kind > 0) & (kind < pl.num_programs(2) - 1))
    def _middle():
      # Hadamard and quantize
      all_xq, all_sx = dict(), dict()
      for mloop in range(m_sub_iters):
        data_m_slc = pl.Slice(mloop * m_subtile_iter_size, m_subtile_iter_size)
        for kloop in range(sk):
          data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)
          x = x_vmem[data_m_slc, data_k_slc]
          xq, sx = quantize_tile_fn(x, quantize_tile_fn_axis, qtype)
          all_xq[(mloop, kloop)] = xq
          all_sx[(mloop, kloop)] = sx

      # Quantized Matmul
      for mloop in range(m_sub_iters):
        data_m_slc = pl.Slice(mloop * m_subtile_iter_size, m_subtile_iter_size)
        for nloop in range(sn):
          data_n_slc = pl.Slice(nloop * n_tile_size, n_tile_size)
          res = accum_vmem[data_m_slc, data_n_slc]
          for kloop in range(sk):
            data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)

            xq = all_xq[(mloop, kloop)]
            sx = all_sx[(mloop, kloop)]
            y = y_vmem[data_k_slc, data_n_slc]
            xy = jnp.matmul(xq, y, preferred_element_type=reduction_dtype)
            sy = sy_vmem[kloop, nind * sn + nloop]
            xys = xy * (sx * sy)
            res += xys
          accum_vmem[data_m_slc, data_n_slc] = res

    # Write results to output buffer.
    @pl.when(kind == pl.num_programs(2) - 1)
    def _write():
      # Quantize
      all_xq, all_sx = dict(), dict()
      for mloop in range(m_sub_iters):
        data_m_slc = pl.Slice(mloop * m_subtile_iter_size, m_subtile_iter_size)
        for kloop in range(sk):
          data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)
          x = x_vmem[data_m_slc, data_k_slc]
          xq, sx = quantize_tile_fn(x, quantize_tile_fn_axis, qtype)
          all_xq[(mloop, kloop)] = xq
          all_sx[(mloop, kloop)] = sx

      # Quantized Matmul
      for mloop in range(m_sub_iters):
        data_m_slc = pl.Slice(mloop * m_subtile_iter_size, m_subtile_iter_size)
        for nloop in range(sn):
          data_n_slc = pl.Slice(nloop * n_tile_size, n_tile_size)
          res = accum_vmem[data_m_slc, data_n_slc]
          for kloop in range(sk):
            data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)

            xq = all_xq[(mloop, kloop)]
            sx = all_sx[(mloop, kloop)]
            y = y_vmem[data_k_slc, data_n_slc]
            xy = jnp.matmul(xq, y, preferred_element_type=reduction_dtype)
            sy = sy_vmem[kloop, nind * sn + nloop]
            xys = xy * (sx * sy)
            res += xys
          o_vmem[data_m_slc, data_n_slc] = res.astype(o_vmem.dtype)

  kernel_body = kernel_body_multiple_k if grid[2] > 1 else kernel_body_single_k

  # Call the kernel.
  pltpu.emit_pipeline(
      kernel_body,
      grid=grid,
      in_specs=[x_spec, y_spec, sy_spec],
      out_specs=o_spec,
      core_axis_name=_CORE_AXIS_NAME,
      dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY, pltpu.ARBITRARY),
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
    quantize_tile_fn=quantize_a_tile,
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
    quantize_tile_fn: Function to quantize a tile of LHS values.

  Returns:
    The output array (m, n)
  """
  # Create the tensor core mesh
  tc_mesh = pltpu.create_tensorcore_mesh(axis_name=_CORE_AXIS_NAME)

  # Create the output type
  out_type = jax.core.ShapedArray((x.shape[0], y.shape[1]), x.dtype)

  # Create the kernel with kwargs
  kernel = functools.partial(
      lhs_fused_qmm_kernel,
      bm=bm,
      bk=bk,
      bn=bn,
      sm_global=sm,
      quantize_tile_fn=quantize_tile_fn,
  )

  accum_buffer = pltpu.VMEM((bm, bn), accum_dtype)

  # Call the kernel
  return pl.kernel(
      kernel,
      out_type=out_type,
      mesh=tc_mesh,
      scratch_types=[accum_buffer],
  )(x, y, sy)
