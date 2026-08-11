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
"""Fused Hadamard quantized matmul.

This file introduces the lhs_fused_hadamard_quantized_matmul kernel, which
quantizes the hadamard transformed LHS matrix and multiplies it with the
pre-quantized RHS matrix. This file also introduces hadamard_transform_rhs to
apply the Hadamard transform to the RHS matrix in a way that is compatible
with lhs_fused_hadamard_quantized_matmul.
"""

import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

HAD_LHS_KEY_FOLD_IN: int = 0
HAD_RHS_KEY_FOLD_IN: int = 1
_CORE_AXIS_NAME = "core"


def _create_base_hadamard_matrix(power: int) -> jax.Array:
  """Returns a bfloat16 Hadamard matrix of size 2^power x 2^power."""
  if power == 0:
    return jnp.array([[1]], dtype=jnp.bfloat16)
  if power < 0:
    raise ValueError("Power must be non-negative.")
  had_block = _create_base_hadamard_matrix(power - 1)
  return jnp.block([[had_block, had_block], [had_block, -had_block]])


def hadamard_matrix(power: int, dtype=jnp.int8) -> jax.Array:
  """Returns a Hadamard matrix of size 2^power x 2^power."""
  return _create_base_hadamard_matrix(power).astype(dtype)


def quantize_a_tile(x, axis, dtype=jnp.int8):
  """Quantizes a tile of LHS values."""
  s = jnp.max(jnp.abs(x), axis=axis, keepdims=True) / (
      jnp.iinfo(dtype).max + 0.5
  )
  x = jnp.rint(x / s).astype(dtype)
  return x, s


def hadamard_quantize_multiply_kernel(
    x_hbm: jax.Ref,
    y_hbm: jax.Ref,
    sy_hbm: jax.Ref,
    had_mat_hbm: jax.Ref,
    key_hbm: jax.Ref,
    o_hbm: jax.Ref,
    accum_vmem: jax.Ref,
    *,
    bm: int,
    bk: int,
    bn: int,
    sm_global: int,
    quantize_tile_fn=quantize_a_tile,
):
  """Hadamard + quantize then matmul kernel."""
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
  had_mat_spec = pl.BlockSpec(had_mat_hbm.shape, lambda a, b, c: (0, 0))
  key_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
  o_spec = pl.BlockSpec((bm, bn), lambda a, b, c: (a, b))

  # Tile sizes corresponding to scale entries
  m_tile_size = pl.cdiv(bm, sm)
  k_tile_size = pl.cdiv(bk, sk)
  n_tile_size = pl.cdiv(bn, sn)

  had_size = had_mat_hbm.shape[0]

  # Check if sm == bm (e.g. lhs should use 1d sub-channel quantization)
  quantize_tile_fn_axis = 1 if sm == bm else None
  m_subtile_iter_size = min(128, bm) if sm == bm else m_tile_size
  m_sub_iters = bm // m_subtile_iter_size
  assert bm % m_subtile_iter_size == 0

  def kernel_body_single_k(
      x_vmem: jax.Ref,
      y_vmem: jax.Ref,
      sy_vmem: jax.Ref,
      had_mat_vmem: jax.Ref,
      key_vmem: jax.Ref,
      o_vmem: jax.Ref,
  ):
    """Hadamard + quantize then matmul when bk == k."""
    nind, kind = pl.program_id(1), pl.program_id(2)
    key = key_vmem[...]
    had_mat = had_mat_vmem[...]

    # Hadamard and quantize
    all_xq, all_sx = dict(), dict()
    for mloop in range(m_sub_iters):
      data_m_slc = pl.Slice(mloop * m_subtile_iter_size, m_subtile_iter_size)
      for kloop in range(sk):
        data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)
        x = x_vmem[data_m_slc, data_k_slc]
        sub_block_idx = kind * sk + kloop

        lhs_key = jax.random.fold_in(jax.random.fold_in(key, 0), sub_block_idx)
        rhs_key = jax.random.fold_in(jax.random.fold_in(key, 1), sub_block_idx)

        dtype = x_vmem.dtype
        lhs_mask = jax.random.bernoulli(lhs_key, shape=(1, had_size))
        lhs_sign_flips = jnp.where(lhs_mask, 1, -1).astype(dtype)
        x_col = x * lhs_sign_flips
        x_rot = jnp.matmul(x_col, had_mat, preferred_element_type=jnp.float32)
        rhs_mask = jax.random.bernoulli(rhs_key, shape=(1, had_size))
        rhs_sign_flips = jnp.where(rhs_mask, 1, -1).astype(dtype)
        x_rot = x_rot * rhs_sign_flips

        xq, sx = quantize_tile_fn(x_rot, quantize_tile_fn_axis, qtype)
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
      had_mat_vmem: jax.Ref,
      key_vmem: jax.Ref,
      o_vmem: jax.Ref,
  ):
    """Hadamard + quantize then matmul when bk != k."""
    nind, kind = pl.program_id(1), pl.program_id(2)
    key = key_vmem[...]
    had_mat = had_mat_vmem[...]

    @pl.when(kind == 0)
    def _init():
      # Hadamard and quantize
      all_xq, all_sx = dict(), dict()
      for mloop in range(m_sub_iters):
        data_m_slc = pl.Slice(mloop * m_subtile_iter_size, m_subtile_iter_size)
        for kloop in range(sk):
          data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)
          x = x_vmem[data_m_slc, data_k_slc]
          sub_block_idx = kind * sk + kloop

          lhs_key = jax.random.fold_in(
              jax.random.fold_in(key, 0), sub_block_idx
          )
          rhs_key = jax.random.fold_in(
              jax.random.fold_in(key, 1), sub_block_idx
          )

          dtype = x_vmem.dtype
          lhs_mask = jax.random.bernoulli(lhs_key, shape=(1, had_size))
          lhs_sign_flips = jnp.where(lhs_mask, 1, -1).astype(dtype)
          x_col = x * lhs_sign_flips
          x_rot = jnp.matmul(x_col, had_mat, preferred_element_type=jnp.float32)
          rhs_mask = jax.random.bernoulli(rhs_key, shape=(1, had_size))
          rhs_sign_flips = jnp.where(rhs_mask, 1, -1).astype(dtype)
          x_rot = x_rot * rhs_sign_flips

          xq, sx = quantize_tile_fn(x_rot, quantize_tile_fn_axis, qtype)
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
          sub_block_idx = kind * sk + kloop

          lhs_key = jax.random.fold_in(
              jax.random.fold_in(key, 0), sub_block_idx
          )
          rhs_key = jax.random.fold_in(
              jax.random.fold_in(key, 1), sub_block_idx
          )

          dtype = x_vmem.dtype
          lhs_mask = jax.random.bernoulli(lhs_key, shape=(1, had_size))
          lhs_sign_flips = jnp.where(lhs_mask, 1, -1).astype(dtype)
          x_col = x * lhs_sign_flips
          x_rot = jnp.matmul(x_col, had_mat, preferred_element_type=jnp.float32)
          rhs_mask = jax.random.bernoulli(rhs_key, shape=(1, had_size))
          rhs_sign_flips = jnp.where(rhs_mask, 1, -1).astype(dtype)
          x_rot = x_rot * rhs_sign_flips

          xq, sx = quantize_tile_fn(x_rot, quantize_tile_fn_axis, qtype)
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

    # # Write results to output buffer.
    @pl.when(kind == pl.num_programs(2) - 1)
    def _write():
      # Hadamard and quantize
      all_xq, all_sx = dict(), dict()
      for mloop in range(m_sub_iters):
        data_m_slc = pl.Slice(mloop * m_subtile_iter_size, m_subtile_iter_size)
        for kloop in range(sk):
          data_k_slc = pl.Slice(kloop * k_tile_size, k_tile_size)
          x = x_vmem[data_m_slc, data_k_slc]
          sub_block_idx = kind * sk + kloop

          lhs_key = jax.random.fold_in(
              jax.random.fold_in(key, 0), sub_block_idx
          )
          rhs_key = jax.random.fold_in(
              jax.random.fold_in(key, 1), sub_block_idx
          )

          dtype = x_vmem.dtype
          lhs_mask = jax.random.bernoulli(lhs_key, shape=(1, had_size))
          lhs_sign_flips = jnp.where(lhs_mask, 1, -1).astype(dtype)
          x_col = x * lhs_sign_flips
          x_rot = jnp.matmul(x_col, had_mat, preferred_element_type=jnp.float32)
          rhs_mask = jax.random.bernoulli(rhs_key, shape=(1, had_size))
          rhs_sign_flips = jnp.where(rhs_mask, 1, -1).astype(dtype)
          x_rot = x_rot * rhs_sign_flips

          xq, sx = quantize_tile_fn(x_rot, quantize_tile_fn_axis, qtype)
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
      in_specs=[x_spec, y_spec, sy_spec, had_mat_spec, key_spec],
      out_specs=o_spec,
      core_axis_name=_CORE_AXIS_NAME,
      dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
  )(x_hbm, y_hbm, sy_hbm, had_mat_hbm, key_hbm, o_hbm)


def hadamard_quantize_multiply(
    x: jax.Array,
    y: jax.Array,
    sy: jax.Array,
    key: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    sm_global: int,
    hadamard_power: int,
    accum_dtype=jnp.float32,
    quantize_tile_fn=quantize_a_tile,
):
  """Hadamard + quantize then matmul.

  Args:
    x: The input matrix of shape (m, k).
    y: The quantized input matrix of shape (k, n).
    sy: The input scale matrix of shape (sk, sn).
    key: The random key to use for sign flips.
    bm: The blockspec for the m dimension.
    bk: The blockspec for the k dimension.
    bn: The blockspec for the n dimension.
    sm_global: The global scale shape for the m dimension.
    hadamard_power: The power of 2 to generate the Hadamard matrix.
    accum_dtype: The dtype to use for the accumulator.
    quantize_tile_fn: The function to use for quantization.

  Returns:
    The output matrix of shape (m, n).
  """
  dtype = x.dtype

  # Create the tensor core mesh
  tc_mesh = pltpu.create_tensorcore_mesh(axis_name=_CORE_AXIS_NAME)

  # Create the output type
  out_type = jax.core.ShapedArray((x.shape[0], y.shape[1]), dtype)

  # Create the hadamard matrix
  had_mat = hadamard_matrix(hadamard_power, dtype=jnp.int32)
  assert had_mat.shape[-1] <= bk and bk % had_mat.shape[-1] == 0

  # Convert key to Pallas key to use hardware PRNG
  p_key = pltpu.to_pallas_key(key)

  # Create the kernel with kwargs
  kernel = functools.partial(
      hadamard_quantize_multiply_kernel,
      bm=bm,
      bk=bk,
      bn=bn,
      sm_global=sm_global,
      quantize_tile_fn=quantize_tile_fn,
  )

  accum_buffer = pltpu.VMEM((bm, bn), accum_dtype)

  # Call the kernel
  return pl.kernel(
      kernel,
      out_type=out_type,
      mesh=tc_mesh,
      scratch_types=[accum_buffer],
  )(x, y, sy, had_mat, p_key)


def _pallas_sample_sign_flips(
    key: jax.Array, folds: list[int], shape: tuple[int, ...], dtype
):
  """Samples sign flips using Pallas keys."""

  def kernel_body(key_ref, o_ref):
    # Load the Pallas key from SMEM
    k = key_ref[...]
    # Fold into the Pallas key inside the kernel
    folded_key = k
    for fold in reversed(folds):
      folded_key = jax.random.fold_in(folded_key, fold)

    mask = jax.random.bernoulli(folded_key, shape=shape)
    sign_flips = jnp.where(mask, 1, -1)
    o_ref[...] = sign_flips.astype(dtype)

  p_key = pltpu.to_pallas_key(key)
  out_shape = jax.ShapeDtypeStruct(shape, dtype)
  return pl.pallas_call(
      kernel_body,
      in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
      out_shape=out_shape,
  )(p_key)


def hadamard_transform_rhs(x: jax.Array, had_power: int, key: jax.Array):
  """Applies the Hadamard transform to the RHS of the matrix.

  This function is compatible with the hadamard_quantize_multiply function since
  it uses the same key-folding scheme.

  Args:
    x: The input matrix of shape (k, m).
    had_power: The power of 2 to generate the Hadamard matrix.
    key: The random key to use for sign flips.

  Returns:
    The Hadamard transformed matrix of shape (k, m).
  """
  k, _ = x.shape
  # p_key = pltpu.to_pallas_key(key)
  had_mat = hadamard_matrix(had_power, dtype=jnp.int32)
  had_size = had_mat.shape[0]
  out_mats = []
  for i in range(k // had_size):
    lhs_folds = [i, HAD_LHS_KEY_FOLD_IN]
    rhs_folds = [i, HAD_RHS_KEY_FOLD_IN]

    # Now we do RHS first since this is the transposed operation.
    dtype = x.dtype
    rhs_sign_flips = _pallas_sample_sign_flips(
        key, rhs_folds, (1, had_size), dtype
    ).reshape(had_size, 1)
    h = rhs_sign_flips * had_mat
    lhs_sign_flips = _pallas_sample_sign_flips(
        key, lhs_folds, (1, had_size), dtype
    )

    h = h * lhs_sign_flips
    out_mats.append(jnp.matmul(h, x[i * had_size : (i + 1) * had_size]))
  return jnp.concatenate(out_mats, axis=0) / had_size
