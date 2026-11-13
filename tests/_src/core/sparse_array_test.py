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
"""Tests for N:M structured sparse array utilities."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from qwix._src.core import qarray
from qwix._src.core import sparse_array
from qwix._src.core import sparsity


class SparseArrayTest(parameterized.TestCase):

  def _assert_equal(self, actual: jax.Array, expected: jax.Array) -> None:
    self.assertEqual(actual.dtype, expected.dtype)
    self.assertEqual(actual.shape, expected.shape)
    self.assertTrue(
        jnp.array_equal(actual, expected), f'{actual=}\n{expected=}'
    )

  def test_compress_decompress_literal(self):
    # Tile a 2x8 block to 16x64 so compressed uint2 indices have shape 16x16,
    # meeting TPU's >= 16 second-minor dimension requirement on all backends.
    x = jnp.tile(
        jnp.array(
            [
                [1.0, -4.0, 3.0, -2.0, 5.0, 6.0, 7.0, -8.0],
                [-9.0, 2.0, 3.0, 4.0, 1.0, -0.0, 2.0, -3.0],
            ],
            dtype=jnp.bfloat16,
        ),
        (8, 8),
    )
    mask = jnp.tile(
        jnp.array([
            [False, False, True, False, False, False, False, True],
            [False, False, False, False, False, True, False, False],
        ]),
        (8, 8),
    )

    values, indices = sparse_array.compress(x, mask, n=1, m=4, axis=1)
    self._assert_equal(
        values,
        jnp.tile(
            jnp.array([[3.0, -8.0], [0.0, -0.0]], dtype=jnp.bfloat16),
            (8, 8),
        ),
    )
    self._assert_equal(
        indices,
        jnp.tile(
            jnp.array([[2, 3], [0, 1]], dtype=jnp.int32),
            (8, 8),
        ).astype(jnp.uint2),
    )
    self._assert_equal(
        jnp.signbit(values),
        jnp.tile(
            jnp.array([[False, True], [False, True]]),
            (8, 8),
        ),
    )

    reconstructed = sparse_array.decompress(values, indices, n=1, m=4, axis=1)
    expected = jnp.where(mask, x, jnp.zeros_like(x))
    self._assert_equal(reconstructed, expected)
    self._assert_equal(jnp.signbit(reconstructed), jnp.signbit(expected))

    # Groups with > 1 True entries select the lowest True offset.
    over_mask = jnp.tile(
        jnp.array([
            [False, True, True, True, True, False, True, True],
            [True, True, False, True, False, True, True, True],
        ]),
        (8, 8),
    )
    over_val, over_idx = sparse_array.compress(x, over_mask, n=1, m=4, axis=1)
    self._assert_equal(
        over_val,
        jnp.tile(
            jnp.array([[-4.0, 5.0], [-9.0, -0.0]], dtype=jnp.bfloat16),
            (8, 8),
        ),
    )
    self._assert_equal(
        over_idx,
        jnp.tile(
            jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
            (8, 8),
        ).astype(jnp.uint2),
    )

    # Abstract shape inference also succeeds on small shapes (< 16).
    small_val, small_idx = jax.eval_shape(
        lambda a, b: sparse_array.compress(a, b, n=1, m=4, axis=0),
        jax.ShapeDtypeStruct((4, 4), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((4, 4), jnp.bool_),
    )
    self.assertEqual(
        (small_val.shape, small_val.dtype), ((1, 4), jnp.float8_e4m3fn)
    )
    self.assertEqual((small_idx.shape, small_idx.dtype), ((1, 4), jnp.uint2))

  @parameterized.named_parameters(
      ('fp8_1_4_2d_axis0', (64, 32), jnp.float8_e4m3fn, 1, 4, 0),
      ('bf16_1_4_2d_axis1', (32, 64), jnp.bfloat16, 1, 4, 1),
      ('int4_1_4_2d_axis0', (64, 32), jnp.int4, 1, 4, 0),
      ('int8_1_3_2d_axis_neg1', (32, 48), jnp.int8, 1, 3, -1),
      ('bf16_1_2_2d_axis1', (32, 32), jnp.bfloat16, 1, 2, 1),
      ('fp8_1_4_3d_axis_neg1', (2, 32, 64), jnp.float8_e4m3fn, 1, 4, -1),
      ('int4_1_4_3d_axis_neg2', (2, 64, 32), jnp.int4, 1, 4, -2),
  )
  def test_roundtrip(self, shape, dtype, n, m, axis):
    raw = (
        (jnp.arange(math.prod(shape), dtype=jnp.int32) * 7 + 3) % 15 - 7
    ).reshape(shape)
    x = raw.astype(dtype)
    norm_axis = axis % len(shape)

    def axis_slice(start: int, stop: int) -> tuple[slice, ...]:
      return (slice(None),) * norm_axis + (slice(start, stop),)

    if jnp.issubdtype(dtype, jnp.floating):
      x = x.at[axis_slice(0, m)].set(jnp.array(-0.0, dtype=dtype))

    mask = sparsity.get_sparsity_mask(
        x.astype(jnp.float32), n_sparsity=n, m_sparsity=m, axis=axis
    )
    # Exercise last-slot selection and an all-False (`< n`) group.
    mask = mask.at[axis_slice(m, 2 * m - 1)].set(False)
    mask = mask.at[axis_slice(2 * m - 1, 2 * m)].set(True)
    mask = mask.at[axis_slice(2 * m, 3 * m)].set(False)

    compress_fn = jax.jit(
        lambda a, b: sparse_array.compress(a, b, n=n, m=m, axis=axis)
    )
    decompress_fn = jax.jit(
        lambda v, idx: sparse_array.decompress(v, idx, n=n, m=m, axis=axis)
    )
    values, indices = compress_fn(x, mask)

    expected_shape = (
        shape[:norm_axis] + (shape[norm_axis] // m,) + shape[norm_axis + 1 :]
    )
    self.assertEqual((values.shape, values.dtype), (expected_shape, dtype))
    self.assertEqual(
        (indices.shape, indices.dtype), (expected_shape, jnp.uint2)
    )

    indices_i32 = indices.astype(jnp.int32)
    self.assertTrue(jnp.all((indices_i32 >= 0) & (indices_i32 < m)))

    reconstructed = decompress_fn(values, indices)
    expected = jnp.where(mask, x, jnp.zeros_like(x))
    self._assert_equal(reconstructed, expected)
    if jnp.issubdtype(dtype, jnp.floating):
      self._assert_equal(
          jnp.signbit(reconstructed.astype(jnp.float32)),
          jnp.signbit(expected.astype(jnp.float32)),
      )

    eager_values, eager_indices = sparse_array.compress(
        x, mask, n=n, m=m, axis=axis
    )
    self._assert_equal(eager_indices, indices)
    self._assert_equal(
        sparse_array.decompress(
            eager_values, eager_indices, n=n, m=m, axis=axis
        ),
        expected,
    )

    @jax.jit
    def fused_roundtrip(a, b):
      v, idx = sparse_array.compress(a, b, n=n, m=m, axis=axis)
      return idx, sparse_array.decompress(v, idx, n=n, m=m, axis=axis)

    fused_indices, fused_out = fused_roundtrip(x, mask)
    self._assert_equal(fused_indices, indices)
    self._assert_equal(fused_out, expected)

  def test_vmap_and_named_sharding(self):
    shape = (2, 2, 32, 64)
    raw = ((jnp.arange(8192, dtype=jnp.int32) * 7 + 3) % 15 - 7).reshape(shape)
    x = raw.astype(jnp.float8_e4m3fn)
    mask = jax.vmap(
        lambda sample: sparsity.get_sparsity_mask(
            sample.astype(jnp.float32), n_sparsity=1, m_sparsity=4, axis=-1
        )
    )(x)
    expected = jnp.where(mask, x, jnp.zeros_like(x))

    def roundtrip(array, input_mask):
      v, idx = sparse_array.compress(array, input_mask, n=1, m=4, axis=-1)
      return sparse_array.decompress(v, idx, n=1, m=4, axis=-1)

    self._assert_equal(jax.jit(jax.vmap(roundtrip))(x, mask), expected)

    for axis_type in (
        jax.sharding.AxisType.Explicit,
        jax.sharding.AxisType.Auto,
    ):
      mesh = jax.make_mesh((1,), ('data',), axis_types=(axis_type,))
      sharding = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec('data', None, None)
      )
      sharded_x = jax.device_put(x[0], sharding)
      sharded_mask = jax.device_put(mask[0], sharding)
      with jax.set_mesh(mesh):
        out = jax.jit(roundtrip)(sharded_x, sharded_mask)
      self._assert_equal(out, expected[0])
      if axis_type == jax.sharding.AxisType.Explicit:
        reduced_spec = jax.sharding.PartitionSpec(
            None, None, None, reduced={'data'}
        )
        unreduced_spec = jax.sharding.PartitionSpec(None, None, None)
        with jax.set_mesh(mesh):
          reduced_x = jax.device_put(x[0], reduced_spec)
          reduced_mask = sparsity.get_sparsity_mask(
              reduced_x.astype(jnp.float32),
              n_sparsity=1,
              m_sparsity=4,
              axis=-1,
          )
          v, idx = jax.jit(
              lambda a, b: sparse_array.compress(a, b, n=1, m=4, axis=-1)
          )(reduced_x, reduced_mask)
          reduced_out = jax.jit(
              lambda a, b: sparse_array.decompress(a, b, n=1, m=4, axis=-1)
          )(v, idx)
          self.assertEqual(
              reduced_out.sharding.spec.reduced, frozenset({'data'})
          )
          eager_reduced_out = roundtrip(reduced_x, reduced_mask)
          self.assertEqual(
              eager_reduced_out.sharding.spec.reduced, frozenset({'data'})
          )
          self._assert_equal(
              jax.sharding.reshard(reduced_out, unreduced_spec), expected[0]
          )
          self._assert_equal(
              jax.sharding.reshard(eager_reduced_out, unreduced_spec),
              expected[0],
          )

  @parameterized.named_parameters(
      dict(testcase_name='compress_n_zero', n=0, regex='0 < n < m <= 4'),
      dict(testcase_name='compress_n_eq_m', n=4, m=4, regex='0 < n < m <= 4'),
      dict(testcase_name='compress_m_gt_4', m=8, regex='0 < n < m <= 4'),
      dict(testcase_name='compress_1d', shape=(16,), regex='rank >= 2'),
      dict(testcase_name='compress_empty', shape=(0, 16), regex='non-empty'),
      dict(testcase_name='compress_axis', axis=2, regex='out of range'),
      dict(
          testcase_name='compress_shape',
          rhs_shape=(16, 8),
          regex='shapes must match',
      ),
      dict(
          testcase_name='compress_mask_dtype',
          rhs_dtype=jnp.int32,
          regex='dtype bool',
      ),
      dict(
          testcase_name='compress_div_m',
          shape=(18, 16),
          regex='not divisible by m=4',
      ),
      dict(
          testcase_name='decompress_n_m',
          is_compress=False,
          n=4,
          m=4,
          regex='0 < n < m <= 4',
      ),
      dict(
          testcase_name='decompress_1d',
          is_compress=False,
          shape=(16,),
          regex='rank >= 2',
      ),
      dict(
          testcase_name='decompress_empty',
          is_compress=False,
          shape=(16, 0),
          regex='non-empty',
      ),
      dict(
          testcase_name='decompress_axis',
          is_compress=False,
          axis=-3,
          regex='out of range',
      ),
      dict(
          testcase_name='decompress_shape',
          is_compress=False,
          rhs_shape=(16, 32),
          regex='shapes must match',
      ),
      dict(
          testcase_name='decompress_dtype',
          is_compress=False,
          rhs_dtype=jnp.uint8,
          regex='dtype uint2',
      ),
      dict(
          testcase_name='compress_n_gt_1',
          n=2,
          err_cls=NotImplementedError,
          regex='Only n == 1',
      ),
      dict(
          testcase_name='decompress_n_gt_1',
          is_compress=False,
          n=2,
          err_cls=NotImplementedError,
          regex='Only n == 1',
      ),
  )
  def test_compress_decompress_errors(
      self,
      *,
      regex,
      err_cls=ValueError,
      is_compress=True,
      shape=(16, 16),
      rhs_shape=None,
      rhs_dtype=None,
      n=1,
      m=4,
      axis=0,
  ):
    rhs_shape = shape if rhs_shape is None else rhs_shape
    if rhs_dtype is None:
      rhs_dtype = jnp.bool_ if is_compress else jnp.uint2
    fn = sparse_array.compress if is_compress else sparse_array.decompress
    with self.assertRaisesRegex(err_cls, regex):
      jax.eval_shape(
          lambda: fn(
              jnp.zeros(shape),
              jnp.zeros(rhs_shape, rhs_dtype),
              n=n,
              m=m,
              axis=axis,
          )
      )

  def test_sparse_array_properties_and_axis_normalization(self):
    vals_1_2 = jnp.ones((16, 16), dtype=jnp.bfloat16)
    idx_1_2 = jnp.zeros((16, 16), dtype=jnp.uint2)
    sa = sparse_array.SparseArray(
        vals_1_2, idx_1_2, sparsity.StructuredSparsity(1, 2, dimension=1)
    )
    self.assertEqual(sa.axis, -1)
    self.assertEqual(sa.shape, (16, 32))
    self.assertEqual(sa.ndim, 2)
    self.assertEqual(sa.dtype, jnp.bfloat16)
    self.assertEqual(sa.astype(jnp.float32).dtype, jnp.float32)
    sparse_array.validate_sparse_array(sa)

    qvals = qarray.QArray(
        jnp.zeros((16, 16), dtype=jnp.float8_e4m3fn),
        jnp.ones((1, 16), dtype=jnp.bfloat16),
    )
    idx_1_4 = jnp.zeros((16, 16), dtype=jnp.uint2)
    qsa = sparse_array.SparseArray(
        qvals, idx_1_4, sparsity.StructuredSparsity(1, 4, dimension=0)
    )
    self.assertEqual(qsa.axis, -2)
    self.assertEqual(qsa.shape, (64, 16))
    self.assertEqual(qsa.dtype, jnp.bfloat16)
    qsa_f32 = qsa.astype(jnp.float32)
    self.assertEqual(qsa_f32.dtype, jnp.float32)
    self.assertEqual(qsa_f32.values.qvalue.dtype, jnp.float8_e4m3fn)
    qsa_per_tensor = sparse_array.SparseArray(
        qarray.QArray(qvals.qvalue, jnp.ones((1, 1), dtype=jnp.bfloat16)),
        idx_1_4,
        qsa.sparsity,
    )
    for valid_qsa in (qsa, qsa_per_tensor):
      sparse_array.validate_sparse_array(valid_qsa)
      jax.eval_shape(sparse_array.validate_sparse_array, valid_qsa)

    with self.assertRaisesRegex(ValueError, 'sparsity.dimension to be set'):
      sparse_array.SparseArray(
          vals_1_2, idx_1_2, sparsity.StructuredSparsity(1, 2)
      )
    for dim in (0, -1):
      with self.assertRaisesRegex(NotImplementedError, 'Only n == 1'):
        sparse_array.SparseArray(
            vals_1_2,
            idx_1_2,
            sparsity.StructuredSparsity(2, 4, dimension=dim),
        )
    with self.assertRaisesRegex(ValueError, 'rank >= 2'):
      sparse_array.SparseArray(
          jax.ShapeDtypeStruct((16,), jnp.float32),
          jax.ShapeDtypeStruct((16,), jnp.uint2),
          sparsity.StructuredSparsity(1, 4, dimension=0),
      )
    with self.assertRaisesRegex(ValueError, 'out of range'):
      sparse_array.SparseArray(
          vals_1_2, idx_1_2, sparsity.StructuredSparsity(1, 2, dimension=2)
      )
    bad_sa = sparse_array.SparseArray(
        vals_1_2, idx_1_2, sparsity.StructuredSparsity(1, 2, dimension=-3)
    )
    with self.assertRaisesRegex(ValueError, 'out of range'):
      _ = bad_sa.shape

  def test_sparse_array_pytree_partitioned_and_nnx(self):
    vals = jnp.ones((16, 16), dtype=jnp.float32)
    idx = jnp.zeros((16, 16), dtype=jnp.uint2)
    sa = sparse_array.SparseArray(
        vals, idx, sparsity.StructuredSparsity(1, 4, dimension=0)
    )
    finite = jax.tree.map(jnp.isfinite, sa)
    all_finite = jax.tree.map(jnp.all, finite)
    self.assertEqual(all_finite.values.shape, ())
    self.assertTrue(all_finite.values)
    self.assertTrue(all_finite.indices)
    nones = jax.tree.map(lambda _: None, sa)
    self.assertIsNone(nones.values)
    self.assertIsNone(nones.indices)
    grad_sa = jax.grad(lambda a: jnp.sum(a.values * 2.0), allow_int=True)(sa)
    self._assert_equal(grad_sa.values, jnp.full((16, 16), 2.0, jnp.float32))
    self.assertEqual(grad_sa.indices.dtype, jax.dtypes.float0)

    p_sa = jax.tree.map(lambda x: nn.Partitioned(x, ('in', 'out')), sa)
    self.assertEqual(p_sa.axis, -2)
    self.assertIsInstance(p_sa.values, nn.Partitioned)
    sparse_array.validate_sparse_array(p_sa)
    unboxed = nn.unbox(p_sa)
    self.assertEqual(unboxed.shape, (64, 16))
    sparse_array.validate_sparse_array(unboxed)

    class Holder(nnx.Module):

      def __init__(self, fill: float):
        qvals = qarray.QArray(
            nnx.Param(jnp.zeros((16, 16), dtype=jnp.float8_e4m3fn)),
            nnx.Param(jnp.full((1, 16), fill, dtype=jnp.bfloat16)),
        )
        self.w = sparse_array.SparseArray(
            qvals,
            nnx.Param(jnp.zeros((16, 16), dtype=jnp.uint2)),
            sparsity.StructuredSparsity(1, 4, dimension=0),
        )

    src = Holder(3.0)
    sparse_array.validate_sparse_array(src.w)
    dst = Holder(1.0)
    pure = nnx.to_pure_dict(nnx.state(src))
    dst_state = nnx.state(dst)
    nnx.replace_by_pure_dict(dst_state, pure)
    nnx.update(dst, dst_state)
    self.assertEqual(dst.w.axis, -2)
    self.assertEqual(dst.w.shape, (64, 16))
    self.assertEqual(dst.w.dtype, jnp.bfloat16)
    self._assert_equal(
        dst.w.values.scale.get_raw_value(),
        jnp.full((1, 16), 3.0, dtype=jnp.bfloat16),
    )
    self._assert_equal(
        qarray.dequantize(dst.w),
        jnp.zeros((64, 16), dtype=jnp.bfloat16),
    )

  @parameterized.named_parameters(
      dict(testcase_name='idx_dtype', idx_dtype=jnp.uint8, regex='dtype uint2'),
      dict(
          testcase_name='zero_point',
          scale_shape=(1, 16),
          has_zp=True,
          regex='zero_point',
      ),
      dict(
          testcase_name='nf4_qtype',
          qval_dtype=jnp.uint4,
          qtype='nf4',
          scale_shape=(1, 16),
          regex='Non-uniform qtype',
      ),
      dict(
          testcase_name='qval_dtype',
          qval_dtype=jnp.float32,
          scale_shape=(1, 16),
          regex='not a valid type for qvalue',
      ),
      dict(testcase_name='scale_ndim', scale_shape=(16,), regex='same rank'),
      dict(
          testcase_name='subchannel_sparse_axis',
          scale_shape=(16, 16),
          regex='Subchannel scale along sparse axis',
      ),
      dict(
          testcase_name='subchannel_other_axis',
          scale_shape=(1, 4),
          regex='per-channel or per-tensor',
      ),
  )
  def test_validate_sparse_array_errors(
      self,
      *,
      regex,
      idx_dtype=jnp.uint2,
      qval_dtype=jnp.int8,
      qtype=None,
      scale_shape=None,
      has_zp=False,
  ):
    def build_and_validate():
      if scale_shape is None:
        vals = jnp.ones((16, 16), jnp.bfloat16)
      else:
        vals = qarray.QArray(
            jnp.zeros((16, 16), qval_dtype),
            jnp.ones(scale_shape, jnp.bfloat16),
            zero_point=jnp.zeros(scale_shape, qval_dtype) if has_zp else None,
            qtype=qtype,
        )
      sa = sparse_array.SparseArray(
          vals,
          jnp.zeros((16, 16), idx_dtype),
          sparsity.StructuredSparsity(1, 4, dimension=-2),
      )
      sparse_array.validate_sparse_array(sa)

    with self.assertRaisesRegex(ValueError, regex):
      jax.eval_shape(build_and_validate)


if __name__ == '__main__':
  absltest.main()
