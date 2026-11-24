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

import functools
import inspect
from typing import cast

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from qwix._src.core import qarray
from qwix._src.core import take


def _make_qtable(
    *,
    shape: tuple[int, ...] = (32, 16),
    qtype: jax.typing.DTypeLike = jnp.int8,
    channelwise_axes: tuple[int, ...] = (0,),
    tiled_axes: dict[int, int] | None = None,
    calibration_method: str = 'absmax',
    scale_dtype: jax.typing.DTypeLike = jnp.float32,
    seed: int = 0,
) -> qarray.QArray:
  rng = np.random.default_rng(seed)
  table = rng.uniform(-1.5, 1.5, size=shape).astype(np.float32)
  return qarray.quantize_api(
      jnp.asarray(table),
      qtype,
      channelwise_axes=channelwise_axes,
      tiled_axes=tiled_axes,
      calibration_method=calibration_method,
      scale_dtype=scale_dtype,
  )


class TakeTest(parameterized.TestCase):

  def test_validation_errors(self):
    qtable = _make_qtable()
    ids = jnp.array([0, 1], dtype=jnp.int32)
    empty_qt = qarray.QArray(
        jnp.zeros((0, 4), jnp.int8), jnp.ones((0, 1), jnp.float32)
    )
    bad_zero_scale = qarray.QArray(
        jnp.zeros((4, 8), jnp.int8), jnp.ones((0, 1), jnp.float32)
    )
    bad_zero_qval_scale = qarray.QArray(
        jnp.zeros((0, 4), jnp.int8), jnp.ones((2, 1), jnp.float32)
    )
    bad_zero_scale_dtype = qarray.QArray(
        jnp.zeros((0, 4), jnp.int8), jnp.zeros((0, 1), jnp.int32)
    )
    bad_zero_zp = qarray.QArray(
        jnp.zeros((4, 4), jnp.int8),
        jnp.ones((1, 1), jnp.float32),
        zero_point=jnp.zeros((0, 1), jnp.int8),
    )

    with self.assertRaisesRegex(NotImplementedError, "'out'"):
      take.take(qtable, ids, axis=0, out='buf')  # type: ignore[arg-type]
    with self.assertRaisesRegex(NotImplementedError, "'raise'"):
      take.take(qtable, ids, axis=0, mode='raise')
    with self.assertRaisesRegex(NotImplementedError, 'out_sharding'):
      take.take(qtable, ids, axis=0, out_sharding=jax.sharding.PartitionSpec())
    with self.assertRaisesRegex(ValueError, 'Invalid mode'):
      take.take(qtable, ids, axis=0, mode='bad')
    with self.assertRaisesRegex(ValueError, 'integer indices'):
      take.take(qtable, jnp.array([0.0], dtype=jnp.float32), axis=0)
    with self.assertRaisesRegex(ValueError, 'integer indices'):
      take.take(qtable, jnp.array([True, False]), axis=0)
    with self.assertRaisesRegex(ValueError, 'scalar fill_value'):
      take.take(
          qtable,
          ids,
          axis=0,
          fill_value=[1.0, 2.0],  # type: ignore[arg-type]
      )
    with self.assertRaisesRegex(ValueError, 'scalar fill_value'):
      take.take(
          qtable,
          ids,
          axis=0,
          mode='clip',
          fill_value=jnp.array([1.0, 2.0]),  # type: ignore[arg-type]
      )
    with self.assertRaisesRegex(ValueError, 'out of bounds'):
      take.take(qtable, ids, axis=2)
    with self.assertRaisesRegex(IndexError, 'empty axis'):
      take.take(empty_qt, ids, axis=0)
    with self.assertRaisesRegex(IndexError, 'empty axis'):
      take.take(empty_qt, ids, axis=None)
    with self.assertRaisesRegex(ValueError, 'broadcastable'):
      take.take(bad_zero_scale, jnp.zeros((0,), jnp.int32), axis=0)
    with self.assertRaisesRegex(ValueError, 'broadcastable'):
      take.take(bad_zero_qval_scale, jnp.zeros((0,), jnp.int32), axis=0)
    with self.assertRaisesRegex(ValueError, 'valid type for scale'):
      take.take(bad_zero_scale_dtype, jnp.zeros((0,), jnp.int32), axis=0)
    with self.assertRaisesRegex(ValueError, 'broadcastable'):
      take.take(bad_zero_zp, ids, axis=0)

  @parameterized.product(
      channelwise_axes=((), (0,)),
      mode=(None, 'fill', 'clip', 'wrap'),
      fill_value=(None, 0.0, -3.5),
  )
  def test_parity_with_jnp_take_2d_axis0(
      self, channelwise_axes, mode, fill_value
  ):
    qtable = _make_qtable(channelwise_axes=channelwise_axes)
    qvalue = qtable.qvalue.at[0, 0].set(jnp.int8(-128))
    qtable = qarray.QArray(
        qvalue, qtable.scale, qtable.zero_point, qtable.qtype
    )
    ref_table = qarray.dequantize(qtable)

    ids = jnp.array([[0, 1, 31, -1], [-32, 32, -33, 5]], dtype=jnp.int32)
    take_fn = jax.jit(
        functools.partial(take.take, axis=0, mode=mode, fill_value=fill_value)
    )
    out = take_fn(qtable, ids)
    expected = jnp.take(
        ref_table, ids, axis=0, mode=mode, fill_value=fill_value
    )
    self.assertEqual(out.dtype, expected.dtype)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))

    # Empty indices return zeros of the expected shape and dtype.
    empty_ids = jnp.zeros((2, 0), dtype=jnp.int32)
    empty_out = take_fn(qtable, empty_ids)
    empty_expected = jnp.take(
        ref_table, empty_ids, axis=0, mode=mode, fill_value=fill_value
    )
    self.assertEqual(empty_out.shape, empty_expected.shape)
    self.assertEqual(empty_out.dtype, empty_expected.dtype)
    np.testing.assert_array_equal(
        np.asarray(empty_out), np.asarray(empty_expected)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='axis_none_per_tensor',
          shape=(8, 4),
          qtype=jnp.int8,
          channelwise_axes=(),
          tiled_axes=None,
          calibration_method='absmax',
          scale_dtype=jnp.float32,
          axis=None,
          ids=[0, 5, -1, 32],
      ),
      dict(
          testcase_name='axis_none_per_row',
          shape=(8, 4),
          qtype=jnp.int8,
          channelwise_axes=(0,),
          tiled_axes=None,
          calibration_method='absmax',
          scale_dtype=jnp.float32,
          axis=None,
          ids=[0, 5, -1, 32],
      ),
      dict(
          testcase_name='axis_none_per_col_fallback',
          shape=(8, 4),
          qtype=jnp.int8,
          channelwise_axes=(1,),
          tiled_axes=None,
          calibration_method='absmax',
          scale_dtype=jnp.float32,
          axis=None,
          ids=[0, 5, -1, 32],
      ),
      dict(
          testcase_name='axis_1_subchannel',
          shape=(8, 16),
          qtype=jnp.int8,
          channelwise_axes=(0,),
          tiled_axes={1: 4},
          calibration_method='absmax',
          scale_dtype=jnp.float32,
          axis=1,
          ids=[[0, 3, 4, 15, -1, 16]],
      ),
      dict(
          testcase_name='axis_neg1_asymmetric_bf16',
          shape=(6, 8),
          qtype=jnp.int8,
          channelwise_axes=(0,),
          tiled_axes=None,
          calibration_method='minmax',
          scale_dtype=jnp.bfloat16,
          axis=-1,
          ids=[0, 7, -2, 8],
      ),
      dict(
          testcase_name='axis_0_nf4_scalar_index',
          shape=(16, 8),
          qtype='nf4',
          channelwise_axes=(0,),
          tiled_axes=None,
          calibration_method='absmax',
          scale_dtype=jnp.float32,
          axis=0,
          ids=-1,
      ),
      dict(
          testcase_name='axis_0_str_int8_qtype',
          shape=(16, 8),
          qtype='int8',
          channelwise_axes=(0,),
          tiled_axes=None,
          calibration_method='absmax',
          scale_dtype=jnp.float32,
          axis=0,
          ids=[0, 3, -1, 16],
      ),
      dict(
          testcase_name='axis_0_np_dtype_int8_qtype',
          shape=(16, 8),
          qtype=np.dtype('int8'),
          channelwise_axes=(0,),
          tiled_axes=None,
          calibration_method='absmax',
          scale_dtype=jnp.float32,
          axis=0,
          ids=[0, 3, -1, 16],
      ),
      dict(
          testcase_name='axis_1_3d_subchannel_along_gathered_axis',
          shape=(4, 12, 6),
          qtype=jnp.int8,
          channelwise_axes=(0,),
          tiled_axes={1: 4},
          calibration_method='absmax',
          scale_dtype=jnp.float32,
          axis=1,
          ids=[0, 3, 4, 11, -1, 12],
      ),
  )
  def test_general_qarray_take(
      self,
      shape,
      qtype,
      channelwise_axes,
      tiled_axes,
      calibration_method,
      scale_dtype,
      axis,
      ids,
  ):
    qtable = _make_qtable(
        shape=shape,
        qtype=qtype,
        channelwise_axes=channelwise_axes,
        tiled_axes=tiled_axes,
        calibration_method=calibration_method,
        scale_dtype=scale_dtype,
    )
    ref_table = qarray.dequantize(qtable)
    ids_arr = jnp.asarray(ids, dtype=jnp.int32)

    out = take.take(qtable, ids_arr, axis=axis, fill_value=0.0)
    expected = jnp.take(ref_table, ids_arr, axis=axis, fill_value=0.0)
    self.assertEqual(out.dtype, expected.dtype)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))

  def test_zero_size_qarray(self):
    empty_rows_qt = qarray.QArray(
        jnp.zeros((0, 4), jnp.int8), jnp.zeros((0, 1), jnp.float32)
    )
    out = take.take(empty_rows_qt, jnp.array([0], jnp.int32), axis=1)
    self.assertEqual(out.shape, (0, 1))
    self.assertEqual(out.dtype, jnp.float32)

    empty_zp_qt = qarray.QArray(
        jnp.zeros((0, 4), jnp.int8),
        jnp.ones((1, 1), jnp.float32),
        zero_point=jnp.zeros((0, 1), jnp.int8),
    )
    out_zp = take.take(empty_zp_qt, jnp.array([0], jnp.int32), axis=1)
    self.assertEqual(out_zp.shape, (0, 1))
    self.assertEqual(out_zp.dtype, jnp.float32)

    empty_cols_qt = qarray.QArray(
        jnp.zeros((4, 0), jnp.int8), jnp.ones((4, 1), jnp.float32)
    )
    out_cols = take.take(empty_cols_qt, jnp.array([0, 1], jnp.int32), axis=0)
    self.assertEqual(out_cols.shape, (2, 0))
    self.assertEqual(out_cols.dtype, jnp.float32)

    out_flat_empty = take.take(
        empty_rows_qt, jnp.zeros((0,), jnp.int32), axis=None
    )
    self.assertEqual(out_flat_empty.shape, (0,))
    self.assertEqual(out_flat_empty.dtype, jnp.float32)

  @parameterized.named_parameters(
      dict(
          testcase_name='axis0_empty_np',
          channelwise_axes=(0,),
          axis=0,
          empty_shape=(0,),
          use_jnp=False,
      ),
      dict(
          testcase_name='axis0_empty_jnp',
          channelwise_axes=(0,),
          axis=0,
          empty_shape=(0,),
          use_jnp=True,
      ),
      dict(
          testcase_name='axis0_empty_2d_float',
          channelwise_axes=(0,),
          axis=0,
          empty_shape=(2, 0),
          use_jnp=False,
      ),
      dict(
          testcase_name='axis_none_per_row',
          channelwise_axes=(0,),
          axis=None,
          empty_shape=(0,),
          use_jnp=False,
      ),
      dict(
          testcase_name='axis_none_per_col_fallback',
          channelwise_axes=(1,),
          axis=None,
          empty_shape=(0,),
          use_jnp=False,
      ),
  )
  def test_empty_untyped_indices(
      self, channelwise_axes, axis, empty_shape, use_jnp
  ):
    qtable = _make_qtable(shape=(8, 4), channelwise_axes=channelwise_axes)
    ref_table = qarray.dequantize(qtable)
    empty_ids = jnp.zeros(empty_shape) if use_jnp else np.zeros(empty_shape)
    out = take.take(qtable, empty_ids, axis=axis)
    expected = jnp.take(ref_table, empty_ids, axis=axis)
    self.assertEqual(out.shape, expected.shape)
    self.assertEqual(out.dtype, expected.dtype)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))

  @parameterized.parameters(None, 'fill', 'clip', 'wrap')
  def test_grad_scale_with_unique_indices_and_oob(self, mode):
    qtable = _make_qtable(shape=(10, 4), channelwise_axes=(0,))
    # Distinct raw indices [9, 100] both clamp to 9 in fill/clip modes.
    ids = jnp.array([9, 100], dtype=jnp.int32)
    weights = jnp.array([2.0, 3.0], dtype=jnp.float32)[:, None]

    def loss_take(scale):
      qa = qarray.QArray(qtable.qvalue, scale, qtable.zero_point, qtable.qtype)
      rows = take.take(
          qa,
          ids,
          axis=0,
          mode=mode,
          unique_indices=True,
          fill_value=0.0,
      )
      return jnp.sum(rows * weights)

    def loss_ref(scale):
      qa = qarray.QArray(qtable.qvalue, scale, qtable.zero_point, qtable.qtype)
      rows = jnp.take(
          qarray.dequantize(qa),
          ids,
          axis=0,
          mode=mode,
          fill_value=0.0,
      )
      return jnp.sum(rows * weights)

    grad_take = jax.grad(loss_take)(qtable.scale)
    grad_ref = jax.grad(loss_ref)(qtable.scale)
    np.testing.assert_allclose(
        np.asarray(grad_take), np.asarray(grad_ref), rtol=1e-6, atol=1e-6
    )

  def test_vmap_and_host_numpy_qarray(self):
    qtable = _make_qtable(shape=(16, 8), channelwise_axes=(0,))
    ref_table = qarray.dequantize(qtable)
    batch_ids = jnp.array([[0, 1, 2], [3, 4, 15]], dtype=jnp.int32)
    out = jax.vmap(
        functools.partial(
            take.take,
            qtable,
            axis=0,
            unique_indices=True,
            indices_are_sorted=True,
        )
    )(batch_ids)
    expected = jnp.take(ref_table, batch_ids, axis=0)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))

    # Host NumPy leaves (as produced by host table quantizers) are accepted.
    host_qtable = jax.tree.map(np.asarray, qtable)
    np.testing.assert_array_equal(
        np.asarray(take.take(host_qtable, batch_ids, axis=0)),
        np.asarray(expected),
    )

  def test_out_sharding_plain_array(self):
    if 'out_sharding' not in inspect.signature(jnp.take).parameters:
      self.skipTest('jnp.take out_sharding requires JAX >= 0.12.1')
    ref_table = qarray.dequantize(_make_qtable(shape=(16, 8)))
    ids = jnp.array([0, 3, -1], dtype=jnp.int32)
    mesh = jax.make_mesh(
        (1,),
        ('x',),
        axis_types=(jax.sharding.AxisType.Explicit,),
        devices=jax.devices()[:1],
    )
    pspec = jax.sharding.PartitionSpec
    with jax.set_mesh(mesh):
      sharded_ref = jax.device_put(ref_table, pspec('x', None))
      sharded_ids = jax.device_put(ids, pspec(None))
      expected = jnp.take(
          sharded_ref, sharded_ids, axis=0, out_sharding=pspec('x', None)
      )
      out_plain = take.take(
          sharded_ref, sharded_ids, axis=0, out_sharding=pspec('x', None)
      )
      np.testing.assert_array_equal(np.asarray(out_plain), np.asarray(expected))
      self.assertEqual(jax.typeof(out_plain).sharding.spec, pspec('x', None))

  @parameterized.parameters(jnp.float32, jnp.bfloat16)
  def test_unquantized_array_passthrough(self, dtype):
    rng = np.random.default_rng(2)
    table = jnp.asarray(rng.uniform(-1.0, 1.0, size=(16, 8)), dtype=dtype)
    ids = jnp.array([[0, -1, 15, 16]], dtype=jnp.int32)
    out = jax.jit(functools.partial(take.take, axis=0, fill_value=0.0))(
        table, ids
    )
    expected = jnp.take(table, ids, axis=0, fill_value=0.0)
    self.assertEqual(out.dtype, expected.dtype)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))
    # Untyped empty array (float dtype) matches jnp.take on plain arrays.
    empty_ids = np.array([])
    np.testing.assert_array_equal(
        np.asarray(take.take(table, empty_ids, axis=0)),
        np.asarray(jnp.take(table, empty_ids, axis=0)),
    )

  def test_ids_dtypes_and_wrapping(self):
    n_small = 16
    qtable_small = _make_qtable(shape=(n_small, 4), seed=3)
    ref_small = np.asarray(qarray.dequantize(qtable_small))

    np.testing.assert_array_equal(
        np.asarray(take.take(qtable_small, jnp.int32(-1), axis=0)),
        ref_small[15],
    )

    for dtype in (jnp.int8, jnp.int16, jnp.int32):
      ids = jnp.array([-17, -16, -1, 0, 15, 16], dtype=dtype)
      out = np.asarray(take.take(qtable_small, ids, axis=0))
      self.assertTrue(np.all(np.isnan(out[0])))
      np.testing.assert_array_equal(out[1:5], ref_small[[0, 15, 0, 15]])
      self.assertTrue(np.all(np.isnan(out[5])))

    for dtype in (jnp.uint8, jnp.uint16, jnp.uint32):
      high_oob = 2**31 if dtype == jnp.uint32 else 16
      ids = jnp.array([0, 15, 16, high_oob], dtype=dtype)
      out = np.asarray(take.take(qtable_small, ids, axis=0))
      np.testing.assert_array_equal(out[:2], ref_small[[0, 15]])
      self.assertTrue(np.all(np.isnan(out[2:])))

    with jax.enable_x64():
      ids_i64 = jnp.array([-(2**40), -16, -1, 0, 15, 2**40], dtype=jnp.int64)
      out_i64 = np.asarray(take.take(qtable_small, ids_i64, axis=0))
      self.assertTrue(np.all(np.isnan(out_i64[0])))
      np.testing.assert_array_equal(out_i64[1:5], ref_small[[0, 15, 0, 15]])
      self.assertTrue(np.all(np.isnan(out_i64[5])))

      ids_u64 = jnp.array([0, 15, 16, 2**63], dtype=jnp.uint64)
      out_u64 = np.asarray(take.take(qtable_small, ids_u64, axis=0))
      np.testing.assert_array_equal(out_u64[:2], ref_small[[0, 15]])
      self.assertTrue(np.all(np.isnan(out_u64[2:])))

    # Large table (N > 65_536): narrow signed/unsigned ids must not wrap N.
    n_large = 70_000
    qvalue_large = jnp.broadcast_to(
        (jnp.arange(n_large, dtype=jnp.int32) % 127).astype(jnp.int8)[:, None],
        (n_large, 2),
    )
    qtable_large = qarray.QArray(
        qvalue_large, jnp.ones((n_large, 1), dtype=jnp.float32), qtype=jnp.int8
    )
    for dtype, vals, expected_rows in (
        (jnp.uint8, [0, 1, 255], [0, 1, 255]),
        (jnp.uint16, [0, 1000, 65535], [0, 1000, 65535]),
        (jnp.int8, [-128, -1, 0, 127], [n_large - 128, n_large - 1, 0, 127]),
        (jnp.int16, [-1000, -1, 1000], [n_large - 1000, n_large - 1, 1000]),
    ):
      out = np.asarray(
          take.take(qtable_large, jnp.array(vals, dtype=dtype), axis=0)
      )
      expected = np.asarray(qvalue_large[np.array(expected_rows)], np.float32)
      np.testing.assert_array_equal(out, expected)

  @parameterized.parameters(None, 'fill', 'clip', 'wrap')
  def test_large_axis_size_x64(self, mode):
    with jax.enable_x64():
      qtable = qarray.QArray(
          cast(jax.Array, jax.ShapeDtypeStruct((2**31 + 1, 4), jnp.int8)),
          cast(jax.Array, jax.ShapeDtypeStruct((1, 1), jnp.float32)),
          qtype=jnp.int8,
      )
      for dtype in (jnp.int8, jnp.int16, jnp.int32, jnp.int64):
        ids = jax.ShapeDtypeStruct((3,), dtype)
        jaxpr = jax.make_jaxpr(functools.partial(take.take, axis=0, mode=mode))(
            qtable, ids
        )
        self.assertEqual(jaxpr.out_avals[0].shape, (3, 4))
        self.assertEqual(jaxpr.out_avals[0].dtype, jnp.float32)
        gather_eqns = [
            eqn for eqn in jaxpr.eqns if eqn.primitive.name == 'gather'
        ]
        self.assertNotEmpty(gather_eqns)
        for eqn in gather_eqns:
          self.assertEqual(
              cast(jax.core.ShapedArray, eqn.invars[1].aval).dtype, jnp.int64
          )

      tiled_qtable = qarray.QArray(
          cast(jax.Array, jax.ShapeDtypeStruct((2**32, 4), jnp.int8)),
          cast(jax.Array, jax.ShapeDtypeStruct((2, 1), jnp.float32)),
          qtype=jnp.int8,
      )
      for dtype, expected_idx_dtype in (
          (jnp.uint8, jnp.int32),
          (jnp.uint32, jnp.uint32),
          (jnp.int32, jnp.int64),
      ):
        ids = jax.ShapeDtypeStruct((3,), dtype)
        jaxpr = jax.make_jaxpr(functools.partial(take.take, axis=0, mode=mode))(
            tiled_qtable, ids
        )
        self.assertEqual(jaxpr.out_avals[0].shape, (3, 4))
        gather_eqns = [
            eqn for eqn in jaxpr.eqns if eqn.primitive.name == 'gather'
        ]
        self.assertLen(gather_eqns, 2)
        for eqn in gather_eqns:
          self.assertEqual(
              cast(jax.core.ShapedArray, eqn.invars[1].aval).dtype,
              expected_idx_dtype,
          )

  @parameterized.parameters(None, 'fill', 'clip', 'wrap')
  def test_large_axis_size_unsigned_x32(self, mode):
    with jax.enable_x64(False):
      qtable_2g = qarray.QArray(
          cast(jax.Array, jax.ShapeDtypeStruct((2**31, 4), jnp.int8)),
          cast(jax.Array, jax.ShapeDtypeStruct((1, 1), jnp.float32)),
          qtype=jnp.int8,
      )
      for dtype, expected_idx_dtype in (
          (jnp.uint8, jnp.int32),
          (jnp.uint16, jnp.int32),
          (jnp.uint32, jnp.uint32),
      ):
        ids = jax.ShapeDtypeStruct((3,), dtype)
        jaxpr = jax.make_jaxpr(functools.partial(take.take, axis=0, mode=mode))(
            qtable_2g, ids
        )
        self.assertEqual(jaxpr.out_avals[0].shape, (3, 4))
        gather_eqns = [
            eqn for eqn in jaxpr.eqns if eqn.primitive.name == 'gather'
        ]
        self.assertNotEmpty(gather_eqns)
        for eqn in gather_eqns:
          self.assertEqual(
              cast(jax.core.ShapedArray, eqn.invars[1].aval).dtype,
              expected_idx_dtype,
          )

      tiled_4g = qarray.QArray(
          cast(jax.Array, jax.ShapeDtypeStruct((2**32, 4), jnp.int8)),
          cast(jax.Array, jax.ShapeDtypeStruct((2, 1), jnp.float32)),
          qtype=jnp.int8,
      )
      for dtype in (jnp.uint8, jnp.uint16, jnp.uint32):
        ids = jax.ShapeDtypeStruct((3,), dtype)
        out = jax.eval_shape(
            functools.partial(take.take, axis=0, mode=mode), tiled_4g, ids
        )
        self.assertEqual(out.shape, (3, 4))
        self.assertEqual(out.dtype, jnp.float32)

  @parameterized.parameters(None, 'fill', 'clip', 'wrap')
  def test_large_axis_size_signed_overflow_x32(self, mode):
    with jax.enable_x64(False):
      qtable = qarray.QArray(
          cast(jax.Array, jax.ShapeDtypeStruct((2**31, 4), jnp.int8)),
          cast(jax.Array, jax.ShapeDtypeStruct((1, 1), jnp.float32)),
          qtype=jnp.int8,
      )
      for dtype in (jnp.int8, jnp.int16, jnp.int32):
        ids = jax.ShapeDtypeStruct((3,), dtype)
        with self.assertRaisesRegex(
            OverflowError, 'exceeds maximum int32 index'
        ):
          jax.eval_shape(
              functools.partial(take.take, axis=0, mode=mode), qtable, ids
          )


if __name__ == '__main__':
  absltest.main()
