# Copyright 2025 Google LLC
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

from typing import Collection, Mapping

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import numerics
from qwix._src.core import qarray
from qwix._src.core import sparsity

jax.config.update('jax_threefry_partitionable', False)


class QArrayTest(parameterized.TestCase):

  def _make_array(self, shape, asymmetric=False):
    zero_point = 1 if asymmetric else 0
    return (
        jax.random.normal(jax.random.key(42), shape, jnp.bfloat16) + zero_point
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='int8',
          array_shape=(10, 256, 16),
          qtype=jnp.int8,
          channelwise_axes=[0],
          tiled_axes=dict(),
          calibration_method='absmax',
          expected_mae=0.00765991,
      ),
      dict(
          testcase_name='int4',
          array_shape=(10, 256, 16),
          qtype=jnp.int4,
          channelwise_axes=[0],
          tiled_axes=dict(),
          calibration_method='absmax',
          expected_mae=0.122559,
      ),
      dict(
          testcase_name='int4_subchannel',
          array_shape=(10, 256, 16),
          qtype=jnp.int4,
          channelwise_axes=[0],
          tiled_axes={1: 32},
          calibration_method='absmax',
          expected_mae=0.122559,
      ),
      dict(
          testcase_name='asymmetric',
          array_shape=(10, 256, 16),
          qtype=jnp.int8,
          channelwise_axes=[0],
          tiled_axes=dict(),
          calibration_method='minmax',
          expected_mae=0.00521851,
      ),
      dict(
          testcase_name='asymmetric_subchannel',
          array_shape=(10, 256, 16),
          qtype=jnp.int8,
          channelwise_axes=[0],
          tiled_axes={1: 32},
          calibration_method='minmax',
          expected_mae=0.00521851,
      ),
      dict(
          testcase_name='nf4_tiled',
          array_shape=(10, 256, 16),
          qtype='nf4',
          channelwise_axes=[0],
          tiled_axes={1: 32},
          calibration_method='absmax',
          expected_mae=0.0986328,
      ),
      dict(
          testcase_name='rms_calibration',
          array_shape=(10, 256, 16),
          qtype=jnp.int8,
          channelwise_axes=[0],
          tiled_axes={1: 1 / 8},
          calibration_method='rms,7',
          expected_mae=0.017334,
      ),
      dict(
          testcase_name='fixed_calibration',
          array_shape=(10, 256, 16),
          qtype=jnp.int8,
          channelwise_axes=[0],
          tiled_axes={},
          calibration_method='fixed,3',
          expected_mae=0.00765991,
      ),
      dict(
          testcase_name='mxfp8',
          array_shape=(10, 128, 64),
          qtype='mxfp8',
          channelwise_axes=[0, 1],
          tiled_axes={2: 32},
          calibration_method='absmax',
          expected_mae=0.0223389,
      ),
      dict(
          testcase_name='mxfp4',
          array_shape=(10, 128, 64),
          qtype='mxfp4',
          channelwise_axes=[0, 1],
          tiled_axes={2: 32},
          calibration_method='absmax',
          expected_mae=0.106445,
      ),
      dict(
          testcase_name='nvfp4',
          array_shape=(10, 128, 64),
          qtype='nvfp4',
          channelwise_axes=[0, 1],
          tiled_axes={2: 16},
          calibration_method='absmax',
          expected_mae=0.0893555,
      ),
  )
  def test_quantize_dequantize(
      self,
      array_shape: tuple[int, ...],
      qtype: jax.typing.DTypeLike,
      channelwise_axes: Collection[int],
      tiled_axes: Mapping[int, int],
      calibration_method: str,
      expected_mae: float,
  ):
    array = self._make_array(array_shape, calibration_method == 'minmax')

    how = qarray.HowToQuantize(
        qtype=qtype,
        channelwise_axes=channelwise_axes,
        tiled_axes=tiled_axes,
        calibration_method=calibration_method,
    )
    q_array = qarray.quantize(array, how)
    dq_array = qarray.dequantize(q_array)

    expected_scale_shape = list(array_shape)
    for axis, tile_size in tiled_axes.items():
      expected_scale_shape[axis] = (
          expected_scale_shape[axis] + tile_size - 1
      ) // tile_size

    if qtype == 'mxfp8':
      self.assertEqual(q_array.qvalue.dtype, jnp.float8_e4m3fn)
      self.assertEqual(q_array.scale.shape, tuple(expected_scale_shape))
      self.assertIsNone(q_array.zero_point)
    elif qtype in ('mxfp4', 'nvfp4'):
      self.assertEqual(q_array.qvalue.dtype, jnp.float4_e2m1fn)
      self.assertEqual(q_array.scale.shape, tuple(expected_scale_shape))
      self.assertIsNone(q_array.zero_point)
    else:
      self.assertEqual(
          q_array.qvalue.dtype, jnp.uint4 if qtype == 'nf4' else qtype
      )

    self.assertEqual(q_array.qvalue.shape, array_shape)

    mae = jnp.abs(array - dq_array).mean() / jnp.abs(array).mean()
    self.assertAlmostEqual(mae, expected_mae, places=5)

  @parameterized.named_parameters(
      dict(
          testcase_name='with_error',
          with_error=True,
      ),
      dict(
          testcase_name='without_error',
          with_error=False,
      ),
  )
  def test_exact_quantization(self, with_error):
    # Verify that 0, 1/255, 2/255, ..., 254/255, 255/255 are quantized to
    # 0, 1, 2, ..., 254, 255 respectively and dequantized to the original values
    # exactly. If with_error, add a small error (1e-7) to the array, which
    # shouldn't affect the quantization result.
    array = jnp.arange(256) / 255.0
    if with_error:
      array += jax.random.uniform(
          jax.random.key(42), array.shape, minval=-1e-7, maxval=1e-7
      )
    how = qarray.HowToQuantize(qtype=jnp.int8, calibration_method='minmax')
    q_array = qarray.quantize(array, how)
    self.assertEqual(q_array.zero_point, jnp.array(-128, dtype=jnp.int8), array)
    expected_q_array = jnp.arange(-128, 128, dtype=jnp.int8)
    self.assertTrue(
        jnp.all(q_array.qvalue == expected_q_array),
        f'{q_array.qvalue} != {expected_q_array}',
    )
    dq_array = qarray.dequantize(q_array)
    if with_error:
      self.assertTrue(
          jnp.allclose(dq_array, array, atol=1e-6),
          f'{dq_array} != {array}\nDiff: {jnp.abs(dq_array - array)}',
      )
    else:
      self.assertTrue(
          jnp.allclose(dq_array, array),
          f'{dq_array} != {array}\nDiff: {jnp.abs(dq_array - array)}',
      )

  def test_get_tiled_axes(self):
    array = qarray.QArray(
        qvalue=jnp.ones((10, 256, 16), jnp.int8),
        scale=jnp.ones((10, 8, 2)),
    )
    self.assertEqual(qarray.get_tiled_axes(array), {1: 32, 2: 8})

  def test_array_methods(self):
    array = self._make_array((2, 2, 6))
    q_array = qarray.quantize(
        array,
        qarray.HowToQuantize(
            qtype=jnp.int8,
            channelwise_axes=[0],
            tiled_axes={2: 2},
            calibration_method='absmax',
        ),
    )
    self.assertEqual(q_array.scale.shape, (2, 1, 3))
    self.assertEqual(q_array.shape, (2, 2, 6))
    self.assertEqual(q_array.ndim, 3)

    with self.subTest('tile_shape'):
      self.assertEqual(q_array.scale_tile_shape, (1, 2, 2))
      self.assertIsNone(q_array.zero_point_tile_shape)

    with self.subTest('reshape'):
      reshaped_array = q_array.reshape(4, 1, 3, 2)
      self.assertEqual(reshaped_array.shape, (4, 1, 3, 2))
      self.assertEqual(reshaped_array.scale.shape, (2, 1, 3, 1))
      self.assertTrue(
          jnp.array_equal(
              qarray.dequantize(reshaped_array),
              qarray.dequantize(q_array).reshape(4, 1, 3, 2),
          )
      )
      self.assertEqual(q_array.reshape(2, 1, 2, 6).scale.shape, (2, 1, 1, 3))
      self.assertEqual(q_array.reshape(1, 4, 6).scale.shape, (1, 2, 3))
      self.assertEqual(reshaped_array.reshape(4, 6).scale.shape, (2, 3))

    with self.subTest('transpose'):
      transposed_array = q_array.transpose(1, 2, 0)
      self.assertEqual(transposed_array.shape, (2, 6, 2))
      self.assertEqual(transposed_array.scale.shape, (1, 3, 2))
      transposed_array = q_array.T
      self.assertEqual(transposed_array.shape, (6, 2, 2))
      self.assertEqual(transposed_array.scale.shape, (3, 1, 2))
      transposed_array = q_array.mT
      self.assertEqual(transposed_array.shape, (2, 6, 2))
      self.assertEqual(transposed_array.scale.shape, (2, 3, 1))

    with self.subTest('slice'):
      sliced_array = q_array[..., 1]
      self.assertEqual(sliced_array.shape, (2, 2))
      self.assertEqual(sliced_array.scale.shape, (2, 1))
      self.assertTrue(
          jnp.array_equal(
              qarray.dequantize(sliced_array),
              qarray.dequantize(q_array)[..., 1],
          )
      )
      self.assertEqual(q_array[0:1, 1:2, 4].scale.shape, (1, 1))
      self.assertEqual(q_array[0].scale.shape, (1, 3))
      self.assertEqual(q_array[..., None].scale.shape, (2, 1, 3, 1))
      self.assertEqual(q_array[None].scale.shape, (1, 2, 1, 3))

    with self.assertRaises(ValueError):
      qarray.validate_qarray(
          qarray.QArray(
              qvalue=jnp.ones((10, 10), jnp.int8),
              scale=jnp.ones((10, 3), jnp.float32),
          )
      )

    with self.subTest('swapaxes'):
      swapaxes_array = q_array.swapaxes(1, 2)
      self.assertEqual(swapaxes_array.shape, (2, 6, 2))
      self.assertEqual(swapaxes_array.scale.shape, (2, 3, 1))

    with self.subTest('astype'):
      astype_array = q_array.astype(jnp.float32)
      self.assertEqual(astype_array.scale.dtype, jnp.float32)

  @parameterized.named_parameters(
      ('absmax_default', 'absmax', True),
      ('absmax_full', 'absmax,1.0', True),
      ('absmax_over', 'absmax,1.2', True),
      ('absmax_clipped', 'absmax,0.9', False),
      ('minmax_default', 'minmax', True),
      ('minmax_full', 'minmax,1.0', True),
      ('minmax_clipped', 'minmax,0.8', False),
      ('fixed', 'fixed,1.0', False),
      ('rms', 'rms,1.0', False),
  )
  def test_is_gradient_clipping_noop(self, method, expected):
    self.assertEqual(qarray.is_gradient_clipping_noop(method), expected)

  def test_clip_gradient_to_calibration(self):
    with self.subTest('optimization_skip_masking'):
      array = jnp.array([100.0])
      grad = jnp.array([1.0])
      calibration = {'absmax': jnp.array([1.0])}
      out_g = qarray.clip_gradient_to_calibration(
          grad, array, calibration, 'absmax,1.0'
      )
      self.assertTrue(jnp.array_equal(out_g, grad))

    with self.subTest('basic_clipping'):
      array = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
      grad = jnp.ones_like(array)
      calibration = {'absmax': jnp.array([1.0])}
      out_g = qarray.clip_gradient_to_calibration(
          grad, array, calibration, 'absmax,0.5'
      )
      expected = jnp.array([0.0, 1.0, 1.0, 1.0, 0.0])
      self.assertTrue(jnp.array_equal(out_g, expected))

    with self.subTest('tiling_broadcast'):
      array = jnp.array([10.0, 10.0, 2.0, 2.0])
      grad = jnp.ones_like(array)
      calibration = {'absmax': jnp.array([2.5, 2.5])}
      out_g = qarray.clip_gradient_to_calibration(
          grad, array, calibration, 'absmax,0.5'
      )
      expected = jnp.array([0.0, 0.0, 1.0, 1.0])
      self.assertTrue(jnp.array_equal(out_g, expected))

    with self.subTest('fixed_range'):
      array = jnp.array([-2.0, 0.0, 2.0])
      grad = jnp.ones_like(array)
      calibration = {'min': jnp.array([-1.0]), 'max': jnp.array([1.0])}
      out_g = qarray.clip_gradient_to_calibration(
          grad, array, calibration, 'fixed,1.0'
      )
      expected = jnp.array([0.0, 1.0, 0.0])
      self.assertTrue(jnp.array_equal(out_g, expected))

  @parameterized.named_parameters(
      dict(
          testcase_name='qarray',
          input_shape=(2, 1, 4),
          target_shape=(2, 3, 4),
          is_qarray=True,
      ),
      dict(
          testcase_name='standard_array',
          input_shape=(1, 4),
          target_shape=(2, 4),
          is_qarray=False,
      ),
      dict(
          testcase_name='tiling',
          input_shape=(8,),
          target_shape=(64,),
          is_qarray=True,
      ),
  )
  def test_broadcast_to_with(self, input_shape, target_shape, is_qarray):
    if is_qarray:
      operand = qarray.quantize_api(
          self._make_array(input_shape),
          jnp.int8,
          channelwise_axes=[0] if input_shape else [],
          tiled_axes={2: 2} if len(input_shape) > 2 else None,
      )
    else:
      operand = jnp.zeros(input_shape)

    res = qarray.broadcast_to(operand, target_shape)
    self.assertEqual(res.shape, target_shape)

    actual = qarray.dequantize(res) if is_qarray else res
    dequant_operand = qarray.dequantize(operand) if is_qarray else operand
    expected = qarray.call_with_generic_broadcast(
        lambda a, b: jnp.broadcast_to(a, b.shape),
        dequant_operand,
        jnp.zeros(target_shape),
    )
    self.assertTrue(jnp.allclose(actual, expected, atol=1e-6))

  def test_sparsify(self):
    x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    # 2:4 sparsity (2 elements non-zero in each block of 4)
    how = sparsity.SparsityRule(
        weight_sparsity_n=2, weight_sparsity_m=4, weight_sparsity_order='R'
    )
    y = qarray.sparsify(x, how)
    expected = jnp.array([[0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 7.0, 8.0]])
    self.assertTrue(jnp.array_equal(y, expected))

  def test_quantize_nan_reproduction(self):
    # This test verifies both:
    # 1. Default Safe Mode: The scale == 0 (or scale < tiny) check prevents NaN.
    # 2. Unsafe Mode (reproduction): Disabling the safety check triggers NaN.
    val = jnp.array(1.18e-38, dtype=jnp.bfloat16)
    self.assertNotEqual(val, 0.0)

    # mxfp8 requires last dimension to be at least 32 (tile size)
    x = jnp.zeros(32, dtype=jnp.bfloat16)
    x = x.at[0].set(val)

    original_reciprocal = qarray.USE_RECIPROCAL_FOR_QUANTIZATION

    # ---- Part 1: Verify default behavior (Safe Mode) ----
    qarray.USE_RECIPROCAL_FOR_QUANTIZATION = False
    try:
      how = qarray.HowToQuantize(
          qtype='mxfp8', calibration_method='absmax', tiled_axes={0: 32}
      )
      q_array = qarray.quantize(x, how)
      # Default should be safe (no NaN)
      self.assertFalse(
          jnp.isnan(q_array.qvalue).any(),
          f'Found unexpected NaN in safe mode: {q_array.qvalue}',
      )
    finally:
      qarray.USE_RECIPROCAL_FOR_QUANTIZATION = original_reciprocal

    # ---- Part 2: Verify unsafe behavior ----
    # (reproduce NaN by monkeypatching to disable safety check)
    original_compute = qarray.compute_scale_zero_point

    def unsafe_compute_scale_zero_point(
        calibration, qtype, *unused_args, **unused_kwargs
    ):
      if 'min' in calibration and 'max' in calibration:
        qmin, qmax = numerics.get_asymmetric_bound(qtype)
        scale = (calibration['max'] - calibration['min']) / (qmax - qmin)
        # scale safety check REMOVED
        zero_point = qmin - calibration['min'] / scale
        zero_point = numerics.convert_to(zero_point, qtype)
      elif 'absmax' in calibration:
        qmax = numerics.get_symmetric_bound(qtype)
        scale = calibration['absmax'] / qmax
        # scale safety check REMOVED
        zero_point = None
      else:
        raise ValueError(f'Unsupported calibration: {calibration}')
      if qtype == 'mxfp8' or qtype == 'mxfp4':
        log2_scale = jnp.ceil(jnp.log2(scale))
        scale = (2**log2_scale).astype(scale.dtype)
      return scale, zero_point

    qarray.compute_scale_zero_point = unsafe_compute_scale_zero_point
    qarray.USE_RECIPROCAL_FOR_QUANTIZATION = False
    try:
      how = qarray.HowToQuantize(
          qtype='mxfp8', calibration_method='absmax', tiled_axes={0: 32}
      )
      q_array = qarray.quantize(x, how)
      # Unsafe mode should produce NaN
      self.assertTrue(
          jnp.isnan(q_array.qvalue).any(),
          f'Expected NaN in unsafe mode, but got {q_array.qvalue}',
      )
    finally:
      qarray.compute_scale_zero_point = original_compute
      qarray.USE_RECIPROCAL_FOR_QUANTIZATION = original_reciprocal

  def test_mxfp_tile_size_validation(self):
    with self.assertRaisesRegex(
        ValueError, 'Format mxfp8 requires `tiled_axes` to be specified.'
    ):
      qarray.HowToQuantize(qtype='mxfp8')

    with self.assertRaisesRegex(
        ValueError, 'Format mxfp8 requires a tile size of 32, but axis 1 got 64'
    ):
      qarray.HowToQuantize(
          qtype='mxfp8',
          tiled_axes={1: 64},
      )

    with self.assertRaisesRegex(
        ValueError, 'Format mxfp8_16 requires `tiled_axes` to be specified.'
    ):
      qarray.HowToQuantize(qtype='mxfp8_16')

    with self.assertRaisesRegex(
        ValueError,
        'Format mxfp8_16 requires a tile size of 16, but axis 1 got 32',
    ):
      qarray.HowToQuantize(
          qtype='mxfp8_16',
          tiled_axes={1: 32},
      )

    with self.assertRaisesRegex(
        ValueError, 'Format nvfp4 requires `tiled_axes` to be specified.'
    ):
      qarray.HowToQuantize(qtype='nvfp4')

    with self.assertRaisesRegex(
        ValueError, 'Format nvfp4 requires a tile size of 16, but axis 1 got 32'
    ):
      qarray.HowToQuantize(
          qtype='nvfp4',
          tiled_axes={1: 32},
      )

    with self.assertRaisesRegex(
        ValueError, 'Format mxint8 requires `tiled_axes` to be specified.'
    ):
      qarray.HowToQuantize(qtype='mxint8')

    with self.assertRaisesRegex(
        ValueError,
        'Format mxint8 requires a tile size of 32, but axis 1 got 16',
    ):
      qarray.HowToQuantize(
          qtype='mxint8',
          tiled_axes={1: 16},
      )

    with self.assertRaisesRegex(
        ValueError, 'Format mxint4 requires `tiled_axes` to be specified.'
    ):
      qarray.HowToQuantize(qtype='mxint4')

    with self.assertRaisesRegex(
        ValueError,
        'Format mxint4 requires a tile size of 32, but axis 1 got 16',
    ):
      qarray.HowToQuantize(
          qtype='mxint4',
          tiled_axes={1: 16},
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='mxfp4',
          qtype='mxfp4',
          test_values=[6.5, 7.5],
          expected_scales=[1.0, 2.0],
      ),
      dict(
          testcase_name='mxfp8',
          qtype='mxfp8',
          test_values=[460.0, 465.0],
          expected_scales=[1.0, 2.0],
      ),
      dict(
          testcase_name='mxfp8_16',
          qtype='mxfp8_16',
          test_values=[460.0, 465.0],
          expected_scales=[1.0, 2.0],
      ),
      dict(
          testcase_name='mxfp4_boundary',
          qtype='mxfp4',
          test_values=[
              6.875,
              6.90625,
              6.9375,
              6.95,
              6.96875,
              7.0,
              7.03125,
              7.0625,
          ],
          expected_scales=[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
      ),
      dict(
          testcase_name='mxfp8_boundary',
          qtype='mxfp8',
          test_values=[458.0, 460.0, 462.0, 462.5, 464.0, 466.0, 468.0, 470.0],
          expected_scales=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
      ),
      dict(
          testcase_name='mxfp8_16_boundary',
          qtype='mxfp8_16',
          test_values=[458.0, 460.0, 462.0, 462.5, 464.0, 466.0, 468.0, 470.0],
          expected_scales=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
      ),
  )
  def test_compute_scale_zero_point_mxfp_oas_bias(
      self, qtype, test_values, expected_scales
  ):
    calibration = {'absmax': jnp.array(test_values)}
    scale, _ = qarray.compute_scale_zero_point(calibration, qtype)
    self.assertTrue(
        jnp.array_equal(scale, jnp.array(expected_scales, dtype=scale.dtype))
    )

  def test_compute_scale_zero_point_mxint_power_of_2(self):
    calibration_i8 = {
        'absmax': jnp.array([63.75, 64.0, 127.5, 127.6, 255.0, 256.0])
    }
    scale, zero_point = qarray.compute_scale_zero_point(
        calibration_i8, 'mxint8'
    )
    self.assertIsNone(zero_point)
    expected_scales = jnp.array(
        [0.5, 1.0, 1.0, 2.0, 2.0, 4.0], dtype=scale.dtype
    )
    self.assertTrue(jnp.array_equal(scale, expected_scales))

    calibration_i4 = {'absmax': jnp.array([3.75, 3.8, 7.5, 7.6, 15.0, 15.1])}
    scale_i4, zero_point_i4 = qarray.compute_scale_zero_point(
        calibration_i4, 'mxint4'
    )
    self.assertIsNone(zero_point_i4)
    expected_scales_i4 = jnp.array(
        [0.5, 1.0, 1.0, 2.0, 2.0, 4.0], dtype=scale_i4.dtype
    )
    self.assertTrue(jnp.array_equal(scale_i4, expected_scales_i4))

  def test_mxint8_quantize_dequantize(self):
    x = jnp.array(
        [[10.0, -20.0, 30.0, -40.0] * 8, [50.0, -60.0, 70.0, -80.0] * 8],
        dtype=jnp.float32,
    )  # shape (2, 32)
    how = qarray.HowToQuantize(
        qtype='mxint8', channelwise_axes=[0], tiled_axes={1: 32}
    )
    q = qarray.quantize(x, how)
    self.assertEqual(q.qvalue.dtype, jnp.int8)
    self.assertEqual(q.scale.shape, (2, 1))
    self.assertIsNone(q.zero_point)
    # Scale must be a power of 2
    log2_scale = jnp.log2(q.scale)
    self.assertTrue(jnp.all(jnp.equal(log2_scale, jnp.round(log2_scale))))
    # Check dequantize
    deq = qarray.dequantize(q)
    self.assertEqual(deq.shape, x.shape)
    # Dequantized values should be close to original
    self.assertTrue(jnp.allclose(deq, x, atol=2.0))

  def test_mxint4_quantize_dequantize(self):
    x = jnp.array(
        [[1.0, -2.0, 3.0, -4.0] * 8, [5.0, -6.0, 7.0, -7.0] * 8],
        dtype=jnp.float32,
    )  # shape (2, 32)
    how = qarray.HowToQuantize(
        qtype='mxint4', channelwise_axes=[0], tiled_axes={1: 32}
    )
    q = qarray.quantize(x, how)
    self.assertEqual(q.qvalue.dtype, jnp.int4)
    self.assertEqual(q.scale.shape, (2, 1))
    self.assertIsNone(q.zero_point)
    # Scale must be a power of 2
    log2_scale = jnp.log2(q.scale)
    self.assertTrue(jnp.all(jnp.equal(log2_scale, jnp.round(log2_scale))))
    # Check dequantize
    deq = qarray.dequantize(q)
    self.assertEqual(deq.shape, x.shape)
    # Dequantized values should be close to original
    self.assertTrue(jnp.allclose(deq, x, atol=1.0))

  def test_mxint8_sqnr(self):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 128), dtype=jnp.float32)
    how = qarray.HowToQuantize(
        qtype='mxint8', channelwise_axes=[0], tiled_axes={1: 32}
    )
    q = qarray.quantize(x, how)
    deq = qarray.dequantize(q)
    signal_power = jnp.mean(jnp.square(x))
    noise_power = jnp.mean(jnp.square(x - deq))
    sqnr = float(10.0 * jnp.log10(signal_power / noise_power))
    self.assertGreater(
        sqnr, 35.0, f'mxint8 SQNR {sqnr:.2f} dB is below expected 35 dB'
    )

  def test_mxint4_sqnr(self):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 128), dtype=jnp.float32)
    how = qarray.HowToQuantize(
        qtype='mxint4', channelwise_axes=[0], tiled_axes={1: 32}
    )
    q = qarray.quantize(x, how)
    deq = qarray.dequantize(q)
    signal_power = jnp.mean(jnp.square(x))
    noise_power = jnp.mean(jnp.square(x - deq))
    sqnr = float(10.0 * jnp.log10(signal_power / noise_power))
    self.assertGreater(
        sqnr, 15.0, f'mxint4 SQNR {sqnr:.2f} dB is below expected 15 dB'
    )

  def test_hierarchical_scaling_mxfp8(self):
    """Verifies hierarchical scaling on mxfp8_16."""
    key = jax.random.key(123)
    x = jax.random.normal(key, (4, 32), dtype=jnp.float32)
    # Give row 0 a much larger dynamic range than row 1
    x = x.at[0].multiply(50.0)
    x = x.at[1].multiply(0.1)

    how_flat = qarray.HowToQuantize(
        qtype='mxfp8_16',
        channelwise_axes=[0],
        tiled_axes={1: 16},
        hierarchical_scaling=False,
    )
    how_hier = qarray.HowToQuantize(
        qtype='mxfp8_16',
        channelwise_axes=[0],
        tiled_axes={1: 16},
        hierarchical_scaling=True,
    )

    q_flat = qarray.quantize(x, how_flat)
    q_hier = qarray.quantize(x, how_hier)

    self.assertEqual(q_hier.scale.shape, (4, 2))
    self.assertEqual(q_hier.qvalue.shape, (4, 32))
    self.assertEqual(q_hier.scale.dtype, x.dtype)

    # Dequantization should closely reconstruct the original array
    deq_hier = qarray.dequantize(q_hier)
    mae_hier = jnp.mean(jnp.abs(deq_hier - x))
    mae_flat = jnp.mean(jnp.abs(qarray.dequantize(q_flat) - x))
    self.assertTrue(jnp.isfinite(mae_hier))
    self.assertLess(mae_hier, 1.0)
    # The reconstruction error should be very close to or better than flat
    self.assertAlmostEqual(float(mae_hier), float(mae_flat), delta=0.5)

  def test_hierarchical_scaling_noop_non_microscaling(self):
    """Verifies that hierarchical_scaling is a no-op on non-microscaling types."""
    x = jax.random.normal(jax.random.key(42), (4, 32), dtype=jnp.float32)

    # int8 without tiled_axes
    how_std = qarray.HowToQuantize(
        qtype='int8',
        channelwise_axes=[0],
        hierarchical_scaling=False,
    )
    how_hier = qarray.HowToQuantize(
        qtype='int8',
        channelwise_axes=[0],
        hierarchical_scaling=True,
    )
    q_std = qarray.quantize(x, how_std)
    q_hier = qarray.quantize(x, how_hier)
    self.assertTrue(jnp.array_equal(q_std.qvalue, q_hier.qvalue))
    self.assertTrue(jnp.array_equal(q_std.scale, q_hier.scale))

  def test_hierarchical_scaling_formats(self):
    """Verifies hierarchical scaling on all supported microscaling formats."""
    key = jax.random.key(42)
    x = jax.random.normal(key, (2, 32), dtype=jnp.float32)

    for qtype, tile_size in [
        ('mxfp8', 32),
        ('mxfp8_16', 16),
        ('mxfp4', 32),
        ('mxint8', 32),
        ('mxint4', 32),
    ]:
      how = qarray.HowToQuantize(
          qtype=qtype,
          channelwise_axes=[0],
          tiled_axes={1: tile_size},
          hierarchical_scaling=True,
      )
      q = qarray.quantize(x, how)
      self.assertEqual(q.scale.shape, (2, 32 // tile_size))
      self.assertEqual(q.scale.dtype, x.dtype)
      deq = qarray.dequantize(q)
      self.assertTrue(jnp.all(jnp.isfinite(deq)))

  def test_hierarchical_scaling_jit(self):
    """Verifies that quantize with hierarchical_scaling works inside jax.jit."""
    x = jax.random.normal(jax.random.key(42), (4, 32), dtype=jnp.float32)
    how = qarray.HowToQuantize(
        qtype='mxfp8_16',
        channelwise_axes=[0],
        tiled_axes={1: 16},
        hierarchical_scaling=True,
    )

    @jax.jit
    def quant_and_dequant(arr):
      q = qarray.quantize(arr, how)
      return qarray.dequantize(q)

    out = quant_and_dequant(x)
    self.assertEqual(out.shape, x.shape)
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_hierarchical_scaling_zeros_and_underflow(self):
    """Verifies robustness to zeros and small values."""
    x_zeros = jnp.zeros((2, 32), dtype=jnp.float32)
    how = qarray.HowToQuantize(
        qtype='mxfp8_16',
        channelwise_axes=[0],
        tiled_axes={1: 16},
        hierarchical_scaling=True,
    )
    q_zeros = qarray.quantize(x_zeros, how)
    self.assertTrue(jnp.all(jnp.isfinite(q_zeros.scale)))
    self.assertTrue(jnp.all(jnp.isfinite(qarray.dequantize(q_zeros))))

    # Very small numbers
    x_small = jnp.ones((2, 32), dtype=jnp.float32) * 1e-12
    q_small = qarray.quantize(x_small, how)
    self.assertTrue(jnp.all(jnp.isfinite(q_small.scale)))
    self.assertTrue(jnp.all(jnp.isfinite(qarray.dequantize(q_small))))

  @parameterized.parameters(
      ('mxfp8_16', 16, 27.0, 33.0, 0.045),
      ('mxfp8', 32, 26.0, 33.0, 0.050),
      ('mxint8', 32, 36.0, 45.0, 0.015),
  )
  def test_quantize_dequantize_sqnr_gaussian(
      self, qtype, tile_size, min_snr, max_snr, max_rel_err
  ):
    """Verifies that quantize/dequantize on N(0, 1) achieves expected SQNR."""
    key = jax.random.key(1234)
    x = jax.random.normal(key, (128, 256), dtype=jnp.float32)

    for hier in (False, True):
      how = qarray.HowToQuantize(
          qtype=qtype,
          channelwise_axes=[0],
          tiled_axes={1: tile_size},
          hierarchical_scaling=hier,
      )
      q = qarray.quantize(x, how)
      deq = qarray.dequantize(q)

      noise = x - deq
      sig_power = jnp.mean(jnp.square(x))
      noise_power = jnp.mean(jnp.square(noise))
      snr = 10.0 * jnp.log10(sig_power / noise_power)
      rel_err = jnp.linalg.norm(noise) / jnp.linalg.norm(x)

      self.assertGreater(float(snr), min_snr)
      self.assertLess(float(snr), max_snr)
      self.assertLess(float(rel_err), max_rel_err)


if __name__ == '__main__':
  absltest.main()
