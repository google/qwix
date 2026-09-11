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
      dict(
          testcase_name='mxint4',
          qtype='mxint4',
          test_values=[7.4, 7.5, 14.9, 15.0],
          expected_scales=[1.0, 2.0, 2.0, 4.0],
      ),
      dict(
          testcase_name='mxint4_boundary',
          qtype='mxint4',
          test_values=[3.7, 3.75, 7.4, 7.5, 14.9, 15.0],
          expected_scales=[0.5, 1.0, 1.0, 2.0, 2.0, 4.0],
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

  def test_compute_scale_zero_point_mxint8_power_of_2(self):
    calibration = {
        'absmax': jnp.array([63.75, 64.0, 127.5, 127.6, 255.0, 256.0])
    }
    scale, zero_point = qarray.compute_scale_zero_point(calibration, 'mxint8')
    self.assertIsNone(zero_point)
    expected_scales = jnp.array(
        [0.5, 1.0, 1.0, 2.0, 2.0, 4.0], dtype=scale.dtype
    )
    self.assertTrue(jnp.array_equal(scale, expected_scales))

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

  def test_scale_method_transitions_mxfp4(self):
    """Tests mxfp4 scale transitions for OAS vs Ceil around boundaries."""
    # Ceil transitions at powers-of-2 multiples of qmax=6.0 (6.0, 12.0).
    # OAS transitions at powers-of-2 multiples of cutoff=7.0 (3.5, 7.0, 14.0).
    test_cases = [
        # Around 3.0 (Ceil transition: 0.5 -> 1.0; OAS: 0.5)
        (3.0, 0.5, 0.5),
        (3.2, 0.5, 1.0),
        # Around 3.5 (OAS transition: 0.5 -> 1.0; Ceil: 1.0)
        (3.4, 0.5, 1.0),
        (3.5, 1.0, 1.0),
        (3.6, 1.0, 1.0),
        # Around 6.0 (Ceil transition: 1.0 -> 2.0; OAS: 1.0)
        (5.8, 1.0, 1.0),
        (6.0, 1.0, 1.0),
        (6.2, 1.0, 2.0),
        # In (6.0, 7.0], Ceil is 2.0, OAS stays at 1.0
        (6.8, 1.0, 2.0),
        # At 7.0 (OAS transition: 1.0 -> 2.0; Ceil: 2.0)
        (7.0, 2.0, 2.0),
        (7.2, 2.0, 2.0),
        # Around 12.0 (Ceil transition: 2.0 -> 4.0; OAS: 2.0)
        (11.8, 2.0, 2.0),
        (12.0, 2.0, 2.0),
        (12.2, 2.0, 4.0),
        # At 14.0 (OAS transition: 2.0 -> 4.0; Ceil: 4.0)
        (13.8, 2.0, 4.0),
        (14.0, 4.0, 4.0),
        (14.2, 4.0, 4.0),
    ]
    vals = [t[0] for t in test_cases]
    expected_oas = [t[1] for t in test_cases]
    expected_ceil = [t[2] for t in test_cases]
    calib = {'absmax': jnp.array(vals)}

    scale_oas, _ = qarray.compute_scale_zero_point(
        calib, 'mxfp4', scale_method='oas'
    )
    scale_ceil, _ = qarray.compute_scale_zero_point(
        calib, 'mxfp4', scale_method='ceil'
    )
    scale_default, _ = qarray.compute_scale_zero_point(
        calib, 'mxfp4', scale_method='default'
    )
    self.assertTrue(
        jnp.array_equal(
            scale_oas, jnp.array(expected_oas, dtype=scale_oas.dtype)
        )
    )
    self.assertTrue(
        jnp.array_equal(
            scale_ceil, jnp.array(expected_ceil, dtype=scale_ceil.dtype)
        )
    )
    # Default for mxfp4 matches OAS
    self.assertTrue(jnp.array_equal(scale_default, scale_oas))

  def test_scale_method_transitions_mxfp8(self):
    """Tests mxfp8 scale transitions for OAS vs Ceil around boundaries."""
    # Ceil transitions at powers-of-2 multiples of qmax=448.0 (224, 448, 896).
    # OAS transitions at powers-of-2 multiples of cutoff=464.0 (232, 464, 928).
    test_cases = [
        # Around 224.0 (Ceil transition: 0.5 -> 1.0; OAS: 0.5)
        (224.0, 0.5, 0.5),
        (228.0, 0.5, 1.0),
        # Around 232.0 (OAS transition: 0.5 -> 1.0; Ceil: 1.0)
        (230.0, 0.5, 1.0),
        (232.0, 1.0, 1.0),
        (235.0, 1.0, 1.0),
        # Around 448.0 (Ceil transition: 1.0 -> 2.0; OAS: 1.0)
        (440.0, 1.0, 1.0),
        (448.0, 1.0, 1.0),
        (450.0, 1.0, 2.0),
        # In (448.0, 464.0], Ceil is 2.0, OAS stays at 1.0
        (460.0, 1.0, 2.0),
        # At 464.0 (OAS transition: 1.0 -> 2.0; Ceil: 2.0)
        (464.0, 2.0, 2.0),
        (470.0, 2.0, 2.0),
        # Around 896.0 (Ceil transition: 2.0 -> 4.0; OAS: 2.0)
        (890.0, 2.0, 2.0),
        (896.0, 2.0, 2.0),
        (900.0, 2.0, 4.0),
        # At 928.0 (OAS transition: 2.0 -> 4.0; Ceil: 4.0)
        (920.0, 2.0, 4.0),
        (928.0, 4.0, 4.0),
        (935.0, 4.0, 4.0),
    ]
    vals = [t[0] for t in test_cases]
    expected_oas = [t[1] for t in test_cases]
    expected_ceil = [t[2] for t in test_cases]
    calib = {'absmax': jnp.array(vals)}

    for qtype in ('mxfp8', 'mxfp8_16'):
      scale_oas, _ = qarray.compute_scale_zero_point(
          calib, qtype, scale_method='oas'
      )
      scale_ceil, _ = qarray.compute_scale_zero_point(
          calib, qtype, scale_method='ceil'
      )
      scale_default, _ = qarray.compute_scale_zero_point(
          calib, qtype, scale_method='default'
      )
      self.assertTrue(
          jnp.array_equal(
              scale_oas, jnp.array(expected_oas, dtype=scale_oas.dtype)
          )
      )
      self.assertTrue(
          jnp.array_equal(
              scale_ceil, jnp.array(expected_ceil, dtype=scale_ceil.dtype)
          )
      )
      # Default for mxfp8/16 matches OAS
      self.assertTrue(jnp.array_equal(scale_default, scale_oas))

  def test_scale_method_transitions_mxint4(self):
    """Tests mxint4 scale transitions for OAS vs Ceil around boundaries."""
    # Ceil transitions when scale = absmax / 7.5 crosses exact powers of 2.
    # OAS matches ceil for non-powers-of-2, but provides +1 bit of headroom
    # at exact powers of 2 (3.75, 7.5, 15.0).
    test_cases = [
        # Around 3.75 (scale = 0.5)
        (3.50, 0.5, 0.5),
        (3.75, 1.0, 0.5),  # OAS rounds up to 1.0; Ceil keeps 0.5
        (4.00, 1.0, 1.0),
        # Around 7.5 (scale = 1.0)
        (7.00, 1.0, 1.0),
        (7.50, 2.0, 1.0),  # OAS rounds up to 2.0; Ceil keeps 1.0
        (8.00, 2.0, 2.0),
        # Around 15.0 (scale = 2.0)
        (14.00, 2.0, 2.0),
        (15.00, 4.0, 2.0),  # OAS rounds up to 4.0; Ceil keeps 2.0
        (16.00, 4.0, 4.0),
    ]
    vals = [t[0] for t in test_cases]
    expected_oas = [t[1] for t in test_cases]
    expected_ceil = [t[2] for t in test_cases]
    calib = {'absmax': jnp.array(vals)}

    scale_oas, _ = qarray.compute_scale_zero_point(
        calib, 'mxint4', scale_method='oas'
    )
    scale_ceil, _ = qarray.compute_scale_zero_point(
        calib, 'mxint4', scale_method='ceil'
    )
    scale_default, _ = qarray.compute_scale_zero_point(
        calib, 'mxint4', scale_method='default'
    )
    self.assertTrue(
        jnp.array_equal(
            scale_oas, jnp.array(expected_oas, dtype=scale_oas.dtype)
        )
    )
    self.assertTrue(
        jnp.array_equal(
            scale_ceil, jnp.array(expected_ceil, dtype=scale_ceil.dtype)
        )
    )
    # Default for mxint4 matches OAS
    self.assertTrue(jnp.array_equal(scale_default, scale_oas))

  def test_scale_method_transitions_mxint8(self):
    """Tests mxint8 scale transitions for OAS vs Ceil around boundaries."""
    # Ceil transitions when scale = absmax / 127.5 crosses exact powers of 2.
    # OAS matches ceil for non-powers-of-2, but provides +1 bit of headroom
    # at exact powers of 2 (63.75, 127.5, 255.0).
    test_cases = [
        # Around 63.75 (scale = 0.5)
        (60.00, 0.5, 0.5),
        (63.75, 1.0, 0.5),  # OAS rounds up to 1.0; Ceil keeps 0.5
        (68.00, 1.0, 1.0),
        # Around 127.5 (scale = 1.0)
        (120.00, 1.0, 1.0),
        (127.50, 2.0, 1.0),  # OAS rounds up to 2.0; Ceil keeps 1.0
        (135.00, 2.0, 2.0),
        # Around 255.0 (scale = 2.0)
        (240.00, 2.0, 2.0),
        (255.00, 4.0, 2.0),  # OAS rounds up to 4.0; Ceil keeps 2.0
        (270.00, 4.0, 4.0),
    ]
    vals = [t[0] for t in test_cases]
    expected_oas = [t[1] for t in test_cases]
    expected_ceil = [t[2] for t in test_cases]
    calib = {'absmax': jnp.array(vals)}

    scale_oas, _ = qarray.compute_scale_zero_point(
        calib, 'mxint8', scale_method='oas'
    )
    scale_ceil, _ = qarray.compute_scale_zero_point(
        calib, 'mxint8', scale_method='ceil'
    )
    scale_default, _ = qarray.compute_scale_zero_point(
        calib, 'mxint8', scale_method='default'
    )
    self.assertTrue(
        jnp.array_equal(
            scale_oas, jnp.array(expected_oas, dtype=scale_oas.dtype)
        )
    )
    self.assertTrue(
        jnp.array_equal(
            scale_ceil, jnp.array(expected_ceil, dtype=scale_ceil.dtype)
        )
    )
    # Default for mxint8 matches Ceil
    self.assertTrue(jnp.array_equal(scale_default, scale_ceil))

  def test_quantize_scale_method_transition_mxint4(self):
    """Verifies quantize with scale_method='ceil' vs 'oas' on mxint4."""
    # At boundary 7.5: Ceil keeps scale=1.0; OAS provides headroom scale=2.0.
    x_at_boundary = jnp.array([[7.5] * 32], dtype=jnp.float32)
    how_ceil = qarray.HowToQuantize(
        qtype='mxint4', tiled_axes={1: 32}, scale_method='ceil'
    )
    how_oas = qarray.HowToQuantize(
        qtype='mxint4', tiled_axes={1: 32}, scale_method='oas'
    )
    q_ceil = qarray.quantize(x_at_boundary, how_ceil)
    q_oas = qarray.quantize(x_at_boundary, how_oas)
    self.assertEqual(float(q_ceil.scale[0, 0]), 1.0)
    self.assertEqual(float(q_oas.scale[0, 0]), 2.0)

    # Below boundary (7.0): both yield scale=1.0.
    x_below = jnp.array([[7.0] * 32], dtype=jnp.float32)
    self.assertEqual(float(qarray.quantize(x_below, how_ceil).scale[0, 0]), 1.0)
    self.assertEqual(float(qarray.quantize(x_below, how_oas).scale[0, 0]), 1.0)

    # Above boundary (8.0): both yield scale=2.0.
    x_above = jnp.array([[8.0] * 32], dtype=jnp.float32)
    self.assertEqual(float(qarray.quantize(x_above, how_ceil).scale[0, 0]), 2.0)
    self.assertEqual(float(qarray.quantize(x_above, how_oas).scale[0, 0]), 2.0)

  def test_quantize_scale_method_transition_mxfp4(self):
    """Verifies quantize with scale_method='ceil' vs 'oas' on mxfp4."""
    # At 6.5 (between 6.0 and 7.0): Ceil scales up to 2.0; OAS stays at 1.0.
    x_between = jnp.array([[6.5] * 32], dtype=jnp.float32)
    how_ceil = qarray.HowToQuantize(
        qtype='mxfp4', tiled_axes={1: 32}, scale_method='ceil'
    )
    how_oas = qarray.HowToQuantize(
        qtype='mxfp4', tiled_axes={1: 32}, scale_method='oas'
    )
    q_ceil = qarray.quantize(x_between, how_ceil)
    q_oas = qarray.quantize(x_between, how_oas)
    self.assertEqual(float(q_ceil.scale[0, 0]), 2.0)
    self.assertEqual(float(q_oas.scale[0, 0]), 1.0)

    # Below 6.0 (5.5): both yield scale=1.0.
    x_below = jnp.array([[5.5] * 32], dtype=jnp.float32)
    self.assertEqual(float(qarray.quantize(x_below, how_ceil).scale[0, 0]), 1.0)
    self.assertEqual(float(qarray.quantize(x_below, how_oas).scale[0, 0]), 1.0)

    # Above 7.0 (7.5): both yield scale=2.0.
    x_above = jnp.array([[7.5] * 32], dtype=jnp.float32)
    self.assertEqual(float(qarray.quantize(x_above, how_ceil).scale[0, 0]), 2.0)
    self.assertEqual(float(qarray.quantize(x_above, how_oas).scale[0, 0]), 2.0)


if __name__ == '__main__':
  absltest.main()
