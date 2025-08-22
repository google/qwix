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
from qwix._src.core import qarray

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

    self.assertEqual(
        q_array.qvalue.dtype, jnp.uint4 if qtype == 'nf4' else qtype
    )
    self.assertEqual(q_array.qvalue.shape, array_shape)

    mae = jnp.abs(array - dq_array).mean() / jnp.abs(array).mean()
    self.assertAlmostEqual(mae, expected_mae)

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
    how = qarray.HowToQuantize(
        qtype=jnp.int8,
        channelwise_axes=[],
        tiled_axes={},
        calibration_method='minmax',
    )
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
        qvalue=jnp.ones((10, 256, 16)),
        scale=jnp.ones((10, 8, 2)),
        zero_point=None,
        qtype=jnp.int8,
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


if __name__ == '__main__':
  absltest.main()
