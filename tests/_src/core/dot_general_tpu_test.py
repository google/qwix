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

from unittest import mock
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import dot_general
from qwix._src.core import qarray


def rel_mae(x, y):
  assert x.dtype == y.dtype and x.shape == y.shape
  return jnp.abs(x - y).mean() / jnp.abs(x).mean()


class DotGeneralTest(parameterized.TestCase):
  """More expensive TPU tests for dot_general, mainly on numerics."""

  def setUp(self):
    super().setUp()
    self._random_key = jax.random.key(42)

  def _make_array(self, shape, asymmetric=False):
    self._random_key, key = jax.random.split(self._random_key)
    if asymmetric:
      return jax.random.uniform(key, shape, jnp.float32)
    return jax.random.normal(key, shape, jnp.float32) / 10

  @parameterized.named_parameters(
      dict(
          testcase_name='int8_int4_sc',
          lhs_shape=(128, 512, 4),
          lhs_tile_sizes=(1, 128, 1),
          lhs_qtype=jnp.int8,
          rhs_shape=(512, 256, 4),
          rhs_tile_sizes=(128, 1, 1),
          rhs_qtype=jnp.int4,
          dimension_numbers=(([1], [0]), ([2], [2])),
          expected_mae=0.12,
      ),
      dict(
          testcase_name='fp8_tiled_ra',
          lhs_shape=(4, 128, 256),
          lhs_tile_sizes=(1, 1, 128),
          lhs_qtype=jnp.float8_e4m3fn,
          rhs_shape=(256, 256),
          rhs_tile_sizes=(128, 128),  # deepseek style.
          rhs_qtype=jnp.float8_e4m3fn,
          dimension_numbers=(([2], [0]), ([], [])),
          expected_mae=0.04,
      ),
      dict(
          testcase_name='nf4',
          lhs_shape=(128, 512, 4),
          lhs_tile_sizes=(1, None, 1),
          lhs_qtype=None,
          rhs_shape=(512, 256, 4),
          rhs_tile_sizes=(None, 1, 1),
          rhs_qtype='nf4',
          dimension_numbers=(([1], [0]), ([2], [2])),
          expected_mae=0.11,
          # Only slow_dot_general supports nf4.
          disable_fast_dot_general=True,
          disable_loop_dot_general=True,
      ),
      dict(
          testcase_name='lhs_asymmetric',
          lhs_shape=(128, 512, 3),
          lhs_qtype=jnp.int8,
          lhs_asymmetric=True,
          rhs_shape=(512, 256, 3),
          rhs_qtype=jnp.int4,
          dimension_numbers=(([1], [0]), ([2], [2])),
          expected_mae=0.19,
          disable_loop_dot_general=True,
      ),
      dict(
          testcase_name='lhs_asymmetric_subchannel',
          lhs_shape=(128, 512, 3),
          lhs_tile_sizes=(1, 1 / 4, 1),
          lhs_qtype=jnp.int8,
          lhs_asymmetric=True,
          rhs_shape=(512, 256, 3),
          rhs_tile_sizes=(1 / 4, 1, 1),
          rhs_qtype=jnp.int4,
          dimension_numbers=(([1], [0]), ([2], [2])),
          expected_mae=0.128906,
          disable_loop_dot_general=True,
      ),
      dict(
          testcase_name='two_contractions',
          lhs_shape=(128, 2, 128, 32, 64),
          lhs_qtype=None,
          rhs_shape=(64, 32, 128, 128),
          rhs_qtype=jnp.int8,
          dimension_numbers=(([3, 4], [1, 0]), ([0], [3])),
          expected_mae=0.013,
      ),
      dict(
          testcase_name='two_contractions_two_sc',
          lhs_shape=(128, 2, 128, 32, 64),
          lhs_tile_sizes=(1, 1, 1, 8, 8),
          lhs_qtype=jnp.int8,
          rhs_shape=(64, 32, 128, 128),
          rhs_tile_sizes=(8, 8, 1, 1),
          rhs_qtype=jnp.int8,
          dimension_numbers=(([3, 4], [1, 0]), ([0], [3])),
          expected_mae=0.01,
      ),
      dict(
          testcase_name='two_contractions_one_sc',
          lhs_shape=(128, 2, 128, 32, 64),
          lhs_tile_sizes=(1, 1, 1, None, 16),
          lhs_qtype=jnp.int8,
          rhs_shape=(64, 32, 128, 128),
          rhs_tile_sizes=(16, None, 1, 1),
          rhs_qtype=jnp.int8,
          dimension_numbers=(([3, 4], [1, 0]), ([0], [3])),
          expected_mae=0.011,
      ),
      dict(
          testcase_name='dequant_on_input',
          lhs_shape=(16, 128, 128),
          lhs_tile_sizes=(1, 1, None),
          lhs_qtype=jnp.int8,
          rhs_shape=(128, 128, 16),
          rhs_tile_sizes=(1, None, 1),
          rhs_qtype=jnp.int8,
          dimension_numbers=(([1, 2], [0, 1]), ([], [])),
          expected_mae=0.01,
          # Only slow_dot_general supports dequant_on_input.
          disable_fast_dot_general=True,
          disable_loop_dot_general=True,
      ),
  )
  def test_dot_general(
      self,
      *,
      lhs_shape: tuple[int, ...],
      lhs_qtype: jax.typing.DTypeLike | None,
      lhs_tile_sizes: tuple[int | None, ...] = (),
      lhs_asymmetric: bool = False,
      rhs_shape: tuple[int, ...],
      rhs_qtype: jax.typing.DTypeLike | None,
      rhs_tile_sizes: tuple[int | None, ...] = (),
      rhs_asymmetric: bool = False,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      expected_mae: float,
      disable_fast_dot_general: bool = False,
      disable_loop_dot_general: bool = False,
  ):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Only run on TPU.')
    lhs = self._make_array(lhs_shape, lhs_asymmetric)
    rhs = self._make_array(rhs_shape, rhs_asymmetric)

    if lhs_qtype:
      lhs_how = qarray.HowToQuantize(
          qtype=lhs_qtype,
          channelwise_axes=(),
          tiled_axes={a: s for a, s in enumerate(lhs_tile_sizes) if s},
          calibration_method='minmax' if lhs_asymmetric else 'absmax',
      )
      q_lhs = qarray.quantize(lhs, lhs_how)
    else:
      q_lhs = lhs

    if rhs_qtype:
      rhs_how = qarray.HowToQuantize(
          qtype=rhs_qtype,
          channelwise_axes=(),
          tiled_axes={a: s for a, s in enumerate(rhs_tile_sizes) if s},
          calibration_method='minmax' if rhs_asymmetric else 'absmax',
      )
      q_rhs = qarray.quantize(rhs, rhs_how)
    else:
      q_rhs = rhs

    @jax.jit
    def _multi_dot_general(lhs, rhs, fp_res):
      slow_res = dot_general._slow_dot_general(lhs, rhs, dimension_numbers)
      if disable_fast_dot_general:
        fast_res = slow_res
      else:
        fast_res = dot_general._fast_dot_general(lhs, rhs, dimension_numbers)
      if disable_loop_dot_general:
        loop_res = slow_res
      else:
        loop_res = dot_general.loop_dot_general(lhs, rhs, dimension_numbers)
      return (
          rel_mae(slow_res, fp_res),
          rel_mae(slow_res, fast_res),
          rel_mae(slow_res, loop_res),
      )

    fp_res = jax.lax.dot_general(lhs, rhs, dimension_numbers)
    fp_mae, fast_mae, loop_mae = _multi_dot_general(q_lhs, q_rhs, fp_res)

    logging.info(
        'fp_mae=%s fast_mae=%s loop_mae=%s', fp_mae, fast_mae, loop_mae
    )
    self.assertLessEqual(fp_mae, expected_mae)
    # The error between slow vs fast, or slow vs loop should be purely due to
    # floating point imprecision, and should be small.
    # On TPU7 / TPU7x, native FP8 matmuls (specifically
    # test_dot_general_fp8_tiled_ra) have slightly different intermediate
    # rounding compared to the bf16 slow path (fast_mae ≈ 0.0045), requiring a
    # 0.006 tolerance on TPU7/TPU7x for FP8 cases while keeping 0.003 for
    # integer cases and other device kinds.
    device_kind = jax.devices()[0].device_kind.split('\n', 1)[0]
    fp8_types = (
        jnp.float8_e4m3fn,
        jnp.float8_e5m2,
        jnp.float8_e4m3b11fnuz,
        jnp.float8_e4m3fnuz,
        jnp.float8_e5m2fnuz,
    )
    is_fp8_case = lhs_qtype in fp8_types or rhs_qtype in fp8_types
    max_tolerance = 0.006 if ('TPU7' in device_kind and is_fp8_case) else 0.003
    self.assertLessEqual(fast_mae, max_tolerance)
    self.assertLessEqual(loop_mae, max_tolerance)

  @parameterized.parameters(128, 256, 512)
  def test_dot_general_subchannel_tile_sizes(self, tile_size: int):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Only run on TPU.')
    lhs = self._make_array((128, 1024))
    rhs = self._make_array((1024, 256))
    q_lhs = qarray.quantize(
        lhs,
        qarray.HowToQuantize(
            qtype=jnp.int8,
            channelwise_axes=(0,),
            tiled_axes={1: tile_size},
        ),
    )
    q_rhs = qarray.quantize(
        rhs,
        qarray.HowToQuantize(
            qtype=jnp.int4,
            channelwise_axes=(1,),
            tiled_axes={0: tile_size},
        ),
    )
    dnums = (((1,), (0,)), ((), ()))
    fp_res = jax.lax.dot_general(lhs, rhs, dnums)
    res = jax.jit(dot_general.dot_general, static_argnums=(2,))(
        q_lhs, q_rhs, dnums
    )
    slow_res = jax.jit(dot_general._slow_dot_general, static_argnums=(2,))(
        q_lhs, q_rhs, dnums
    )
    self.assertLessEqual(rel_mae(slow_res, fp_res), 0.13)
    self.assertLessEqual(rel_mae(slow_res, res), 0.003)

  def test_get_min_tile_size_to_dequant_on_output_live_tpu(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Only run on TPU.')
    expected_by_kind = {
        'TPU v5 lite': 128,
        'TPU v5p': 128,
        'TPU v6 lite': 256,
        'TPU v6e': 256,
        'TPU7': 512,
        'TPU7x': 512,
        'TPU8i': 256,
        'TPU8t': 256,
    }
    device_kind = jax.devices()[0].device_kind.split('\n', 1)[0]
    self.assertIn(device_kind, expected_by_kind)
    expected_threshold = expected_by_kind[device_kind]
    self.assertEqual(
        dot_general.get_min_tile_size_to_dequant_on_output(),
        expected_threshold,
    )

    k = 1024
    dnums = (([1], [0]), ([], []))
    for tile_size in (128, 256, 512):
      num_tiles = k // tile_size
      lhs = qarray.QArray(
          jnp.ones((16, k), jnp.int8),
          jnp.ones((16, num_tiles), jnp.bfloat16),
      )
      rhs = qarray.QArray(
          jnp.ones((k, 16), jnp.int4),
          jnp.ones((num_tiles, 16), jnp.bfloat16),
      )
      with (
          mock.patch.object(
              dot_general,
              '_fast_dot_general',
              wraps=dot_general._fast_dot_general,
          ) as mock_fast,
          mock.patch.object(
              dot_general,
              '_slow_dot_general',
              wraps=dot_general._slow_dot_general,
          ) as mock_slow,
      ):
        res = dot_general.dot_general(lhs, rhs, dnums)
        self.assertEqual(res.shape, (16, 16))
        if tile_size >= expected_threshold:
          mock_fast.assert_called_once()
          mock_slow.assert_not_called()
        else:
          mock_slow.assert_called_once()
          mock_fast.assert_not_called()


if __name__ == '__main__':
  absltest.main()
