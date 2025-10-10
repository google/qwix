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

import logging
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import qarray
from qwix._src.core import ragged_dot


def rel_mae(a, b):
  assert a.dtype == b.dtype and a.shape == b.shape
  return jnp.abs(a - b).mean() / jnp.abs(a).mean()


class RaggedDotTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._random_key = jax.random.key(42)

  def _make_array(self, shape, asymmetric=False):
    self._random_key, key = jax.random.split(self._random_key)
    if asymmetric:
      return jax.random.uniform(key, shape, jnp.float32)
    return jax.random.normal(key, shape, jnp.float32)

  @parameterized.named_parameters(
      dict(
          testcase_name='int8',
          lhs_shape=(128, 256),
          lhs_how=qarray.HowToQuantize(qtype=jnp.int8),
          rhs_shape=(4, 256, 64),
          rhs_how=qarray.HowToQuantize(qtype=jnp.int8),
          group_sizes=(64, 32, 16, 16),
          expected_mae=0.03,
      ),
      dict(
          testcase_name='channelwise',
          lhs_shape=(128, 256),
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.float8_e5m2,
              channelwise_axes=(0,),
          ),
          rhs_shape=(4, 256, 64),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.float8_e5m2,
              channelwise_axes=(2,),
          ),
          group_sizes=(128, 100, 0, 28),
          expected_mae=0.08,
      ),
      dict(
          testcase_name='rhs_group_and_out_channelwise',
          lhs_shape=(128, 256),
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.float8_e5m2,
              channelwise_axes=(0,),
          ),
          rhs_shape=(4, 256, 64),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.float8_e5m2,
              channelwise_axes=(0, 2),
          ),
          group_sizes=(128, 100, 0, 28),
          expected_mae=0.08,
      ),
  )
  def test_ragged_dot(
      self,
      *,
      lhs_shape: tuple[int, ...],
      lhs_how: qarray.HowToQuantize | None,
      rhs_shape: tuple[int, ...],
      rhs_how: qarray.HowToQuantize | None,
      group_sizes: tuple[int, ...],
      expected_mae: float,
  ):
    group_sizes = jnp.array(group_sizes)
    lhs = self._make_array(lhs_shape, asymmetric=False)
    rhs = self._make_array(rhs_shape, asymmetric=False)
    q_lhs = qarray.quantize(lhs, lhs_how) if lhs_how else lhs
    q_rhs = qarray.quantize(rhs, rhs_how) if rhs_how else rhs

    @jax.jit
    def _jitted_ragged_dot(lhs, rhs, fp_res):
      q_res = ragged_dot.ragged_dot(lhs, rhs, group_sizes)
      return rel_mae(q_res, fp_res)

    fp_res = jax.lax.ragged_dot(lhs, rhs, group_sizes)
    fp_mae = _jitted_ragged_dot(q_lhs, q_rhs, fp_res)

    logging.info('fp_mae=%s', fp_mae)
    self.assertLessEqual(fp_mae, expected_mae)

  @parameterized.named_parameters(
      dict(
          testcase_name='int8',
          lhs_shape=(128, 256),
          lhs_how=qarray.HowToQuantize(qtype=jnp.int8),
          rhs_shape=(4, 256, 64),
          rhs_how=qarray.HowToQuantize(qtype=jnp.int8),
          group_sizes=(64, 32, 16, 16),
          expected_mae=0.03,
      ),
      dict(
          testcase_name='lhs_asymmetric',
          lhs_shape=(128, 256),
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              calibration_method='minmax',
          ),
          rhs_shape=(4, 256, 64),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              calibration_method='absmax',
          ),
          group_sizes=(50, 50, 28, 0),
          expected_mae=0.07,
          disable_fast_ragged_dot=True,
      ),
      dict(
          testcase_name='rhs_group_channelwise',
          lhs_shape=(128, 256),
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              calibration_method='absmax',
          ),
          rhs_shape=(4, 256, 64),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              channelwise_axes=(0,),
              calibration_method='absmax',
          ),
          group_sizes=(128, 0, 0, 0),
          expected_mae=0.03,
          disable_fast_ragged_dot=True,
      ),
      dict(
          testcase_name='rhs_contracting_tiled',
          lhs_shape=(128, 256),
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              calibration_method='absmax',
          ),
          rhs_shape=(4, 256, 64),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              tiled_axes={1: 128},
              calibration_method='absmax',
          ),
          group_sizes=(10, 20, 30, 68),
          expected_mae=0.03,
          disable_fast_ragged_dot=True,
      ),
      dict(
          testcase_name='channelwise',
          lhs_shape=(128, 256),
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.float8_e5m2,
              channelwise_axes=(0,),
          ),
          rhs_shape=(4, 256, 64),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.float8_e5m2,
              channelwise_axes=(2,),
          ),
          group_sizes=(128, 100, 0, 28),
          expected_mae=0.08,
      ),
      dict(
          testcase_name='rhs_group_and_out_channelwise',
          lhs_shape=(128, 256),
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.float8_e5m2,
              channelwise_axes=(0,),
          ),
          rhs_shape=(4, 256, 64),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.float8_e5m2,
              channelwise_axes=(0, 2),
          ),
          group_sizes=(128, 100, 0, 28),
          expected_mae=0.08,
      ),
  )
  def test_ragged_dot_general(
      self,
      *,
      lhs_shape: tuple[int, ...],
      lhs_how: qarray.HowToQuantize | None,
      rhs_shape: tuple[int, ...],
      rhs_how: qarray.HowToQuantize | None,
      group_sizes: tuple[int, ...],
      expected_mae: float,
      disable_fast_ragged_dot: bool = False,
  ):
    group_sizes = jnp.array(group_sizes)
    lhs_asymmetric = (
        lhs_how.calibration_method == 'minmax' if lhs_how else False
    )
    rhs_asymmetric = (
        rhs_how.calibration_method == 'minmax' if rhs_how else False
    )
    lhs = self._make_array(lhs_shape, lhs_asymmetric)
    rhs = self._make_array(rhs_shape, rhs_asymmetric)
    q_lhs = qarray.quantize(lhs, lhs_how) if lhs_how else lhs
    q_rhs = qarray.quantize(rhs, rhs_how) if rhs_how else rhs

    @jax.jit
    def _jitted_ragged_dot_general(lhs, rhs, fp_res):
      slow_res = ragged_dot._slow_ragged_dot_general(
          lhs, rhs, group_sizes, ragged_dot._BASIC_RAGGED_DOT_DIMENSION_NUMBERS
      )
      if disable_fast_ragged_dot:
        fast_res = slow_res
      else:
        fast_res = ragged_dot._fast_ragged_dot_general(
            lhs,
            rhs,
            group_sizes,
            ragged_dot._BASIC_RAGGED_DOT_DIMENSION_NUMBERS,
        )
      return (
          rel_mae(slow_res, fp_res),
          rel_mae(slow_res, fast_res),
      )

    fp_res = jax.lax.ragged_dot(lhs, rhs, group_sizes)
    fp_mae, fast_mae = _jitted_ragged_dot_general(q_lhs, q_rhs, fp_res)

    logging.info('fp_mae=%s fast_mae=%s', fp_mae, fast_mae)
    self.assertLessEqual(fp_mae, expected_mae)
    self.assertLessEqual(fast_mae, 0.003)

  @parameterized.named_parameters(
      dict(
          testcase_name='fast',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              calibration_method='absmax',
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              calibration_method='absmax',
          ),
          expect_fast=True,
      ),
      dict(
          testcase_name='slow_lhs',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              tiled_axes={1: 64},
              calibration_method='absmax',
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              calibration_method='absmax',
          ),
          expect_fast=False,
      ),
      dict(
          testcase_name='rhs_group_and_out_channelwise_fast',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.float8_e5m2,
              channelwise_axes=(0,),
              calibration_method='absmax',
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.float8_e5m2,
              channelwise_axes=(0, 2),
              calibration_method='absmax',
          ),
          expect_fast=True,
      ),
      dict(
          testcase_name='slow_rhs_k_channelwise',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              calibration_method='absmax',
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              channelwise_axes=(1,),
              calibration_method='absmax',
          ),
          expect_fast=False,
      ),
      dict(
          testcase_name='slow_rhs_zp',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              calibration_method='minmax',
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              calibration_method='minmax',
          ),
          expect_fast=False,
      ),
  )
  @mock.patch.object(ragged_dot, '_slow_ragged_dot_general', autospec=True)
  @mock.patch.object(ragged_dot, '_fast_ragged_dot_general', autospec=True)
  def test_ragged_dot_general_implementation(
      self,
      mock_fast,
      mock_slow,
      *,
      lhs_how: qarray.HowToQuantize | None,
      rhs_how: qarray.HowToQuantize | None,
      expect_fast: bool,
  ):
    mock_fast.return_value = jnp.ones((1, 1), jnp.float32)
    mock_slow.return_value = jnp.ones((1, 1), jnp.float32)

    lhs_shape = (128, 256)
    rhs_shape = (2, 256, 64)
    group_sizes = jnp.array((100, 28))

    lhs = jax.random.normal(jax.random.key(0), lhs_shape, jnp.float32)
    rhs = jax.random.normal(jax.random.key(1), rhs_shape, jnp.float32)

    q_lhs = qarray.quantize(lhs, lhs_how) if lhs_how else lhs
    q_rhs = qarray.quantize(rhs, rhs_how) if rhs_how else rhs

    ragged_dot.ragged_dot_general(
        q_lhs,
        q_rhs,
        group_sizes,
        ragged_dot._BASIC_RAGGED_DOT_DIMENSION_NUMBERS,
    )
    if expect_fast:
      mock_fast.assert_called_once()
      mock_slow.assert_not_called()
    else:
      mock_fast.assert_not_called()
      mock_slow.assert_called_once()


if __name__ == '__main__':
  absltest.main()
