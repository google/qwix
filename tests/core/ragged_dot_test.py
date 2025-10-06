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
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import qarray
from qwix._src.core import ragged_dot


def mae(a, b):
  assert a.dtype == b.dtype and a.shape == b.shape
  return jnp.abs(a - b).mean() / jnp.abs(a).mean()


class RaggedDotTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no_channelwise',
          lhs_how=qarray.HowToQuantize(qtype=jnp.int8, channelwise_axes=[]),
          rhs_how=qarray.HowToQuantize(qtype=jnp.int8, channelwise_axes=[]),
      ),
      dict(
          testcase_name='channelwise',
          lhs_how=qarray.HowToQuantize(qtype=jnp.int8, channelwise_axes=[0]),
          rhs_how=qarray.HowToQuantize(qtype=jnp.int8, channelwise_axes=[2]),
      ),
      dict(
          testcase_name='more_channelwise',
          lhs_how=qarray.HowToQuantize(qtype=jnp.int8, channelwise_axes=[0]),
          rhs_how=qarray.HowToQuantize(qtype=jnp.int8, channelwise_axes=[0, 2]),
      ),
  )
  def test_ragged_dot(
      self,
      lhs_how,
      rhs_how,
      disable_fast_path=False,
  ):
    lhs = jax.random.normal(jax.random.key(0), (256, 16), jnp.bfloat16)
    rhs = jax.random.normal(jax.random.key(1), (10, 16, 64), jnp.bfloat16)
    group_sizes = jnp.array([10, 20, 30, 40, 0, 115, 6, 7, 1, 27], jnp.int32)

    fp_res = jax.lax.ragged_dot(lhs, rhs, group_sizes)

    qlhs = qarray.quantize(lhs, lhs_how)
    qrhs = qarray.quantize(rhs, rhs_how)

    slow_res = ragged_dot._slow_ragged_dot(qlhs, qrhs, group_sizes)
    self.assertLess(mae(slow_res, fp_res), 0.02)

    if not disable_fast_path:
      fast_res = ragged_dot._fast_ragged_dot(qlhs, qrhs, group_sizes)
      self.assertLess(mae(fast_res, slow_res), 0.005)

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
  @mock.patch.object(ragged_dot, '_slow_ragged_dot', autospec=True)
  @mock.patch.object(ragged_dot, '_fast_ragged_dot', autospec=True)
  def test_ragged_dot_implementation(
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

    ragged_dot.ragged_dot(q_lhs, q_rhs, group_sizes)
    if expect_fast:
      mock_fast.assert_called_once()
      mock_slow.assert_not_called()
    else:
      mock_fast.assert_not_called()
      mock_slow.assert_called_once()


if __name__ == '__main__':
  absltest.main()
