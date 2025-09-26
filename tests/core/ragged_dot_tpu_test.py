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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import qarray
from qwix._src.core import ragged_dot


def rel_mae(x, y):
  assert x.dtype == y.dtype and x.shape == y.shape
  return jnp.abs(x - y).mean() / jnp.abs(x).mean()


class RaggedDotTpuTest(parameterized.TestCase):
  """More expensive TPU tests for ragged_dot, mainly on numerics."""

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
          lhs_qtype=jnp.int8,
          rhs_shape=(4, 256, 64),
          rhs_qtype=jnp.int8,
          group_sizes=(64, 32, 16, 16),
          expected_mae=0.03,
      ),
      dict(
          testcase_name='lhs_asymmetric',
          lhs_shape=(128, 256),
          lhs_qtype=jnp.int8,
          lhs_asymmetric=True,
          rhs_shape=(4, 256, 64),
          rhs_qtype=jnp.int8,
          group_sizes=(50, 50, 28, 0),
          expected_mae=0.07,
          disable_fast_ragged_dot=True,
      ),
      dict(
          testcase_name='rhs_group_channelwise',
          lhs_shape=(128, 256),
          lhs_qtype=jnp.int8,
          rhs_shape=(4, 256, 64),
          rhs_qtype=jnp.int8,
          rhs_channelwise_axes=(0,),
          group_sizes=(128, 0, 0, 0),
          expected_mae=0.03,
          disable_fast_ragged_dot=True,
      ),
      dict(
          testcase_name='rhs_contracting_tiled',
          lhs_shape=(128, 256),
          lhs_qtype=jnp.int8,
          rhs_shape=(4, 256, 64),
          rhs_qtype=jnp.int8,
          rhs_tiled_axes={1: 128},
          group_sizes=(10, 20, 30, 68),
          expected_mae=0.03,
          disable_fast_ragged_dot=True,
      ),
  )
  def test_ragged_dot(
      self,
      *,
      lhs_shape: tuple[int, ...],
      lhs_qtype: jax.typing.DTypeLike | None,
      lhs_asymmetric: bool = False,
      rhs_shape: tuple[int, ...],
      rhs_qtype: jax.typing.DTypeLike | None,
      rhs_channelwise_axes: tuple[int, ...] = (),
      rhs_tiled_axes: dict[int, int] | None = None,
      group_sizes: tuple[int, ...],
      expected_mae: float,
      disable_fast_ragged_dot: bool = False,
  ):
    lhs = self._make_array(lhs_shape, lhs_asymmetric)
    rhs = self._make_array(rhs_shape, False)
    rhs_tiled_axes = rhs_tiled_axes or {}
    group_sizes = jnp.array(group_sizes)

    if lhs_qtype:
      lhs_how = qarray.HowToQuantize(
          qtype=lhs_qtype,
          channelwise_axes=(),
          tiled_axes={},
          calibration_method='minmax' if lhs_asymmetric else 'absmax',
      )
      q_lhs = qarray.quantize(lhs, lhs_how)
    else:
      q_lhs = lhs

    if rhs_qtype:
      rhs_how = qarray.HowToQuantize(
          qtype=rhs_qtype,
          channelwise_axes=rhs_channelwise_axes,
          tiled_axes=rhs_tiled_axes,
          calibration_method='absmax',
      )
      q_rhs = qarray.quantize(rhs, rhs_how)
    else:
      q_rhs = rhs

    @jax.jit
    def _multi_ragged_dot(lhs, rhs, fp_res):
      slow_res = ragged_dot._slow_ragged_dot(lhs, rhs, group_sizes)
      if disable_fast_ragged_dot:
        fast_res = slow_res
      else:
        fast_res = ragged_dot._fast_ragged_dot(lhs, rhs, group_sizes)
      return (
          rel_mae(slow_res, fp_res),
          rel_mae(slow_res, fast_res),
      )

    fp_res = jax.lax.ragged_dot(lhs, rhs, group_sizes)
    fp_mae, fast_mae = _multi_ragged_dot(q_lhs, q_rhs, fp_res)

    logging.info('fp_mae=%s fast_mae=%s', fp_mae, fast_mae)
    self.assertLessEqual(fp_mae, expected_mae)
    self.assertLessEqual(fast_mae, 0.003)


if __name__ == '__main__':
  absltest.main()
