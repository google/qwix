# Copyright 2024 Google LLC
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
"""TPU tests for conv_general."""

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import conv_general
from qwix._src.core import qarray


def mae(lhs: jax.Array, rhs: jax.Array) -> float:
  assert lhs.dtype == rhs.dtype and lhs.shape == rhs.shape
  return jnp.abs(lhs - rhs).mean() / jnp.abs(lhs).mean()


class ConvGeneralTest(parameterized.TestCase):

  def _make_array(self, shape, asymmetric=False):
    zero_point = 1 if asymmetric else 0
    return (
        jax.random.normal(jax.random.key(42), shape, jnp.bfloat16) + zero_point
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='int8',
          lhs_shape=(10, 5, 40, 30),
          rhs_shape=(2, 5, 3, 3),
          qtype=jnp.int8,
          dimension_numbers=('NCHW', 'OIHW', 'NCHW'),  # jax.lax default
          expected_mae=0.009,
      ),
      dict(
          testcase_name='int8_flax',
          lhs_shape=(10, 40, 30, 5),
          rhs_shape=(3, 3, 5, 2),
          qtype=jnp.int8,
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'),  # flax style
          expected_mae=0.009,
      ),
      dict(
          testcase_name='int4',
          lhs_shape=(10, 5, 40, 30),
          rhs_shape=(2, 5, 3, 3),
          qtype=jnp.int4,
          dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
          expected_mae=0.16,
      ),
      dict(
          testcase_name='feature_group_count',
          lhs_shape=(10, 15, 40, 30),
          rhs_shape=(3, 5, 3, 3),
          qtype=jnp.int8,
          dimension_numbers=('NCHW', 'OIHW', 'NCHW'),  # jax.lax default
          conv_kwargs={'feature_group_count': 3},
          expected_mae=0.009,
      ),
      dict(
          testcase_name='rhs_dilation',
          lhs_shape=(10, 6, 40, 30),
          rhs_shape=(2, 6, 3, 3),
          qtype=jnp.int8,
          dimension_numbers=('NCHW', 'OIHW', 'NCHW'),  # jax.lax default
          conv_kwargs={'rhs_dilation': (2, 2)},
          expected_mae=0.01,
      ),
      dict(
          testcase_name='lhs_asymmetric',
          lhs_shape=(10, 5, 40, 30),
          rhs_shape=(2, 5, 3, 3),
          qtype=jnp.int8,
          dimension_numbers=('NCHW', 'OIHW', 'NCHW'),  # jax.lax default
          lhs_asymmetric=True,
          expected_mae=0.01,
      ),
      dict(
          testcase_name='rhs_dilation_lhs_asymmetric',
          lhs_shape=(10, 5, 40, 30),
          rhs_shape=(2, 5, 3, 3),
          qtype=jnp.int8,
          dimension_numbers=('NCHW', 'OIHW', 'NCHW'),  # jax.lax default
          lhs_asymmetric=True,
          conv_kwargs={'rhs_dilation': (2, 2)},
          expected_mae=0.01,
      ),
  )
  def test_numerics(
      self,
      *,
      lhs_shape: tuple[int, ...],
      rhs_shape: tuple[int, ...],
      qtype: jax.typing.DTypeLike,
      dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers,
      lhs_asymmetric: bool = False,
      conv_kwargs: dict[str, Any] | None = None,
      expected_mae: float = 0.01,
  ):
    dimension_numbers = jax.lax.conv_dimension_numbers(
        lhs_shape, rhs_shape, dimension_numbers
    )
    if conv_kwargs is None:
      conv_kwargs = {}
    conv_kwargs.setdefault('window_strides', (1, 1))
    conv_kwargs.setdefault('padding', 'SAME')
    conv_kwargs['dimension_numbers'] = dimension_numbers

    lhs = self._make_array(lhs_shape, asymmetric=lhs_asymmetric)
    rhs = self._make_array(rhs_shape)

    def quantize(array, for_lhs):
      calibration_method = 'minmax' if for_lhs and lhs_asymmetric else 'absmax'
      how = conv_general.get_how_to_quantize(
          dimension_numbers=dimension_numbers,
          for_lhs=for_lhs,
          qtype=qtype,
          calibration_method=calibration_method,
      )
      return qarray.quantize(array, how)

    @jax.jit
    def f(lhs, rhs):
      fp_res = jax.lax.conv_general_dilated(lhs, rhs, **conv_kwargs)
      rhs = quantize(rhs, for_lhs=False)
      lhs = quantize(lhs, for_lhs=True)
      slow_res = conv_general._slow_conv_general_dilated(
          lhs, rhs, **conv_kwargs
      )
      fast_res = conv_general._fast_conv_general_dilated(
          lhs, rhs, **conv_kwargs
      )
      return mae(fp_res, slow_res), mae(fast_res, slow_res)

    fp_slow_mae, fast_slow_mae = f(lhs, rhs)
    self.assertLessEqual(fp_slow_mae, expected_mae)
    # The error between slow vs fast should be purely due to floating point
    # imprecision, and should be small.
    self.assertLessEqual(fast_slow_mae, 0.004)


if __name__ == '__main__':
  absltest.main()
