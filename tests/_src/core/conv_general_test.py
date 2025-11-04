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
"""CPU tests for conv_general."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from qwix._src.core import conv_general
from qwix._src.core import qarray


class ConvGeneralTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='i8_i8',
          lhs_dtype=jnp.int8,
          rhs_dtype=jnp.int8,
          expected_output_dtype=jnp.int32,
      ),
      dict(
          testcase_name='i8_i8f32',
          lhs_dtype=jnp.int8,
          rhs_dtype=(jnp.int8, jnp.float32),
          expected_output_dtype=jnp.float32,
      ),
  )
  def test_output_dtype(self, lhs_dtype, rhs_dtype, expected_output_dtype):
    lhs_shape = (1, 3, 10, 10)
    rhs_shape = (5, 3, 2, 2)
    dnums = ('NCHW', 'OIHW', 'NCHW')

    if isinstance(lhs_dtype, tuple):
      lhs = qarray.QArray(
          jnp.ones(lhs_shape, lhs_dtype[0]),
          jnp.ones((1, 3, 1, 1), lhs_dtype[1]),
      )
    else:
      lhs = jnp.ones(lhs_shape, lhs_dtype)
    if isinstance(rhs_dtype, tuple):
      rhs = qarray.QArray(
          jnp.ones(rhs_shape, rhs_dtype[0]),
          jnp.ones((5, 1, 1, 1), rhs_dtype[1]),
      )
    else:
      rhs = jnp.ones(rhs_shape, rhs_dtype)

    result = conv_general.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=dnums,
    )
    self.assertEqual(result.dtype, expected_output_dtype)


if __name__ == '__main__':
  absltest.main()
