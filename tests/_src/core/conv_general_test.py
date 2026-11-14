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
import jax
from jax import numpy as jnp
import numpy as np
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

  @parameterized.named_parameters(
      dict(
          testcase_name='feature_groups_nchw',
          lhs_shape=(4, 6, 8, 8),
          rhs_shape=(6, 2, 3, 3),
          dnums=('NCHW', 'OIHW', 'NCHW'),
          feature_group_count=3,
          batch_group_count=1,
      ),
      dict(
          testcase_name='batch_groups_nchw',
          lhs_shape=(6, 4, 8, 8),
          rhs_shape=(6, 4, 3, 3),
          dnums=('NCHW', 'OIHW', 'NCHW'),
          feature_group_count=1,
          batch_group_count=3,
      ),
      dict(
          testcase_name='batch_groups_nhwc',
          lhs_shape=(4, 8, 8, 3),
          rhs_shape=(3, 3, 3, 6),
          dnums=('NHWC', 'HWIO', 'NHWC'),
          feature_group_count=1,
          batch_group_count=2,
      ),
      dict(
          testcase_name='batch_groups_per_tensor_nhwc',
          lhs_shape=(4, 8, 8, 3),
          rhs_shape=(3, 3, 3, 6),
          dnums=('NHWC', 'HWIO', 'NHWC'),
          feature_group_count=1,
          batch_group_count=2,
          per_tensor=True,
      ),
  )
  def test_grouped_conv_fast_matches_slow(
      self,
      lhs_shape,
      rhs_shape,
      dnums,
      feature_group_count,
      batch_group_count,
      per_tensor=False,
  ):
    dnums = jax.lax.conv_dimension_numbers(lhs_shape, rhs_shape, dnums)
    lhs_fp = jnp.arange(jnp.prod(jnp.array(lhs_shape)), dtype=jnp.float32)
    lhs_fp = jnp.sin(lhs_fp).reshape(lhs_shape)
    batch_axis = dnums.lhs_spec[0]
    batch_scales = (
        2.0 ** jnp.arange(lhs_shape[batch_axis], dtype=jnp.float32)
    ).reshape([-1 if i == batch_axis else 1 for i in range(len(lhs_shape))])
    lhs_fp = lhs_fp * batch_scales
    rhs_fp = jnp.arange(jnp.prod(jnp.array(rhs_shape)), dtype=jnp.float32)
    rhs_fp = jnp.cos(rhs_fp).reshape(rhs_shape)
    lhs_how = conv_general.get_how_to_quantize(
        dimension_numbers=dnums, for_lhs=True, qtype=jnp.int8
    )
    rhs_how = conv_general.get_how_to_quantize(
        dimension_numbers=dnums, for_lhs=False, qtype=jnp.int8
    )
    if per_tensor:
      lhs_how = qarray.HowToQuantize(qtype=jnp.int8)
      rhs_how = qarray.HowToQuantize(qtype=jnp.int8)
    lhs_q = qarray.quantize(lhs_fp, lhs_how)
    rhs_q = qarray.quantize(rhs_fp, rhs_how)
    fast_out = conv_general._fast_conv_general_dilated(
        lhs_q,
        rhs_q,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=dnums,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
    )
    slow_out = conv_general._slow_conv_general_dilated(
        lhs_q,
        rhs_q,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=dnums,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
    )
    self.assertEqual(fast_out.shape, slow_out.shape)
    np.testing.assert_allclose(fast_out, slow_out, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
