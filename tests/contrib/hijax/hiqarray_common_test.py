# Copyright 2026 Google LLC
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

from absl.testing import absltest
import jax.numpy as jnp
from qwix.contrib.hijax import hiqarray_common as hqc


class QuantizationMetadataTest(absltest.TestCase):

  def test_get_shape_and_axes_1(self):
    x = jnp.ones((16, 32, 64))
    (
        data_compatible_shape,
        quant_compatible_shape,
        tiled_reduction_axes,
        full_reduction_axes,
    ) = hqc.QuantizationMetadata._get_reduction_shape_and_axes(
        x.shape, (0, 1, 2), (-1, -1, -1)
    )
    self.assertEqual(data_compatible_shape, (16, 32, 64))
    self.assertEqual(quant_compatible_shape, (1, 1, 1))
    self.assertEmpty(tiled_reduction_axes)
    self.assertEqual(full_reduction_axes, (0, 1, 2))

  def test_get_shape_and_axes_2(self):
    x = jnp.ones((16, 32, 64))
    (
        data_compatible_shape,
        quant_compatible_shape,
        tiled_reduction_axes,
        full_reduction_axes,
    ) = hqc.QuantizationMetadata._get_reduction_shape_and_axes(
        x.shape, (0, 1, 2), (2, 4, 8)
    )
    self.assertEqual(data_compatible_shape, (8, 2, 8, 4, 8, 8))
    self.assertEqual(quant_compatible_shape, (8, 1, 8, 1, 8, 1))
    self.assertEqual(tiled_reduction_axes, (1, 3, 5))
    self.assertEmpty(full_reduction_axes)

  def test_get_shape_and_axes_3(self):
    shape = (16, 32, 64)
    (
        data_compatible_shape,
        quant_compatible_shape,
        tiled_reduction_axes,
        full_reduction_axes,
    ) = hqc.QuantizationMetadata._get_reduction_shape_and_axes(
        shape, (0, 1), (-1, 4)
    )
    self.assertEqual(data_compatible_shape, (16, 8, 4, 64))
    self.assertEqual(quant_compatible_shape, (1, 8, 1, 64))
    self.assertEqual(tiled_reduction_axes, (2,))
    self.assertEqual(full_reduction_axes, (0,))

    metadata = hqc.QuantizationMetadata.init(
        shape,
        {0: -1, 1: 4},
        jnp.float32,
        jnp.float32,
    )

    q_shape = metadata.quant_compatible_shape
    self.assertEqual(len(data_compatible_shape), len(q_shape))
    self.assertTrue(
        all(
            ti == qi or qi == 1
            for ti, qi in zip(data_compatible_shape, q_shape)
        )
    )

  def test_get_shape_and_axes_4(self):
    # Verify example in docstring
    original_shape = (16, 32, 64)
    quant_info = {0: 2, 1: -1}
    qmd = hqc.QuantizationMetadata.init(
        original_shape, quant_info, jnp.float32, jnp.float32
    )
    self.assertEqual(qmd.data_compatible_shape, (8, 2, 32, 64))
    self.assertEqual(qmd.quant_compatible_shape, (8, 1, 1, 64))
    self.assertEqual(qmd._tiled_reduction_axes, (1,))
    self.assertEqual(qmd._full_reduction_axes, (1,))

  def test_get_quant_shape(self):
    tmp_shape = (16, 32, 64)
    tiled_reduction_axes = ()
    full_reduction_axes = (0, 1, 2)
    new_shape = hqc.QuantizationMetadata._get_quant_shape(
        tmp_shape, tiled_reduction_axes, full_reduction_axes
    )
    self.assertEqual(new_shape, (1, 1, 1))


if __name__ == "__main__":
  absltest.main()
