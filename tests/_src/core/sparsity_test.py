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

"""Tests for sparsity core functionalities."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from qwix._src.core import sparsity

dataclass = dataclasses.dataclass


class PruningFunctionalityTest(parameterized.TestCase):

  def test_prune_inputs_n_m(self):
    inputs = jnp.array(np.random.rand(10, 2, 4))
    prune_rate = (1, 4)

    out = sparsity.prune_inputs_n_m(
        inputs, n=prune_rate[0], m=prune_rate[1], order='R'
    )
    self.assertEqual(out.shape[0], inputs.shape[0])
    self.assertEqual(out.shape[1], inputs.shape[1])
    self.assertEqual(out.shape[2], inputs.shape[2])

    # Only 20 non-zero elements must exist after pruning.
    num_non_zero_elems = 0.25 * inputs.size
    self.assertEqual(out[out != 0].shape[0], num_non_zero_elems)
    self.assertEqual(
        list(np.argmax(inputs, axis=2).flatten()),
        list(np.argmax(out != 0, axis=2).flatten()),
    )

  def test_n_m_pruning_mask(self):
    inputs = jnp.array(np.random.rand(10, 2, 4))
    prune_rate = (1, 4)
    mask = sparsity.get_sparsity_mask(
        inputs, n_sparsity=prune_rate[0], m_sparsity=prune_rate[1], order='R'
    )
    self.assertEqual(
        list(np.argmax(inputs, axis=2).flatten()),
        list(np.argmax(mask == 1, axis=2).flatten()),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='2d_row_wise_pruning',
          order='R',
          inputs=np.arange(1, 73).reshape(6, 12),
          exp_output=[
              [0, 2, 3, 0, 5, 6, 0, 8, 9, 0, 11, 12],
              [0, 14, 15, 0, 17, 18, 0, 20, 21, 0, 23, 24],
              [0, 26, 27, 0, 29, 30, 0, 32, 33, 0, 35, 36],
              [0, 38, 39, 0, 41, 42, 0, 44, 45, 0, 47, 48],
              [0, 50, 51, 0, 53, 54, 0, 56, 57, 0, 59, 60],
              [0, 62, 63, 0, 65, 66, 0, 68, 69, 0, 71, 72],
          ],
          n_sparsity=2,
          m_sparsity=3,
      ),
      dict(
          testcase_name='3d_row_wise_pruning',
          order='R',
          inputs=np.arange(1, 73).reshape(2, 6, 6),
          exp_output=[
              [
                  [0, 2, 3, 0, 5, 6],
                  [0, 8, 9, 0, 11, 12],
                  [0, 14, 15, 0, 17, 18],
                  [0, 20, 21, 0, 23, 24],
                  [0, 26, 27, 0, 29, 30],
                  [0, 32, 33, 0, 35, 36],
              ],
              [
                  [0, 38, 39, 0, 41, 42],
                  [0, 44, 45, 0, 47, 48],
                  [0, 50, 51, 0, 53, 54],
                  [0, 56, 57, 0, 59, 60],
                  [0, 62, 63, 0, 65, 66],
                  [0, 68, 69, 0, 71, 72],
              ],
          ],
          n_sparsity=2,
          m_sparsity=3,
      ),
      dict(
          testcase_name='2d_column_wise_pruning',
          order='C',
          inputs=np.arange(1, 73).reshape(6, 12),
          exp_output=[
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
              [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
          ],
          n_sparsity=2,
          m_sparsity=3,
      ),
      dict(
          testcase_name='3d_column_wise_pruning',
          order='C',
          inputs=np.arange(1, 65).reshape(4, 4, 4),
          exp_output=[
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16],
              ],
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [25, 26, 27, 28],
                  [29, 30, 31, 32],
              ],
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [41, 42, 43, 44],
                  [45, 46, 47, 48],
              ],
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [57, 58, 59, 60],
                  [61, 62, 63, 64],
              ],
          ],
          n_sparsity=2,
          m_sparsity=4,
      ),
      dict(
          testcase_name='3d_column_wise_pruning2',
          order='C',
          inputs=np.arange(1, 33).reshape(2, 4, 4),
          exp_output=[
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16],
              ],
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [25, 26, 27, 28],
                  [29, 30, 31, 32],
              ],
          ],
          n_sparsity=2,
          m_sparsity=4,
      ),
  )
  def test_pruning(self, order, inputs, exp_output, n_sparsity, m_sparsity):
    inputs = jnp.array(inputs)
    output = sparsity.prune_inputs_n_m(
        inputs, n=n_sparsity, m=m_sparsity, order=order
    )
    np.testing.assert_array_equal(output, exp_output)

  @parameterized.named_parameters(
      dict(
          testcase_name='apply_column_mask',
          inputs=np.arange(1, 7).reshape(2, 3),
          mask=[[0, 1, 0]],
          exp_output=[
              [0, 2, 0],
              [0, 5, 0],
          ],
      ),
      dict(
          testcase_name='apply_row_mask',
          inputs=np.arange(1, 7).reshape(2, 3),
          mask=[[0], [1]],
          exp_output=[
              [0, 0, 0],
              [4, 5, 6],
          ],
      ),
  )
  def test_apply_channelwise_mask(self, inputs, mask, exp_output):
    inputs = jnp.array(inputs)
    mask = jnp.array(mask)
    output = sparsity.apply_sparsity(inputs, mask, True)
    np.testing.assert_array_equal(output, exp_output)


class BlockPruningFunctionalityTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='block_size_1', block_size=1),
      dict(testcase_name='block_size_2', block_size=2),
      dict(testcase_name='block_size_4', block_size=4),
  )
  def test_prune_inputs_n_m(self, block_size):
    inputs = jnp.array(np.random.rand(10, 2, 4))
    prune_rate = (1, 2)

    block_mask = sparsity.get_sparsity_mask(
        inputs,
        n_sparsity=prune_rate[0],
        m_sparsity=prune_rate[1],
        order='R',
        block_size=block_size,
    )
    out = sparsity.apply_sparsity(inputs, block_mask)
    self.assertEqual(out.shape[0], inputs.shape[0])
    self.assertEqual(out.shape[1], inputs.shape[1])
    self.assertEqual(out.shape[2], inputs.shape[2])

    # Only 40 non-zero elements must exist after pruning.
    num_non_zero_elems = 0.5 * inputs.size
    self.assertEqual(out[out != 0].shape[0], num_non_zero_elems)

  @parameterized.named_parameters(
      dict(
          testcase_name='2d_row_wise_pruning',
          order='R',
          inputs=np.arange(1, 73).reshape(6, 12),
          exp_output=[
              [0, 0, 3, 4, 5, 6, 0, 0, 9, 10, 11, 12],
              [0, 0, 15, 16, 17, 18, 0, 0, 21, 22, 23, 24],
              [0, 0, 27, 28, 29, 30, 0, 0, 33, 34, 35, 36],
              [0, 0, 39, 40, 41, 42, 0, 0, 45, 46, 47, 48],
              [0, 0, 51, 52, 53, 54, 0, 0, 57, 58, 59, 60],
              [0, 0, 63, 64, 65, 66, 0, 0, 69, 70, 71, 72],
          ],
          n_sparsity=2,
          m_sparsity=3,
          block_size=2,
      ),
      dict(
          testcase_name='2d_row_wise_pruning_2',
          order='R',
          inputs=np.arange(1, 73).reshape(6, 12),
          exp_output=[
              [0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12],
              [0, 0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24],
              [0, 0, 0, 0, 29, 30, 31, 32, 33, 34, 35, 36],
              [0, 0, 0, 0, 41, 42, 43, 44, 45, 46, 47, 48],
              [0, 0, 0, 0, 53, 54, 55, 56, 57, 58, 59, 60],
              [0, 0, 0, 0, 65, 66, 67, 68, 69, 70, 71, 72],
          ],
          n_sparsity=2,
          m_sparsity=3,
          block_size=4,
      ),
      dict(
          testcase_name='3d_row_wise_pruning',
          order='R',
          inputs=np.arange(1, 73).reshape(2, 6, 6),
          exp_output=[
              [
                  [0, 0, 3, 4, 5, 6],
                  [0, 0, 9, 10, 11, 12],
                  [0, 0, 15, 16, 17, 18],
                  [0, 0, 21, 22, 23, 24],
                  [0, 0, 27, 28, 29, 30],
                  [0, 0, 33, 34, 35, 36],
              ],
              [
                  [0, 0, 39, 40, 41, 42],
                  [0, 0, 45, 46, 47, 48],
                  [0, 0, 51, 52, 53, 54],
                  [0, 0, 57, 58, 59, 60],
                  [0, 0, 63, 64, 65, 66],
                  [0, 0, 69, 70, 71, 72],
              ],
          ],
          n_sparsity=2,
          m_sparsity=3,
          block_size=2,
      ),
      dict(
          testcase_name='3d_row_wise_pruning_2',
          order='R',
          inputs=np.arange(1, 73).reshape(1, 6, 12),
          exp_output=[
              [
                  [0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12],
                  [0, 0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24],
                  [0, 0, 0, 0, 29, 30, 31, 32, 33, 34, 35, 36],
                  [0, 0, 0, 0, 41, 42, 43, 44, 45, 46, 47, 48],
                  [0, 0, 0, 0, 53, 54, 55, 56, 57, 58, 59, 60],
                  [0, 0, 0, 0, 65, 66, 67, 68, 69, 70, 71, 72],
              ],
          ],
          n_sparsity=2,
          m_sparsity=3,
          block_size=4,
      ),
      dict(
          testcase_name='2d_column_wise_pruning',
          order='C',
          inputs=np.arange(1, 73).reshape(6, 12),
          exp_output=[
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
              [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
              [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
              [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
          ],
          n_sparsity=2,
          m_sparsity=3,
          block_size=2,
      ),
      dict(
          testcase_name='2d_column_wise_pruning_2',
          order='C',
          inputs=np.arange(1, 73).reshape(12, 6),
          exp_output=[
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [25, 26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35, 36],
              [37, 38, 39, 40, 41, 42],
              [43, 44, 45, 46, 47, 48],
              [49, 50, 51, 52, 53, 54],
              [55, 56, 57, 58, 59, 60],
              [61, 62, 63, 64, 65, 66],
              [67, 68, 69, 70, 71, 72],
          ],
          n_sparsity=2,
          m_sparsity=3,
          block_size=4,
      ),
      dict(
          testcase_name='3d_column_wise_pruning',
          order='C',
          inputs=np.arange(1, 65).reshape(2, 8, 4),
          exp_output=[
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [17, 18, 19, 20],
                  [21, 22, 23, 24],
                  [25, 26, 27, 28],
                  [29, 30, 31, 32],
              ],
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [49, 50, 51, 52],
                  [53, 54, 55, 56],
                  [57, 58, 59, 60],
                  [61, 62, 63, 64],
              ],
          ],
          n_sparsity=2,
          m_sparsity=4,
          block_size=2,
      ),
      dict(
          testcase_name='3d_column_wise_pruning_2',
          order='C',
          inputs=np.arange(1, 65).reshape(1, 16, 4),
          exp_output=[
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [33, 34, 35, 36],
                  [37, 38, 39, 40],
                  [41, 42, 43, 44],
                  [45, 46, 47, 48],
                  [49, 50, 51, 52],
                  [53, 54, 55, 56],
                  [57, 58, 59, 60],
                  [61, 62, 63, 64],
              ],
          ],
          n_sparsity=2,
          m_sparsity=4,
          block_size=4,
      ),
  )
  def test_block_pruning(
      self, order, inputs, exp_output, n_sparsity, m_sparsity, block_size
  ):
    inputs = jnp.array(inputs)
    block_mask = sparsity.get_sparsity_mask(
        inputs, n_sparsity, m_sparsity, order=order, block_size=block_size
    )
    output = sparsity.apply_sparsity(inputs, block_mask)
    np.testing.assert_array_equal(output, exp_output)


class StructuredCompression1To4Test(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='f32_axis0', dtype=jnp.float32, axis=0),
      dict(testcase_name='bf16_axis1', dtype=jnp.bfloat16, axis=1),
      dict(testcase_name='f8e4m3_axis0', dtype=jnp.float8_e4m3fn, axis=0),
      dict(testcase_name='f8e5m2_axis_neg1', dtype=jnp.float8_e5m2, axis=-1),
      dict(testcase_name='int4_axis0', dtype=jnp.int4, axis=0),
  )
  def test_compress_decompress_roundtrip(self, dtype, axis):
    rng = np.random.default_rng(42)
    raw = rng.integers(-7, 8, size=(16, 12)).astype(np.float32)
    # Add distinct magnitudes within each group of 4 so argmax is unique.
    arr = jnp.array(raw).astype(dtype)
    values, indices = sparsity.compress_1_4(arr, axis=axis)
    norm_axis = axis % arr.ndim
    expected_shape = list(arr.shape)
    expected_shape[norm_axis] //= 4
    self.assertEqual(values.shape, tuple(expected_shape))
    self.assertEqual(values.dtype, dtype)
    self.assertEqual(indices.shape, tuple(expected_shape))
    self.assertEqual(indices.dtype, jnp.uint8)

    decompressed = sparsity.decompress_1_4(values, indices, axis=axis)
    self.assertEqual(decompressed.shape, arr.shape)
    self.assertEqual(decompressed.dtype, dtype)

    # Compare with dense 1:4 pruning along the same axis.
    moved = jnp.moveaxis(arr, norm_axis, -1)
    grouped = moved.reshape(*moved.shape[:-1], -1, 4)
    idx = jnp.argmax(jnp.abs(grouped.astype(jnp.float32)), axis=-1)
    mask = jax_one_hot_bool(idx, 4)
    expected_dense = jnp.moveaxis(
        jnp.where(mask, grouped, jnp.zeros_like(grouped)).reshape(moved.shape),
        -1,
        norm_axis,
    )
    np.testing.assert_array_equal(
        np.asarray(decompressed, dtype=np.float32),
        np.asarray(expected_dense, dtype=np.float32),
    )

  def test_pack_unpack_u2_indices(self):
    rng = np.random.default_rng(0)
    indices = jnp.array(rng.integers(0, 4, size=(8, 16), dtype=np.uint8))
    for pack_axis in (0, 1, -1):
      packed = sparsity.pack_u2_indices(indices, pack_axis=pack_axis)
      norm_axis = pack_axis % indices.ndim
      expected_shape = list(indices.shape)
      expected_shape[norm_axis] //= 4
      self.assertEqual(packed.shape, tuple(expected_shape))
      self.assertEqual(packed.dtype, jnp.uint8)
      unpacked = sparsity.unpack_u2_indices(packed, pack_axis=pack_axis)
      np.testing.assert_array_equal(unpacked, indices)

  def test_compress_invalid_shape_raises(self):
    x = jnp.ones((6, 8), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, 'divisible by 4'):
      sparsity.compress_1_4(x, axis=0)

  def test_pack_u2_invalid_shape_raises(self):
    idx = jnp.zeros((6, 8), dtype=jnp.uint8)
    with self.assertRaisesRegex(ValueError, 'divisible by 4'):
      sparsity.pack_u2_indices(idx, pack_axis=0)


def jax_one_hot_bool(indices, num_classes):
  return indices[..., None] == jnp.arange(num_classes, dtype=indices.dtype)


if __name__ == '__main__':
  absltest.main()
