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


class AxisPruningFunctionalityTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='2d_axis0_2_4',
          inputs=[
              [1.0, -4.0],
              [-3.0, 1.0],
              [2.0, -3.0],
              [-4.0, 2.0],
              [5.0, -1.0],
              [-1.0, 6.0],
              [2.0, -7.0],
              [-6.0, 3.0],
          ],
          n=2,
          m=4,
          axis=0,
          expected_mask=[
              [False, True],
              [True, False],
              [False, True],
              [True, False],
              [True, False],
              [False, True],
              [False, True],
              [True, False],
          ],
      ),
      dict(
          testcase_name='2d_axis1_2_4',
          inputs=[
              [1.0, -4.0, 3.0, -2.0, 5.0, -1.0, 2.0, -6.0],
              [-5.0, 2.0, -1.0, 4.0, -3.0, 6.0, -7.0, 1.0],
          ],
          n=2,
          m=4,
          axis=1,
          expected_mask=[
              [False, True, True, False, True, False, False, True],
              [True, False, False, True, False, True, True, False],
          ],
      ),
      dict(
          testcase_name='2d_axis0_1_4',
          inputs=[
              [1.0, -4.0],
              [-3.0, 1.0],
              [2.0, -3.0],
              [-4.0, 2.0],
          ],
          n=1,
          m=4,
          axis=0,
          expected_mask=[
              [False, True],
              [False, False],
              [False, False],
              [True, False],
          ],
      ),
      dict(
          testcase_name='2d_axis0_3_4',
          inputs=[
              [1.0, -4.0],
              [-3.0, 1.0],
              [2.0, -3.0],
              [-4.0, 2.0],
          ],
          n=3,
          m=4,
          axis=0,
          expected_mask=[
              [False, True],
              [True, False],
              [True, True],
              [True, True],
          ],
      ),
      dict(
          testcase_name='3d_middle_axis_2_4',
          inputs=[
              [[1.0, -4.0], [-3.0, 1.0], [2.0, -3.0], [-4.0, 2.0]],
              [[5.0, -1.0], [-1.0, 6.0], [2.0, -7.0], [-6.0, 3.0]],
          ],
          n=2,
          m=4,
          axis=1,
          expected_mask=[
              [[False, True], [True, False], [False, True], [True, False]],
              [[True, False], [False, True], [False, True], [True, False]],
          ],
      ),
      dict(
          testcase_name='3d_negative_axis_2_4',
          inputs=[
              [[1.0, -4.0], [-3.0, 1.0], [2.0, -3.0], [-4.0, 2.0]],
              [[5.0, -1.0], [-1.0, 6.0], [2.0, -7.0], [-6.0, 3.0]],
          ],
          n=2,
          m=4,
          axis=-2,
          expected_mask=[
              [[False, True], [True, False], [False, True], [True, False]],
              [[True, False], [False, True], [False, True], [True, False]],
          ],
      ),
  )
  def test_get_sparsity_mask_along_axis(
      self, inputs, n, m, axis, expected_mask
  ):
    inputs = jnp.asarray(inputs, dtype=jnp.float32)
    expected_mask = jnp.asarray(expected_mask, dtype=jnp.bool_)
    mask = sparsity.get_sparsity_mask(
        inputs, n_sparsity=n, m_sparsity=m, axis=axis
    )
    self.assertEqual(mask.shape, inputs.shape)
    self.assertEqual(mask.dtype, jnp.bool_)
    np.testing.assert_array_equal(mask, expected_mask)

    pruned = sparsity.prune_inputs_n_m(inputs, n=n, m=m, axis=axis)
    np.testing.assert_array_equal(pruned, jnp.where(expected_mask, inputs, 0.0))

  def test_axis_ties_keep_lower_indices(self):
    ties = jnp.ones((4, 2), dtype=jnp.float32)
    tie_mask = sparsity.get_sparsity_mask(
        ties, n_sparsity=2, m_sparsity=4, axis=0
    )
    np.testing.assert_array_equal(
        tie_mask,
        jnp.array([[True, True], [True, True], [False, False], [False, False]]),
    )

  def test_axis_errors(self):
    x = jnp.ones((6, 4), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, 'not divisible by M=4'):
      sparsity.get_sparsity_mask(x, n_sparsity=2, m_sparsity=4, axis=0)
    for bad_axis in (2, -3):
      with self.assertRaisesRegex(ValueError, 'out of range'):
        sparsity.get_sparsity_mask(x, n_sparsity=2, m_sparsity=4, axis=bad_axis)
    with self.assertRaisesRegex(ValueError, 'block_size and offset'):
      sparsity.get_sparsity_mask(
          jnp.ones((8, 4)), n_sparsity=2, m_sparsity=4, block_size=2, axis=0
      )
    with self.assertRaisesRegex(ValueError, 'block_size and offset'):
      sparsity.get_sparsity_mask(
          jnp.ones((8, 4)), n_sparsity=2, m_sparsity=4, offset=2, axis=0
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='2d_axis1_1_4_stride2',
          inputs=[
              [
                  1.0,
                  2.0,
                  -3.0,
                  6.0,
                  -9.0,
                  -4.0,
                  5.0,
                  8.0,
                  7.0,
                  -1.0,
                  -2.0,
                  -8.0,
                  4.0,
                  3.0,
                  -6.0,
                  5.0,
              ],
              [2.0] * 16,
          ],
          n=1,
          m=4,
          axis=1,
          stride=2,
          expected_mask=[
              [
                  False,
                  False,
                  False,
                  False,
                  True,
                  False,
                  False,
                  True,
                  True,
                  False,
                  False,
                  True,
                  False,
                  False,
                  False,
                  False,
              ],
              [
                  True,
                  True,
                  False,
                  False,
                  False,
                  False,
                  False,
                  False,
                  True,
                  True,
                  False,
                  False,
                  False,
                  False,
                  False,
                  False,
              ],
          ],
      ),
      dict(
          testcase_name='2d_axis0_2_4_stride2',
          inputs=[
              [1.0, -8.0],
              [4.0, 3.0],
              [-5.0, 2.0],
              [-2.0, -9.0],
              [3.0, 6.0],
              [-6.0, 1.0],
              [-7.0, -1.0],
              [1.0, 5.0],
          ],
          n=2,
          m=4,
          axis=0,
          stride=2,
          expected_mask=[
              [False, True],
              [True, False],
              [True, False],
              [False, True],
              [False, True],
              [True, False],
              [True, False],
              [False, True],
          ],
      ),
      dict(
          testcase_name='3d_negative_middle_axis_1_2_stride3',
          inputs=[
              [
                  [9.0, 1.0, -2.0, 5.0],
                  [2.0, -6.0, 4.0, 1.0],
                  [-8.0, 2.0, 3.0, -4.0],
                  [3.0, -8.0, -7.0, 2.0],
                  [-5.0, 3.0, 1.0, -9.0],
                  [1.0, 7.0, -6.0, 4.0],
              ],
              [
                  [1.0, -9.0, 6.0, 2.0],
                  [7.0, 1.0, -8.0, 3.0],
                  [3.0, 8.0, 1.0, -9.0],
                  [-4.0, 2.0, 1.0, -7.0],
                  [2.0, -5.0, 4.0, -6.0],
                  [-7.0, 4.0, -5.0, 2.0],
              ],
          ],
          n=1,
          m=2,
          axis=-2,
          stride=3,
          expected_mask=[
              [
                  [True, False, False, True],
                  [False, True, True, False],
                  [True, False, False, True],
                  [False, True, True, False],
                  [True, False, False, True],
                  [False, True, True, False],
              ],
              [
                  [False, True, True, False],
                  [True, False, True, False],
                  [False, True, False, True],
                  [True, False, False, True],
                  [False, True, False, True],
                  [True, False, True, False],
              ],
          ],
      ),
  )
  def test_get_sparsity_mask_with_stride(
      self, inputs, n, m, axis, stride, expected_mask
  ):
    inputs = jnp.asarray(inputs, dtype=jnp.float32)
    mask = sparsity.get_sparsity_mask(
        inputs, n_sparsity=n, m_sparsity=m, axis=axis, stride=stride
    )
    self.assertEqual(mask.dtype, jnp.bool_)
    np.testing.assert_array_equal(mask, expected_mask)

  @parameterized.named_parameters(
      ('zero_no_axis', dict(stride=0), 'stride must be positive'),
      (
          'negative_with_axis',
          dict(axis=0, stride=-2),
          'stride must be positive',
      ),
      ('stride_without_axis', dict(stride=2), 'requires an explicit axis'),
      (
          'not_divisible_by_m_times_stride',
          dict(axis=0, stride=2),
          r'not divisible by M\*stride=8',
      ),
  )
  def test_stride_errors(self, kwargs, error_regex):
    x = jnp.ones((4, 4), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, error_regex):
      sparsity.get_sparsity_mask(x, 2, 4, **kwargs)


class StructuredSparsityTest(parameterized.TestCase):

  def test_defaults_and_dimension(self):
    sp = sparsity.StructuredSparsity(1, 4)
    self.assertEqual((sp.n, sp.m, sp.stride, sp.dimension), (1, 4, 1, None))
    sp_dim = sparsity.StructuredSparsity(2, 4, dimension=-1)
    self.assertEqual((sp_dim.n, sp_dim.m, sp_dim.dimension), (2, 4, -1))

  @parameterized.named_parameters(
      ('n_zero', 0, 4, 1, ValueError, '0 < n < m <= 4'),
      ('n_eq_m', 4, 4, 1, ValueError, '0 < n < m <= 4'),
      ('m_gt_4', 2, 8, 1, ValueError, '0 < n < m <= 4'),
      ('stride_zero', 1, 4, 0, ValueError, 'stride must be positive'),
      ('stride_neg', 1, 4, -1, ValueError, 'stride must be positive'),
      ('stride_two', 1, 4, 2, NotImplementedError, 'Only stride == 1'),
  )
  def test_invalid_args(self, n, m, stride, err_cls, regex):
    with self.assertRaisesRegex(err_cls, regex):
      sparsity.StructuredSparsity(n, m, stride=stride)


if __name__ == '__main__':
  absltest.main()
