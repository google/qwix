# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from qwix.core import dot_general
from qwix.core import pallas
from qwix.core import qarray


class PallasTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="channelwise",
          block_shape=(128, 128),
          qvalue_shape=(1024, 1024),
          scale_shape=(1024, 1),
          expected_scale_block_shape=(128, 1),
          expected_scale_index_map={(1, 1): (1, 0)},
      ),
      dict(
          testcase_name="blockwise",
          block_shape=(256, 256),
          qvalue_shape=(1024, 1024),
          scale_shape=(32, 8),  # 32x128 tiling.
          expected_scale_block_shape=(8, 2),
          expected_scale_index_map={(1, 1): (1, 1)},
      ),
  )
  def test_update_block_specs_for_qarray(
      self,
      block_shape,
      qvalue_shape,
      scale_shape,
      expected_scale_block_shape,
      expected_scale_index_map,
  ):
    block_spec = pl.BlockSpec(block_shape, lambda *args: tuple(args))
    array = qarray.QArray(
        qvalue=jax.ShapeDtypeStruct(qvalue_shape, jnp.int8),
        scale=jax.ShapeDtypeStruct(scale_shape, jnp.float32),
        zero_point=None,
        qtype="int8",
    )
    new_block_spec = pallas._update_block_specs_for_qarray(block_spec, array)
    self.assertIsInstance(new_block_spec, qarray.QArray)
    self.assertEqual(new_block_spec.qvalue, block_spec)
    self.assertIsInstance(new_block_spec.scale, pl.BlockSpec)
    self.assertEqual(
        new_block_spec.scale.block_shape, expected_scale_block_shape
    )
    for k, v in expected_scale_index_map.items():
      self.assertEqual(new_block_spec.scale.index_map(*k), v)

  @parameterized.named_parameters(
      dict(
          testcase_name="no_transform",
          arg_shape=(4, 128),
          block_shape=(4, 128),
          expected_new_block_shape=(4, 128),
      ),
      dict(
          testcase_name="transpose",
          block_shape=(128, 8),
          arg_shape=(128, 16),
          expected_new_block_shape=(8, 128),
      ),
      dict(
          testcase_name="reshape",
          block_shape=(8, 8),
          arg_shape=(32, 32),
          expected_new_block_shape=(1, 1, 1, 64),
      ),
  )
  def test_transform_block_specs_for_tpu(
      self, block_shape, arg_shape, expected_new_block_shape
  ):
    block_spec = pl.BlockSpec(block_shape, lambda *args: tuple(args))
    arg = jnp.zeros(arg_shape, dtype=jnp.float32)
    new_block_spec, new_array, restore_fn = (
        pallas._transform_block_specs_for_tpu(block_spec, arg)
    )
    self.assertTrue(
        pallas._can_fit_tpu_requirements(
            new_block_spec.block_shape, new_array.shape
        )
    )
    self.assertEqual(new_block_spec.block_shape, expected_new_block_shape)
    restored = restore_fn(
        jnp.zeros(expected_new_block_shape, dtype=jnp.float32)
    )
    self.assertEqual(restored.shape, block_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="tensorwise",
          input_shape=(256, 256),
          tiled_axes={},
      ),
      dict(
          testcase_name="channelwise0",
          input_shape=(256, 256),
          tiled_axes={0: 1},
      ),
      dict(
          testcase_name="channelwise1",
          input_shape=(256, 256),
          tiled_axes={1: 1},
      ),
      dict(
          testcase_name="blockwise0",
          input_shape=(256, 256),
          tiled_axes={0: 128},
      ),
      dict(
          testcase_name="blockwise1",
          input_shape=(256, 256),
          tiled_axes={1: 128},
      ),
      dict(
          testcase_name="blockwise2",
          input_shape=(256, 256),
          tiled_axes={0: 128, 1: 128},
      ),
      dict(
          testcase_name="blockwise_and_channelwise",
          input_shape=(256, 256),
          tiled_axes={0: 128, 1: 1},
      ),
      dict(
          testcase_name="3d",
          input_shape=(4, 256, 256),
          tiled_axes={1: 1, 2: 128},
      ),
      dict(
          # This test case triggers transpose codepath.
          testcase_name="transpose",
          input_shape=(128, 2048),
          tiled_axes={0: 1, 1: 128},
          bs=(128, 1024),
      ),
  )
  def test_pallas_dequantize(self, input_shape, tiled_axes, bs=(128, 128)):
    """Comprehensive tests for the pallas_call function."""

    def dequantize_kernel(q_ref, out_ref):
      out_ref[...] = qarray.dequantize(jax.tree.map(lambda x: x[...], q_ref))

    def dequantize_pallas(q: qarray.QArray):
      block_shape = (1,) * (q.qvalue.ndim - 2) + bs

      return pallas.pallas_call(
          dequantize_kernel,
          out_shape=jax.ShapeDtypeStruct(q.qvalue.shape, q.scale.dtype),
          in_specs=[pl.BlockSpec(block_shape, lambda *args: args)],
          out_specs=pl.BlockSpec(block_shape, lambda *args: args),
          grid=tuple(i // bs for i, bs in zip(q.qvalue.shape, block_shape)),
      )(q)

    x = jax.random.uniform(jax.random.key(0), input_shape, jnp.float32)
    how = qarray.HowToQuantize(
        qtype="int8",
        channelwise_axes=[],
        tiled_axes=tiled_axes,
        batch_axes=[],
        calibration_method="absmax",
    )
    qx = qarray.quantize(x, how)
    self.assertTrue(jnp.allclose(dequantize_pallas(qx), qarray.dequantize(qx)))

  def test_pallas_matmul(self):
    """A basic example of using Qwix pallas_call to implement matmul."""

    def pallas_matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
      @pl.when(pl.program_id(2) == 0)
      def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

      x = jax.tree.map(lambda x: x[...], x_ref)
      y = jax.tree.map(lambda x: x[...], y_ref)
      # NOTE: Qwix's dot_general is not generally supported inside pallas
      # kernels, as the reshape and transpose operations will trigger errors
      # when block sizes != subchannel tiled sizes.
      acc_ref[...] += dot_general.dot_general(x, y, (((1,), (0,)), ((), ())))

      @pl.when(pl.program_id(2) == nsteps - 1)
      def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)

    def pallas_matmul(
        x: qarray.QArray,
        y: qarray.QArray,
        *,
        bm: int = 128,
        bk: int = 128,
        bn: int = 128,
    ):
      m, k = x.qvalue.shape
      _, n = y.qvalue.shape

      return pallas.pallas_call(
          functools.partial(pallas_matmul_kernel, nsteps=k // bk),
          out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
          grid=(m // bm, n // bn, k // bk),
          in_specs=[
              pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
              pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
          ],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
          scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel", "parallel", "arbitrary")
          ),
      )(x, y)

    x_how = qarray.HowToQuantize(
        qtype="int8",
        channelwise_axes=[],
        tiled_axes={0: 1, 1: 128},
        batch_axes=[],
        calibration_method="absmax",
    )
    qx = qarray.quantize(
        jax.random.uniform(jax.random.key(0), (256, 256), jnp.float32), x_how
    )
    y_how = qarray.HowToQuantize(
        qtype="int8",
        channelwise_axes=[],
        tiled_axes={0: 128, 1: 1},
        batch_axes=[],
        calibration_method="absmax",
    )
    qy = qarray.quantize(
        jax.random.uniform(jax.random.key(1), (256, 256), jnp.float32), y_how
    )

    self.assertTrue(
        jnp.allclose(
            pallas_matmul(qx, qy),
            dot_general.dot_general(qx, qy, (((1,), (0,)), ((), ()))),
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="weight_only_single_scale",
          lhs_shape=(8, 128),
          lhs_dtype=jnp.bfloat16,
          lhs_scale_shape=None,
          rhs_shape=(256, 128),
          rhs_scale_shape=(1, 1),
          dimension_numbers=(([1], [1]), ([], [])),
      ),
      # Not implemented: Broadcast in both sublanes and lanes.
      #   "vector.broadcast" (vector<1x1xbf16>) -> vector<8x256xbf16>
      # dict(
      #     testcase_name="single_scale_bf16",
      #     lhs_shape=(8, 128),
      #     lhs_scale_shape=(1, 1),
      #     lhs_scale_dtype=jnp.bfloat16,
      #     rhs_shape=(256, 128),
      #     rhs_scale_shape=(1, 1),
      #     rhs_scale_dtype=jnp.bfloat16,
      #     dimension_numbers=(([1], [1]), ([], [])),
      # ),
      dict(
          testcase_name="channelwise_fp32",
          lhs_shape=(8, 128),
          lhs_scale_shape=(8, 1),
          rhs_shape=(128, 256),
          rhs_scale_shape=(1, 256),
          dimension_numbers=(([1], [0]), ([], [])),
      ),
      dict(
          testcase_name="channelwise_bf16",
          lhs_shape=(8, 16, 128),
          lhs_scale_shape=(8, 16, 1),
          lhs_scale_dtype=jnp.bfloat16,
          rhs_shape=(256, 128),
          rhs_scale_shape=(256, 1),
          rhs_scale_dtype=jnp.bfloat16,
          dimension_numbers=(([2], [1]), ([], [])),
      ),
      dict(
          testcase_name="subchannel",
          lhs_shape=(8, 16, 512),
          lhs_scale_shape=(8, 16, 4),
          rhs_shape=(512, 256),
          rhs_scale_shape=(4, 256),
          dimension_numbers=(([2], [0]), ([], [])),
      ),
  )
  def test_pallas_dot_general(
      self,
      *,
      lhs_shape,
      lhs_dtype=jnp.int8,
      lhs_scale_shape,
      lhs_scale_dtype=jnp.float32,
      rhs_shape,
      rhs_dtype=jnp.int8,
      rhs_scale_shape,
      rhs_scale_dtype=jnp.float32,
      dimension_numbers,
  ):
    """Test what kind of dot_general can be called in pallas kernels."""

    def pl_kernel(x, y, o):
      o[...] = dot_general.loop_dot_general(
          jax.tree.map(lambda x: x[...], x),
          jax.tree.map(lambda x: x[...], y),
          dimension_numbers,
      )

    if lhs_scale_shape is None:
      lhs = self._make_array(lhs_shape, lhs_dtype)
    else:
      lhs = qarray.QArray(
          self._make_array(lhs_shape, lhs_dtype),
          self._make_array(lhs_scale_shape, lhs_scale_dtype),
          None,
          lhs_dtype,
      )
    rhs = qarray.QArray(
        self._make_array(rhs_shape, rhs_dtype),
        self._make_array(rhs_scale_shape, rhs_scale_dtype),
        None,
        rhs_dtype,
    )
    jax_output = dot_general.dot_general(lhs, rhs, dimension_numbers)
    output_shape = jax.ShapeDtypeStruct(jax_output.shape, jax_output.dtype)
    pallas_output = pl.pallas_call(pl_kernel, output_shape)(lhs, rhs)
    self.assertLess(
        jnp.abs(jax_output - pallas_output).mean() / jnp.abs(jax_output).mean(),
        2e-3,  # bf16 has a lot error.
    )

  def _make_array(self, shape, dtype):
    try:
      iinfo = jnp.iinfo(dtype)
      return jax.random.randint(
          jax.random.key(0), shape, int(iinfo.min), int(iinfo.max), dtype
      )
    except ValueError:
      # dtype is float.
      return jax.random.normal(jax.random.key(0), shape, dtype)


if __name__ == "__main__":
  absltest.main()
