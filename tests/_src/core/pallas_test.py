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
import itertools
import re

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from qwix._src.core import dot_general
from qwix._src.core import pallas
from qwix._src.core import qarray

_TILE_SHAPE = (128, 256)
_FUSED_M = 768
_FUSED_K = 1024
_FUSED_N = 640
_FUSED_BM = 256
_FUSED_BK = 512
_FUSED_BN = 128
_QUANTIZE_TILE_CASES = [
    dict(
        testcase_name=(
            f"{jnp.dtype(dt).name}_{jnp.dtype(qt).name}_{cal}"
            f"_cw{''.join(map(str, cw)) or 'none'}"
        ),
        dtype=dt,
        qtype=qt,
        calibration_method=cal,
        channelwise_axes=cw,
    )
    for dt, (qt, cal), cw in itertools.product(
        (jnp.bfloat16, jnp.float32),
        (
            (jnp.int8, "absmax"),
            (jnp.float8_e4m3fn, "absmax"),
            (jnp.int8, "minmax"),
        ),
        ((0,), (1,), ()),
    )
]


class PallasTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    platform = jax.devices()[0].platform
    if platform == "tpu":
      cls.pallas_call_kwargs = dict()
    else:
      cls.pallas_call_kwargs = dict(interpret=True)

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
      dict(
          testcase_name="block_shape_none",
          block_shape=(None, 256),
          qvalue_shape=(1024, 1024),
          scale_shape=(1024, 8),  # 1x128 tiling.
          expected_scale_block_shape=(None, 2),
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
    new_block_spec = pallas.update_block_specs_for_qarray(block_spec, array)
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
          arg_shape=(4, 4, 128),
          block_shape=(None, 4, 128),
          expected_new_block_shape=(None, 4, 128),
          expected_new_index_map={(0, 1, 2): (0, 1, 2)},
      ),
      dict(
          testcase_name="transpose",
          arg_shape=(4, 128, 16),
          block_shape=(None, 128, 8),
          expected_new_block_shape=(None, 8, 128),
          expected_new_index_map={(1, 2, 3): (1, 3, 2)},
      ),
      dict(
          testcase_name="transpose_for_optimal",
          arg_shape=(256, 1),
          block_shape=(128, 1),
          expected_new_block_shape=(1, 128),
          expected_new_index_map={(1, 0): (0, 1)},
      ),
      dict(
          testcase_name="reshape",
          block_shape=(8, 8),
          arg_shape=(32, 32),
          expected_new_block_shape=(1, 1, 1, 64),
          expected_new_index_map={(1, 2): (1, 2, 0, 0)},
      ),
  )
  def test_transform_block_specs_for_tpu(
      self,
      arg_shape,
      block_shape,
      expected_new_block_shape,
      expected_new_index_map,
  ):
    block_spec = pl.BlockSpec(block_shape, lambda *args: tuple(args))
    arg = jnp.zeros(arg_shape, dtype=jnp.float32)
    new_block_spec, new_array, restore_fn = (
        pallas.transform_block_specs_for_tpu(block_spec, arg)
    )
    self.assertTrue(
        pallas._can_fit_tpu_requirements(
            new_block_spec.block_shape, new_array.shape
        )
    )
    self.assertEqual(new_block_spec.block_shape, expected_new_block_shape)
    for k, v in expected_new_index_map.items():
      self.assertEqual(new_block_spec.index_map(*k), v)
    restored = restore_fn(
        jnp.zeros(
            [s for s in expected_new_block_shape if s is not None],
            dtype=jnp.float32,
        )
    )
    self.assertEqual(
        restored.shape, tuple(s for s in block_shape if s is not None)
    )

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

    def dequantize_kernel(scalar_ref, q_ref, out_ref):
      del scalar_ref  # Unused.
      out_ref[...] = qarray.dequantize(q_ref[...])

    def dequantize_pallas(q: qarray.QArray):
      block_shape = (1,) * (q.qvalue.ndim - 2) + bs

      return pallas.pallas_call(
          dequantize_kernel,
          out_shape=jax.ShapeDtypeStruct(q.qvalue.shape, q.scale.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              grid=tuple(i // bs for i, bs in zip(q.qvalue.shape, block_shape)),
              in_specs=[pl.BlockSpec(block_shape, lambda *args: args[:-1])],
              out_specs=pl.BlockSpec(block_shape, lambda *args: args[:-1]),
          ),
          **self.pallas_call_kwargs,
      )(jnp.ones((1,), jnp.float32), q)

    x = jax.random.uniform(jax.random.key(0), input_shape, jnp.float32)
    how = qarray.HowToQuantize(
        qtype="int8",
        tiled_axes=tiled_axes,
    )
    qx = qarray.quantize(x, how)
    self.assertTrue(jnp.allclose(dequantize_pallas(qx), qarray.dequantize(qx)))

  def test_pallas_batch_matmul(self):
    """A basic example of using Qwix pallas_call to implement batch_matmul."""

    def pallas_matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):

      @pl.when(pl.program_id(3) == 0)
      def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

      acc_ref[...] += dot_general.loop_dot_general(
          x_ref[...], y_ref[...], (((1,), (0,)), ((), ()))
      )

      @pl.when(pl.program_id(3) == nsteps - 1)
      def _write():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)

    def pallas_batch_matmul(
        x: qarray.QArray,
        y: qarray.QArray,
        *,
        bm: int = 128,
        bk: int = 128,
        bn: int = 128,
    ):
      b, m, k = x.qvalue.shape
      _, _, n = y.qvalue.shape

      return pallas.pallas_call(
          functools.partial(pallas_matmul_kernel, nsteps=k // bk),
          out_shape=jax.ShapeDtypeStruct((b, m, n), jnp.float32),
          grid=(b, m // bm, n // bn, k // bk),
          in_specs=[
              pl.BlockSpec((None, bm, bk), lambda b, i, j, k: (b, i, k)),
              pl.BlockSpec((None, bk, bn), lambda b, i, j, k: (b, k, j)),
          ],
          out_specs=pl.BlockSpec((None, bm, bn), lambda b, i, j, k: (b, i, j)),
          scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
          **self.pallas_call_kwargs,
      )(x, y)

    x_how = qarray.HowToQuantize(
        qtype="int8",
        channelwise_axes=[0, 1],
        tiled_axes={2: 128},
    )
    qx = qarray.quantize(
        jax.random.uniform(jax.random.key(0), (4, 256, 256), jnp.float32), x_how
    )
    y_how = qarray.HowToQuantize(
        qtype="int8",
        channelwise_axes=[2],
        tiled_axes={1: 128},
    )
    qy = qarray.quantize(
        jax.random.uniform(jax.random.key(1), (4, 256, 256), jnp.float32), y_how
    )

    self.assertTrue(
        jnp.allclose(
            pallas_batch_matmul(qx, qy),
            dot_general.dot_general(qx, qy, (((2,), (1,)), ((0,), (0,)))),
        )
    )

  def _run_pallas_dot_general(self, lhs, rhs, dimension_numbers):
    """Runs loop_dot_general inside pl.pallas_call and returns (xla, pallas)."""

    def pl_kernel(x, y, o):
      o[...] = dot_general.loop_dot_general(x[...], y[...], dimension_numbers)

    jax_output = dot_general.dot_general(lhs, rhs, dimension_numbers)
    output_shape = jax.ShapeDtypeStruct(jax_output.shape, jax_output.dtype)
    pallas_output = pl.pallas_call(
        pl_kernel, output_shape, **self.pallas_call_kwargs
    )(lhs, rhs)
    return jax_output, pallas_output

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
          rhs_scale_shape=(4, 2),
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
    jax_output, pallas_output = self._run_pallas_dot_general(
        lhs, rhs, dimension_numbers
    )
    self.assertLess(
        jnp.abs(jax_output - pallas_output).mean() / jnp.abs(jax_output).mean(),
        2e-3,  # bf16 has a lot error.
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="single_scale_f32",
          qtype=jnp.int8,
          lhs_shape=(128, 256),
          rhs_shape=(256, 128),
          lhs_channelwise_axes=(),
          rhs_channelwise_axes=(),
          scale_dtype=jnp.float32,
          dimension_numbers=(([1], [0]), ([], [])),
      ),
      dict(
          testcase_name="single_scale_bf16",
          qtype=jnp.int8,
          lhs_shape=(8, 128),
          rhs_shape=(256, 128),
          lhs_channelwise_axes=(),
          rhs_channelwise_axes=(),
          dimension_numbers=(([1], [1]), ([], [])),
      ),
      dict(
          testcase_name="fp8_channelwise_f32",
          qtype=jnp.float8_e4m3fn,
          lhs_shape=(128, 256),
          rhs_shape=(256, 128),
          lhs_channelwise_axes=(0,),
          rhs_channelwise_axes=(1,),
          scale_dtype=jnp.float32,
          dimension_numbers=(([1], [0]), ([], [])),
      ),
      dict(
          testcase_name="mxfp8_f32_scale",
          qtype="mxfp8",
          lhs_shape=(128, 256),
          rhs_shape=(256, 128),
          lhs_channelwise_axes=(0,),
          rhs_channelwise_axes=(1,),
          lhs_tiled_axes={1: 32},
          rhs_tiled_axes={0: 32},
          scale_dtype=jnp.float32,
          dimension_numbers=(([1], [0]), ([], [])),
      ),
  )
  def test_pallas_dot_general_quantized(
      self,
      *,
      qtype,
      lhs_shape,
      rhs_shape,
      lhs_channelwise_axes,
      rhs_channelwise_axes,
      lhs_tiled_axes=None,
      rhs_tiled_axes=None,
      scale_dtype=None,
      dimension_numbers,
  ):
    """Tests loop_dot_general on host-quantized tiles matching quantize_tile."""
    lhs = qarray.quantize_api(
        jax.random.normal(jax.random.key(0), lhs_shape, jnp.bfloat16),
        qtype,
        channelwise_axes=lhs_channelwise_axes,
        tiled_axes=lhs_tiled_axes,
        scale_dtype=scale_dtype,
    )
    rhs = qarray.quantize_api(
        jax.random.normal(jax.random.key(1), rhs_shape, jnp.bfloat16),
        qtype,
        channelwise_axes=rhs_channelwise_axes,
        tiled_axes=rhs_tiled_axes,
        scale_dtype=scale_dtype,
    )
    jax_output, pallas_output = self._run_pallas_dot_general(
        lhs, rhs, dimension_numbers
    )
    self.assertLess(
        jnp.abs(jax_output - pallas_output).mean() / jnp.abs(jax_output).mean(),
        2e-3,  # bf16 has a lot error.
    )

  def _compile_ref(self, fn):
    # Mosaic matches op-by-op XLA on TPU; interpret mode compiles through
    # XLA:CPU and matches jitted code.
    return fn if jax.devices()[0].platform == "tpu" else jax.jit(fn)

  @parameterized.named_parameters(_QUANTIZE_TILE_CASES)
  def test_quantize_tile(
      self, dtype, qtype, calibration_method, channelwise_axes
  ):
    """Tests qarray.quantize_tile in pl.pallas_call against qarray.quantize_api."""
    x = jax.random.normal(jax.random.key(0), _TILE_SHAPE, dtype)
    expected = self._compile_ref(
        lambda arr: qarray.quantize_api(
            arr,
            qtype,
            channelwise_axes=channelwise_axes,
            calibration_method=calibration_method,
        )
    )(x)

    def quantize_kernel(x_ref, q_ref, s_ref, *zp_refs):
      qa = qarray.quantize_tile(
          x_ref[...],
          qtype,
          channelwise_axes=channelwise_axes,
          calibration_method=calibration_method,
      )
      q_ref[...] = qa.qvalue
      s_ref[...] = qa.scale
      if zp_refs:
        zp_refs[0][...] = qa.zero_point

    out_shape = [
        jax.ShapeDtypeStruct(expected.qvalue.shape, expected.qvalue.dtype),
        jax.ShapeDtypeStruct(expected.scale.shape, expected.scale.dtype),
    ]
    if expected.zero_point is not None:
      out_shape.append(
          jax.ShapeDtypeStruct(
              expected.zero_point.shape, expected.zero_point.dtype
          )
      )

    qvalue, scale, *zp = pl.pallas_call(
        quantize_kernel,
        out_shape=tuple(out_shape),
        **self.pallas_call_kwargs,
    )(x)

    self.assertEqual(qvalue.dtype, expected.qvalue.dtype)
    self.assertEqual(scale.dtype, expected.scale.dtype)
    self.assertTrue(jnp.array_equal(qvalue, expected.qvalue))
    self.assertTrue(jnp.array_equal(scale, expected.scale))
    if expected.zero_point is not None:
      self.assertEqual(zp[0].dtype, expected.zero_point.dtype)
      self.assertTrue(jnp.array_equal(zp[0], expected.zero_point))

  @parameterized.product(
      dtype=(jnp.bfloat16, jnp.float32),
      channelwise_axes=((0,), (1,)),
  )
  def test_quantize_tile_mxfp8(self, dtype, channelwise_axes):
    """Tests qarray.quantize_tile with mxfp8 inside pl.pallas_call."""
    x = jax.random.normal(jax.random.key(0), _TILE_SHAPE, dtype)
    tiled_axes = {1: 32} if channelwise_axes == (0,) else {0: 32}
    expected = qarray.quantize_api(
        x, "mxfp8", channelwise_axes=channelwise_axes, tiled_axes=tiled_axes
    )

    def quantize_kernel(x_ref, q_ref, s_ref):
      qa = qarray.quantize_tile(
          x_ref[...], "mxfp8", channelwise_axes=channelwise_axes
      )
      q_ref[...] = qa.qvalue
      s_ref[...] = qa.scale

    qvalue, scale = pl.pallas_call(
        quantize_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(expected.qvalue.shape, expected.qvalue.dtype),
            jax.ShapeDtypeStruct(expected.scale.shape, expected.scale.dtype),
        ),
        **self.pallas_call_kwargs,
    )(x)

    self.assertEqual(qvalue.dtype, expected.qvalue.dtype)
    self.assertEqual(scale.dtype, expected.scale.dtype)
    self.assertTrue(jnp.array_equal(qvalue, expected.qvalue))
    self.assertTrue(jnp.array_equal(scale, expected.scale))

  @parameterized.parameters(jnp.bfloat16, jnp.float32)
  def test_quantize_tile_emit_pipeline(self, dtype):
    """Tests qarray.quantize_tile inside pl.kernel + pltpu.emit_pipeline."""
    if jax.devices()[0].platform != "tpu":
      self.skipTest("TPU required for pltpu.emit_pipeline")

    x = jax.random.normal(jax.random.key(0), (256, 256), dtype)
    expected = qarray.quantize_api(x, jnp.int8, channelwise_axes=(0,))

    def pipeline_step(x_ref, q_ref, s_ref):
      qa = qarray.quantize_tile(x_ref[...], jnp.int8, channelwise_axes=(0,))
      q_ref[...] = qa.qvalue
      s_ref[...] = qa.scale

    mesh = pltpu.TensorCoreMesh(axis_name="core", num_cores=1)

    @pl.kernel(
        out_type=(
            jax.ShapeDtypeStruct(expected.qvalue.shape, expected.qvalue.dtype),
            jax.ShapeDtypeStruct(expected.scale.shape, expected.scale.dtype),
        ),
        mesh=mesh,
    )
    def run_kernel(x_ref, q_ref, s_ref):
      pltpu.emit_pipeline(
          pipeline_step,
          grid=(2,),
          in_specs=[pl.BlockSpec(_TILE_SHAPE, lambda i: (i, 0))],
          out_specs=[
              pl.BlockSpec(_TILE_SHAPE, lambda i: (i, 0)),
              pl.BlockSpec((_TILE_SHAPE[0], 1), lambda i: (i, 0)),
          ],
      )(x_ref, q_ref, s_ref)

    qvalue, scale = jax.jit(run_kernel)(x)

    self.assertEqual(qvalue.dtype, expected.qvalue.dtype)
    self.assertEqual(scale.dtype, expected.scale.dtype)
    self.assertTrue(jnp.array_equal(qvalue, expected.qvalue))
    self.assertTrue(jnp.array_equal(scale, expected.scale))

  @parameterized.parameters(((0,),), ((),))
  def test_quantize_tile_bf16_rms(self, channelwise_axes):
    """Tests qarray.quantize_tile with bf16 rms calibration in pl.pallas_call."""
    x = jax.random.normal(jax.random.key(0), _TILE_SHAPE, jnp.bfloat16)
    expected = self._compile_ref(
        lambda arr: qarray.quantize_api(
            arr,
            jnp.int8,
            channelwise_axes=channelwise_axes,
            calibration_method="rms,7",
        )
    )(x)

    def quantize_kernel(x_ref, q_ref, s_ref):
      qa = qarray.quantize_tile(
          x_ref[...],
          jnp.int8,
          channelwise_axes=channelwise_axes,
          calibration_method="rms,7",
      )
      q_ref[...] = qa.qvalue
      s_ref[...] = qa.scale

    qvalue, scale = pl.pallas_call(
        quantize_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(expected.qvalue.shape, expected.qvalue.dtype),
            jax.ShapeDtypeStruct(expected.scale.shape, expected.scale.dtype),
        ),
        **self.pallas_call_kwargs,
    )(x)

    self.assertEqual(qvalue.dtype, expected.qvalue.dtype)
    self.assertEqual(scale.dtype, expected.scale.dtype)
    self.assertTrue(jnp.array_equal(qvalue, expected.qvalue))
    self.assertTrue(jnp.array_equal(scale, expected.scale))

  def _make_fused_inputs_and_expected(
      self, *, dtype, channelwise_axes, scale_dtype
  ):
    """Builds inputs and host loop_dot_general reference for fused tests."""
    x = jax.random.normal(jax.random.key(0), (_FUSED_M, _FUSED_K), dtype)
    w = qarray.quantize_api(
        jax.random.normal(jax.random.key(1), (_FUSED_K, _FUSED_N), jnp.float32),
        jnp.int8,
        channelwise_axes=(1,),
    )
    tiled_axes = (
        {1: _FUSED_BK}
        if channelwise_axes == (0,)
        else {0: _FUSED_BM, 1: _FUSED_BK}
    )

    def _ref(x_arr, w_arr):
      x_q_ref = qarray.quantize_api(
          x_arr,
          jnp.int8,
          channelwise_axes=channelwise_axes,
          tiled_axes=tiled_axes,
          scale_dtype=scale_dtype,
      )
      return dot_general.loop_dot_general(
          x_q_ref,
          w_arr,
          (((1,), (0,)), ((), ())),
          preferred_element_type=jnp.float32,
      )

    expected = self._compile_ref(_ref)(x, w)
    return x, w, expected

  def _fused_matmul_step(
      self,
      k_idx,
      x_ref,
      wq_ref,
      ws_ref,
      z_ref,
      acc_ref,
      *,
      channelwise_axes,
      scale_dtype,
  ):
    @pl.when(k_idx == 0)
    def _init():
      acc_ref[...] = jnp.zeros_like(acc_ref)

    x_q = qarray.quantize_tile(
        x_ref[...],
        jnp.int8,
        channelwise_axes=channelwise_axes,
        scale_dtype=scale_dtype,
    )
    w_q = qarray.QArray(wq_ref[...], ws_ref[...], None, jnp.int8)
    acc_ref[...] += dot_general.loop_dot_general(
        x_q,
        w_q,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    @pl.when(k_idx == _FUSED_K // _FUSED_BK - 1)
    def _write():
      z_ref[...] = acc_ref[...].astype(z_ref.dtype)

  def _assert_fused_kernel(self, jitted_kernel, x, w_qvalue, w_scale):
    """Asserts the compiled HLO fuses quantize and matmul into one kernel."""
    if jax.devices()[0].platform != "tpu":
      return
    hlo_text = jitted_kernel.lower(x, w_qvalue, w_scale).compile().as_text()
    # 1. Exactly one Mosaic custom-call kernel.
    custom_calls = re.findall(
        r'custom-call\(([^)]*)\),[^\n]*?custom_call_target="tpu_custom_call"',
        hlo_text,
    )
    self.assertLen(custom_calls, 1, hlo_text)
    # 2. No standalone XLA dot / convolution / reduce ops outside the kernel.
    for line in hlo_text.splitlines():
      prefix = line.split(", metadata=", 1)[0].split(", backend_config=", 1)[0]
      self.assertIsNone(
          re.match(
              r"^\s*(?:ROOT\s+)?%[\w.-]+\s*=\s*.+?\s+"
              r"(?:dot|convolution|reduce(?:-[\w-]+)?)\(",
              prefix,
          ),
          f"Unexpected standalone XLA op: {line}",
      )
    # 3. Kernel receives unquantized bf16/f32 x and no s8[768,1024] exists.
    x_operand = custom_calls[0].split(",", 1)[0].strip()
    hlo_dtype = "bf16" if x.dtype == jnp.bfloat16 else "f32"
    m, k = x.shape
    self.assertRegex(
        hlo_text,
        rf"(?m)^\s*{re.escape(x_operand)}\s*=\s*{hlo_dtype}\[{m},\s*{k}\]",
    )
    self.assertNotRegex(hlo_text, rf"\bs8\[{m},\s*{k}\]")

  @parameterized.product(
      dtype=(jnp.bfloat16, jnp.float32),
      channelwise_axes=((0,), ()),
      scale_dtype=(None, jnp.float32),
  )
  def test_fused_quantize_tile_dot_general(
      self, dtype, channelwise_axes, scale_dtype
  ):
    """Tests fused qpl.quantize_tile + qpl.dot_general in pl.pallas_call.

    Runs qpl.quantize_tile + qpl.dot_general (= qarray.quantize_tile +
    dot_general.loop_dot_general) fused inside one Pallas kernel.
    """
    x, w, expected = self._make_fused_inputs_and_expected(
        dtype=dtype, channelwise_axes=channelwise_axes, scale_dtype=scale_dtype
    )

    def fused_kernel(x_ref, wq_ref, ws_ref, z_ref, acc_ref):
      self._fused_matmul_step(
          pl.program_id(2),
          x_ref,
          wq_ref,
          ws_ref,
          z_ref,
          acc_ref,
          channelwise_axes=channelwise_axes,
          scale_dtype=scale_dtype,
      )

    jitted_kernel = jax.jit(
        pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct((_FUSED_M, _FUSED_N), jnp.float32),
            grid=(
                _FUSED_M // _FUSED_BM,
                _FUSED_N // _FUSED_BN,
                _FUSED_K // _FUSED_BK,
            ),
            in_specs=[
                pl.BlockSpec(
                    (_FUSED_BM, _FUSED_BK), lambda i, j, k_idx: (i, k_idx)
                ),
                pl.BlockSpec(
                    (_FUSED_BK, _FUSED_BN), lambda i, j, k_idx: (k_idx, j)
                ),
                pl.BlockSpec((1, _FUSED_BN), lambda i, j, k_idx: (0, j)),
            ],
            out_specs=pl.BlockSpec(
                (_FUSED_BM, _FUSED_BN), lambda i, j, k_idx: (i, j)
            ),
            scratch_shapes=[pltpu.VMEM((_FUSED_BM, _FUSED_BN), jnp.float32)],
            **self.pallas_call_kwargs,
        )
    )
    self._assert_fused_kernel(jitted_kernel, x, w.qvalue, w.scale)
    actual = jitted_kernel(x, w.qvalue, w.scale)

    if dtype == jnp.bfloat16 and scale_dtype is None:
      # int32 * bf16 scale casts the partial dot product to bf16 inside
      # loop_dot_general, so VPU and XLA can differ by ~1 bf16 ULP.
      tol = 2e-3
    elif jax.devices()[0].platform != "tpu":
      # Observed mean relative 2.4e-8 (max 1.5e-5 = 1 f32 ULP): tiled K
      # accumulation in the kernel vs one reference matmul on XLA:CPU.
      tol = 1e-6
    else:
      tol = None

    if tol is not None:
      mean_rel = jnp.abs(actual - expected).mean() / jnp.abs(expected).mean()
      self.assertLess(
          mean_rel,
          tol,
          f"mean_rel={mean_rel},"
          f" max_diff={jnp.max(jnp.abs(actual - expected))}",
      )
    else:
      self.assertTrue(
          jnp.array_equal(actual, expected),
          f"max diff={jnp.max(jnp.abs(actual - expected))}",
      )

  def test_fused_quantize_tile_dot_general_emit_pipeline(self):
    """Tests fused qpl.quantize_tile + qpl.dot_general in emit_pipeline.

    Runs qpl.quantize_tile + qpl.dot_general (= qarray.quantize_tile +
    dot_general.loop_dot_general) fused in one Pallas kernel via
    pl.kernel + pltpu.emit_pipeline.
    """
    if jax.devices()[0].platform != "tpu":
      self.skipTest("TPU required for pltpu.emit_pipeline")

    x, w, expected = self._make_fused_inputs_and_expected(
        dtype=jnp.bfloat16,
        channelwise_axes=(0,),
        scale_dtype=jnp.float32,
    )
    mesh = pltpu.TensorCoreMesh(axis_name="core", num_cores=1)

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((_FUSED_M, _FUSED_N), jnp.float32),
        scratch_types=(pltpu.VMEM((_FUSED_BM, _FUSED_BN), jnp.float32),),
        mesh=mesh,
    )
    def run_kernel(x_ref, wq_ref, ws_ref, z_ref, acc_ref):
      def pipeline_step(step, x_vmem, wq_vmem, ws_vmem, z_vmem):
        self._fused_matmul_step(
            step.index[2],
            x_vmem,
            wq_vmem,
            ws_vmem,
            z_vmem,
            acc_ref,
            channelwise_axes=(0,),
            scale_dtype=jnp.float32,
        )

      pltpu.emit_pipeline(
          pipeline_step,
          grid=(
              _FUSED_M // _FUSED_BM,
              _FUSED_N // _FUSED_BN,
              _FUSED_K // _FUSED_BK,
          ),
          in_specs=[
              pl.BlockSpec(
                  (_FUSED_BM, _FUSED_BK), lambda i, j, k_idx: (i, k_idx)
              ),
              pl.BlockSpec(
                  (_FUSED_BK, _FUSED_BN), lambda i, j, k_idx: (k_idx, j)
              ),
              pl.BlockSpec((1, _FUSED_BN), lambda i, j, k_idx: (0, j)),
          ],
          out_specs=pl.BlockSpec(
              (_FUSED_BM, _FUSED_BN), lambda i, j, k_idx: (i, j)
          ),
          core_axis_name="core",
          dimension_semantics=(
              pltpu.PARALLEL,
              pltpu.ARBITRARY,
              pltpu.ARBITRARY,
          ),
          _explicit_indices=True,
      )(x_ref, wq_ref, ws_ref, z_ref)

    jitted_kernel = jax.jit(run_kernel)
    self._assert_fused_kernel(jitted_kernel, x, w.qvalue, w.scale)
    actual = jitted_kernel(x, w.qvalue, w.scale)
    self.assertTrue(
        jnp.array_equal(actual, expected),
        f"max diff={jnp.max(jnp.abs(actual - expected))}",
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
