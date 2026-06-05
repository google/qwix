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
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import dot_general
from qwix._src.core import dot_general_qt
from qwix._src.core import qarray

jax.config.update('jax_threefry_partitionable', False)


class TwoDimScalesTest(parameterized.TestCase):
  """Tests for two dimensional scales."""

  def test_quantize_2d_scales(self):
    """Tests quantization with 2d scales."""
    shape = (128, 128)
    array = jax.random.normal(jax.random.key(0), shape, dtype=jnp.bfloat16)

    # 2D tiles of shape (8, 8)
    tiled_axes = {0: 8, 1: 8}

    how = qarray.HowToQuantize(
        qtype=jnp.int8,
        tiled_axes=tiled_axes,
        calibration_method='absmax',
    )

    q_array = qarray.quantize(array, how)

    # Verify shapes
    self.assertEqual(q_array.qvalue.shape, shape)

    # Scale shape should be (128//8, 128//8) -> (16, 16)
    expected_scale_shape = (16, 16)
    self.assertEqual(q_array.scale.shape, expected_scale_shape)

    # Dequantize and check shape
    dq_array = qarray.dequantize(q_array)
    self.assertEqual(dq_array.shape, shape)

  def test_dot_general_2d_scales(self):
    """Tests dot_general implementations with 2d scales."""
    lhs_shape = (128, 128)
    rhs_shape = (128, 128)

    lhs_array = jax.random.normal(
        jax.random.key(0), lhs_shape, dtype=jnp.float32
    )
    rhs_array = jax.random.normal(
        jax.random.key(1), rhs_shape, dtype=jnp.float32
    )

    # 2D tiles of shape (8, 8)
    lhs_how = qarray.HowToQuantize(
        qtype=jnp.int8,
        tiled_axes={0: 8, 1: 8},
        calibration_method='absmax',
    )
    rhs_how = qarray.HowToQuantize(
        qtype=jnp.int8,
        tiled_axes={0: 8, 1: 8},
        calibration_method='absmax',
    )

    lhs_q = qarray.quantize(lhs_array, lhs_how)
    rhs_q = qarray.quantize(rhs_array, rhs_how)

    dn = (((1,), (0,)), ((), ()))

    res_true = jnp.dot(qarray.dequantize(lhs_q), qarray.dequantize(rhs_q))
    res_fast = dot_general._fast_dot_general(lhs_q, rhs_q, dn)
    res_slow = dot_general._slow_dot_general(lhs_q, rhs_q, dn)
    res_loop = dot_general.loop_dot_general(lhs_q, rhs_q, dn)

    self.assertTrue(jnp.allclose(res_fast, res_true, rtol=1e-4, atol=1e-4))
    self.assertTrue(jnp.allclose(res_slow, res_true, rtol=1e-4, atol=1e-4))
    self.assertTrue(jnp.allclose(res_loop, res_true, rtol=1e-4, atol=1e-4))

  def test_dot_general_qt_2d_scales(self):
    """Tests dot_general_qt with 2d scales."""
    lhs_shape = (128, 128)
    rhs_shape = (128, 128)

    lhs_array = jax.random.normal(
        jax.random.key(0), lhs_shape, dtype=jnp.float32
    )
    rhs_array = jax.random.normal(
        jax.random.key(1), rhs_shape, dtype=jnp.float32
    )

    # We want to use 2D tiles of shape (8, 8)
    tile_size = {0: 8, 1: 8}

    config = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype=jnp.int8,
        rhs_qtype=jnp.int8,
        tile_size=tile_size,
    )

    dn = (((1,), (0,)), ((), ()))

    res_qt = dot_general_qt.dot_general_qt(lhs_array, rhs_array, dn, config)
    lhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dn,
        ndims=(2, 2),
        for_lhs=True,
        tile_size=tile_size,
        qtype=config.lhs_qtype,
    )
    rhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dn,
        ndims=(2, 2),
        for_lhs=False,
        tile_size=tile_size,
        qtype=config.rhs_qtype,
    )
    lhs_q = qarray.quantize(lhs_array, lhs_how)
    rhs_q = qarray.quantize(rhs_array, rhs_how)
    res_true = dot_general.dot_general(lhs_q, rhs_q, dn)

    self.assertTrue(jnp.allclose(res_qt, res_true, rtol=1e-4, atol=1e-4))

  @parameterized.named_parameters(
      dict(
          testcase_name='dlhs_float_drhs_float',
          dlhs_grad_qtype=None,
          drhs_grad_qtype=None,
      ),
      dict(
          testcase_name='dlhs_int8_drhs_float',
          dlhs_grad_qtype=jnp.int8,
          drhs_grad_qtype=None,
      ),
      dict(
          testcase_name='dlhs_float_drhs_int8',
          dlhs_grad_qtype=None,
          drhs_grad_qtype=jnp.int8,
      ),
      dict(
          testcase_name='dlhs_int8_drhs_int8',
          dlhs_grad_qtype=jnp.int8,
          drhs_grad_qtype=jnp.int8,
      ),
  )
  def test_dot_general_qt_2d_scales_backward(
      self, dlhs_grad_qtype, drhs_grad_qtype
  ):
    """Tests backwards for dot_general_qt with 2d scales."""
    lhs_shape = (128, 128)
    rhs_shape = (128, 128)

    lhs_array = jax.random.normal(
        jax.random.key(0), lhs_shape, dtype=jnp.float32
    )
    rhs_array = jax.random.normal(
        jax.random.key(1), rhs_shape, dtype=jnp.float32
    )

    tile_size = {0: 8, 1: 8}

    config = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype=jnp.int8,
        rhs_qtype=jnp.int8,
        tile_size=tile_size,
        dlhs_grad_qtype=dlhs_grad_qtype,
        dlhs_tile_size=tile_size if dlhs_grad_qtype is not None else None,
        drhs_grad_qtype=drhs_grad_qtype,
        drhs_tile_size=tile_size if drhs_grad_qtype is not None else None,
    )

    dn = (((1,), (0,)), ((), ()))

    def loss_fn_qt(x, y):
      return dot_general_qt.dot_general_qt(x, y, dn, config).mean()

    out_val, qt_grads = jax.value_and_grad(loss_fn_qt, argnums=(0, 1))(
        lhs_array, rhs_array
    )
    del out_val, qt_grads


if __name__ == '__main__':
  absltest.main()
