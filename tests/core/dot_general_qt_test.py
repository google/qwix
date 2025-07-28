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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix.core import dot_general
from qwix.core import dot_general_qt
from qwix.core import qarray


def _fake_quant(
    array: jax.Array,
    how: qarray.HowToQuantize,
) -> jax.Array:
  calibration = qarray.calibrate(array, how)
  scale, zero_point = qarray.compute_scale_zero_point(calibration, how.qtype)
  q_array = qarray.quantize_with_scale_zero_point(array, how, scale, zero_point)
  dq_array = qarray.dequantize(q_array)
  ste_array = qarray.clip_to_calibration(array, calibration, how.tiled_axes)
  return ste_array + jax.lax.stop_gradient(dq_array - ste_array)


def dot_general_fq(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: dot_general_qt.DotGeneralQtConfig,
):
  """dot_general implemented with fake quantization."""
  lhs_how = dot_general.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=True,
      qtype=config.lhs_qtype,
      tile_size=config.tile_size,
      calibration_method=config.lhs_calibration_method,
      batch_axes=config.lhs_batch_axes,
  )
  rhs_how = dot_general.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=False,
      qtype=config.rhs_qtype,
      tile_size=config.tile_size,
      calibration_method=config.rhs_calibration_method,
      batch_axes=config.rhs_batch_axes,
  )
  lhs_fq = _fake_quant(lhs, lhs_how)
  rhs_fq = _fake_quant(rhs, rhs_how)
  return jax.lax.dot_general(
      lhs_fq,
      rhs_fq,
      dimension_numbers,
  )


class DotGeneralQtTest(parameterized.TestCase):
  """Test class for dot_general_qt."""

  @parameterized.named_parameters(
      dict(
          testcase_name='int8',
          lhs_qtype='int8',
          rhs_qtype='int8',
          # If we set precision=HIGHEST in fq implementation above, then
          # expected_mae_fq_out will become 1e-7 but expected_mae_fq_grads will
          # be non-zero.
          expected_mae_fq_out=0.02,
          expected_mae_fq_grads=0.0,
          expected_mae_fp_out=0.06,
          expected_mae_fp_grads=0.02,
      ),
      dict(
          testcase_name='int4',
          lhs_qtype='int4',
          rhs_qtype='int4',
          expected_mae_fq_out=0.04,
          expected_mae_fq_grads=0.0,
          expected_mae_fp_out=0.5,
          expected_mae_fp_grads=0.5,
      ),
      dict(
          testcase_name='fp8_bwd',
          lhs_shape=(8, 8),
          rhs_shape=(8, 8),
          lhs_qtype='float8_e4m3',
          rhs_qtype='float8_e4m3',
          bwd_qtype='float8_e4m3',
          expected_mae_fq_out=0.0002,
          expected_mae_fq_grads=0.02,
          expected_mae_fp_out=0.04,
          expected_mae_fp_grads=0.04,
      ),
      dict(
          testcase_name='fp8_bwd_sc128',
          lhs_shape=(512, 512),
          rhs_shape=(512, 512),
          lhs_qtype='float8_e4m3',
          rhs_qtype='float8_e4m3',
          bwd_qtype='float8_e4m3',
          tile_size=128,
          expected_mae_fq_out=0.005,
          expected_mae_fq_grads=0.2,
          expected_mae_fp_out=0.4,
          expected_mae_fp_grads=0.4,
      ),
      dict(
          testcase_name='fp8_bwd_param_grad_tiling',
          lhs_shape=(64, 64),
          rhs_shape=(64, 64),
          lhs_qtype='float8_e4m3',
          rhs_qtype='float8_e4m3',
          bwd_qtype='float8_e4m3',
          bwd_drhs_tile_size=32,
          expected_mae_fq_out=0.0033,
          expected_mae_fq_grads=0.14,
          expected_mae_fp_out=0.2,
          expected_mae_fp_grads=0.2,
      ),
  )
  def test_grad_against_fq(
      self,
      *,
      lhs_shape=(2, 4),
      rhs_shape=(4, 2),
      lhs_qtype,
      rhs_qtype,
      bwd_qtype=None,
      tile_size=None,
      bwd_drhs_tile_size=None,
      expected_mae_fq_out,
      expected_mae_fq_grads,
      expected_mae_fp_out,
      expected_mae_fp_grads,
  ):
    lhs = jax.random.normal(jax.random.key(0), lhs_shape, jnp.float32)
    rhs = jax.random.normal(jax.random.key(1), rhs_shape, jnp.float32)
    dimension_numbers = (((1,), (0,)), ((), ()))
    config = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype=lhs_qtype,
        rhs_qtype=rhs_qtype,
        bwd_qtype=bwd_qtype,
        tile_size=tile_size,
        bwd_drhs_tile_size=bwd_drhs_tile_size,
    )

    def loss_fn_fq(lhs_arr, rhs_arr):
      return jnp.sum(
          dot_general_fq(lhs_arr, rhs_arr, dimension_numbers, config)
      )

    def loss_fn_qt(lhs_arr, rhs_arr):
      return jnp.sum(
          dot_general_qt.dot_general_qt(
              lhs_arr, rhs_arr, dimension_numbers, config
          )
      )

    def loss_fn_fp(lhs_arr, rhs_arr):
      return jnp.sum(jax.lax.dot_general(lhs_arr, rhs_arr, dimension_numbers))

    fq_out, fq_grads = jax.value_and_grad(loss_fn_fq, argnums=(0, 1))(lhs, rhs)
    qt_out, qt_grads = jax.value_and_grad(loss_fn_qt, argnums=(0, 1))(lhs, rhs)
    fp_out, fp_grads = jax.value_and_grad(loss_fn_fp, argnums=(0, 1))(lhs, rhs)

    print('-' * 20, self._testMethodName, '-' * 20)
    print('fq result:', fq_out, fq_grads)
    print('qt result:', qt_out, qt_grads)
    print('fp result:', fp_out, fp_grads)

    mae = lambda x, y: jnp.mean(jnp.abs((x - y) / y))
    print(
        'qt vs fq:',
        mae(qt_out, fq_out),
        mae(qt_grads[0], fq_grads[0]),
        mae(qt_grads[1], fq_grads[1]),
    )
    print(
        'qt vs fp:',
        mae(qt_out, fp_out),
        mae(qt_grads[0], fp_grads[0]),
        mae(qt_grads[1], fp_grads[1]),
    )

    # fq and QT results should be close.
    self.assertLessEqual(mae(qt_out, fq_out), expected_mae_fq_out)
    self.assertLessEqual(mae(qt_grads[0], fq_grads[0]), expected_mae_fq_grads)
    self.assertLessEqual(mae(qt_grads[1], fq_grads[1]), expected_mae_fq_grads)

    # QT and FP results should be close in a larger tolerance.
    self.assertLessEqual(mae(qt_out, fp_out), expected_mae_fp_out)
    self.assertLessEqual(mae(qt_grads[0], fp_grads[0]), expected_mae_fp_grads)
    self.assertLessEqual(mae(qt_grads[1], fp_grads[1]), expected_mae_fp_grads)


if __name__ == '__main__':
  absltest.main()
