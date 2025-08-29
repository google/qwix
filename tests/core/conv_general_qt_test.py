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

from collections.abc import Sequence
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import conv_general
from qwix._src.core import conv_general_qt
from qwix._src.core import qarray


def _fake_quant(
    array: jax.Array,
    how: qarray.HowToQuantize,
) -> jax.Array:
  """Generic fake quantization function using a Straight-Through Estimator."""
  calibration = qarray.calibrate(array, how)
  scale, zero_point = qarray.compute_scale_zero_point(calibration, how.qtype)
  q_array = qarray.quantize_with_scale_zero_point(
      array, how.qtype, scale, zero_point
  )
  dq_array = qarray.dequantize(q_array)
  # The STE passes the original full-precision value for the gradient.
  ste_array = qarray.clip_to_calibration(array, calibration, how.tiled_axes)
  return ste_array + jax.lax.stop_gradient(dq_array - ste_array)


def conv_general_fq(
    lhs: jax.Array,
    rhs: jax.Array,
    config: conv_general_qt.ConvGeneralQtConfig,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers | None,
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
):
  """conv_general_dilated implemented with fake quantization."""
  dnums = jax.lax.conv_dimension_numbers(
      lhs.shape, rhs.shape, dimension_numbers
  )

  lhs_how = conv_general.get_how_to_quantize(
      dimension_numbers=dnums,
      for_lhs=True,
      qtype=config.fwd_qtype,
      calibration_method=config.fwd_calibration_method,
  )
  rhs_how = conv_general.get_how_to_quantize(
      dimension_numbers=dnums,
      for_lhs=False,
      qtype=config.fwd_qtype,
      calibration_method=config.fwd_calibration_method,
  )
  lhs_fq = _fake_quant(lhs, lhs_how)
  rhs_fq = _fake_quant(rhs, rhs_how)
  return jax.lax.conv_general_dilated(
      lhs_fq,
      rhs_fq,
      window_strides,
      padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      dimension_numbers=dnums,
  )


class ConvGeneralQtTest(parameterized.TestCase):
  """Test class for conv_general_qt."""

  @parameterized.named_parameters(
      dict(
          testcase_name='int8_nhwc',
          data_format='NHWC',
          fwd_qtype='int8',
          expected_mae_fq_out=0.01,
          expected_mae_dlhs_fq_grads=0.015,
          expected_mae_drhs_fq_grads=0.05,
          expected_mae_fp_out=0.08,
          expected_mae_dlhs_fp_grads=0.06,
          expected_mae_drhs_fp_grads=0.06,
      ),
      dict(
          testcase_name='int4_nhwc',
          data_format='NHWC',
          fwd_qtype='int4',
          expected_mae_fq_out=0.01,
          expected_mae_dlhs_fq_grads=0.1,
          expected_mae_drhs_fq_grads=0.25,
          expected_mae_fp_out=0.5,
          expected_mae_dlhs_fp_grads=0.1,
          expected_mae_drhs_fp_grads=0.5,
      ),
      dict(
          testcase_name='fp8_bwd_nhwc',
          data_format='NHWC',
          fwd_qtype='float8_e4m3',
          bwd_qtype='float8_e4m3',
          expected_mae_fq_out=0.01,
          expected_mae_dlhs_fq_grads=0.12,
          expected_mae_drhs_fq_grads=0.35,
          expected_mae_fp_out=0.05,
          expected_mae_dlhs_fp_grads=0.12,
          expected_mae_drhs_fp_grads=0.35,
      ),
      dict(
          testcase_name='fp8_bwd_nhwc_dilated',
          data_format='NHWC',
          fwd_qtype='float8_e4m3',
          bwd_qtype='float8_e4m3',
          padding=((0, 0), (0, 0)),
          lhs_dilation=(2, 2),
          rhs_dilation=(2, 2),
          expected_mae_fq_out=0.01,
          expected_mae_dlhs_fq_grads=0.15,
          expected_mae_drhs_fq_grads=0.35,
          expected_mae_fp_out=0.05,
          expected_mae_dlhs_fp_grads=0.15,
          expected_mae_drhs_fp_grads=0.35,
      ),
      dict(
          testcase_name='int8_nchw',
          data_format='NCHW',
          fwd_qtype='int8',
          expected_mae_fq_out=0.01,
          expected_mae_dlhs_fq_grads=0.015,
          expected_mae_drhs_fq_grads=0.05,
          expected_mae_fp_out=0.08,
          expected_mae_dlhs_fp_grads=0.06,
          expected_mae_drhs_fp_grads=0.06,
      ),
      dict(
          testcase_name='int4_nchw',
          data_format='NCHW',
          fwd_qtype='int4',
          expected_mae_fq_out=0.01,
          expected_mae_dlhs_fq_grads=0.16,
          expected_mae_drhs_fq_grads=0.95,
          expected_mae_fp_out=0.5,
          expected_mae_dlhs_fp_grads=0.1,
          expected_mae_drhs_fp_grads=0.5,
      ),
      dict(
          testcase_name='fp8_bwd_nchw',
          data_format='NCHW',
          fwd_qtype='float8_e4m3',
          bwd_qtype='float8_e4m3',
          expected_mae_fq_out=0.01,
          expected_mae_dlhs_fq_grads=0.12,
          expected_mae_drhs_fq_grads=0.35,
          expected_mae_fp_out=0.05,
          expected_mae_dlhs_fp_grads=0.12,
          expected_mae_drhs_fp_grads=0.35,
      ),
  )
  def test_grad_against_fq(
      self,
      *,
      data_format,
      fwd_qtype,
      bwd_qtype=None,
      padding='SAME',
      lhs_dilation=None,
      rhs_dilation=None,
      expected_mae_fq_out,
      expected_mae_dlhs_fq_grads,
      expected_mae_drhs_fq_grads,
      expected_mae_fp_out,
      expected_mae_dlhs_fp_grads,
      expected_mae_drhs_fp_grads,
  ):
    window_strides = (1, 1)
    if data_format == 'NCHW':
      lhs_shape = (4, 8, 16, 16)  # N, C_in, H, W
      rhs_shape = (32, 8, 3, 3)  # C_out, C_in, H_k, W_k
      dimension_numbers = ('NCHW', 'OIHW', 'NCHW')
    elif data_format == 'NHWC':
      lhs_shape = (4, 16, 16, 8)  # N, H, W, C_in
      rhs_shape = (3, 3, 8, 32)  # H_k, W_k, C_in, C_out
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    else:
      raise ValueError(f'Unknown data_format: {data_format}')

    lhs = jax.random.normal(jax.random.key(0), lhs_shape, jnp.float32)
    rhs = jax.random.normal(jax.random.key(1), rhs_shape, jnp.float32)
    config = conv_general_qt.ConvGeneralQtConfig(
        fwd_qtype=fwd_qtype,
        bwd_qtype=bwd_qtype,
        bwd_use_original_residuals=True,
    )

    def loss_fn_fq(lhs_arr, rhs_arr):
      return jnp.sum(
          conv_general_fq(
              lhs_arr,
              rhs_arr,
              config,
              window_strides,
              padding,
              dimension_numbers,
              lhs_dilation=lhs_dilation,
              rhs_dilation=rhs_dilation,
          )
      )

    def loss_fn_qt(lhs_arr, rhs_arr):
      return jnp.sum(
          conv_general_qt.conv_general_qt(
              lhs=lhs_arr,
              rhs=rhs_arr,
              config=config,
              window_strides=window_strides,
              padding=padding,
              dimension_numbers=dimension_numbers,
              lhs_dilation=lhs_dilation,
              rhs_dilation=rhs_dilation,
          )
      )

    def loss_fn_fp(lhs_arr, rhs_arr):
      return jnp.sum(
          jax.lax.conv_general_dilated(
              lhs_arr,
              rhs_arr,
              window_strides,
              padding,
              dimension_numbers=dimension_numbers,
              lhs_dilation=lhs_dilation,
              rhs_dilation=rhs_dilation,
          )
      )

    fq_out, fq_grads = jax.value_and_grad(loss_fn_fq, argnums=(0, 1))(lhs, rhs)
    qt_out, qt_grads = jax.value_and_grad(loss_fn_qt, argnums=(0, 1))(lhs, rhs)
    fp_out, fp_grads = jax.value_and_grad(loss_fn_fp, argnums=(0, 1))(lhs, rhs)

    mae = lambda x, y: jnp.mean(jnp.abs((x - y) / y))

    self.assertLessEqual(mae(qt_out, fq_out), expected_mae_fq_out)
    self.assertLessEqual(
        mae(qt_grads[0], fq_grads[0]), expected_mae_dlhs_fq_grads
    )
    self.assertLessEqual(
        mae(qt_grads[1], fq_grads[1]), expected_mae_drhs_fq_grads
    )

    self.assertLessEqual(mae(qt_out, fp_out), expected_mae_fp_out)
    self.assertLessEqual(
        mae(qt_grads[0], fp_grads[0]), expected_mae_dlhs_fp_grads
    )
    self.assertLessEqual(
        mae(qt_grads[1], fp_grads[1]), expected_mae_drhs_fp_grads
    )


if __name__ == '__main__':
  absltest.main()
