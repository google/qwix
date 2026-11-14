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


def mae(lhs: jax.Array, rhs: jax.Array) -> float:
  assert lhs.dtype == rhs.dtype and lhs.shape == rhs.shape
  return jnp.abs(lhs - rhs).mean() / jnp.abs(lhs).mean()


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
    feature_group_count: int = 1,
    batch_group_count: int = 1,
):
  """conv_general_dilated implemented with fake quantization."""
  dnums = jax.lax.conv_dimension_numbers(
      lhs.shape, rhs.shape, dimension_numbers
  )

  lhs_how = conv_general.get_how_to_quantize(
      dimension_numbers=dnums,
      for_lhs=True,
      qtype=config.lhs_qtype,
      calibration_method=config.lhs_calibration_method,
  )
  rhs_how = conv_general.get_how_to_quantize(
      dimension_numbers=dnums,
      for_lhs=False,
      qtype=config.rhs_qtype,
      calibration_method=config.rhs_calibration_method,
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
      feature_group_count=feature_group_count,
      batch_group_count=batch_group_count,
  )


class ConvGeneralQtTest(parameterized.TestCase):
  """Test class for conv_general_qt."""

  @parameterized.named_parameters(
      dict(
          testcase_name='int8_nhwc',
          data_format='NHWC',
          fwd_qtype='int8',
          expected_maes=[0.00055, 0, 0, 0.1, 0.005, 0.008],
      ),
      dict(
          testcase_name='int4_nhwc',
          data_format='NHWC',
          fwd_qtype='int4',
          expected_maes=[0.003, 0, 0, 0.12, 0.06, 0.12],
      ),
      dict(
          testcase_name='fp8_bwd_nhwc',
          data_format='NHWC',
          fwd_qtype='float8_e4m3',
          bwd_qtype='float8_e4m3',
          expected_maes=[0.004, 0.05, 0.06, 0.07, 0.05, 0.06],
      ),
      dict(
          testcase_name='fp8_bwd_nhwc_dilated',
          data_format='NHWC',
          fwd_qtype='float8_e4m3',
          bwd_qtype='float8_e4m3',
          padding=((0, 0), (0, 0)),
          lhs_dilation=(2, 2),
          rhs_dilation=(2, 2),
          expected_maes=[0.02, 0.04, 0.05, 0.007, 0.05, 0.05],
      ),
      dict(
          testcase_name='int8_bwd_nchw',
          data_format='NCHW',
          fwd_qtype='int8',
          bwd_qtype='int8',
          expected_maes=[0.002, 0.005, 0.003, 0.009, 0.008, 0.02],
      ),
      dict(
          testcase_name='int4_bwd_nchw',
          data_format='NCHW',
          fwd_qtype='int4',
          bwd_qtype='int4',
          expected_maes=[0.002, 0.05, 0.06, 0.28, 0.12, 0.13],
      ),
      dict(
          testcase_name='fp8_bwd_nchw',
          data_format='NCHW',
          fwd_qtype='float8_e4m3',
          bwd_qtype='float8_e4m3',
          expected_maes=[0.002, 0.04, 0.04, 0.02, 0.05, 0.06],
      ),
      dict(
          testcase_name='int8_nhwc_feature_groups',
          data_format='NHWC',
          fwd_qtype='int8',
          feature_group_count=4,
          # Index 0 is 6.1e-7 on CPU, 0.00282 on TPU (conv_general_fq uses
          # default bf16 MXU precision vs exact int32 accumulation in QT,
          # amplified by near-zero scalar cancellation in jnp.sum).
          expected_maes=[0.003, 0, 0, 0.025, 0.005, 0.008],
      ),
      dict(
          testcase_name='int8_bwd_nhwc_depthwise',
          data_format='NHWC',
          fwd_qtype='int8',
          bwd_qtype='int8',
          feature_group_count=8,
          # Index 0 is 2.7e-7 on CPU, 0.0148 on TPU for the same bf16 MXU vs
          # int32 scalar-sum reason above.
          expected_maes=[0.016, 0.003, 0.005, 0.02, 0.004, 0.009],
      ),
      dict(
          testcase_name='fp8_bwd_nchw_feature_groups',
          data_format='NCHW',
          fwd_qtype='float8_e4m3',
          bwd_qtype='float8_e4m3',
          feature_group_count=4,
          expected_maes=[0.0008, 0.025, 0.025, 0.03, 0.035, 0.035],
      ),
      dict(
          testcase_name='int8_nchw_batch_groups',
          data_format='NCHW',
          fwd_qtype='int8',
          batch_group_count=2,
          expected_maes=[1e-5, 0, 0, 0.015, 0.007, 0.006],
      ),
      dict(
          testcase_name='int8_bwd_nhwc_batch_groups',
          data_format='NHWC',
          fwd_qtype='int8',
          bwd_qtype='int8',
          batch_group_count=2,
          # Index 0 is 1.9e-7 on CPU, 0.00201 on TPU.
          expected_maes=[0.0021, 0.004, 0.005, 0.003, 0.007, 0.009],
      ),
      dict(
          testcase_name='fp8_bwd_nchw_batch_groups_full',
          data_format='NCHW',
          fwd_qtype='float8_e4m3',
          bwd_qtype='float8_e4m3',
          batch_group_count=4,
          expected_maes=[0.001, 0.025, 0.002, 0.04, 0.04, 0.025],
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
      feature_group_count=1,
      batch_group_count=1,
      expected_maes,
  ):
    if fwd_qtype == 'int4' and jax.devices()[0].platform != 'tpu':
      self.skipTest('int4 requires TPU.')
    if (
        rhs_dilation is not None
        and bwd_qtype == 'float8_e4m3'
        and jax.devices()[0].platform != 'tpu'
    ):
      # For this dilated fp8 backward conv, the mae(fp_out, qt_out) component
      # (4th) diverges to >1.0 on the XLA CPU backend while every other
      # component stays in range. A tolerance that large would make the
      # assertion meaningless, so skip only this specific parameterization
      # off TPU.
      self.skipTest('Dilated fp8 bwd conv diverges on non-TPU (XLA CPU).')
    window_strides = (1, 1)
    if data_format == 'NCHW':
      lhs_shape = (4, 8, 16, 16)  # N, C_in, H, W
      rhs_shape = (
          32,
          8 // feature_group_count,
          3,
          3,
      )  # C_out, C_in/G, H_k, W_k
      dimension_numbers = ('NCHW', 'OIHW', 'NCHW')
    elif data_format == 'NHWC':
      lhs_shape = (4, 16, 16, 8)  # N, H, W, C_in
      rhs_shape = (
          3,
          3,
          8 // feature_group_count,
          32,
      )  # H_k, W_k, C_in/G, C_out
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    else:
      raise ValueError(f'Unknown data_format: {data_format}')

    lhs = jax.random.normal(jax.random.key(0), lhs_shape, jnp.float32)
    rhs = jax.random.normal(jax.random.key(1), rhs_shape, jnp.float32)
    config = conv_general_qt.ConvGeneralQtConfig(
        lhs_qtype=fwd_qtype,
        rhs_qtype=fwd_qtype,
        dlhs_grad_qtype=bwd_qtype,
        drhs_grad_qtype=bwd_qtype,
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
              feature_group_count=feature_group_count,
              batch_group_count=batch_group_count,
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
              feature_group_count=feature_group_count,
              batch_group_count=batch_group_count,
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
              feature_group_count=feature_group_count,
              batch_group_count=batch_group_count,
          )
      )

    @jax.jit
    def f(lhs, rhs):
      fq_out, fq_grads = jax.value_and_grad(loss_fn_fq, argnums=(0, 1))(
          lhs, rhs
      )
      qt_out, qt_grads = jax.value_and_grad(loss_fn_qt, argnums=(0, 1))(
          lhs, rhs
      )
      fp_out, fp_grads = jax.value_and_grad(loss_fn_fp, argnums=(0, 1))(
          lhs, rhs
      )
      return (
          mae(fq_out, qt_out),
          mae(fq_grads[0], qt_grads[0]),
          mae(fq_grads[1], qt_grads[1]),
          mae(fp_out, qt_out),
          mae(fp_grads[0], qt_grads[0]),
          mae(fp_grads[1], qt_grads[1]),
      )

    maes = f(lhs, rhs)
    maes = [float(x) for x in maes]

    for x, y in zip(maes, expected_maes):
      self.assertLessEqual(x, y, msg=f'{maes} < {expected_maes}')

  @parameterized.parameters(None, 'int8')
  def test_batch_group_scale_folding_vjp(self, bwd_qtype):
    lhs = jax.random.normal(jax.random.key(0), (6, 4, 8, 8), jnp.float32)
    lhs = lhs * (2.0 ** jnp.arange(6, dtype=jnp.float32)).reshape(6, 1, 1, 1)
    rhs = jax.random.normal(jax.random.key(1), (6, 4, 3, 3), jnp.float32)
    g = jax.random.normal(jax.random.key(2), (2, 6, 8, 8), jnp.float32)
    dnums = ('NCHW', 'OIHW', 'NCHW')
    config = conv_general_qt.ConvGeneralQtConfig(
        lhs_qtype='int8',
        rhs_qtype='int8',
        dlhs_grad_qtype=bwd_qtype,
        drhs_grad_qtype=bwd_qtype,
    )
    fq_out, fq_vjp = jax.vjp(
        lambda l, r: conv_general_fq(
            l, r, config, (1, 1), 'SAME', dnums, batch_group_count=3
        ),
        lhs,
        rhs,
    )
    qt_out, qt_vjp = jax.vjp(
        lambda l, r: conv_general_qt.conv_general_qt(
            l,
            r,
            config,
            (1, 1),
            'SAME',
            dimension_numbers=dnums,
            batch_group_count=3,
        ),
        lhs,
        rhs,
    )
    fq_dlhs, fq_drhs = fq_vjp(g)
    qt_dlhs, qt_drhs = qt_vjp(g)
    # Matches fast_slow_mae <= 0.004 in conv_general_tpu_test (bf16 MXU in
    # conv_general_fq vs exact int32 accumulation in conv_general_qt; 0.0022 on
    # TPU, <1e-6 on CPU).
    self.assertLessEqual(mae(fq_out, qt_out), 0.004)
    grad_tol = 0 if bwd_qtype is None else 0.01
    self.assertLessEqual(mae(fq_dlhs, qt_dlhs), grad_tol)
    self.assertLessEqual(mae(fq_drhs, qt_drhs), grad_tol)


if __name__ == '__main__':
  absltest.main()
