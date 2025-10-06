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
from qwix._src.core import qarray
from qwix._src.core import ragged_dot_qt


def _mae(a, b):
  """Computes mean absolute error normalized by the mean absolute value of a."""
  return jnp.mean(jnp.abs(a - b)) / jnp.mean(jnp.abs(a))


def _fake_quant(array: jax.Array, how: qarray.HowToQuantize) -> jax.Array:
  """Simulates quantization in full precision using STE."""
  if not how.qtype:
    return array
  calibration = qarray.calibrate(array, how)
  scale, zp = qarray.compute_scale_zero_point(calibration, how.qtype)
  q_arr = qarray.quantize_with_scale_zero_point(array, how.qtype, scale, zp)
  dq_arr = qarray.dequantize(q_arr)
  # The straight-through estimator (STE) part
  return array + jax.lax.stop_gradient(dq_arr - array)


def ragged_dot_fq(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    config: ragged_dot_qt.RaggedDotQtConfig,
) -> jax.Array:
  """Ragged dot implemented with fake quantization for baseline comparison."""
  lhs_how = qarray.HowToQuantize(qtype=config.lhs_qtype, channelwise_axes=[0])
  rhs_how = qarray.HowToQuantize(qtype=config.rhs_qtype, channelwise_axes=[2])
  lhs_fq = _fake_quant(lhs, lhs_how)
  rhs_fq = _fake_quant(rhs, rhs_how)
  return jax.lax.ragged_dot(lhs_fq, rhs_fq, group_sizes)


class RaggedDotQtTest(parameterized.TestCase):
  """Test class for ragged_dot_qt."""

  @parameterized.named_parameters(
      dict(
          testcase_name="fp8",
          lhs_qtype=jnp.float8_e4m3,
          rhs_qtype=jnp.float8_e4m3,
          expected_mae_fq_out=1e-6,
          expected_mae_fq_dlhs=1e-6,
          expected_mae_fq_drhs=1e-6,
          expected_mae_fp_out=0.02,
          expected_mae_fp_dlhs=0.03,
          expected_mae_fp_drhs=0.03,
      ),
      dict(
          testcase_name="fp8_bwd",
          lhs_qtype=jnp.float8_e4m3,
          rhs_qtype=jnp.float8_e4m3,
          bwd_qtype=jnp.float8_e4m3,
          expected_mae_fq_out=1e-6,
          expected_mae_fq_dlhs=0.03,
          expected_mae_fq_drhs=0.03,
          expected_mae_fp_out=0.02,
          expected_mae_fp_dlhs=0.05,
          expected_mae_fp_drhs=0.05,
      ),
      dict(
          testcase_name="int8",
          lhs_qtype=jnp.int8,
          rhs_qtype=jnp.int8,
          expected_mae_fq_out=1e-6,
          expected_mae_fq_dlhs=1e-6,
          expected_mae_fq_drhs=1e-6,
          expected_mae_fp_out=0.02,
          expected_mae_fp_dlhs=0.01,
          expected_mae_fp_drhs=0.01,
      ),
  )
  def test_grad_against_fq_and_fp(
      self,
      lhs_qtype,
      rhs_qtype,
      expected_mae_fq_out,
      expected_mae_fq_dlhs,
      expected_mae_fq_drhs,
      expected_mae_fp_out,
      expected_mae_fp_dlhs,
      expected_mae_fp_drhs,
      bwd_qtype=None,
  ):
    lhs = jax.random.normal(jax.random.key(0), (256, 64), jnp.float32)
    rhs = jax.random.normal(jax.random.key(1), (8, 64, 128), jnp.float32)
    group_sizes = jnp.array([10, 20, 30, 40, 50, 60, 31, 15], jnp.int32)
    config = ragged_dot_qt.RaggedDotQtConfig(
        lhs_qtype=lhs_qtype,
        rhs_qtype=rhs_qtype,
        dlhs_grad_qtype=bwd_qtype,
        drhs_grad_qtype=bwd_qtype,
    )

    loss_fn_qt = lambda l, r: jnp.sum(
        ragged_dot_qt.ragged_dot_qt(l, r, group_sizes, config)
    )
    loss_fn_fq = lambda l, r: jnp.sum(ragged_dot_fq(l, r, group_sizes, config))
    loss_fn_fp = lambda l, r: jnp.sum(jax.lax.ragged_dot(l, r, group_sizes))

    qt_out, (qt_dlhs, qt_drhs) = jax.value_and_grad(loss_fn_qt, argnums=(0, 1))(
        lhs, rhs
    )
    fq_out, (fq_dlhs, fq_drhs) = jax.value_and_grad(loss_fn_fq, argnums=(0, 1))(
        lhs, rhs
    )
    fp_out, (fp_dlhs, fp_drhs) = jax.value_and_grad(loss_fn_fp, argnums=(0, 1))(
        lhs, rhs
    )

    # QT and FQ results should be close.
    self.assertLessEqual(_mae(qt_out, fq_out), expected_mae_fq_out)
    self.assertLessEqual(_mae(qt_dlhs, fq_dlhs), expected_mae_fq_dlhs)
    self.assertLessEqual(_mae(qt_drhs, fq_drhs), expected_mae_fq_drhs)

    # QT and FP results should be close in a larger tolerance.
    self.assertLessEqual(_mae(qt_out, fp_out), expected_mae_fp_out)
    self.assertLessEqual(_mae(qt_dlhs, fp_dlhs), expected_mae_fp_dlhs)
    self.assertLessEqual(_mae(qt_drhs, fp_drhs), expected_mae_fp_drhs)


if __name__ == "__main__":
  absltest.main()
