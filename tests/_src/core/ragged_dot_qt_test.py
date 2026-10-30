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
          lhs_qtype=jnp.float8_e4m3fn,
          rhs_qtype=jnp.float8_e4m3fn,
          expected_mae_fq_out=5e-3,
          expected_mae_fq_dlhs=0.035,
          expected_mae_fq_drhs=0.035,
          expected_mae_fp_out=0.02,
          expected_mae_fp_dlhs=0.03,
          expected_mae_fp_drhs=0.03,
      ),
      dict(
          testcase_name="fp8_bwd",
          lhs_qtype=jnp.float8_e4m3fn,
          rhs_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e4m3fn,
          expected_mae_fq_out=0.025,
          expected_mae_fq_dlhs=0.045,
          expected_mae_fq_drhs=0.045,
          expected_mae_fp_out=0.04,
          expected_mae_fp_dlhs=0.05,
          expected_mae_fp_drhs=0.05,
      ),
      dict(
          testcase_name="int8",
          lhs_qtype=jnp.int8,
          rhs_qtype=jnp.int8,
          expected_mae_fq_out=5e-3,
          expected_mae_fq_dlhs=5e-3,
          expected_mae_fq_drhs=5e-3,
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

    @jax.jit
    def f(lhs, rhs):
      qt_out, (qt_dlhs, qt_drhs) = jax.value_and_grad(
          loss_fn_qt, argnums=(0, 1)
      )(lhs, rhs)
      fq_out, (fq_dlhs, fq_drhs) = jax.value_and_grad(
          loss_fn_fq, argnums=(0, 1)
      )(lhs, rhs)
      fp_out, (fp_dlhs, fp_drhs) = jax.value_and_grad(
          loss_fn_fp, argnums=(0, 1)
      )(lhs, rhs)
      return (
          _mae(qt_out, fq_out),
          _mae(qt_dlhs, fq_dlhs),
          _mae(qt_drhs, fq_drhs),
          _mae(qt_out, fp_out),
          _mae(qt_dlhs, fp_dlhs),
          _mae(qt_drhs, fp_drhs),
      )

    maes = f(lhs, rhs)
    expected_maes = (
        expected_mae_fq_out,
        expected_mae_fq_dlhs,
        expected_mae_fq_drhs,
        expected_mae_fp_out,
        expected_mae_fp_dlhs,
        expected_mae_fp_drhs,
    )
    for x, y in zip(maes, expected_maes):
      self.assertLessEqual(x, y, msg=f"{maes} vs {expected_maes}")

  def test_traced_group_sizes(self):
    lhs = jax.random.normal(jax.random.key(0), (256, 64), jnp.float32)
    rhs = jax.random.normal(jax.random.key(1), (8, 64, 128), jnp.float32)
    config = ragged_dot_qt.RaggedDotQtConfig(
        lhs_qtype=jnp.float8_e4m3fn,
        rhs_qtype=jnp.float8_e4m3fn,
    )

    @jax.jit
    def f(group_sizes):
      return ragged_dot_qt.ragged_dot_qt(lhs, rhs, group_sizes, config)

    group_sizes = jnp.array([10, 20, 30, 40, 50, 60, 31, 15], jnp.int32)
    out = f(group_sizes)

    self.assertEqual(out.shape, (256, 128))

  def test_multipass_ragged_dot_qt(self):
    lhs = jax.random.normal(jax.random.key(0), (64, 32), jnp.float32)
    rhs = jax.random.normal(jax.random.key(1), (4, 32, 16), jnp.float32)
    group_sizes = jnp.array([16, 16, 16, 16], jnp.int32)

    cfg_1p = ragged_dot_qt.RaggedDotQtConfig(
        lhs_qtype=jnp.float8_e4m3fn,
        rhs_qtype=jnp.float8_e4m3fn,
        dlhs_grad_qtype=jnp.float8_e4m3fn,
        drhs_grad_qtype=jnp.float8_e4m3fn,
    )
    cfg_3p = ragged_dot_qt.RaggedDotQtConfig(
        multipass_mode="three_pass_fp8",
        dlhs_multipass_mode="three_pass_fp8",
        drhs_multipass_mode="three_pass_fp8",
    )

    ref_fn = lambda l, r: jnp.sum(jax.lax.ragged_dot(l, r, group_sizes) ** 2)
    fn_1p = lambda l, r: jnp.sum(
        ragged_dot_qt.ragged_dot_qt(l, r, group_sizes, cfg_1p) ** 2
    )
    fn_3p = lambda l, r: jnp.sum(
        ragged_dot_qt.ragged_dot_qt(l, r, group_sizes, cfg_3p) ** 2
    )

    ref_out = jax.lax.ragged_dot(lhs, rhs, group_sizes)
    out_1p = ragged_dot_qt.ragged_dot_qt(lhs, rhs, group_sizes, cfg_1p)
    out_3p = ragged_dot_qt.ragged_dot_qt(lhs, rhs, group_sizes, cfg_3p)

    self.assertLess(_mae(ref_out, out_3p), _mae(ref_out, out_1p))

    ref_dlhs, ref_drhs = jax.grad(ref_fn, argnums=(0, 1))(lhs, rhs)
    dlhs_1p, drhs_1p = jax.grad(fn_1p, argnums=(0, 1))(lhs, rhs)
    dlhs_3p, drhs_3p = jax.grad(fn_3p, argnums=(0, 1))(lhs, rhs)

    self.assertLess(_mae(ref_dlhs, dlhs_3p), _mae(ref_dlhs, dlhs_1p))
    self.assertLess(_mae(ref_drhs, drhs_3p), _mae(ref_drhs, drhs_1p))


if __name__ == "__main__":
  absltest.main()
