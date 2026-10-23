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

"""End-to-end test: one training step of a dense layer on the fused kernels.

This is the way a model would use `fused_fp8_matmul`: differentiate through
it with `jax.vjp` (or `jax.grad`), consume the output and both gradients. The
test runs the four projection shapes the kernels were built for, at full
size, and checks two things a training user cares about:

*   Accuracy: the output and both gradients stay within FP8 quantization noise
    of the same step in bfloat16.
*   The compiled program contains no full-size layout copies. An earlier
    hand-written wrapper around the same kernels made XLA insert two
    full-tensor copies per step (about 270 us on the largest shape); this
    guards against that coming back, without timing anything.

Performance is measured separately, not asserted here.
"""

import re

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix.kernels.fused_fp8_matmul import api
from qwix.kernels.fused_fp8_matmul import test_utils

# The projection shapes of the training workload this kernel was built for,
# (m, k, n), at full size.
SHAPES = {
    "wi_1": (32768, 3072, 4096),
    "wo": (32768, 4096, 3072),
    "qkv_proj": (32768, 3072, 2048),
    "o_proj": (32768, 1792, 3072),
}

# Both operands of every matmul in the step are quantized to FP8, so each
# result lands within FP8 noise of the bfloat16 step; see
# `test_utils.FP8_NOISE_TOL`. Measured on TPU7x: out 0.0385, d_act 0.0384,
# d_w 0.0391 on every shape.
_TOL = test_utils.FP8_NOISE_TOL


def _training_step(matmul, act, w, cot):
  """One step of `out = act @ w`: forward, then both gradients."""
  out, vjp_fn = jax.vjp(matmul, act, w)
  d_act, d_w = vjp_fn(cot)
  return out, d_act, d_w


def _full_size_copies(hlo: str, m: int) -> list[str]:
  """Returns the `copy` instructions whose operand has `m` rows."""
  return [
      line.strip()
      for line in hlo.splitlines()
      if re.search(r"= \S+ copy\(", line) and f"[{m}," in line
  ]


class TrainingStepTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    device = jax.devices()[0]
    if not api.fp8_matmul_supported_on(device):
      self.skipTest(f"FP8 matmul is not supported on {device.device_kind}.")

  def _inputs(self, m, k, n):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
    act = jax.random.normal(k1, (m, k), jnp.bfloat16)
    w = jax.random.normal(k2, (k, n), jnp.bfloat16)
    cot = jax.random.normal(k3, (m, n), jnp.bfloat16)
    return act, w, cot

  @parameterized.named_parameters(
      (name, *shape) for name, shape in SHAPES.items()
  )
  def test_step_matches_bf16_within_fp8_noise(self, m, k, n):
    act, w, cot = self._inputs(m, k, n)
    step_bf16 = jax.jit(lambda a, b, c: _training_step(jnp.matmul, a, b, c))
    step_fp8 = jax.jit(
        lambda a, b, c: _training_step(api.fused_fp8_matmul, a, b, c)
    )
    expected = step_bf16(act, w, cot)
    actual = step_fp8(act, w, cot)
    for name, got, want in zip(("out", "d_act", "d_w"), actual, expected):
      with self.subTest(name):
        self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype))
        self.assertLess(test_utils.relative_error(got, want), _TOL)

  @parameterized.named_parameters(
      (name, *shape) for name, shape in SHAPES.items()
  )
  def test_step_has_no_full_size_layout_copies(self, m, k, n):
    act, w, cot = self._inputs(m, k, n)
    step = jax.jit(
        lambda a, b, c: _training_step(api.fused_fp8_matmul, a, b, c)
    )
    hlo = step.lower(act, w, cot).compile().as_text() or ""
    self.assertEmpty(_full_size_copies(hlo, m))

  def test_step_with_frozen_quantized_weight(self):
    """Serving-style use: the weight is pre-quantized and gets no gradient."""
    m, k, n = SHAPES["qkv_proj"]
    act, w, cot = self._inputs(m, k, n)
    w_q = test_utils.quantize_per_column(w)
    step_fp8 = jax.jit(
        lambda a, b, c: _training_step(api.fused_fp8_matmul, a, b, c)
    )
    out, d_act, d_w = step_fp8(act, w_q, cot)
    out_plain, d_act_plain, _ = step_fp8(act, w, cot)
    # The weight was quantized by qwix here and by the op in the plain case.
    self.assertLess(test_utils.relative_error(out, out_plain), _TOL)
    self.assertLess(test_utils.relative_error(d_act, d_act_plain), _TOL)
    self.assertEqual(float(jnp.abs(d_w.scale).max()), 0.0)


if __name__ == "__main__":
  absltest.main()
