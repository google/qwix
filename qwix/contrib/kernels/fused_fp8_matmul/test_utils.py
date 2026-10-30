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

"""Shared shapes, tolerances and correctness suite for the fused FP8 matmul."""

from collections.abc import Callable

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from qwix._src.core import qarray
from qwix.contrib.kernels.fused_fp8_matmul import reference

# (m, k, n) shapes. The first four are the projection shapes of the training
# workload this kernel was built for, at reduced `m`; the rest exercise large
# `k` (where a kernel may split the contraction), a block size that is not a
# power of two, and small shapes.
SHAPES = (
    (2048, 3072, 4096),
    (2048, 3072, 12288),
    (2048, 12288, 3072),
    (2048, 3072, 3072),
    (512, 256, 256),
    (1024, 1024, 384),
    (256, 6144, 512),
)

# Above this `k`, implementations are allowed to quantize `lhs` per k block
# rather than per full row, so they only agree with the reference up to FP8
# quantization noise rather than up to rounding of the final cast.
_SINGLE_ROW_SCALE_MAX_K = 4096

# Relative error (in norm) between two results computed with the *same* FP8
# quantization algorithm by different compilers. XLA on TPU may keep the bf16
# scaling multiply in f32 and round once to FP8, while an eager computation or
# a Pallas kernel rounds to bf16 first; elements that straddle an FP8 rounding
# boundary then differ by one FP8 ulp. Measured at ~1.3% on TPU7x but 2.4% on
# TPU v6e (no FP8 MXU, so XLA fuses the dot differently), which is why the
# suite compares jitted results with jitted results. 2% is tight enough to
# catch a wrong scale or tile index (tens of percent) without flaking.
SAME_QUANTIZATION_TOL = 2e-2

# Relative error (in norm) between two results whose operands were quantized
# to FP8 by different but equivalent methods, e.g. a different scale
# granularity. An e4m3 value has a 3-bit mantissa, so rounding gives each
# quantized operand ~2.5% RMS relative error; two independent quantizations of
# the same operand therefore differ by ~3.5%. 5% leaves margin without hiding
# a real bug (a wrong scale shows up as tens of percent).
FP8_NOISE_TOL = 5e-2

# Elementwise tolerance for reconstructing an operand from its FP8 value and
# scale: half an e4m3 ulp is 6.25% at the bottom of a binade, plus a little
# from rounding the scale to bf16 inside the quantizer.
FP8_RECONSTRUCTION_RTOL = 0.1

# `matmul_fn(lhs, rhs) -> out`: the implementation under test.
MatmulFn = Callable[[jax.Array, jax.Array | qarray.QArray], jax.Array]


def random_inputs(m, k, n, dtype, seed=0):
  k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
  lhs = jax.random.normal(k1, (m, k), jnp.float32).astype(dtype)
  rhs = (jax.random.normal(k2, (k, n), jnp.float32) * 0.02).astype(dtype)
  return lhs, rhs


def quantize_per_column(rhs: jax.Array) -> qarray.QArray:
  """Quantizes `rhs` the way a Qwix user would for this op.

  The scale is left in the default dtype (the weight's own, so `bfloat16` for
  a `bfloat16` weight) on purpose: that is what reaches the op from
  `qwix.quantize` unless the user asks otherwise. Runs under `jax.jit`, like
  the op's own quantization of a plain weight, so the two produce the same
  FP8 values (see `SAME_QUANTIZATION_TOL`).

  Args:
    rhs: The `(k, n)` weight.

  Returns:
    The FP8 `QArray` with one scale per output column, of shape `(1, n)`.
  """
  return _jitted_quantize_per_column(rhs)


@jax.jit
def _jitted_quantize_per_column(rhs: jax.Array) -> qarray.QArray:
  return qarray.quantize_api(rhs, jnp.float8_e4m3fn, channelwise_axes=(1,))


def f32_matmul(lhs, rhs):
  return jnp.matmul(
      lhs.astype(jnp.float32),
      rhs.astype(jnp.float32),
      precision=jax.lax.Precision.HIGHEST,
  )


def relative_error(actual, expected) -> float:
  actual = np.asarray(actual, np.float32)
  expected = np.asarray(expected, np.float32)
  return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def reference_matmul(lhs, rhs):
  return reference.fused_fp8_matmul_reference(lhs, rhs)[0]


class FusedFp8MatmulTestBase(parameterized.TestCase):
  """Correctness suite shared by all fused FP8 matmul implementations.

  Subclasses pass the function under test as `matmul_fn`. It is compared
  against the XLA reference tightly, and against an exact f32 matmul loosely
  (FP8 quantization error).
  """

  def __init__(self, *args, matmul_fn: MatmulFn):
    super().__init__(*args)
    self._matmul_fn = matmul_fn

  def _matmul(self, lhs, rhs):
    return jax.jit(self._matmul_fn)(lhs, rhs)

  def assert_close_to_reference(self, actual, expected, k):
    # With one scale per row both sides quantize the same way; above
    # `_SINGLE_ROW_SCALE_MAX_K` a kernel may quantize per k block instead.
    tol = (
        SAME_QUANTIZATION_TOL if k <= _SINGLE_ROW_SCALE_MAX_K else FP8_NOISE_TOL
    )
    self.assertLess(relative_error(actual, expected), tol)

  @parameterized.product(shape=SHAPES, dtype=(jnp.bfloat16, jnp.float32))
  def test_matches_reference(self, shape, dtype):
    m, k, n = shape
    lhs, rhs = random_inputs(m, k, n, dtype)
    actual = self._matmul(lhs, rhs)
    expected = jax.jit(reference_matmul)(lhs, rhs)
    self.assertEqual(actual.shape, expected.shape)
    self.assertEqual(actual.dtype, dtype)
    self.assert_close_to_reference(actual, expected, k=k)

  @parameterized.parameters(*SHAPES)
  def test_close_to_f32_matmul(self, m, k, n):
    lhs, rhs = random_inputs(m, k, n, jnp.bfloat16)
    actual = self._matmul(lhs, rhs)
    # Both operands are quantized (see `FP8_NOISE_TOL`), so the result lands
    # ~3.5% from the exact matmul.
    self.assertLess(relative_error(actual, f32_matmul(lhs, rhs)), 0.05)

  def test_qarray_rhs_matches_plain_rhs(self):
    """A qwix-quantized rhs gives the same result as quantizing in-op."""
    m, k, n = 1024, 3072, 4096
    lhs, rhs = random_inputs(m, k, n, jnp.bfloat16)
    rhs_q = quantize_per_column(rhs)
    self.assertEqual(rhs_q.scale.shape, (1, n))
    actual = self._matmul(lhs, rhs_q)
    # The op quantizes a plain weight with the same jitted `qwix.quantize`
    # call, so the two should agree up to compiler differences.
    expected = self._matmul(lhs, rhs)
    self.assertLess(relative_error(actual, expected), SAME_QUANTIZATION_TOL)

  def test_grad_matches_reference(self):
    """Gradients w.r.t. both operands match the reference VJP."""
    m, k, n = 512, 1024, 2048
    lhs, rhs = random_inputs(m, k, n, jnp.bfloat16)
    dout = jax.random.normal(jax.random.PRNGKey(3), (m, n), jnp.bfloat16)

    def loss(fn, a, b):
      return jnp.sum(fn(a, b).astype(jnp.float32) * dout.astype(jnp.float32))

    def reference_grads(a, b):
      _, residuals = reference.fused_fp8_matmul_reference(a, b)
      return reference.fused_fp8_matmul_vjp_reference(residuals, dout)

    actual = jax.jit(jax.grad(lambda a, b: loss(self._matmul_fn, a, b), (0, 1)))
    dlhs, drhs = actual(lhs, rhs)
    dlhs_ref, drhs_ref = jax.jit(reference_grads)(lhs, rhs)
    self.assertEqual((dlhs.shape, dlhs.dtype), (lhs.shape, lhs.dtype))
    self.assertEqual((drhs.shape, drhs.dtype), (rhs.shape, rhs.dtype))
    self.assert_close_to_reference(dlhs, dlhs_ref, k=k)
    self.assert_close_to_reference(drhs, drhs_ref, k=k)

  def test_grad_with_mixed_dtypes(self):
    """A float32 rhs gets a float32 gradient even when lhs is bfloat16."""
    m, k, n = 256, 512, 512
    lhs, rhs = random_inputs(m, k, n, jnp.bfloat16)
    rhs = rhs.astype(jnp.float32)

    def loss(a, b):
      return jnp.sum(self._matmul_fn(a, b).astype(jnp.float32))

    dlhs, drhs = jax.jit(jax.grad(loss, (0, 1)))(lhs, rhs)
    self.assertEqual((dlhs.shape, dlhs.dtype), (lhs.shape, jnp.bfloat16))
    self.assertEqual((drhs.shape, drhs.dtype), (rhs.shape, jnp.float32))

  def test_grad_with_qarray_rhs(self):
    """Differentiating w.r.t. lhs works when rhs is a pre-quantized QArray."""
    m, k, n = 512, 1024, 1024
    lhs, rhs = random_inputs(m, k, n, jnp.bfloat16)
    rhs_q = quantize_per_column(rhs)

    def loss(a, b):
      return jnp.sum(self._matmul_fn(a, b).astype(jnp.float32))

    dlhs = jax.jit(jax.grad(loss))(lhs, rhs_q)
    dlhs_ref = jax.jit(jax.grad(loss))(lhs, rhs)
    self.assertEqual((dlhs.shape, dlhs.dtype), (lhs.shape, lhs.dtype))
    # `rhs` was quantized by qwix in one case and by the op in the other.
    self.assertLess(relative_error(dlhs, dlhs_ref), FP8_NOISE_TOL)
    # The pre-quantized weight gets a zero gradient rather than an error.
    drhs_q = jax.jit(jax.grad(loss, argnums=1))(lhs, rhs_q)
    self.assertEqual(drhs_q.qvalue.shape, rhs_q.qvalue.shape)
    np.testing.assert_array_equal(np.asarray(drhs_q.scale, np.float32), 0.0)

  def test_rejects_bad_inputs(self):
    """Shape and dtype errors are raised eagerly with clear messages."""
    lhs = jnp.zeros((256, 512), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "Contracting dims differ"):
      self._matmul_fn(lhs, jnp.zeros((256, 512), jnp.bfloat16))
    with self.assertRaisesRegex(ValueError, "2-D"):
      self._matmul_fn(lhs[None], jnp.zeros((512, 256), jnp.bfloat16))
    with self.assertRaisesRegex(ValueError, "qwix.QArray"):
      self._matmul_fn(lhs, jnp.zeros((512, 256), jnp.float8_e4m3fn))
    with self.assertRaisesRegex(ValueError, "one scale per output column"):
      per_row = qarray.quantize_api(
          jnp.ones((512, 256), jnp.bfloat16),
          jnp.float8_e4m3fn,
          channelwise_axes=(0,),
      )
      self._matmul_fn(lhs, per_row)
