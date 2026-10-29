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

"""Tests for `fused_fp8_matmul`, the user-facing function."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix.contrib.kernels import fused_fp8_matmul as ffm
from qwix.contrib.kernels.fused_fp8_matmul import forward_kernel
from qwix.contrib.kernels.fused_fp8_matmul import test_utils


def _has_fp8_tpu() -> bool:
  return ffm.fp8_matmul_supported_on(jax.devices()[0])


class XlaImplementationTest(test_utils.FusedFp8MatmulTestBase):
  """The plain-XLA path runs on every backend."""

  def __init__(self, *args):
    super().__init__(
        *args,
        matmul_fn=functools.partial(ffm.fused_fp8_matmul, implementation="xla"),
    )

  def test_rejects_config(self):
    lhs, rhs = test_utils.random_inputs(256, 256, 256, jnp.bfloat16)
    config = ffm.Config(block_m=256, block_k=256, block_n=256)
    with self.assertRaisesRegex(ValueError, "implementation='pallas'"):
      ffm.fused_fp8_matmul(lhs, rhs, config=config, implementation="xla")


class PallasImplementationTest(test_utils.FusedFp8MatmulTestBase):
  """The kernel path, with the default (heuristic) tiling."""

  def __init__(self, *args):
    super().__init__(*args, matmul_fn=ffm.fused_fp8_matmul)

  def setUp(self):
    super().setUp()
    if not _has_fp8_tpu():
      self.skipTest(
          f"FP8 matmul is not supported on {jax.devices()[0].device_kind}."
      )

  @parameterized.parameters(
      # (m, k, n, block_m, block_k, block_n)
      (1024, 3072, 4096, 512, 3072, 4096),  # single k block, bn == n
      (1024, 3072, 4096, 256, 1024, 1024),  # multi-k with an n grid
  )
  def test_explicit_config(self, m, k, n, block_m, block_k, block_n):
    config = ffm.Config(block_m=block_m, block_k=block_k, block_n=block_n)
    lhs, rhs = test_utils.random_inputs(m, k, n, jnp.bfloat16)
    actual = jax.jit(functools.partial(ffm.fused_fp8_matmul, config=config))(
        lhs, rhs
    )
    expected = jax.jit(test_utils.reference_matmul)(lhs, rhs)
    if block_k == k:
      tol = test_utils.SAME_QUANTIZATION_TOL
    else:
      tol = test_utils.FP8_NOISE_TOL
    self.assertLess(test_utils.relative_error(actual, expected), tol)

  def test_grad_with_multi_k_config(self):
    # With more than one k block the kernel cannot emit the per-row residuals,
    # so the op recomputes them in XLA; the gradient must still match.
    m, k, n = 512, 2048, 1024
    config = ffm.Config(block_m=256, block_k=512, block_n=1024)
    lhs, rhs = test_utils.random_inputs(m, k, n, jnp.bfloat16)

    def loss(fn, a, b):
      return jnp.sum(fn(a, b).astype(jnp.float32))

    kernel_fn = functools.partial(ffm.fused_fp8_matmul, config=config)
    xla_fn = functools.partial(ffm.fused_fp8_matmul, implementation="xla")
    dlhs, drhs = jax.jit(jax.grad(functools.partial(loss, kernel_fn), (0, 1)))(
        lhs, rhs
    )
    dlhs_ref, drhs_ref = jax.jit(
        jax.grad(functools.partial(loss, xla_fn), (0, 1))
    )(lhs, rhs)
    self.assert_close_to_reference(dlhs, dlhs_ref, k=k)
    self.assert_close_to_reference(drhs, drhs_ref, k=k)

  def test_grad_with_f32_activations_fits_in_vmem(self):
    # Differentiating makes the kernel emit residuals, and f32 activation and
    # output blocks are twice the size of bf16 ones. At this shape the bf16
    # tiling (bm = 1024) overflows VMEM for f32; the heuristics must see the
    # dtype and back off.
    m, k, n = 2048, 3072, 3072
    lhs, rhs = test_utils.random_inputs(m, k, n, jnp.float32)
    self.assertEqual(
        ffm.get_heuristics_config(m, k, n, lhs_dtype=jnp.float32).block_m, 512
    )

    def loss(fn, a, b):
      return jnp.sum(fn(a, b))

    xla_fn = functools.partial(ffm.fused_fp8_matmul, implementation="xla")
    grads = jax.jit(
        jax.grad(functools.partial(loss, ffm.fused_fp8_matmul), (0, 1))
    )
    grads_ref = jax.jit(jax.grad(functools.partial(loss, xla_fn), (0, 1)))
    for actual, expected in zip(grads(lhs, rhs), grads_ref(lhs, rhs)):
      self.assert_close_to_reference(actual, expected, k=k)

  def test_rejects_block_that_does_not_divide_shape(self):
    lhs, rhs = test_utils.random_inputs(512, 1024, 1024, jnp.bfloat16)
    config = ffm.Config(block_m=384, block_k=1024, block_n=1024)
    with self.assertRaisesRegex(ValueError, "divisible by bm"):
      ffm.fused_fp8_matmul(lhs, rhs, config=config)


class ArgumentsTest(parameterized.TestCase):
  """Argument handling that does not need an FP8 TPU."""

  def test_pallas_on_unsupported_device_raises(self):
    if _has_fp8_tpu():
      self.skipTest("This test needs a device without an FP8 MXU.")
    lhs, rhs = test_utils.random_inputs(256, 256, 256, jnp.bfloat16)
    with self.assertRaisesRegex(NotImplementedError, "implementation='xla'"):
      ffm.fused_fp8_matmul(lhs, rhs)

  def test_pallas_rejects_dimensions_not_multiple_of_128(self):
    # The alignment check runs before the device check, so this holds on CPU.
    lhs, rhs = test_utils.random_inputs(200, 256, 256, jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "multiples of 128"):
      ffm.fused_fp8_matmul(lhs, rhs)

  def test_rejects_unknown_implementation(self):
    lhs, rhs = test_utils.random_inputs(256, 256, 256, jnp.bfloat16)
    # Passed through a dict so that the static type checkers, which know the
    # parameter is a `Literal`, let the runtime check be exercised.
    bad_kwargs = {"implementation": "triton"}
    with self.assertRaisesRegex(ValueError, "implementation must be"):
      ffm.fused_fp8_matmul(lhs, rhs, **bad_kwargs)

  def test_config_rejects_unaligned_blocks(self):
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      ffm.Config(block_m=100, block_k=256, block_n=256)

  @parameterized.parameters(*test_utils.SHAPES, (32768, 3072, 4096))
  def test_heuristics_config_divides_shape_and_fits(self, m, k, n):
    config = ffm.get_heuristics_config(m, k, n)
    self.assertEqual(
        (m % config.block_m, k % config.block_k, n % config.block_n), (0, 0, 0)
    )
    self.assertLessEqual(
        forward_kernel.forward_vmem_bytes(
            config.block_m, config.block_k, config.block_n, k, n, lhs_itemsize=2
        ),
        forward_kernel.VMEM_BUDGET_BYTES,
    )


if __name__ == "__main__":
  absltest.main()
