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

"""Tests for `qwix.contrib.kernels.hadamard` (XLA and Pallas TPU)."""

import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from qwix._src import interception
from qwix._src.core import dot_general
from qwix._src.core import dot_general_qt
from qwix._src.core import einsum as qwix_einsum
from qwix._src.core import qarray
from qwix.contrib.kernels import hadamard as hk
from qwix.contrib.kernels.hadamard import kernel
from qwix.contrib.kernels.hadamard import op


def _rel_l2_error(actual: jax.Array, expected: jax.Array) -> float:
  diff = (actual.astype(jnp.float32) - expected.astype(jnp.float32)).ravel()
  ref = expected.astype(jnp.float32).ravel()
  denom = float(jnp.linalg.norm(ref))
  if denom == 0.0:
    return float(jnp.linalg.norm(diff))
  return float(jnp.linalg.norm(diff)) / denom


def _round_to_bfloat16(v: np.ndarray) -> np.ndarray:
  """Rounds float64 `v` to the nearest bfloat16 value, ties to even."""
  # Rounding the 8-bit significand directly avoids float64 -> float32 ->
  # bfloat16 double rounding.
  mantissa, exponent = np.frexp(v)
  return np.ldexp(np.round(mantissa * 256.0), exponent - 8)


class HadamardOpTest(parameterized.TestCase):
  """Functional tests executed against both `"xla"` and `"pallas"`."""

  def _require_backend(self, implementation: op.Implementation) -> None:
    if implementation == "pallas" and not hk.hadamard_pallas_supported():
      self.skipTest(
          f"Pallas TPU RHT is not supported on {jax.devices()[0].device_kind}."
      )

  @parameterized.product(
      case=(
          # (shape, axis, block_size, dtype, with_signs)
          ((128, 256), -1, None, jnp.bfloat16, True),
          ((256, 2048), -1, None, jnp.bfloat16, True),  # two-stage, 16 blocks
          ((192, 768), -1, None, jnp.bfloat16, True),  # unaligned m + Paley-I
          ((1, 1536), -1, None, jnp.bfloat16, False),  # single row
          ((16, 3200), -1, None, jnp.bfloat16, True),  # Paley-I order 200 * 16
          ((4, 16, 768), 2, None, jnp.bfloat16, True),  # 3-D, last axis
          ((2, 256, 6), 1, None, jnp.float32, True),  # middle axis + float32
          ((128, 1024), -1, 128, jnp.bfloat16, True),  # Block-Hadamard (b=128)
          ((64, 768), -1, 256, jnp.float32, True),  # Block-Hadamard (b=256)
      ),
      implementation=("xla", "pallas"),
  )
  def test_hadamard_rotate_forward_and_inverse_match_reference(
      self, case, implementation
  ):
    self._require_backend(implementation)
    shape, axis, block_size, dtype, with_signs = case
    k0, k1 = jax.random.split(jax.random.key(11))
    x = jax.random.normal(k0, shape, dtype=dtype)
    rot_len = shape[axis]
    signs = hk.random_signs(k1, rot_len, dtype=dtype) if with_signs else None

    fn = jax.jit(
        functools.partial(
            hk.hadamard_rotate,
            block_size=block_size,
            axis=axis,
            implementation=implementation,
        )
    )
    inv_fn = jax.jit(
        functools.partial(
            hk.hadamard_rotate,
            block_size=block_size,
            axis=axis,
            inverse=True,
            implementation=implementation,
        )
    )

    y = fn(x, signs=signs)
    y_ref = hk.hadamard_rotate_reference(
        x, signs=signs, block_size=block_size, axis=axis
    )
    x_rec = inv_fn(y, signs=signs)

    tol = 1e-2 if dtype == jnp.bfloat16 else 1e-5
    self.assertEqual(y.shape, x.shape)
    self.assertEqual(y.dtype, dtype)
    self.assertLess(_rel_l2_error(y, y_ref), tol)
    self.assertLess(_rel_l2_error(x_rec, x), tol)

  @parameterized.product(
      case=(
          # (qtype, dtype, shape, axis, block_size)
          ("int8", jnp.bfloat16, (192, 768), -1, None),
          (jnp.int8, jnp.bfloat16, (32, 3200), -1, None),
          (np.dtype("int8"), jnp.float32, (64, 256), -1, None),
          (jnp.float8_e4m3fn, jnp.bfloat16, (4, 16, 768), 2, None),
          ("float8_e4m3fn", jnp.bfloat16, (2, 256, 8), 1, 128),
          (jnp.float8_e5m2, jnp.bfloat16, (128, 512), -1, None),
      ),
      implementation=("xla", "pallas"),
  )
  def test_hadamard_rotate_quantize_matches_reference_and_feeds_dot_general(
      self, case, implementation
  ):
    self._require_backend(implementation)
    qtype, dtype, shape, axis, block_size = case
    k0, k1, k2 = jax.random.split(jax.random.key(23), 3)
    x = jax.random.normal(k0, shape, dtype=dtype)
    k = shape[axis]
    signs = hk.random_signs(k1, k, dtype=dtype)

    xq = jax.jit(
        functools.partial(
            hk.hadamard_rotate_quantize,
            qtype=qtype,
            block_size=block_size,
            axis=axis,
            implementation=implementation,
        )
    )(x, signs=signs)
    expected = jax.jit(
        functools.partial(
            hk.hadamard_rotate_quantize_reference,
            qtype=qtype,
            block_size=block_size,
            axis=axis,
        )
    )(x, signs=signs)

    qarray.validate_qarray(xq)
    self.assertEqual(xq.qvalue.shape, expected.qvalue.shape)
    self.assertEqual(xq.qvalue.dtype, expected.qvalue.dtype)
    self.assertEqual(xq.scale.shape, expected.scale.shape)
    self.assertEqual(xq.scale.dtype, dtype)

    deq = qarray.dequantize(xq)
    deq_ref = qarray.dequantize(expected)
    # On TPU the reference's bfloat16 scales can be one ulp off, moving ~12.5%
    # of int8 values by one step; e5m2 has twice e4m3's step (observed 0.0277).
    same_quant_tol = 3.5e-2 if xq.qvalue.dtype == jnp.float8_e5m2 else 2e-2
    self.assertLess(_rel_l2_error(deq, deq_ref), same_quant_tol)

    # Verify compatibility with `qwix.dot_general` and `qwix.einsum` against
    # rotated weight QArrays (both tight agreement with reference QArray matmul
    # and accuracy vs exact float32 matmul).
    if len(shape) == 3 and axis == 2:
      _, n_tokens, h_dim = shape
      w = jax.random.normal(k2, (n_tokens, h_dim, 64), dtype=dtype) * 0.1
      w_rot = hk.hadamard_rotate_reference(
          w, signs=signs, block_size=block_size, axis=1
      )
      wq_i4 = qarray.quantize_api(w_rot, jnp.int4, channelwise_axes=(0, 2))
      out_q = qwix_einsum.einsum("bnh,nho->bno", xq, wq_i4)
      out_q_ref = qwix_einsum.einsum("bnh,nho->bno", expected, wq_i4)
      out_fp = jnp.einsum(
          "bnh,nho->bno",
          x.astype(jnp.float32),
          w.astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )
      self.assertLess(_rel_l2_error(out_q, out_q_ref), same_quant_tol)
      # Per-channel symmetric int4 absmax on Gaussian weights has ~0.14 RMS
      # quantization error (observed 0.129-0.144 across seeds/shapes).
      self.assertLess(_rel_l2_error(out_q, out_fp), 0.16)
    elif len(shape) == 2 and axis == -1:
      w = jax.random.normal(k2, (k, 128), dtype=dtype) * 0.1
      w_rot = hk.hadamard_rotate_reference(
          w, signs=signs, block_size=block_size, axis=0
      )
      dnums_2d = (((1,), (0,)), ((), ()))
      wq_i8 = qarray.quantize_api(w_rot, jnp.int8, channelwise_axes=(1,))
      out_w8 = dot_general.dot_general(xq, wq_i8, dnums_2d)
      wq_i4 = qarray.quantize_api(w_rot, jnp.int4, channelwise_axes=(1,))
      out_w4 = dot_general.dot_general(xq, wq_i4, dnums_2d)
      out_w4_ref = dot_general.dot_general(expected, wq_i4, dnums_2d)
      out_fp = jnp.matmul(
          x.astype(jnp.float32),
          w.astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )
      # `float8_e5m2` has a 2-bit mantissa (twice the step size of 3-bit e4m3),
      # giving ~5.3% RMS error vs exact float32 (observed 0.0530).
      if xq.qvalue.dtype == jnp.int8:
        w8_tol = 0.015
      elif xq.qvalue.dtype == jnp.float8_e4m3fn:
        w8_tol = 0.04
      else:
        w8_tol = 0.07
      self.assertLess(_rel_l2_error(out_w8, out_fp), w8_tol)
      self.assertLess(_rel_l2_error(out_w4, out_w4_ref), same_quant_tol)
      # Per-channel symmetric int4 absmax on Gaussian weights has ~0.14 RMS
      # quantization error (observed 0.120-0.144 across seeds/shapes).
      self.assertLess(_rel_l2_error(out_w4, out_fp), 0.16)

  @parameterized.product(
      signs_kind=(None, "bf16_signs", "int8_signs"),
      implementation=("xla", "pallas"),
  )
  def test_hadamard_rotate_custom_vjp_grad(self, signs_kind, implementation):
    self._require_backend(implementation)
    k0, k1, k2 = jax.random.split(jax.random.key(31), 3)
    x = jax.random.normal(k0, (128, 768), dtype=jnp.float32)
    dout = jax.random.normal(k1, (128, 768), dtype=jnp.float32)
    if signs_kind is None:
      signs = None
    elif signs_kind == "bf16_signs":
      signs = hk.random_signs(k2, 768, dtype=jnp.bfloat16)
    else:
      signs = hk.random_signs(k2, 768, dtype=jnp.int8)

    def loss(a, s, g):
      y = hk.hadamard_rotate(a, signs=s, implementation=implementation)
      return jnp.sum(y * g)

    def loss_ref(a, s, g):
      return jnp.sum(hk.hadamard_rotate_reference(a, signs=s) * g)

    dx = jax.jit(jax.grad(loss))(x, signs, dout)
    dx_ref = jax.jit(jax.grad(loss_ref))(x, signs, dout)
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-4, atol=1e-4)

  @parameterized.product(
      batched_signs=(False, True),
      implementation=("xla", "pallas"),
  )
  def test_vmap_support(self, batched_signs, implementation):
    self._require_backend(implementation)
    k0, k1 = jax.random.split(jax.random.key(41))
    x = jax.random.normal(k0, (4, 48, 256), dtype=jnp.bfloat16)
    if batched_signs:
      keys = jax.random.split(k1, 4)
      signs = jax.vmap(lambda key: hk.random_signs(key, 256))(keys)
      in_axes = (0, 0)
    else:
      signs = hk.random_signs(k1, 256)
      in_axes = (0, None)

    vmapped_rht = jax.jit(
        jax.vmap(
            lambda a, s: hk.hadamard_rotate(
                a, signs=s, implementation=implementation
            ),
            in_axes=in_axes,
        )
    )
    vmapped_ref = jax.vmap(
        lambda a, s: hk.hadamard_rotate_reference(a, signs=s), in_axes=in_axes
    )
    actual = vmapped_rht(x, signs)
    expected = vmapped_ref(x, signs)
    self.assertLess(_rel_l2_error(actual, expected), 1e-2)

    vmapped_q = jax.jit(
        jax.vmap(
            lambda a, s: hk.hadamard_rotate_quantize(
                a, qtype=jnp.int8, signs=s, implementation=implementation
            ),
            in_axes=in_axes,
        )
    )
    xq = vmapped_q(x, signs)
    self.assertEqual(xq.qvalue.shape, (4, 48, 256))
    self.assertEqual(xq.scale.shape, (4, 48, 1))

  @parameterized.parameters("xla", "pallas")
  def test_rotation_matmuls_are_not_intercepted(self, implementation):
    # `qwix.quantize_model` providers intercept these ops; the rotation's own
    # matmuls are internal to the op and must never reach them.
    self._require_backend(implementation)

    def fail(*args, **kwargs):
      del args, kwargs
      raise AssertionError("A rotation matmul was intercepted.")

    interceptor = lambda: interception.Interceptor(
        mapping={"jax.lax.dot_general": fail, "jax.numpy.einsum": fail}, id=1
    )
    k0, k1 = jax.random.split(jax.random.key(89))
    x = jax.random.normal(k0, (128, 768), dtype=jnp.bfloat16)
    signs = hk.random_signs(k1, 768, dtype=jnp.bfloat16)

    def rotate_and_quantize(a):
      y, vjp = jax.vjp(
          lambda b: hk.hadamard_rotate(
              b, signs=signs, implementation=implementation
          ),
          a,
      )
      (dx,) = vjp(y)
      xq = hk.hadamard_rotate_quantize(
          a, qtype=jnp.int8, signs=signs, implementation=implementation
      )
      return y, dx, xq

    y, dx, xq = interception.wrap_func_intercepted(
        rotate_and_quantize, interceptor, disable_jit=True
    )(x)
    self.assertLess(
        _rel_l2_error(y, hk.hadamard_rotate_reference(x, signs=signs)), 1e-2
    )
    self.assertLess(_rel_l2_error(dx, x), 1e-2)
    self.assertEqual(xq.qvalue.dtype, jnp.int8)

    w_rot = jax.random.normal(k0, (768, 128), dtype=jnp.bfloat16) * 0.05
    dnums = (((1,), (0,)), ((), ()))
    for use_orig in (False, True):
      qt_cfg = dot_general_qt.DotGeneralQtConfig(
          lhs_qtype="int8",
          rhs_qtype="int4",
          dlhs_grad_qtype="int8",
          drhs_grad_qtype="int8",
          use_original_residuals=use_orig,
      )
      out_qt, vjp_qt = interception.wrap_func_intercepted(
          lambda a, w, cfg=qt_cfg: jax.vjp(
              lambda b, u: hk.hadamard_dot_general_qt(
                  b,
                  u,
                  dnums,
                  cfg,
                  signs=signs,
                  implementation=implementation,
              ),
              a,
              w,
          ),
          interceptor,
          disable_jit=True,
      )(x, w_rot)
      dlhs_qt, drhs_qt = interception.wrap_func_intercepted(
          vjp_qt, interceptor, disable_jit=True
      )(out_qt)
      self.assertEqual(dlhs_qt.shape, x.shape)
      self.assertEqual(drhs_qt.shape, w_rot.shape)

  @parameterized.product(
      qt_case=(
          # Fused path (`_can_fuse_lhs_quantize_in_qt`): 3-D, int4 weights,
          # int8 activations and gradients, non-leading lhs batch axis.
          dict(
              is_3d=True,
              lhs_qtype="int8",
              use_original_residuals=False,
              bwd_qtype="int8",
              block_size=None,
              expect_fused=True,
          ),
          # Fused path with diagonal Block-Hadamard (`block_size=128`).
          dict(
              is_3d=True,
              lhs_qtype="int8",
              use_original_residuals=False,
              bwd_qtype="int8",
              block_size=128,
              expect_fused=True,
          ),
          # Fused path: 2-D, int4 weights, FP8 activations and gradients.
          dict(
              is_3d=False,
              lhs_qtype="float8_e4m3fn",
              use_original_residuals=False,
              bwd_qtype="float8_e5m2",
              block_size=None,
              expect_fused=True,
          ),
          # Composed path: float backward with original residuals.
          dict(
              is_3d=True,
              lhs_qtype="int8",
              use_original_residuals=True,
              bwd_qtype=None,
              block_size=None,
              expect_fused=False,
          ),
      ),
      implementation=("xla", "pallas"),
  )
  def test_hadamard_dot_general_qt(self, qt_case, implementation):
    self._require_backend(implementation)
    k0, k1, k2, k3 = jax.random.split(jax.random.key(53), 4)
    if qt_case["is_3d"]:
      # 3-D contraction with a batch axis (`lhs` (B, N, H), `rhs` (N, H, O));
      # `dot_general` places batch dimensions first so output is `(N, B, O)`.
      lhs = jax.random.normal(k0, (32, 6, 768), dtype=jnp.bfloat16)
      rhs_rot = jax.random.normal(k1, (6, 768, 128), dtype=jnp.bfloat16) * 0.05
      dout = jax.random.normal(k2, (6, 32, 128), dtype=jnp.bfloat16)
      dnums = (((2,), (1,)), ((1,), (0,)))
      rot_axis = 2
    else:
      lhs = jax.random.normal(k0, (64, 768), dtype=jnp.bfloat16)
      rhs_rot = jax.random.normal(k1, (768, 128), dtype=jnp.bfloat16) * 0.05
      dout = jax.random.normal(k2, (64, 128), dtype=jnp.bfloat16)
      dnums = (((1,), (0,)), ((), ()))
      rot_axis = 1
    signs = hk.random_signs(k3, 768, dtype=jnp.bfloat16)

    qt_config = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype=qt_case["lhs_qtype"],
        rhs_qtype="int4",
        rhs_calibration_method="rms,2.9",
        dlhs_grad_qtype=qt_case["bwd_qtype"],
        drhs_grad_qtype=qt_case["bwd_qtype"],
        use_original_residuals=qt_case["use_original_residuals"],
    )

    def loss_fused(a, w, s, g):
      out = hk.hadamard_dot_general_qt(
          a,
          w,
          dnums,
          qt_config,
          signs=s,
          block_size=qt_case["block_size"],
          implementation=implementation,
      )
      return jnp.sum(out.astype(jnp.float32) * g.astype(jnp.float32)), out

    def loss_composed(a, w, s, g):
      a_rot = hk.hadamard_rotate_reference(
          a, signs=s, block_size=qt_case["block_size"], axis=rot_axis
      )
      out = dot_general_qt.dot_general_qt(a_rot, w, dnums, qt_config)
      return jnp.sum(out.astype(jnp.float32) * g.astype(jnp.float32)), out

    with mock.patch.object(
        op, "_hadamard_qt", wraps=op._hadamard_qt
    ) as mock_fused:
      (_, out_f), (dlhs_f, drhs_f) = jax.jit(
          jax.value_and_grad(loss_fused, argnums=(0, 1), has_aux=True)
      )(lhs, rhs_rot, signs, dout)
    self.assertEqual(mock_fused.called, qt_case["expect_fused"])
    (_, out_c), (dlhs_c, drhs_c) = jax.jit(
        jax.value_and_grad(loss_composed, argnums=(0, 1), has_aux=True)
    )(lhs, rhs_rot, signs, dout)

    self.assertLess(_rel_l2_error(out_f, out_c), 2e-2)
    self.assertLess(_rel_l2_error(dlhs_f, dlhs_c), 2e-2)
    # FP8: the one-step `lhs_q` differences above meet the coarse `e5m2`
    # cotangent in `drhs` (observed 0.0263 on TPU7x, 0.0261 on TPU v5e).
    drhs_tol = 3e-2 if qt_case["bwd_qtype"] == "float8_e5m2" else 2e-2
    self.assertLess(_rel_l2_error(drhs_f, drhs_c), drhs_tol)

  @parameterized.parameters("xla", "pallas")
  def test_hadamard_dot_general_qt_grad_under_checkpoint(self, implementation):
    # `_hadamard_qt` sets `optimize_remat=True`; check that recomputing its
    # forward pass under `jax.checkpoint` gives the same value and gradients.
    self._require_backend(implementation)
    k0, k1, k2, k3 = jax.random.split(jax.random.key(59), 4)
    lhs = jax.random.normal(k0, (64, 768), dtype=jnp.bfloat16)
    rhs_rot = jax.random.normal(k1, (768, 128), dtype=jnp.bfloat16) * 0.05
    dout = jax.random.normal(k2, (64, 128), dtype=jnp.bfloat16)
    signs = hk.random_signs(k3, 768, dtype=jnp.bfloat16)
    dnums = (((1,), (0,)), ((), ()))
    qt_config = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype="int8",
        rhs_qtype="int4",
        dlhs_grad_qtype="int8",
        drhs_grad_qtype="int8",
    )

    def loss(a, w):
      out = hk.hadamard_dot_general_qt(
          a, w, dnums, qt_config, signs=signs, implementation=implementation
      )
      return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32)), out

    value_and_grad = lambda f: jax.jit(
        jax.value_and_grad(f, argnums=(0, 1), has_aux=True)
    )
    with mock.patch.object(
        op, "_hadamard_qt", wraps=op._hadamard_qt
    ) as mock_fused:
      (_, out_remat), grads_remat = value_and_grad(jax.checkpoint(loss))(
          lhs, rhs_rot
      )
    self.assertTrue(mock_fused.called)
    (_, out_direct), grads_direct = value_and_grad(loss)(lhs, rhs_rot)

    for actual, expected in zip(
        (out_remat, *grads_remat), (out_direct, *grads_direct)
    ):
      self.assertLess(_rel_l2_error(actual, expected), 1e-3)


class PallasKernelTest(parameterized.TestCase):
  """Pallas-specific tiling, VMEM multi-panel, and `shard_map` tests."""

  def setUp(self):
    super().setUp()
    if not hk.hadamard_pallas_supported():
      self.skipTest(
          f"Pallas TPU RHT is not supported on {jax.devices()[0].device_kind}."
      )

  @parameterized.parameters(
      # Explicit multi-panel `block_n < k` exercises `rot_cache` + full-row
      # scale reduction across multiple panels (including Paley-I k=3200).
      (256, 768, 128, 256),
      (192, 3200, 128, 640),
  )
  def test_explicit_multi_panel_config_matches_single_panel(
      self, m, k, block_m, block_n
  ):
    k0, k1 = jax.random.split(jax.random.key(67))
    x = jax.random.normal(k0, (m, k), dtype=jnp.bfloat16)
    signs = hk.random_signs(k1, k, dtype=jnp.bfloat16)
    cfg = hk.Config(block_m=block_m, block_n=block_n)

    y_panel = jax.jit(functools.partial(hk.hadamard_rotate, config=cfg))(
        x, signs=signs
    )
    y_single = jax.jit(hk.hadamard_rotate)(x, signs=signs)
    x_rec = jax.jit(
        functools.partial(hk.hadamard_rotate, inverse=True, config=cfg)
    )(y_panel, signs=signs)
    np.testing.assert_array_equal(np.asarray(y_panel), np.asarray(y_single))
    self.assertLess(_rel_l2_error(x_rec, x), 1e-2)

    xq_panel = jax.jit(
        functools.partial(
            hk.hadamard_rotate_quantize, qtype=jnp.int8, config=cfg
        )
    )(x, signs=signs)
    xq_single = jax.jit(
        functools.partial(hk.hadamard_rotate_quantize, qtype=jnp.int8)
    )(x, signs=signs)
    # Multi-panel computes the exact full-row absmax across all panels, so both
    # qvalue and scale must match the single-panel kernel bit-for-bit.
    np.testing.assert_array_equal(
        np.asarray(xq_panel.qvalue), np.asarray(xq_single.qvalue)
    )
    np.testing.assert_array_equal(
        np.asarray(xq_panel.scale), np.asarray(xq_single.scale)
    )

  @parameterized.parameters(
      (128, 256, jnp.bfloat16),
      (256, 2048, jnp.bfloat16),
      (128, 1024, jnp.float32),
  )
  def test_two_stage_matches_dense_single_panel(self, m, k, dtype):
    # `H_{k/128} (x) H_128 == H_k` for Sylvester orders, so the two-stage and
    # the dense `(k, k)` kernels apply the same matrix and only reorder the
    # float32 sums.
    k0, k1 = jax.random.split(jax.random.key(73))
    x = jax.random.normal(k0, (m, k), dtype=dtype)
    signs = hk.random_signs(k1, k, dtype=dtype)
    dense = hk.Config(block_m=128, block_n=k)
    tol = 1e-3 if dtype == jnp.bfloat16 else 1e-5
    rec_tol = 1e-2 if dtype == jnp.bfloat16 else 1e-5

    y_two = jax.jit(hk.hadamard_rotate)(x, signs=signs)
    y_dense = jax.jit(functools.partial(hk.hadamard_rotate, config=dense))(
        x, signs=signs
    )
    self.assertLess(_rel_l2_error(y_two, y_dense), tol)
    x_rec = jax.jit(functools.partial(hk.hadamard_rotate, inverse=True))(
        y_two, signs=signs
    )
    self.assertLess(_rel_l2_error(x_rec, x), rec_tol)

    xq_two = jax.jit(
        functools.partial(hk.hadamard_rotate_quantize, qtype=jnp.int8)
    )(x, signs=signs)
    xq_dense = jax.jit(
        functools.partial(
            hk.hadamard_rotate_quantize, qtype=jnp.int8, config=dense
        )
    )(x, signs=signs)
    # Reordered float32 sums can move a value that sits on an int8 rounding
    # boundary by one step (observed up to 2.6e-4 rel-L2 for float32 inputs on
    # TPU v5e and TPU7x; bfloat16 inputs matched bit-for-bit).
    mismatch = np.mean(np.asarray(xq_two.qvalue) != np.asarray(xq_dense.qvalue))
    self.assertLess(mismatch, 1e-3)
    self.assertLess(
        _rel_l2_error(qarray.dequantize(xq_two), qarray.dequantize(xq_dense)),
        2e-3,
    )

  @parameterized.named_parameters(
      ("sylvester_default", 512, None, True),
      ("sylvester_block_n_k", 512, 512, False),
      ("paley_default", 768, None, False),
  )
  def test_dispatches_kernel_layout(self, k, block_n, expect_two_stage):
    x = jnp.ones((128, k), dtype=jnp.bfloat16)
    cfg = hk.Config(block_m=128, block_n=block_n)
    with (
        mock.patch.object(
            kernel,
            "_hadamard_two_stage_kernel",
            wraps=kernel._hadamard_two_stage_kernel,
        ) as mock_two_stage,
        mock.patch.object(
            kernel,
            "_hadamard_single_panel_kernel",
            wraps=kernel._hadamard_single_panel_kernel,
        ) as mock_single_panel,
    ):
      jax.jit(
          functools.partial(
              hk.hadamard_rotate_quantize, qtype=jnp.int8, config=cfg
          )
      )(x)
    if expect_two_stage:
      mock_two_stage.assert_called()
      mock_single_panel.assert_not_called()
    else:
      mock_single_panel.assert_called()
      mock_two_stage.assert_not_called()

  @parameterized.parameters(
      # (m, k, dtype, expect_two_stage). The float32 Paley-I H for k=4480 is
      # 80 MB, so on 64 MiB VMEM the heuristic must split `block_n`.
      (4096, 1024, jnp.bfloat16, True),
      (4096, 3200, jnp.bfloat16, False),  # Paley-I 200 * 16
      (128, 4096, jnp.float32, True),
      (128, 4480, jnp.float32, False),
  )
  def test_heuristic_tiles_compile_and_run(self, m, k, dtype, expect_two_stage):
    k0, k1 = jax.random.split(jax.random.key(71))
    x = jax.random.normal(k0, (m, k), dtype=dtype)
    signs = hk.random_signs(k1, k, dtype=dtype)
    cfg = hk.get_heuristics_config(m, k, lhs_dtype=dtype, quantized=True)
    self.assertEqual(cfg.block_n is None, expect_two_stage)
    xq = jax.jit(
        functools.partial(hk.hadamard_rotate_quantize, qtype=jnp.float8_e4m3fn)
    )(x, signs=signs)
    ref = hk.hadamard_rotate_quantize_reference(
        x, qtype=jnp.float8_e4m3fn, signs=signs
    )
    # The bfloat16 bound is `same_quant_tol` above: on TPU the jitted reference
    # rounds its bfloat16 scales less exactly than the kernel.
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-2
    self.assertLess(
        _rel_l2_error(qarray.dequantize(xq), qarray.dequantize(ref)), tol
    )

  @parameterized.product(
      layout=(
          # (k, block_size, block_n)
          (1024, None, None),  # two-stage
          (768, None, None),  # single panel, Paley-I 12 * 64
          (768, None, 256),  # multi-panel
          (1024, 128, None),  # block
      ),
      qtype_and_bound=((jnp.int8, 127.5), (jnp.float8_e4m3fn, 448.0)),
  )
  def test_quantization_is_correctly_rounded(self, layout, qtype_and_bound):
    # Expect correctly rounded scale = bf16(absmax(y) / bound) and
    # q = bf16(y / scale), computed here in float64.
    k, block_size, block_n = layout
    qtype, bound = qtype_and_bound
    k0, k1 = jax.random.split(jax.random.key(97))
    x = jax.random.normal(k0, (256, k), dtype=jnp.bfloat16)
    signs = hk.random_signs(k1, k, dtype=jnp.bfloat16)
    cfg = hk.Config(block_m=128, block_n=block_n)

    y = jax.jit(
        functools.partial(hk.hadamard_rotate, block_size=block_size, config=cfg)
    )(x, signs=signs)
    xq = jax.jit(
        functools.partial(
            hk.hadamard_rotate_quantize,
            qtype=qtype,
            block_size=block_size,
            config=cfg,
        )
    )(x, signs=signs)

    y = np.asarray(y).astype(np.float64)
    scale = _round_to_bfloat16(np.max(np.abs(y), axis=1, keepdims=True) / bound)
    q = _round_to_bfloat16(y / scale)
    if qtype == jnp.int8:
      q = np.clip(np.round(q), -128, 127)
    expected_q = q.astype(np.float32).astype(qtype)
    np.testing.assert_array_equal(np.asarray(xq.scale, np.float64), scale)
    np.testing.assert_array_equal(
        np.asarray(xq.qvalue).astype(np.float32), expected_q.astype(np.float32)
    )

  def test_shard_map_data_parallel(self):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices), ("data",))
    pspec = jax.sharding.PartitionSpec("data", None)
    n_dev = len(devices)
    k0, k1 = jax.random.split(jax.random.key(83))
    x = jax.random.normal(k0, (n_dev * 128, 512), dtype=jnp.bfloat16)
    signs = hk.random_signs(k1, 512, dtype=jnp.bfloat16)

    @jax.jit
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(pspec, jax.sharding.PartitionSpec(None)),
        out_specs=(pspec, pspec),
    )
    def sharded_fn(x_shard, s):
      y = hk.hadamard_rotate(x_shard, signs=s)
      xq = hk.hadamard_rotate_quantize(x_shard, qtype=jnp.int8, signs=s)
      return y, qarray.dequantize(xq)

    actual_rht, actual_q = sharded_fn(x, signs)
    expected_rht = hk.hadamard_rotate_reference(x, signs=signs)
    expected_q = qarray.dequantize(
        hk.hadamard_rotate_quantize_reference(x, qtype=jnp.int8, signs=signs)
    )
    self.assertLess(_rel_l2_error(actual_rht, expected_rht), 1e-2)
    self.assertLess(_rel_l2_error(actual_q, expected_q), 2e-2)


class ValidationAndHeuristicsTest(parameterized.TestCase):

  def test_pallas_on_cpu_raises_helpful_not_implemented_error(self):
    if hk.hadamard_pallas_supported():
      self.skipTest("Runs only when no TPU target is active.")
    x = jnp.ones((128, 256), dtype=jnp.bfloat16)
    w = jnp.ones((256, 64), dtype=jnp.bfloat16)
    qt_cfg = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype="int8", rhs_qtype="int8"
    )
    with self.assertRaisesRegex(NotImplementedError, "implementation='xla'"):
      hk.hadamard_rotate(x)
    with self.assertRaisesRegex(NotImplementedError, "implementation='xla'"):
      hk.hadamard_rotate_quantize(x)
    with self.assertRaisesRegex(NotImplementedError, "implementation='xla'"):
      hk.hadamard_dot_general_qt(x, w, (((1,), (0,)), ((), ())), qt_cfg)

  def test_pallas_rejects_unaligned_rotated_dimension_before_device_check(self):
    # 64 has a Sylvester factorization (2**6) so `validate_hadamard_args`
    # passes, while `_validate_dispatch` rejects `64 % 128 != 0` even on CPU.
    x = jnp.ones((128, 64), dtype=jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      hk.hadamard_rotate(x, implementation="pallas")
    # XLA path supports 64 without error.
    y = hk.hadamard_rotate(x, implementation="xla")
    self.assertEqual(y.shape, (128, 64))

  def test_xla_rejects_pallas_config(self):
    x = jnp.ones((128, 256), dtype=jnp.bfloat16)
    cfg = hk.Config(block_m=128)
    with self.assertRaisesRegex(ValueError, "implementation='pallas'"):
      hk.hadamard_rotate(x, config=cfg, implementation="xla")

  def test_rejects_invalid_configs_and_arguments(self):
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      hk.Config(block_m=64)
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      hk.Config(block_m=128, block_n=192)
    with self.assertRaisesRegex(ValueError, "block_n must be None"):
      kernel.validate_pallas_config(
          512, block_size=128, config=hk.Config(block_m=128, block_n=128)
      )
    with self.assertRaisesRegex(ValueError, "divisible by config.block_n"):
      kernel.validate_pallas_config(
          768, block_size=None, config=hk.Config(block_m=128, block_n=512)
      )

    x = jnp.ones((128, 256), dtype=jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "Unsupported qtype"):
      hk.hadamard_rotate_quantize(x, qtype="nf4", implementation="xla")
    with self.assertRaisesRegex(ValueError, "signs must have shape"):
      hk.hadamard_rotate(x, signs=jnp.ones((128,)), implementation="xla")
    with self.assertRaisesRegex(ValueError, "divisible by positive block_size"):
      hk.hadamard_rotate(x, block_size=192, implementation="xla")
    with self.assertRaisesRegex(ValueError, "implementation must be"):
      bad_kwargs = {"implementation": "triton"}
      hk.hadamard_rotate(x, **bad_kwargs)
    x3 = jnp.ones((4, 16, 256), dtype=jnp.bfloat16)
    w3 = jnp.ones((16, 256, 32), dtype=jnp.bfloat16)
    qt_cfg = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype="int8", rhs_qtype="int8"
    )
    with self.assertRaisesRegex(ValueError, "one contracting axis"):
      hk.hadamard_dot_general_qt(
          x3, w3, (((1, 2), (0, 1)), ((), ())), qt_cfg, implementation="xla"
      )

  def test_hadamard_pallas_supported_queries_tpu_info(self):
    fake_tpu_info = mock.Mock()
    fake_tpu_info.is_matmul_supported.return_value = True
    with mock.patch.object(
        kernel.pltpu, "get_tpu_info", return_value=fake_tpu_info
    ):
      self.assertTrue(hk.hadamard_pallas_supported(jnp.bfloat16))
      fake_tpu_info.is_matmul_supported.assert_called_once_with(
          jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.bfloat16)
      )
    with mock.patch.object(
        kernel.pltpu, "get_tpu_info", side_effect=ValueError("no TPU")
    ):
      self.assertFalse(hk.hadamard_pallas_supported(jnp.bfloat16))

  @parameterized.parameters(
      # (k, block_size, block_n, expected_layout)
      (1024, None, None, "two_stage"),
      (1024, None, 1024, "single_panel"),
      (3200, None, None, "single_panel"),  # Paley-I 200 * 16
      (3200, None, 640, "multi_panel"),
      (1024, 128, None, "block"),
      (128, None, None, "single_panel"),  # one block: nothing to butterfly
  )
  def test_choose_layout(self, k, block_size, block_n, expected_layout):
    self.assertEqual(
        kernel._choose_layout(k, block_size=block_size, block_n=block_n),
        expected_layout,
    )

  @parameterized.parameters(
      # (m, k, block_size, expected_block_m)
      (24576, 1024, 128, 1024),
      (1536, 1024, 128, 512),  # 1024 would pad 1536 rows to 2048
      (1536, 1024, None, 512),
      (1152, 3200, None, 128),  # 9 * 128: only 128 divides
      (192, 768, None, 256),  # no aligned divisor: pads to 256
  )
  def test_heuristics_block_m_prefers_divisors_of_m(
      self, m, k, block_size, expected_block_m
  ):
    with mock.patch.object(
        kernel, "default_vmem_limit_bytes", return_value=64 * 1024 * 1024
    ):
      cfg = hk.get_heuristics_config(
          m, k, block_size=block_size, quantized=True
      )
    self.assertEqual(cfg.block_m, expected_block_m)

  @parameterized.parameters(
      # (m, k, dtype, vmem_mib, expected_layout)
      (4096, 3200, jnp.bfloat16, 64, "single_panel"),  # Paley-I 200 * 16
      (4096, 4096, jnp.bfloat16, 64, "two_stage"),
      (4096, 4096, jnp.float32, 64, "two_stage"),
      (4096, 4480, jnp.float32, 64, "multi_panel"),  # 80 MB H splits panels
      (4096, 4480, jnp.float32, 128, "single_panel"),  # fits on 128 MiB VMEM
      (4096, 3200, jnp.bfloat16, 16, "multi_panel"),  # 16 MiB picks block_n=640
  )
  def test_heuristics_fit_vmem_budget_across_tpu_generations(
      self, m, k, dtype, vmem_mib, expected_layout
  ):
    with mock.patch.object(
        kernel, "default_vmem_limit_bytes", return_value=vmem_mib * 1024 * 1024
    ):
      cfg = hk.get_heuristics_config(m, k, lhs_dtype=dtype, quantized=True)
    self.assertEqual(
        kernel._choose_layout(k, block_size=None, block_n=cfg.block_n),
        expected_layout,
    )

  @parameterized.parameters(
      ("TPU v5 lite", 128 * 1024 * 1024),
      ("TPU7x", 64 * 1024 * 1024),
  )
  def test_compile_only_targets_abstract_mesh_device(
      self, device_kind, vmem_bytes
  ):
    # Tracing for a TPU target from another host, e.g. CPU: the support check
    # and the VMEM budget follow the abstract mesh, not the local device.
    device = jax.sharding.AbstractDevice(
        device_kind=device_kind, num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh((1,), ("x",), abstract_device=device)
    with jax.sharding.use_abstract_mesh(mesh):
      self.assertTrue(hk.hadamard_pallas_supported(jnp.bfloat16))
      self.assertEqual(kernel.default_vmem_limit_bytes(), vmem_bytes)


if __name__ == "__main__":
  absltest.main()
