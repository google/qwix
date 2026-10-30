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

"""Numerics and plain-XLA reference for the fused FP8 matmul.

The fused FP8 matmul computes `lhs @ rhs` with both operands in
`float8_e4m3fn` on the MXU:

* `lhs` (activations, shape `(m, k)`) is quantized inside the op with one
  absmax scale per row. Row scales are the layout that lets the scale be
  factored out of the contraction over `k`.
* `rhs` (weights, shape `(k, n)`) is quantized with one absmax scale per output
  column. It can be passed either already quantized, as a `qwix.QArray`, or as
  a plain array that the op quantizes per column before the matmul.

The output is `(lhs_q @ rhs_q) * (lhs_scale * rhs_scale)` cast to `lhs.dtype`.

The backward pass uses FP8 training numerics: the activation gradient
`dlhs = dout @ rhs^T` quantizes the cotangent to FP8 with one scale per row
(after folding in the weight's column scales), and the weight gradient
`drhs = lhs^T @ dout` quantizes it with one scale per column (after folding in
the activation's row scales), so both run on the FP8 MXU like the forward pass.

Everything in this module is plain `jax.numpy`. The functions here are the
reference the Pallas kernels are tested against, and the fallback used where no
kernel applies. The activation quantizer is shared with the kernels, which call
it on VMEM tiles. A kernel reproduces these functions exactly when its
contracting axis is a single block; where it tiles the contracting axis it
quantizes per block and agrees with the reference to within FP8 quantization
noise.

TODO(stellasyan): Replace the activation quantizer with the Qwix in-kernel
quantization APIs once they land.
"""

import jax
import jax.numpy as jnp
from qwix._src.core import qarray

QArray = qarray.QArray

FP8_DTYPE = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8_DTYPE).max)  # 448.0

# `(lhs_q, lhs_scale, rhs_q, rhs_scale)`: the FP8 operands and their scales, so
# the backward pass does not have to re-quantize anything.
#   lhs_q:     `(m, k)` float8_e4m3fn
#   lhs_scale: `(m, 1)` float32
#   rhs_q:     `(k, n)` float8_e4m3fn
#   rhs_scale: `(n,)`   float32
Residuals = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def quantize_rows_fp8(
    x: jax.Array, eps: float = 1e-7
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `x` to FP8 with one absmax scale per row (last axis).

  This is the in-kernel activation quantizer as well as the reference one: the
  forward kernel calls it on each VMEM tile. The scaling deliberately stays in
  `x`'s own dtype. Writing the obvious `x / scale` with a float32 scale
  promotes the whole tile to float32, which at the tile sizes the kernel wants
  is an 8 MB temporary against 64 MiB of VMEM and is what made the large-tile
  configurations fail to compile. Only the tiny per-row scale is float32. There
  is no clamp: scaling by exactly `FP8_MAX / absmax` puts every element inside
  `[-FP8_MAX, FP8_MAX]` by construction.

  Args:
    x: The array to quantize.
    eps: Lower bound on the per-row absmax, so an all-zero row gets a finite
      scale.

  Returns:
    `(xq, scale)` with `scale` of shape `x.shape[:-1] + (1,)` in float32 such
    that `x ~= xq * scale`.
  """
  max_abs = jnp.maximum(jnp.max(jnp.abs(x), axis=-1, keepdims=True), eps)
  scale = (max_abs / FP8_MAX).astype(jnp.float32)
  inv = (FP8_MAX / max_abs).astype(x.dtype)
  return (x * inv).astype(FP8_DTYPE), scale


def quantize_cols_fp8(
    g: jax.Array, eps: float = 1e-7
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `g` to FP8 with one absmax scale per column (axis 0 reduced).

  The column-wise twin of `quantize_rows_fp8`, used for the cotangent in the
  weight gradient (weights themselves go through `qwix.quantize`, see
  `rhs_as_fp8`). The scaling stays in `g`'s dtype; the dRHS kernel calls this
  on VMEM blocks.

  Args:
    g: The array to quantize.
    eps: Lower bound on the per-column absmax, so an all-zero column gets a
      finite scale.

  Returns:
    `(gq, scale)` with `scale` of shape `g.shape[1:]` in float32 such that
    `g ~= gq * scale`.
  """
  max_abs = jnp.maximum(jnp.max(jnp.abs(g), axis=0), eps)
  scale = (max_abs / FP8_MAX).astype(jnp.float32)
  inv = (FP8_MAX / max_abs).astype(g.dtype)
  return (g * inv).astype(FP8_DTYPE), scale


def rhs_as_fp8(rhs: jax.Array | QArray) -> tuple[jax.Array, jax.Array]:
  """Returns `(rhs_q, rhs_scale)` with `rhs_scale` of shape `(n,)`, float32.

  A plain weight is quantized with `qwix.quantize`, one absmax scale per output
  column, so a weight quantized ahead of time and one quantized here take the
  same path. `qwix.quantize` carries the scale in the weight's dtype by default
  (so `bfloat16` for a `bfloat16` weight); the kernels take float32 scales, so
  it is widened here.

  Args:
    rhs: The weight, either a plain array or an FP8 `QArray` with one scale per
      output column.
  """
  if not isinstance(rhs, QArray):
    rhs = qarray.quantize_api(rhs, FP8_DTYPE, channelwise_axes=(1,))
  return rhs.qvalue, rhs.scale.reshape(-1).astype(jnp.float32)


def fused_fp8_matmul_reference(
    lhs: jax.Array, rhs: jax.Array | QArray
) -> tuple[jax.Array, Residuals]:
  """XLA reference for the forward pass. Returns `(out, residuals)`."""
  lhs_q, lhs_scale = quantize_rows_fp8(lhs)
  rhs_q, rhs_scale = rhs_as_fp8(rhs)
  acc = jnp.matmul(lhs_q, rhs_q, preferred_element_type=jnp.float32)
  out = (acc * (lhs_scale * rhs_scale)).astype(lhs.dtype)
  return out, (lhs_q, lhs_scale, rhs_q, rhs_scale)


def fused_fp8_dlhs_reference(
    dout: jax.Array, rhs_q: jax.Array, rhs_scale: jax.Array
) -> jax.Array:
  """XLA reference for the activation gradient `dlhs = dout @ rhs^T` in FP8.

  The contraction runs over `n`, so the weight's per-column scale cannot be
  factored out of it: it is folded into the cotangent first. The scaled
  cotangent is then quantized with one scale per row, exactly like the
  activation in the forward pass, and that row scale is applied to the FP8
  product.

  Args:
    dout: Cotangent of the output, shape `(m, n)`, `bfloat16` or `float32`.
    rhs_q: The `(k, n)` FP8 weight.
    rhs_scale: The `(n,)` float32 per-column weight scales.

  Returns:
    `dlhs` of shape `(m, k)` in `dout.dtype`.
  """
  dout_q, dout_scale = quantize_rows_fp8(dout * rhs_scale.astype(dout.dtype))
  # Contract the last dims directly rather than spelling it `rhs_q.T`: on TPUs
  # without an FP8 MXU (v6e), XLA's handling of a transposed FP8 operand made
  # the jitted VJP take minutes to compile.
  acc = jax.lax.dot_general(
      dout_q,
      rhs_q,
      dimension_numbers=(((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  return (acc * dout_scale).astype(dout.dtype)


def fused_fp8_drhs_reference(
    lhs_q: jax.Array, lhs_scale: jax.Array, dout: jax.Array
) -> jax.Array:
  """XLA reference for the weight gradient `drhs = lhs^T @ dout` in FP8.

  The contraction runs over `m`, so the activation's per-row scale cannot be
  factored out of it: it is folded into the cotangent first. The scaled
  cotangent is then quantized with one scale per column, like the weight in
  the forward pass, and that column scale is applied to the FP8 product of the
  activation residual and the quantized cotangent.

  Args:
    lhs_q: The `(m, k)` FP8 activation.
    lhs_scale: The `(m, 1)` float32 per-row activation scales.
    dout: Cotangent of the output, shape `(m, n)`, `bfloat16` or `float32`.

  Returns:
    `drhs` of shape `(k, n)` in `dout.dtype`.
  """
  dout_q, dout_scale = quantize_cols_fp8(dout * lhs_scale.astype(dout.dtype))
  acc = jax.lax.dot_general(
      lhs_q,
      dout_q,
      dimension_numbers=(((0,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  return (acc * dout_scale).astype(dout.dtype)


def fused_fp8_matmul_vjp_reference(
    residuals: Residuals, dout: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """XLA reference VJP from the FP8 residuals.

  Args:
    residuals: `(lhs_q, lhs_scale, rhs_q, rhs_scale)` from the forward pass.
    dout: Cotangent of the output, shape `(m, n)`.

  Returns:
    `(dlhs, drhs)` in `dout.dtype`, see `fused_fp8_dlhs_reference` and
    `fused_fp8_drhs_reference`.
  """
  lhs_q, lhs_scale, rhs_q, rhs_scale = residuals
  return (
      fused_fp8_dlhs_reference(dout, rhs_q, rhs_scale),
      fused_fp8_drhs_reference(lhs_q, lhs_scale, dout),
  )


def validate_operands(lhs: jax.Array, rhs: jax.Array | QArray) -> None:
  """Checks shapes and dtypes, raising `ValueError` with a clear message.

  Args:
    lhs: Activations of shape `(m, k)` in `bfloat16` or `float32`.
    rhs: Weights of shape `(k, n)`. Either a `qwix.QArray` already quantized to
      `float8_e4m3fn` with one scale per output column (scale shape `(1, n)`),
      or a plain floating-point array.
  """
  if lhs.ndim != 2:
    raise ValueError(f"lhs must be 2-D (m, k), got shape {lhs.shape}.")
  if rhs.ndim != 2:
    raise ValueError(f"rhs must be 2-D (k, n), got shape {rhs.shape}.")
  m, k = lhs.shape
  k_rhs, n = rhs.shape
  if k_rhs != k:
    raise ValueError(
        f"Contracting dims differ: lhs is ({m}, {k}), rhs is ({k_rhs}, {n})."
    )
  if lhs.dtype not in (jnp.bfloat16, jnp.float32):
    raise ValueError(f"lhs must be bfloat16 or float32, got {lhs.dtype}.")
  if isinstance(rhs, QArray):
    if rhs.qvalue.dtype != FP8_DTYPE:
      raise ValueError(
          f"rhs must be quantized to {FP8_DTYPE}, got {rhs.qtype}."
      )
    if rhs.zero_point is not None:
      raise ValueError("rhs must be symmetrically quantized (no zero point).")
    if rhs.scale.shape not in ((1, n), (n,)):
      raise ValueError(
          "rhs must have one scale per output column, i.e. scale shape"
          f" (1, {n}); got {rhs.scale.shape} for a ({k}, {n}) weight."
      )
  elif not jnp.issubdtype(rhs.dtype, jnp.floating):
    raise ValueError(f"rhs must be floating point, got {rhs.dtype}.")
  elif rhs.dtype == FP8_DTYPE:
    raise ValueError(
        "An already-quantized rhs must be passed as a qwix.QArray so its"
        " scale is known."
    )
