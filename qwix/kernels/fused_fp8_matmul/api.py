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

"""User-facing fused FP8 matmul.

Example:

  from qwix.kernels import fused_fp8_matmul as ffm

  out = ffm.fused_fp8_matmul(activations, weights)   # (m, k) @ (k, n)
  grads = jax.grad(loss)(activations, weights)       # differentiable

See `reference.py` for the numerics and `forward_kernel.py` for the kernel.
"""

import dataclasses
import functools
from typing import Literal

import jax
import jax.numpy as jnp
from qwix.kernels.fused_fp8_matmul import dlhs_kernel
from qwix.kernels.fused_fp8_matmul import drhs_kernel
from qwix.kernels.fused_fp8_matmul import forward_kernel
from qwix.kernels.fused_fp8_matmul import reference

QArray = reference.QArray
Residuals = reference.Residuals

# Pallas blocks are (sublane, lane) tiled; 128 is the lane width.
_BLOCK_ALIGNMENT = 128


@dataclasses.dataclass(frozen=True)
class Config:
  """Tiling for one of the Pallas kernels: block sizes along `m`, `k`, `n`.

  The axes are those of the matmul `lhs (m, k) @ rhs (k, n)`, whichever kernel
  the config is for. Each block size must divide its dimension.

  For the forward kernel: `block_k == k` runs without an accumulator and lets
  the kernel emit the quantized activations as residuals for the backward
  pass; `block_n == n` avoids re-fetching each activation tile once per n
  block.

  For the dLHS kernel (`dlhs = dout @ rhs^T`, contracting over `n`):
  `block_n == n` quantizes each cotangent row against a single scale and needs
  no accumulator; with `block_k < k` the kernel caches the quantized cotangent
  across k blocks.

  For the dRHS kernel (`drhs = lhs^T @ dout`, contracting over `m`): each m
  block of the cotangent is quantized against its own column scales, so
  `block_m == m` gives one scale per column and needs no accumulator;
  `block_k` must also be a multiple of 128 (the lane axis of the FP8
  activation block).

  Attributes:
    block_m: Block size along `m`.
    block_k: Block size along `k`.
    block_n: Block size along `n`.
  """

  block_m: int
  block_k: int
  block_n: int

  def __post_init__(self):
    for name in ("block_m", "block_k", "block_n"):
      value = getattr(self, name)
      if value < _BLOCK_ALIGNMENT or value % _BLOCK_ALIGNMENT:
        raise ValueError(
            f"{name} must be a positive multiple of {_BLOCK_ALIGNMENT}, got"
            f" {value}."
        )


def get_heuristics_config(
    m: int, k: int, n: int, *, lhs_dtype: jax.typing.DTypeLike = jnp.bfloat16
) -> Config:
  """Returns the tiling the measured heuristics pick for `(m, k, n)`.

  Args:
    m: The row dimension of `lhs`.
    k: The contracting dimension.
    n: The column dimension of `rhs`.
    lhs_dtype: The activation dtype. A `float32` activation (and so output)
      block takes twice the VMEM of a `bfloat16` one, which can rule out the
      largest row block.
  """
  bk = forward_kernel.default_block_k(k)
  bm, bn = forward_kernel.fit_forward_tiles(
      m, k, n, bk, lhs_itemsize=jnp.dtype(lhs_dtype).itemsize
  )
  return Config(block_m=bm, block_k=bk, block_n=bn)


def get_dlhs_heuristics_config(
    m: int, k: int, n: int, *, dtype: jax.typing.DTypeLike = jnp.bfloat16
) -> Config:
  """Returns the tiling the measured heuristics pick for the dLHS kernel.

  Args:
    m: The row dimension of the cotangent and of `dlhs`.
    k: The column dimension of `dlhs`.
    n: The contracting dimension (columns of the cotangent).
    dtype: The cotangent dtype, which is `lhs.dtype`. See
      `get_heuristics_config` for why it matters.
  """
  bm, bk, bn = dlhs_kernel.fit_dlhs_tiles(
      m, k, n, cotangent_itemsize=jnp.dtype(dtype).itemsize
  )
  return Config(block_m=bm, block_k=bk, block_n=bn)


def get_drhs_heuristics_config(m: int, k: int, n: int) -> Config:
  """Returns the tiling the measured heuristics pick for the dRHS kernel.

  Args:
    m: The contracting dimension (rows of the activation and cotangent).
    k: The row dimension of `drhs`.
    n: The column dimension of `drhs`.
  """
  bm, bk, bn = drhs_kernel.fit_drhs_tiles(m, k, n)
  return Config(block_m=bm, block_k=bk, block_n=bn)


def fp8_matmul_supported_on(device: jax.Device) -> bool:
  """Whether the Pallas kernel can run on `device` (a TPU with an FP8 MXU)."""
  return forward_kernel.fp8_matmul_supported_on(device)


@dataclasses.dataclass(frozen=True)
class _Static:
  """Trace-time constants threaded through the `custom_vjp` as a nondiff arg.

  Attributes:
    implementation: `"pallas"` or `"xla"`.
    config: The forward kernel tiling; `None` means the heuristics choose it.
    dlhs_config: The dLHS kernel tiling; `None` means the heuristics choose it.
    drhs_config: The dRHS kernel tiling; `None` means the heuristics choose it.
    rhs_dtype: The dtype of a plain `rhs`, which its gradient must be cast to;
      `None` when `rhs` is a `QArray`, which gets no gradient.
  """

  implementation: str
  config: Config | None
  dlhs_config: Config | None
  drhs_config: Config | None
  rhs_dtype: jnp.dtype | None


def fused_fp8_matmul(
    lhs: jax.Array,
    rhs: jax.Array | QArray,
    *,
    config: Config | None = None,
    dlhs_config: Config | None = None,
    drhs_config: Config | None = None,
    implementation: Literal["pallas", "xla"] = "pallas",
) -> jax.Array:
  """Fused FP8 matmul: `lhs @ rhs` with in-op row quantization of `lhs`.

  Both operands reach the MXU in `float8_e4m3fn`. `lhs` is quantized inside the
  op with one absmax scale per row; `rhs` with one scale per output column
  (either here, or beforehand by `qwix.quantize`). The result is
  `(lhs_q @ rhs_q) * (lhs_scale * rhs_scale)` in `lhs.dtype`.

  The function is differentiable with respect to both operands. The backward
  pass uses FP8 training numerics, see
  `reference.fused_fp8_matmul_vjp_reference`.
  A `QArray` `rhs` receives a zero gradient: the gradient of a pre-quantized
  weight is not defined by this op; pass the unquantized weight to train it.

  Args:
    lhs: Activations of shape `(m, k)` in `bfloat16` or `float32`.
    rhs: Weights of shape `(k, n)`. Either a `qwix.QArray` already quantized to
      `float8_e4m3fn` with one scale per output column (scale shape `(1, n)`),
      or a plain floating-point array which the op quantizes per column before
      the matmul.
    config: Tiling for the Pallas forward kernel. `None` picks the measured
      heuristics for the shape. Only valid with `implementation="pallas"`.
    dlhs_config: Tiling for the Pallas dLHS kernel in the backward pass. `None`
      picks the measured heuristics. Only valid with `implementation="pallas"`.
    drhs_config: Tiling for the Pallas dRHS kernel in the backward pass. `None`
      picks the measured heuristics. Only valid with `implementation="pallas"`.
    implementation: `"pallas"` runs the TPU kernel and requires a TPU whose MXU
      takes FP8 operands (TPU7x and later); `"xla"` runs the plain-XLA reference
      with the same numerics on any backend.

  Returns:
    `lhs @ rhs` of shape `(m, n)` in `lhs.dtype`.

  Raises:
    ValueError: On shape or dtype mismatches, or a block size that does not
      divide the shape.
    NotImplementedError: With `implementation="pallas"` on a device without an
      FP8 MXU.
  """
  reference.validate_operands(lhs, rhs)
  if implementation == "pallas":
    device = jax.devices()[0]
    if not fp8_matmul_supported_on(device):
      raise NotImplementedError(
          f"The fused FP8 matmul kernel needs a TPU with an FP8 MXU; {device}"
          " has none. Pass implementation='xla' for the plain-XLA reference."
      )
  elif implementation == "xla":
    if config is not None or dlhs_config is not None or drhs_config is not None:
      raise ValueError(
          "config, dlhs_config and drhs_config only apply to"
          " implementation='pallas'."
      )
  else:
    raise ValueError(
        f"implementation must be 'pallas' or 'xla', got {implementation!r}."
    )
  rhs_dtype = None if isinstance(rhs, QArray) else rhs.dtype
  static = _Static(implementation, config, dlhs_config, drhs_config, rhs_dtype)
  return _fused_fp8_matmul(static, lhs, rhs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _fused_fp8_matmul(
    static: _Static, lhs: jax.Array, rhs: jax.Array | QArray
) -> jax.Array:
  # The primal: used when nothing differentiates through the call, so the
  # kernel is not asked to write the residuals out.
  return _forward(static, lhs, rhs, with_residuals=False)[0]


def _fwd(
    static: _Static, lhs: jax.Array, rhs: jax.Array | QArray
) -> tuple[jax.Array, Residuals]:
  out, residuals = _forward(static, lhs, rhs, with_residuals=True)
  assert residuals is not None
  return out, residuals


def _forward(
    static: _Static,
    lhs: jax.Array,
    rhs: jax.Array | QArray,
    *,
    with_residuals: bool,
) -> tuple[jax.Array, Residuals | None]:
  """Forward pass. Returns `(out, residuals)`; residuals only if asked for."""
  if static.implementation == "xla":
    out, residuals = reference.fused_fp8_matmul_reference(lhs, rhs)
    return out, (residuals if with_residuals else None)

  m, k = lhs.shape
  n = rhs.shape[1]
  config = static.config or get_heuristics_config(m, k, n, lhs_dtype=lhs.dtype)
  rhs_q, rhs_scale = reference.rhs_as_fp8(rhs)
  run = functools.partial(
      forward_kernel.fused_fp8_matmul_pallas,
      lhs,
      rhs_q,
      rhs_scale,
      bm=config.block_m,
      bk=config.block_k,
      bn=config.block_n,
  )
  if not with_residuals:
    return run(return_xq=False), None
  # The kernel can only emit the quantized activations as residuals when it
  # quantizes each row against a single scale, i.e. with one k block. The
  # heuristics pick `block_k == k` whenever it fits; for a hand-picked config
  # with smaller k blocks, keep the configured tiling and recompute the
  # residuals in XLA instead (one extra read of `lhs`).
  if config.block_k == k:
    out, lhs_q, lhs_scale = run(return_xq=True)
  else:
    out = run(return_xq=False)
    lhs_q, lhs_scale = reference.quantize_rows_fp8(lhs)
  return out, (lhs_q, lhs_scale, rhs_q, rhs_scale)


def _bwd(
    static: _Static, residuals: Residuals, dout: jax.Array
) -> tuple[jax.Array, jax.Array | None]:
  """Backward pass. Returns `(dlhs, drhs)`; `drhs` is `None` for a `QArray`."""
  lhs_q, lhs_scale, rhs_q, rhs_scale = residuals
  m, n = dout.shape
  k = rhs_q.shape[0]
  # `dout` has the forward output's dtype, which is `lhs.dtype`, so `dlhs`
  # comes out in the right dtype without a cast.
  if static.implementation == "xla":
    dlhs = reference.fused_fp8_dlhs_reference(dout, rhs_q, rhs_scale)
  else:
    config = static.dlhs_config or get_dlhs_heuristics_config(
        m, k, n, dtype=dout.dtype
    )
    dlhs = dlhs_kernel.fused_fp8_dlhs_pallas(
        dout,
        rhs_q,
        rhs_scale,
        bm=config.block_m,
        bk=config.block_k,
        bn=config.block_n,
    )
  if static.rhs_dtype is None:
    return dlhs, None
  if static.implementation == "xla":
    drhs = reference.fused_fp8_drhs_reference(lhs_q, lhs_scale, dout)
    return dlhs, drhs.astype(static.rhs_dtype)
  config = static.drhs_config or get_drhs_heuristics_config(m, k, n)
  drhs = drhs_kernel.fused_fp8_drhs_pallas(
      lhs_q,
      lhs_scale,
      dout,
      bm=config.block_m,
      bk=config.block_k,
      bn=config.block_n,
      out_dtype=static.rhs_dtype,
  )
  return dlhs, drhs


# `optimize_remat` lets JAX drop the residual outputs when the forward pass is
# recomputed under `jax.checkpoint`, rather than keeping a second copy of them.
_fused_fp8_matmul.defvjp(_fwd, _bwd, optimize_remat=True)
