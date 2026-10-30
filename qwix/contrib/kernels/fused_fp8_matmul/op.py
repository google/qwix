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

"""The user-facing fused FP8 matmul op.

Example:

  from qwix.contrib.kernels import fused_fp8_matmul as ffm

  out = ffm.fused_fp8_matmul(activations, weights)   # (m, k) @ (k, n)
  grads = jax.grad(loss)(activations, weights)       # differentiable

`fused_fp8_matmul` is the training entry point: a `jax.custom_vjp` that runs
the forward kernel and, in the backward pass, the dLHS and dRHS kernels,
accepting a plain or pre-quantized weight. Callers that want the kernels
themselves, with explicit tile sizes, can import them directly from
`forward_kernel.py`, `dlhs_kernel.py` and `drhs_kernel.py`; nothing here is
needed for that.

Chip support: the kernels run on any TPU whose MXU takes FP8 operands. The
tile heuristics were measured on TPU7x and are only tuned for it; on other
chips they are a starting point, and `config` overrides them.

See `reference.py` for the numerics and `forward_kernel.py` for the kernel,
its tile heuristics and `Config`.
"""

import dataclasses
import functools
from typing import Literal

import jax
import jax.numpy as jnp
from qwix.contrib.kernels.fused_fp8_matmul import dlhs_kernel
from qwix.contrib.kernels.fused_fp8_matmul import drhs_kernel
from qwix.contrib.kernels.fused_fp8_matmul import forward_kernel
from qwix.contrib.kernels.fused_fp8_matmul import reference

QArray = reference.QArray
Residuals = reference.Residuals
Config = forward_kernel.Config


@dataclasses.dataclass(frozen=True)
class _Static:
  """Trace-time constants threaded through the `custom_vjp` as a nondiff arg.

  For the dRHS kernel (`drhs = lhs^T @ dout`, contracting over `m`): each m
  block of the cotangent is quantized against its own column scales, so
  `block_m == m` gives one scale per column and needs no accumulator.

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

  The forward kernel quantizes `lhs` per full row only when it takes `k` as a
  single block, which the default heuristics do for `k <= 4096` with
  `k % 256 == 0`. Otherwise it splits `k` and quantizes each row per k block.
  Likewise the dLHS kernel quantizes the cotangent per n block when it splits
  `n`, and the dRHS kernel per m block when it splits `m` (always, at training
  sizes). In those cases a kernel agrees with the plain-XLA reference to within
  FP8 quantization noise rather than exactly.

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
    implementation: `"pallas"` runs the Pallas kernels and requires a TPU whose
      MXU takes FP8 operands; `m`, `k` and `n` must be multiples of 128. The
      kernels and their tile heuristics were measured on TPU7x; other
      FP8-capable chips run them untuned unless configs are given. `"xla"` runs
      the plain-XLA reference on any backend.

  Returns:
    `lhs @ rhs` of shape `(m, n)` in `lhs.dtype`.

  Raises:
    ValueError: On shape or dtype mismatches, a dimension that is not a
      multiple of 128 on the Pallas path, or a block size that does not divide
      the shape.
    NotImplementedError: With `implementation="pallas"` on a device without an
      FP8 MXU.
  """
  reference.validate_operands(lhs, rhs)
  if implementation == "pallas":
    m, k = lhs.shape
    n = rhs.shape[1]
    align = forward_kernel.BLOCK_ALIGNMENT
    if m % align or k % align or n % align:
      raise ValueError(
          "The Pallas path tiles every dimension in multiples of"
          f" {align}; got (m, k, n) = ({m}, {k}, {n}). Pad the operands, or"
          " pass implementation='xla'."
      )
    device = jax.devices()[0]
    if not forward_kernel.fp8_matmul_supported_on(device):
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
  config = static.config or forward_kernel.get_heuristics_config(
      m, k, n, lhs_dtype=lhs.dtype
  )
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
    config = static.dlhs_config or dlhs_kernel.get_dlhs_heuristics_config(
        m, k, n, lhs_dtype=dout.dtype
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
  config = static.drhs_config or drhs_kernel.get_drhs_heuristics_config(
      m, k, n, lhs_dtype=dout.dtype
  )
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
