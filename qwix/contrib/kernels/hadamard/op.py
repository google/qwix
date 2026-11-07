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

"""User-facing Hadamard rotation and VMEM-fused quantization ops.

Example:

  from qwix.contrib.kernels import hadamard as hk

  signs = hk.random_signs(key, x.shape[-1])
  y = hk.hadamard_rotate(x, signs=signs)  # differentiable
  x_q = hk.hadamard_rotate_quantize(x, qtype=jnp.int8, signs=signs)  # QArray

Provides three entry points:

* `hadamard_rotate`: differentiable forward/inverse (randomized) Hadamard
  rotation (`@jax.custom_vjp` whose backward pass runs the inverse rotation
  without saving activation residuals).
* `hadamard_rotate_quantize`: single-pass VMEM-fused Hadamard rotation +
  row-wise symmetric quantization returning a `qwix.QArray` ready for
  `qwix.dot_general` or `qwix.einsum`.
* `hadamard_dot_general_qt`: QAT `dot_general_qt` entry point that fuses LHS
  Hadamard rotation + LHS symmetric quantization in VMEM during the forward pass
  and applies the inverse rotation to `dlhs` in the backward pass.

The rotation is internal to these ops: its matmuls are never intercepted by a
quantization provider, e.g. inside `qwix.quantize_model`.

Relationship to `qwix.contrib.kernels.fused_hadamard_quantize`:
`fused_hadamard_quantize` is a separate forward-only kernel for explicit Pallas
grid tiling (`bm, bn, sm, sn`) that computes **2-D block-level scales** of shape
`(sm, sn, 1, 1)` as raw `(xq, scale)` arrays (paired with
`fused_hadamard_quantized_matmul`). This `hadamard` package instead targets
Qwix's standard **row-wise (per-channel) `QArray`** contract and
`dot_general_qt` QAT training, adding Rademacher sign flips, automatic tile
heuristics, non-power-of-two Paley-I orders, FP8/INT8 qtypes, and custom-VJP
backward passes.

Chip support: the Pallas kernels run on any TPU whose MXU takes the input
dtype. Powers of two `k >= 256` use a two-stage layout (see `kernel.py`); other
orders multiply by the dense `(k, k)` matrix, which is MXU-bound and can be
slower than a plain-XLA rotation. The tile heuristics were tuned on TPU v5e and
TPU7x; `config` overrides them.
"""

import dataclasses
import functools
import math
from typing import Literal

import jax
import jax.numpy as jnp
from qwix._src import interception
from qwix._src.core import dot_general_qt
from qwix._src.core import numerics
from qwix._src.core import qarray
from qwix.contrib.kernels.hadamard import kernel
from qwix.contrib.kernels.hadamard import reference

Config = kernel.Config
QArray = qarray.QArray
Implementation = Literal["pallas", "xla"]


@dataclasses.dataclass(frozen=True)
class _HadamardStatic:
  """Static metadata threaded through `_hadamard_rotate` as a nondiff arg."""

  axis: int
  block_size: int | None
  inverse: bool
  config: Config | None
  implementation: Implementation


@dataclasses.dataclass(frozen=True)
class _HadamardQtStatic:
  """Static metadata threaded through `_hadamard_qt` as a nondiff argument."""

  dimension_numbers: jax.lax.DotDimensionNumbers
  axis: int
  lhs_qdtype: jnp.dtype
  block_size: int | None
  config: Config | None
  implementation: Implementation


def _validate_dispatch(
    x: jax.Array,
    *,
    norm_axis: int,
    block_size: int | None,
    config: Config | None,
    implementation: Implementation,
) -> None:
  """Validates `implementation`, `config`, alignment, and TPU support."""
  if implementation == "pallas":
    kernel.validate_pallas_config(
        x.shape[norm_axis], block_size=block_size, config=config
    )
    if not kernel.hadamard_pallas_supported(x.dtype):
      raise NotImplementedError(
          "The Pallas RHT kernel requires a TPU backend supporting"
          f" {x.dtype} MXU matmuls. Pass implementation='xla' for the"
          " plain-XLA reference."
      )
  elif implementation == "xla":
    if config is not None:
      raise ValueError("config only applies to implementation='pallas'.")
  else:
    raise ValueError(
        f"implementation must be 'pallas' or 'xla', got {implementation!r}."
    )


@interception.disable_interceptions
def _apply_hadamard_rotate_impl(
    static: _HadamardStatic,
    x: jax.Array,
    signs: jax.Array | None,
) -> jax.Array:
  """Executes unquantized forward or inverse RHT on `x`.

  Interceptions are disabled because the rotation's matmuls (`jnp.einsum` on
  the XLA path, `jnp.matmul` inside the Pallas kernel) are internal to this op;
  a quantization provider would otherwise quantize them as model layers.
  """
  if static.implementation == "xla":
    return reference.hadamard_rotate_reference(
        x,
        signs=signs,
        block_size=static.block_size,
        axis=static.axis,
        inverse=static.inverse,
    )
  moved = jnp.moveaxis(x, static.axis, -1)
  lead_shape = moved.shape[:-1]
  k = moved.shape[-1]
  m = math.prod(lead_shape)
  out_2d = kernel.hadamard_pallas_2d(
      moved.reshape(m, k),
      signs=signs,
      block_size=static.block_size,
      inverse=static.inverse,
      qtype=None,
      config=static.config,
  )
  return jnp.moveaxis(out_2d.reshape(lead_shape + (k,)), -1, static.axis)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _hadamard_rotate(
    static: _HadamardStatic,
    x: jax.Array,
    signs: jax.Array | None,
) -> jax.Array:
  return _apply_hadamard_rotate_impl(static, x, signs)


def _hadamard_rotate_fwd(
    static: _HadamardStatic,
    x: jax.Array,
    signs: jax.Array | None,
) -> tuple[jax.Array, jax.Array | None]:
  return _apply_hadamard_rotate_impl(static, x, signs), signs


def _hadamard_rotate_bwd(
    static: _HadamardStatic,
    saved_signs: jax.Array | None,
    dout: jax.Array,
) -> tuple[jax.Array, None]:
  bwd_static = dataclasses.replace(static, inverse=not static.inverse)
  dx = _apply_hadamard_rotate_impl(bwd_static, dout, saved_signs)
  return dx, None


_hadamard_rotate.defvjp(_hadamard_rotate_fwd, _hadamard_rotate_bwd)


def hadamard_rotate(
    x: jax.Array,
    *,
    signs: jax.Array | None = None,
    block_size: int | None = None,
    axis: int = -1,
    inverse: bool = False,
    config: Config | None = None,
    implementation: Implementation = "pallas",
) -> jax.Array:
  """Applies the (inverse) Randomized Hadamard Transform along `axis`.

  Differentiable via `jax.custom_vjp`: because the RHT is orthogonal, the
  backward pass applies `hadamard_rotate(..., inverse=not inverse)` to the
  cotangent without storing `x` as a residual.

  Args:
    x: Input tensor of rank >= 1 in `bfloat16` or `float32`.
    signs: Optional 1-D Rademacher (`+1` / `-1`) vector of shape
      `(x.shape[axis],)`. Treated as non-differentiable.
    block_size: Optional diagonal Hadamard block size dividing `x.shape[axis]`.
      `None` applies the full Kronecker Hadamard transform across `axis`.
    axis: Axis along which to rotate.
    inverse: If True, applies the orthogonal inverse transform.
    config: Optional Pallas tiling `Config` (`implementation="pallas"` only).
    implementation: `"pallas"` for the Mosaic TPU kernel or `"xla"` for the
      plain-XLA reference, which runs on any backend.

  Returns:
    Rotated tensor with the same shape and dtype as `x`.
  """
  norm_axis = reference.validate_hadamard_args(
      x, signs=signs, block_size=block_size, axis=axis
  )
  _validate_dispatch(
      x,
      norm_axis=norm_axis,
      block_size=block_size,
      config=config,
      implementation=implementation,
  )
  static = _HadamardStatic(
      axis=norm_axis,
      block_size=block_size,
      inverse=inverse,
      config=config,
      implementation=implementation,
  )
  return _hadamard_rotate(static, x, signs)


def hadamard_rotate_quantize(
    x: jax.Array,
    *,
    qtype: jax.typing.DTypeLike = jnp.int8,
    signs: jax.Array | None = None,
    block_size: int | None = None,
    axis: int = -1,
    config: Config | None = None,
    implementation: Implementation = "pallas",
) -> QArray:
  """Rotates `x` along `axis` and symmetrically quantizes in a single VMEM pass.

  Produces one symmetric absmax scale per slice orthogonal to `axis`
  (`channelwise_axes = [i for i in range(x.ndim) if i != norm_axis]`), matching
  the LHS quantization layout expected by `qwix.dot_general` and `qwix.einsum`
  when contracting along `axis`.

  Args:
    x: Input tensor of rank >= 1 in `bfloat16` or `float32`.
    qtype: Target symmetric quantized dtype (`int8`, `float8_e4m3fn`, or
      `float8_e5m2`).
    signs: Optional 1-D Rademacher (`+1` / `-1`) vector of shape
      `(x.shape[axis],)`.
    block_size: Optional diagonal Hadamard block size dividing `x.shape[axis]`.
    axis: Contraction axis to rotate and reduce over for scale calibration.
    config: Optional Pallas tiling `Config` (`implementation="pallas"` only).
    implementation: `"pallas"` for the fused Mosaic TPU kernel or `"xla"` for
      the plain-XLA reference, which runs on any backend.

  Returns:
    A `qwix.QArray` with `qvalue` of shape `x.shape` in `qtype` and `scale` of
    shape `x.shape[:norm_axis] + (1,) + x.shape[norm_axis + 1:]` in `x.dtype`.
  """
  qdtype = reference.normalize_qtype(qtype)
  norm_axis = reference.validate_hadamard_args(
      x, signs=signs, block_size=block_size, axis=axis
  )
  _validate_dispatch(
      x,
      norm_axis=norm_axis,
      block_size=block_size,
      config=config,
      implementation=implementation,
  )
  return _hadamard_rotate_quantize_impl(
      x,
      signs,
      qdtype=qdtype,
      block_size=block_size,
      axis=norm_axis,
      config=config,
      implementation=implementation,
  )


@interception.disable_interceptions
def _hadamard_rotate_quantize_impl(
    x: jax.Array,
    signs: jax.Array | None,
    *,
    qdtype: jnp.dtype,
    block_size: int | None,
    axis: int,
    config: Config | None,
    implementation: Implementation,
) -> QArray:
  """Runs `hadamard_rotate_quantize` with active interceptions disabled."""
  if implementation == "xla":
    return reference.hadamard_rotate_quantize_reference(
        x,
        qtype=qdtype,
        signs=signs,
        block_size=block_size,
        axis=axis,
    )

  moved = jnp.moveaxis(x, axis, -1)
  lead_shape = moved.shape[:-1]
  k = moved.shape[-1]
  m = math.prod(lead_shape)
  xq_2d, sx_2d = kernel.hadamard_pallas_2d(
      moved.reshape(m, k),
      signs=signs,
      block_size=block_size,
      inverse=False,
      qtype=qdtype,
      config=config,
  )
  xq = jnp.moveaxis(xq_2d.reshape(lead_shape + (k,)), -1, axis)
  sx = jnp.moveaxis(sx_2d.reshape(lead_shape + (1,)), -1, axis)
  return QArray(xq, sx, qtype=qdtype)


def _can_fuse_lhs_quantize_in_qt(
    lhs: jax.Array,
    qt_config: dot_general_qt.DotGeneralQtConfig,
) -> bool:
  """Returns whether LHS RHT + quantization can fuse into a single VMEM pass."""
  return (
      numerics.should_quantize(lhs.dtype)
      and reference.is_supported_qtype(qt_config.lhs_qtype)
      and not qt_config.use_original_residuals
      and qt_config.multipass_mode is None
      and qt_config.dlhs_multipass_mode is None
      and qt_config.drhs_multipass_mode is None
      and not qt_config.tile_size
      and not qt_config.lhs_disable_channelwise_axes
      and qt_config.lhs_collect_quant_stat is None
      and qt_config.lhs_calibration_method == "absmax"
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _hadamard_qt(
    static: _HadamardQtStatic,
    lhs: jax.Array,
    rhs: jax.Array,
    signs: jax.Array | None,
    rhs_calibration: dict[str, jax.Array] | None,
    qt_config: dot_general_qt.DotGeneralQtConfig,
) -> jax.Array:
  """Custom-VJP wrapper fusing LHS RHT + quantization with `dot_general_qt`."""
  out, _ = _hadamard_qt_fwd(static, lhs, rhs, signs, rhs_calibration, qt_config)
  return out


@interception.disable_interceptions
def _hadamard_qt_fwd(
    static: _HadamardQtStatic,
    lhs: jax.Array,
    rhs: jax.Array,
    signs: jax.Array | None,
    rhs_calibration: dict[str, jax.Array] | None,
    qt_config: dot_general_qt.DotGeneralQtConfig,
) -> tuple[
    jax.Array, tuple[dot_general_qt.DotGeneralQtResiduals, jax.Array | None]
]:
  """Forward pass for `_hadamard_qt`: fuses LHS RHT + quantization in VMEM."""
  lhs_q = _hadamard_rotate_quantize_impl(
      lhs,
      signs,
      qdtype=static.lhs_qdtype,
      block_size=static.block_size,
      axis=static.axis,
      config=static.config,
      implementation=static.implementation,
  )
  out, qt_residuals = dot_general_qt.dot_general_qt_fwd(
      lhs_q,
      rhs,
      None,
      rhs_calibration,
      static.dimension_numbers,
      qt_config,
  )
  return out, (qt_residuals, signs)


@interception.disable_interceptions
def _hadamard_qt_bwd(
    static: _HadamardQtStatic,
    residuals: tuple[dot_general_qt.DotGeneralQtResiduals, jax.Array | None],
    dout: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None, None]:
  """Backward pass for `_hadamard_qt`: runs `dot_general_qt_bwd` + inv RHT."""
  qt_residuals, saved_signs = residuals
  dlhs_rot, drhs = dot_general_qt.dot_general_qt_bwd(qt_residuals, dout)
  bwd_hadamard_static = _HadamardStatic(
      axis=static.axis,
      block_size=static.block_size,
      inverse=True,
      config=static.config,
      implementation=static.implementation,
  )
  dlhs = _apply_hadamard_rotate_impl(bwd_hadamard_static, dlhs_rot, saved_signs)
  return dlhs, drhs, None, None, None


# `optimize_remat` lets JAX drop the residual outputs when the forward pass is
# recomputed under `jax.checkpoint`, rather than keeping a second copy of them.
_hadamard_qt.defvjp(_hadamard_qt_fwd, _hadamard_qt_bwd, optimize_remat=True)


def hadamard_dot_general_qt(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    qt_config: dot_general_qt.DotGeneralQtConfig,
    *,
    signs: jax.Array | None = None,
    block_size: int | None = None,
    config: Config | None = None,
    implementation: Implementation = "pallas",
) -> jax.Array:
  """Quantized `dot_general_qt` with VMEM-fused LHS RHT rotation.

  Expects `rhs` to already live in the rotated basis (e.g. weights rotated once
  at initialization, or rotated explicitly via
  `hadamard_rotate(rhs, signs=signs, axis=rhs_axis)`).

  When `qt_config` quantizes the lhs to `int8` or FP8 with channelwise absmax
  scales and keeps quantized residuals (`use_original_residuals=False`), `lhs`
  is rotated and quantized in a single VMEM pass without materializing the
  rotated floating-point activation in HBM. Otherwise (e.g.
  `use_original_residuals=True` or subchannel-tiled lhs scales), composes
  `hadamard_rotate(lhs, ...)` with `dot_general_qt.dot_general_qt`.

  Args:
    lhs: Unrotated LHS activation of rank >= 1 in `bfloat16` or `float32`.
    rhs: RHS operand in the rotated basis.
    dimension_numbers: `jax.lax.DotDimensionNumbers` with a single contracting
      axis.
    qt_config: `DotGeneralQtConfig` controlling forward/backward quantization.
    signs: Optional 1-D Rademacher (`+1` / `-1`) vector along the contracting
      axis.
    block_size: Optional diagonal Hadamard block size.
    config: Optional Pallas tiling `Config` (`implementation="pallas"` only).
    implementation: `"pallas"` or `"xla"`, as in `hadamard_rotate`.

  Returns:
    The contracted output tensor.
  """
  (lhs_ca, rhs_ca), _ = dimension_numbers
  if len(lhs_ca) != 1 or len(rhs_ca) != 1:
    raise ValueError(
        "hadamard_dot_general_qt requires exactly one contracting axis, got"
        f" lhs_ca={lhs_ca}, rhs_ca={rhs_ca}."
    )
  norm_axis = reference.validate_hadamard_args(
      lhs, signs=signs, block_size=block_size, axis=lhs_ca[0]
  )
  _validate_dispatch(
      lhs,
      norm_axis=norm_axis,
      block_size=block_size,
      config=config,
      implementation=implementation,
  )

  if not _can_fuse_lhs_quantize_in_qt(lhs, qt_config):
    lhs_rot = hadamard_rotate(
        lhs,
        signs=signs,
        block_size=block_size,
        axis=norm_axis,
        config=config,
        implementation=implementation,
    )
    return dot_general_qt.dot_general_qt(
        lhs_rot, rhs, dimension_numbers, qt_config
    )

  assert qt_config.lhs_qtype is not None
  lhs_qdtype = reference.normalize_qtype(qt_config.lhs_qtype)

  if qt_config.sparsity_rule is not None:
    rhs = qarray.sparsify(rhs, qt_config.sparsity_rule)
  rhs_calibration = dot_general_qt.calibrate_operand(
      rhs,
      for_lhs=False,
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      config=qt_config,
  )

  static = _HadamardQtStatic(
      dimension_numbers=dimension_numbers,
      axis=norm_axis,
      lhs_qdtype=lhs_qdtype,
      block_size=block_size,
      config=config,
      implementation=implementation,
  )
  return _hadamard_qt(static, lhs, rhs, signs, rhs_calibration, qt_config)
