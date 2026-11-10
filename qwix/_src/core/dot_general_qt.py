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
"""Quantized jax.lax.dot_general with quantized backpropagation support."""

import dataclasses
import functools
from typing import Any, Callable

import jax
import numpy as np
from qwix._src import interception
from qwix._src.core import dot_general
from qwix._src.core import numerics
from qwix._src.core import qarray
from qwix._src.core import sparsity
from qwix._src.core import stochastic_rounding


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class DotGeneralQtConfig:
  """Configuration for dot_general_qt."""

  # Forward pass.
  lhs_qtype: jax.typing.DTypeLike | None = None
  rhs_qtype: jax.typing.DTypeLike | None = None
  tile_size: int | float | None = None
  lhs_calibration_method: str = 'absmax'
  rhs_calibration_method: str = 'absmax'
  lhs_collect_quant_stat: Callable[[Any], Any] | None = None
  rhs_collect_quant_stat: Callable[[Any], Any] | None = None
  lhs_disable_channelwise_axes: bool = False
  rhs_disable_channelwise_axes: bool = False
  multipass_mode: Any = None

  # Backward pass (dlhs).
  dlhs_grad_qtype: jax.typing.DTypeLike | None = None  # incoming gradient
  dlhs_grad_calibration_method: str = 'absmax'
  dlhs_tile_size: int | float | None = None
  dlhs_stochastic_rounding_noise_fn: stochastic_rounding.NoiseFn | None = None
  dlhs_grad_disable_channelwise_axes: bool = False
  dlhs_multipass_mode: Any = None

  # Backward pass (drhs).
  drhs_grad_qtype: jax.typing.DTypeLike | None = None  # incoming gradient
  drhs_grad_calibration_method: str = 'absmax'
  drhs_tile_size: int | float | None = None
  drhs_stochastic_rounding_noise_fn: stochastic_rounding.NoiseFn | None = None
  drhs_grad_disable_channelwise_axes: bool = False
  drhs_multipass_mode: Any = None

  # Whether not to clip the gradients to the calibration ranges of the quantized
  # inputs. Enabling this improves the performance but may decrease the
  # numerical stability. Note that for the default absmax calibration method,
  # the gradient clipping is always skipped.
  disable_gradient_clipping: bool = False

  # Whether to use original (unquantized) residuals or quantized residuals
  # during the backward pass. Enabling this will allow separate quantization
  # schema for the residuals. When constructed via QtProvider, this defaults to:
  # 1. True if only fwd pass is quantized.
  # 2. False if both fwd and bwd passes are quantized.
  # 3. Users can override use_original_residuals via additional_qt_config.
  use_original_residuals: bool = False

  # Quantization for residuals. These only take effect when residuals are not
  # already quantized, i.e., either use_original_residuals is True or the
  # corresponding qtype in the fwd pass is None.
  dlhs_residual_qtype: jax.typing.DTypeLike | None = None
  dlhs_residual_calibration_method: str = 'absmax'
  dlhs_residual_disable_channelwise_axes: bool = False
  drhs_residual_qtype: jax.typing.DTypeLike | None = None
  drhs_residual_calibration_method: str = 'absmax'
  drhs_residual_disable_channelwise_axes: bool = False

  sparsity_rule: sparsity.SparsityRule | None = None
  out_sharding: (
      jax.sharding.NamedSharding | jax.sharding.PartitionSpec | None
  ) = None

  def tree_flatten(self):
    children = (
        self.dlhs_stochastic_rounding_noise_fn,
        self.drhs_stochastic_rounding_noise_fn,
    )
    aux_data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
    del aux_data['dlhs_stochastic_rounding_noise_fn']
    del aux_data['drhs_stochastic_rounding_noise_fn']
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    dlhs_stochastic_rounding_noise_fn, drhs_stochastic_rounding_noise_fn = (
        children
    )
    return cls(
        dlhs_stochastic_rounding_noise_fn=dlhs_stochastic_rounding_noise_fn,
        drhs_stochastic_rounding_noise_fn=drhs_stochastic_rounding_noise_fn,
        **aux_data,
    )


def _ranges_like(*xs):
  start = 0
  for x in xs:
    yield tuple(range(start, start + len(x)))
    start += len(x)


def _update_dimension_numbers_for_backward(
    fwd_dimension_numbers: jax.lax.DotDimensionNumbers,
    fwd_ndims: tuple[int, int],
    *,
    for_dlhs: bool,
) -> tuple[jax.lax.DotDimensionNumbers, tuple[int, ...]]:
  """Generates a new dimension number for backward pass.

  The dimension number is used to calculate dlhs = dot(g, rhs), if for_dlhs is
  True, or drhs = dot(g, lhs), otherwise.

  Args:
    fwd_dimension_numbers: Dimension numbers from the forward pass.
    fwd_ndims: Number of dimensions of the operands in the forward pass.
    for_dlhs: Whether the backward pass is for dlhs or drhs.

  Returns:
    A tuple of (dimension numbers for backward pass, output transpose to be
    applied to the result of the backward dot to match the original dimensions).
  """
  # We use the following conventions:
  #   g: incoming gradient.
  #   x: output gradient, dlhs if for_dlhs, drhs otherwise.
  #   y: residual, rhs if for_dlhs, lhs otherwise.
  if for_dlhs:
    (x_ca, y_ca), (x_ba, y_ba) = fwd_dimension_numbers
    x_ndim, y_ndim = fwd_ndims
  else:
    (y_ca, x_ca), (y_ba, x_ba) = fwd_dimension_numbers
    y_ndim, x_ndim = fwd_ndims

  x_ra = tuple(sorted(set(range(x_ndim)) - set(x_ca) - set(x_ba)))
  y_ra = tuple(sorted(set(range(y_ndim)) - set(y_ca) - set(y_ba)))

  if for_dlhs:
    g_ba, _, g_ca = _ranges_like(x_ba, x_ra, y_ra)
  else:
    g_ba, g_ca, _ = _ranges_like(x_ba, y_ra, x_ra)
  dnums = ((g_ca, y_ra), (g_ba, y_ba))

  x_ca_sorted_by_y = tuple(np.take(x_ca, np.argsort(y_ca)))
  out_transpose_axes = tuple(np.argsort(tuple(x_ba) + x_ra + x_ca_sorted_by_y))
  return dnums, out_transpose_axes


def _is_block_scaled(operand: qarray.MaybeQArray) -> bool:
  """Returns True if the operand uses block-scaled or tiled quantization.

  Args:
    operand: The potentially quantized operand.

  Returns:
    True if the operand uses block-scaled or tiled quantization.
  """
  if not isinstance(operand, qarray.QArray):
    return False
  return (
      bool(qarray.get_tiled_axes(operand))
      or numerics.get_microscaled_block_size(operand.qtype) is not None
  )


def _requires_unquantized_residual(
    config: DotGeneralQtConfig, operand_qt: qarray.MaybeQArray
) -> bool:
  """Returns True if the backward contraction cannot reuse the forward operand.

  Block-scaled residuals cannot be reused because quantization scales defined
  for the forward contraction axis do not align with the new contraction axis
  in the backward pass. Multi-pass modes also require original residuals.

  Args:
    config: The quantization configuration.
    operand_qt: The potentially quantized operand.

  Returns:
    True if the backward contraction cannot reuse the forward operand.
  """
  # `_check_caller_quantized` relies on this function to reject a caller-
  # quantized `QArray` operand, which has no unquantized value to keep. Any new
  # case in which the backward pass needs the unquantized operand must therefore
  # be added here, or such an operand would silently be used in its place.
  return (
      config.use_original_residuals
      or config.multipass_mode is not None
      or config.dlhs_multipass_mode is not None
      or config.drhs_multipass_mode is not None
      or _is_block_scaled(operand_qt)
  )


def _get_residual_for_backward(
    config: DotGeneralQtConfig,
    operand_in: jax.Array | None,
    operand_qt: qarray.MaybeQArray,
) -> qarray.MaybeQArray:
  """Returns the residual to be used in the backward pass.

  We reuse the quantized residual from the forward pass if possible. However,
  reusing tiled residuals is mathematically incorrect because the quantization
  scales (defined for one contraction axis) do not align with the new
  contraction axis in the backward pass.

  Args:
    config: The quantization configuration.
    operand_in: The original, unquantized operand, or None if not retained.
    operand_qt: The potentially quantized operand.

  Returns:
    The residual to be used in the backward pass.
  """
  if _requires_unquantized_residual(config, operand_qt):
    assert operand_in is not None
    return operand_in
  return operand_qt


def _needs_original_residual(
    config: DotGeneralQtConfig,
    operand_qt: qarray.MaybeQArray,
    calibration: dict[str, jax.Array] | None,
    calibration_method: str,
) -> bool:
  """Returns True if the unquantized operand must be retained in residuals.

  Args:
    config: The quantization configuration.
    operand_qt: The potentially quantized operand.
    calibration: The calibration dictionary or None.
    calibration_method: The calibration method string.

  Returns:
    True if the unquantized operand must be retained in residuals.
  """
  # 1. Contraction residual:
  # The unquantized operand must be retained if:
  #   a) The user explicitly opted into original residuals
  #      (config.use_original_residuals).
  #   b) The operand has tiled axes (transposed backward axes cannot align with
  #      tiled scales).
  #   c) The operand is an MXFP/microscaling type (block-level scales also
  #      require original operands).
  if _requires_unquantized_residual(config, operand_qt):
    return True

  # 2. Gradient clipping:
  # Straight-through estimator clipping requires the unquantized operand
  # unless clipping is a mathematical no-op (full-range absmax/minmax).
  if not config.disable_gradient_clipping and calibration is not None:
    if not qarray.is_gradient_clipping_noop(calibration_method):
      return True

  return False


def _check_caller_quantized(
    name: str,
    operand: qarray.MaybeQArray,
    calibration: dict[str, jax.Array] | None,
    config: DotGeneralQtConfig,
) -> None:
  """Raises if a caller-quantized operand cannot be used as is.

  Args:
    name: The operand name, 'lhs' or 'rhs'.
    operand: The potentially quantized operand.
    calibration: The calibration passed for the operand.
    config: The quantization configuration.

  Raises:
    ValueError: If `operand` is a `QArray` with a calibration, or the backward
      pass requires its unquantized value.
  """
  if not isinstance(operand, qarray.QArray):
    return
  if calibration is not None:
    raise ValueError(
        f'{name} is already a QArray, so {name}_calibration must be None.'
    )
  if _requires_unquantized_residual(config, operand):
    raise ValueError(
        f'{name} is already a QArray, but the backward pass requires the'
        f' unquantized {name} (use_original_residuals, multi-pass or block'
        ' scales).'
    )


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class DotGeneralQtResiduals:
  """Residuals of `dot_general_qt_fwd`, to be passed to `dot_general_qt_bwd`.

  Opaque to callers: the fields are qwix-internal and may change.
  `dimension_numbers` is kept in PyTree `aux_data` so that JAX transformations
  that trace residuals (e.g. `optimize_remat` / `jax.checkpoint`) keep the axis
  indices static rather than turning them into tracers.
  """

  lhs_in: jax.Array | None
  rhs_in: jax.Array | None
  lhs: qarray.MaybeQArray
  rhs: qarray.MaybeQArray
  lhs_calibration: dict[str, jax.Array] | None
  rhs_calibration: dict[str, jax.Array] | None
  dimension_numbers: jax.lax.DotDimensionNumbers
  config: DotGeneralQtConfig

  def tree_flatten(self):
    children = (
        self.lhs_in,
        self.rhs_in,
        self.lhs,
        self.rhs,
        self.lhs_calibration,
        self.rhs_calibration,
        self.config,
    )
    return children, self.dimension_numbers

  @classmethod
  def tree_unflatten(cls, dimension_numbers, children):
    (
        lhs_in,
        rhs_in,
        lhs,
        rhs,
        lhs_calibration,
        rhs_calibration,
        config,
    ) = children
    return cls(
        lhs_in=lhs_in,
        rhs_in=rhs_in,
        lhs=lhs,
        rhs=rhs,
        lhs_calibration=lhs_calibration,
        rhs_calibration=rhs_calibration,
        dimension_numbers=dimension_numbers,
        config=config,
    )


# See test_scan_custom_vjp in interception_test.py for why we need to manually
# disable interceptions for dot_general_qt_fwd.
@interception.disable_interceptions
def dot_general_qt_fwd(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    lhs_calibration: dict[str, jax.Array] | None,
    rhs_calibration: dict[str, jax.Array] | None,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
) -> tuple[jax.Array, DotGeneralQtResiduals]:
  """Forward pass for dot_general_qt custom VJP.

  Callers that run the forward and backward passes themselves pass the returned
  residuals, which are opaque, to `dot_general_qt_bwd`.

  Either operand may be a `QArray` quantized by the caller, with a None
  calibration. It is then used as is, also as its backward residual, so `config`
  must not require the unquantized operand (no `use_original_residuals`, no
  multi-pass, no block scales), and its gradient is not clipped to a calibration
  range.

  Args:
    lhs: The lhs, unquantized or already a `QArray`.
    rhs: The rhs, unquantized or already a `QArray`.
    lhs_calibration: The lhs calibration, or None if `lhs` is not quantized
      here.
    rhs_calibration: The rhs calibration, or None if `rhs` is not quantized
      here.
    dimension_numbers: The dimension numbers of the dot_general.
    config: The dot_general_qt config.

  Returns:
    The output and the opaque residuals for `dot_general_qt_bwd`.

  Raises:
    ValueError: If a `QArray` operand has a calibration, or `config` requires
      its unquantized value.
  """
  _check_caller_quantized('lhs', lhs, lhs_calibration, config)
  _check_caller_quantized('rhs', rhs, rhs_calibration, config)
  lhs_in, rhs_in = lhs, rhs
  if lhs_calibration is not None:
    scale, zero_point = qarray.compute_scale_zero_point(
        lhs_calibration, config.lhs_qtype  # pyrefly: ignore[bad-argument-type]
    )
    lhs = qarray.quantize_with_scale_zero_point(  # pyrefly: ignore[bad-assignment]
        lhs, config.lhs_qtype, scale, zero_point  # pyrefly: ignore[bad-argument-type]
    )
  if rhs_calibration is not None:
    scale, zero_point = qarray.compute_scale_zero_point(
        rhs_calibration, config.rhs_qtype  # pyrefly: ignore[bad-argument-type]
    )
    rhs = qarray.quantize_with_scale_zero_point(  # pyrefly: ignore[bad-assignment]
        rhs, config.rhs_qtype, scale, zero_point  # pyrefly: ignore[bad-argument-type]
    )
  saved_lhs_in = None
  if _needs_original_residual(
      config, lhs, lhs_calibration, config.lhs_calibration_method
  ):
    saved_lhs_in = lhs_in

  saved_rhs_in = None
  if _needs_original_residual(
      config, rhs, rhs_calibration, config.rhs_calibration_method
  ):
    saved_rhs_in = rhs_in

  saved_lhs_calibration = (
      None if config.disable_gradient_clipping else lhs_calibration
  )
  saved_rhs_calibration = (
      None if config.disable_gradient_clipping else rhs_calibration
  )

  residuals = DotGeneralQtResiduals(
      # Never a QArray: _check_caller_quantized rejects a QArray operand whose
      # unquantized value would be needed.
      lhs_in=saved_lhs_in,  # pyrefly: ignore[bad-argument-type]
      rhs_in=saved_rhs_in,  # pyrefly: ignore[bad-argument-type]
      lhs=lhs,
      rhs=rhs,
      lhs_calibration=saved_lhs_calibration,
      rhs_calibration=saved_rhs_calibration,
      dimension_numbers=dimension_numbers,
      config=config,
  )
  if config.multipass_mode is not None:
    # Multi-pass residual decomposition requires unquantized inputs (lhs_in,
    # rhs_in) to compute successive residual passes (A_0 = quant(A),
    # A_1 = quant(A - dequant(A_0))). Passing already-quantized QArrays (lhs,
    # rhs) would cause residual decomposition to zero out subsequent passes.
    out = dot_general.dot_general(
        lhs_in,
        rhs_in,
        dimension_numbers,
        multipass_mode=config.multipass_mode,
        tile_size=config.tile_size,
    )
  else:
    out = dot_general.dot_general(
        lhs, rhs, dimension_numbers, out_sharding=config.out_sharding
    )
  return out, residuals


@interception.disable_interceptions
def dot_general_qt_bwd(
    residuals: DotGeneralQtResiduals,
    g: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Backward pass for dot_general_qt.

  Args:
    residuals: The opaque residuals returned by `dot_general_qt_fwd`, which
      carry the forward dimension numbers.
    g: The gradient of the output.

  Returns:
    The gradients `(dlhs, drhs)` of the lhs and the rhs.
  """
  lhs_in = residuals.lhs_in
  rhs_in = residuals.rhs_in
  lhs = residuals.lhs
  rhs = residuals.rhs
  lhs_calibration = residuals.lhs_calibration
  rhs_calibration = residuals.rhs_calibration
  fwd_dimension_numbers = residuals.dimension_numbers
  config = residuals.config

  def _compute_gradient_for_operand(g: jax.Array, *, for_dlhs: bool):
    """Compute dot_general for gradient and other_fwd_operand."""
    bwd_dnums, transpose_axes = _update_dimension_numbers_for_backward(
        fwd_dimension_numbers, (lhs.ndim, rhs.ndim), for_dlhs=for_dlhs
    )
    primal = lhs if for_dlhs else rhs
    x_sharding = jax.typeof(
        primal.qvalue if isinstance(primal, qarray.QArray) else primal
    ).sharding
    bwd_out_sharding = None
    if jax.sharding.AxisType.Explicit in x_sharding.mesh.axis_types:
      x_partitions = x_sharding.spec.partitions
      bwd_out_sharding = jax.sharding.PartitionSpec(
          *(
              x_partitions[i] if i < len(x_partitions) else None
              for i in np.argsort(transpose_axes)
          ),
          unreduced=x_sharding.spec.reduced,
          reduced=x_sharding.spec.unreduced,
      )
    if for_dlhs:
      g_qtype = config.dlhs_grad_qtype
      g_tile_size = config.dlhs_tile_size
      g_calibration_method = config.dlhs_grad_calibration_method
      g_noise_fn = config.dlhs_stochastic_rounding_noise_fn
      g_disable_channelwise_axes = config.dlhs_grad_disable_channelwise_axes
      y = _get_residual_for_backward(config, rhs_in, rhs)
      y_qtype = config.dlhs_residual_qtype
      y_calibration_method = config.dlhs_residual_calibration_method
      y_disable_channelwise_axes = config.dlhs_residual_disable_channelwise_axes
    else:
      g_qtype = config.drhs_grad_qtype
      g_tile_size = config.drhs_tile_size
      g_calibration_method = config.drhs_grad_calibration_method
      g_noise_fn = config.drhs_stochastic_rounding_noise_fn
      g_disable_channelwise_axes = config.drhs_grad_disable_channelwise_axes
      y = _get_residual_for_backward(config, lhs_in, lhs)
      y_qtype = config.drhs_residual_qtype
      y_calibration_method = config.drhs_residual_calibration_method
      y_disable_channelwise_axes = config.drhs_residual_disable_channelwise_axes

    multipass_mode = (
        config.dlhs_multipass_mode if for_dlhs else config.drhs_multipass_mode
    )
    if multipass_mode is not None:
      grad_res = dot_general.dot_general(
          g,
          y,
          dimension_numbers=bwd_dnums,
          multipass_mode=multipass_mode,
          tile_size=g_tile_size,
      )
      return jax.lax.transpose(grad_res, transpose_axes)

    if g_qtype and numerics.should_quantize(g.dtype):
      if isinstance(y, qarray.QArray):
        # Scale shifting for quantized residuals (use_original_residuals=False)
        # Apply the scale of y to g, this trick avoids requantizing y because
        # the y from fwd pass has different channelwise_axes.
        assert y.zero_point is None and y.qtype == y.qvalue.dtype
        (g_ca, y_ca), (g_ba, y_ba) = bwd_dnums
        g = dot_general.fold_scale_into(
            g, y.scale, (*g_ca, *g_ba), (*y_ca, *y_ba)
        )
        y = y.qvalue

      g_how = dot_general.get_how_to_quantize(
          dimension_numbers=bwd_dnums,
          ndims=(g.ndim, y.ndim),
          for_lhs=True,
          qtype=g_qtype,
          tile_size=g_tile_size,
          calibration_method=g_calibration_method,
          noise_fn=g_noise_fn,
      )
      if g_disable_channelwise_axes:
        g_how = dataclasses.replace(g_how, channelwise_axes=[])

      g = qarray.quantize(g, g_how)  # pyrefly: ignore[bad-assignment]

    if (
        isinstance(y, jax.Array)
        and y_qtype
        and numerics.should_quantize(y.dtype)
    ):
      y_how = dot_general.get_how_to_quantize(
          dimension_numbers=bwd_dnums,
          ndims=(g.ndim, y.ndim),
          for_lhs=False,
          qtype=y_qtype,
          tile_size=g_tile_size,
          calibration_method=y_calibration_method,
      )
      if y_disable_channelwise_axes:
        y_how = dataclasses.replace(y_how, channelwise_axes=[])
      y = qarray.quantize(y, y_how)

    grad_res = dot_general.dot_general(
        g, y, bwd_dnums, out_sharding=bwd_out_sharding
    )
    return jax.lax.transpose(grad_res, transpose_axes)

  dlhs = _compute_gradient_for_operand(g, for_dlhs=True)
  drhs = _compute_gradient_for_operand(g, for_dlhs=False)

  if not config.disable_gradient_clipping:
    if lhs_calibration is not None and lhs_in is not None:
      dlhs = qarray.clip_gradient_to_calibration(
          dlhs, lhs_in, lhs_calibration, config.lhs_calibration_method
      )
    if rhs_calibration is not None and rhs_in is not None:
      drhs = qarray.clip_gradient_to_calibration(
          drhs, rhs_in, rhs_calibration, config.rhs_calibration_method
      )

  return dlhs, drhs


def _dot_general_qt_custom_vjp_bwd(
    dimension_numbers: jax.lax.DotDimensionNumbers,
    residuals: DotGeneralQtResiduals,
    g: jax.Array,
):
  """Adapts `dot_general_qt_bwd` to the signature `custom_vjp` requires."""
  del dimension_numbers  # Carried by the residuals.
  dlhs, drhs = dot_general_qt_bwd(residuals, g)
  return dlhs, drhs, None, None, None


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def dot_general_qt_fwd_bwd(
    lhs: jax.Array,
    rhs: jax.Array,
    lhs_calibration: dict[str, jax.Array] | None,
    rhs_calibration: dict[str, jax.Array] | None,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
) -> jax.Array:
  """dot_general custom VJP."""
  result, _ = dot_general_qt_fwd(
      lhs, rhs, lhs_calibration, rhs_calibration, dimension_numbers, config
  )
  return result


dot_general_qt_fwd_bwd.defvjp(
    dot_general_qt_fwd, _dot_general_qt_custom_vjp_bwd
)


def calibrate_operand(
    x: qarray.MaybeQArray,
    *,
    for_lhs: bool,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    ndims: tuple[int, int],
    config: DotGeneralQtConfig,
) -> dict[str, jax.Array] | None:
  """Calibrates one operand of `dot_general_qt`.

  Callers that quantize an operand themselves (e.g. inside a fused kernel) pass
  the resulting `QArray`, whose calibration is None, and the calibration of the
  other operand to `dot_general_qt_fwd`.

  Args:
    x: The operand, unquantized or already a `QArray`.
    for_lhs: Whether `x` is the lhs.
    dimension_numbers: The dimension numbers of the dot_general.
    ndims: The ranks `(lhs.ndim, rhs.ndim)` of both operands.
    config: The dot_general_qt config.

  Returns:
    The calibration of `x`, or None if `x` is already a `QArray` or `config`
    leaves it unquantized.
  """
  if isinstance(x, qarray.QArray):
    return None
  if for_lhs:
    qtype = config.lhs_qtype
    calibration_method = config.lhs_calibration_method
    disable_channelwise_axes = config.lhs_disable_channelwise_axes
    collect_quant_stat = config.lhs_collect_quant_stat
  else:
    qtype = config.rhs_qtype
    calibration_method = config.rhs_calibration_method
    disable_channelwise_axes = config.rhs_disable_channelwise_axes
    collect_quant_stat = config.rhs_collect_quant_stat
  if not qtype or not numerics.should_quantize(x.dtype):
    return None
  how = dot_general.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=ndims,
      for_lhs=for_lhs,
      qtype=qtype,
      tile_size=config.tile_size,
      calibration_method=calibration_method,
  )
  if disable_channelwise_axes:
    how = dataclasses.replace(how, channelwise_axes=[])
  calibration = qarray.calibrate(jax.lax.stop_gradient(x), how)
  if collect_quant_stat:
    calibration = collect_quant_stat(calibration)
  return calibration


def dot_general_qt(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
    *,
    out_sharding: (
        jax.sharding.NamedSharding | jax.sharding.PartitionSpec | None
    ) = None,
) -> jax.Array:
  """Quantized dot_general with backpropagation support."""
  if out_sharding is not None:
    config = dataclasses.replace(config, out_sharding=out_sharding)
  calibrate = functools.partial(
      calibrate_operand,
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      config=config,
  )
  lhs_calibration = calibrate(lhs, for_lhs=True)
  # Sparsify rhs before calibration and quantization to make sure we
  # are quantizing the remaining values correctly
  if config.sparsity_rule is not None:
    rhs = qarray.sparsify(rhs, config.sparsity_rule)
  rhs_calibration = calibrate(rhs, for_lhs=False)
  return dot_general_qt_fwd_bwd(
      lhs, rhs, lhs_calibration, rhs_calibration, dimension_numbers, config
  )
