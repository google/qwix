"""PTQ provider that pads to tile before quantization and compute.

This provider centralizes tiling-related padding at the provider layer so
qarray/core do not need to pad. It pads operands along the tiled axes (when a
tile_size is configured) BEFORE quantization, ensuring scale shapes are valid
and compute paths can split axes without remainder. For contracting-axis tiling
no output cropping is required.

Currently supports:
  - dot_general
  - einsum (via dot_general under the hood)

Notes:
  - We only tile contracting axes based on existing get_how_to_quantize() in
    core; if non-contracting tiling is desired, additional output-cropping
    logic must be added.
  - conv_general_dilated is left unchanged (no subchannel tiling there).
"""

from __future__ import annotations

from typing import Sequence, Callable
import dataclasses

import jax
from jax import numpy as jnp

from qwix._src import flax_util
from qwix._src import qconfig
from qwix._src.core import qarray
from qwix._src.core import dot_general as core_dot
from qwix._src.core import einsum as core_einsum
from flax import linen as nn
from flax import nnx
from qwix._src.providers.ptq import PtqProvider, WithAux
from qwix._src.providers import ptq as _ptq


def _compute_pad_width(array: jax.Array, tiled_axes: dict[int, int | float]) -> list[tuple[int, int]]:
  """Compute end-padding widths so each tiled axis is divisible by tile size.

  Args:
    array: input array.
    tiled_axes: mapping from axis index to tile size (int) or fraction (float).

  Returns:
    pad_width list compatible with jnp.pad: [(before, after), ...]. Only pads
    at the end (after) as needed.
  """
  nd = array.ndim
  pad_width = [(0, 0)] * nd
  if not tiled_axes:
    return pad_width
  for axis, tile in tiled_axes.items():
    dim = array.shape[axis]
    tile_size = tile
    if isinstance(tile_size, float):
      tile_size = round(dim * tile_size)
    if tile_size is None or tile_size <= 0:
      continue
    rem = dim % tile_size
    if rem:
      pad_width[axis] = (0, tile_size - rem)
  return pad_width


def _maybe_pad(x: jax.Array, tiled_axes: dict[int, int | float]) -> tuple[jax.Array, list[tuple[int, int]]]:
  """Pad `x` along `tiled_axes` if needed. Returns (padded_x, pad_width)."""
  pw = _compute_pad_width(x, tiled_axes)
  if any(b or a for (b, a) in pw):
    x = jnp.pad(x, pw, constant_values=0)
  return x, pw


class PtqPadProvider(PtqProvider):
  """PTQ provider that ensures tiling divisibility by padding pre-quantization."""

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array | WithAux[qarray.QArray],
      dimension_numbers: jax.lax.DotDimensionNumbers,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      *,
      out_sharding: jax.sharding.NamedSharding | None = None,
  ) -> jax.Array:
    rule, op_id = self._get_current_rule_and_op_id('dot_general')
    if rule is None or rule.weight_qtype is None:
      return jax.lax.dot_general(
          lhs,
          rhs,
          dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
          out_sharding=out_sharding,
      )

    # Follow PtqProvider flow exactly, inserting padding right before quantization.
    get_how = lambda for_lhs, qtype, calib: core_dot.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(len(lhs.shape), len((rhs.array if isinstance(rhs, WithAux) else rhs).shape)),
        for_lhs=for_lhs,
        qtype=qtype,
        tile_size=rule.tile_size,
        calibration_method=calib,
    )

    # Prepare rhs.
    rhs_how = None
    if isinstance(rhs, WithAux):  # weight, already quantized
      rhs_how = rhs.how
      rhs = rhs.array
    elif (weight_name := flax_util.find_param(rhs)) is not None:  # weight, not quantized
      rhs_how = get_how(False, rule.weight_qtype, rule.weight_calibration_method)
      rhs = create_quantized_param(weight_name, rhs, rhs_how).array
    elif rule.act_qtype is not None:  # rhs is activation
      rhs_how = get_how(False, rule.act_qtype, rule.act_calibration_method)
      rhs = quantize_act(rhs, rhs_how, rule, op_id + '_rhs')

    # Prepare lhs (activation).
    lhs_how = None
    if rule.act_qtype is not None:
      lhs_how = get_how(True, rule.act_qtype, rule.act_calibration_method)
      lhs = quantize_act(lhs, lhs_how, rule, op_id + '_lhs')

    # Pad qvalues transiently before compute.
    if rhs_how and rhs_how.tiled_axes:
      padded_qvalue, _ = _maybe_pad(rhs.qvalue, dict(rhs_how.tiled_axes))
      rhs = dataclasses.replace(rhs, qvalue=padded_qvalue)
    if lhs_how and lhs_how.tiled_axes:
      padded_qvalue, _ = _maybe_pad(lhs.qvalue, dict(lhs_how.tiled_axes))
      lhs = dataclasses.replace(lhs, qvalue=padded_qvalue)

    return core_dot.dot_general(
        lhs, rhs, dimension_numbers, out_sharding=out_sharding
    )

  def einsum(
      self,
      einsum_str: str,
      *operands: jax.Array,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      _dot_general: Callable[..., jax.Array] = jax.lax.dot_general,  # noqa: N803
      out_sharding=None,
  ) -> jax.Array:
    rule, op_id = self._get_current_rule_and_op_id('einsum')
    if rule is None or rule.weight_qtype is None:
      return jax.numpy.einsum(
          einsum_str,
          *operands,
          precision=precision,
          preferred_element_type=preferred_element_type,
          _dot_general=_dot_general,
          out_sharding=out_sharding,
      )
    if not isinstance(einsum_str, str) or len(operands) != 2:
      raise ValueError(f'Unsupported einsum format: {einsum_str=} {operands=}')

    lhs, rhs = operands

    # Follow PtqProvider flow; insert padding right before quantization.
    get_how = lambda for_lhs, qtype, calib: core_einsum.get_how_to_quantize(
        einsum_str=einsum_str,
        ndims=(len(lhs.shape), len(rhs.shape)),
        for_lhs=for_lhs,
        qtype=qtype,
        tile_size=rule.tile_size,
        calibration_method=calib,
    )

    # Prepare rhs.
    rhs_how = None
    if isinstance(rhs, WithAux):  # already quantized weight
      rhs_how = rhs.how
      rhs = rhs.array
    elif (weight_name := flax_util.find_param(rhs)) is not None:
      rhs_how = get_how(False, rule.weight_qtype, rule.weight_calibration_method)
      rhs = create_quantized_param(weight_name, rhs, rhs_how).array
    elif rule.act_qtype is not None:
      rhs_how = get_how(False, rule.act_qtype, rule.act_calibration_method)
      rhs = quantize_act(rhs, rhs_how, rule, op_id + '_rhs')

    # Prepare lhs (activation).
    lhs_how = None
    if rule.act_qtype is not None:
      lhs_how = get_how(True, rule.act_qtype, rule.act_calibration_method)
      lhs = quantize_act(lhs, lhs_how, rule, op_id + '_lhs')

    # Pad qvalues transiently before compute.
    if rhs_how and rhs_how.tiled_axes:
      padded_qvalue, _ = _maybe_pad(rhs.qvalue, dict(rhs_how.tiled_axes))
      rhs = dataclasses.replace(rhs, qvalue=padded_qvalue)
    if lhs_how and lhs_how.tiled_axes:
      padded_qvalue, _ = _maybe_pad(lhs.qvalue, dict(lhs_how.tiled_axes))
      lhs = dataclasses.replace(lhs, qvalue=padded_qvalue)

    return core_einsum.einsum(einsum_str, lhs, rhs)

  def get_intercept_map(self):
    # Override maps for dot_general/einsum, but keep others from PtqProvider.
    return super().get_intercept_map() | {
        'jax.lax.dot_general': self.dot_general,
        'jax.numpy.einsum': self.einsum,
    }

def _calibrate_on_padded(array: jax.Array, how: qarray.HowToQuantize):
  if how.tiled_axes:
    padded, _ = _maybe_pad(array, dict(how.tiled_axes))
  else:
    padded = array
  callibration = qarray.calibrate(padded, how)
  return callibration

def quantize_act(
    array: jax.Array,
    how: qarray.HowToQuantize,
    rule: qconfig.QuantizationRule,
    act_name: str,
) -> qarray.QArray:
  """Quantize activations using padded calibration but original shapes.

  For dynamic: calibrate on padded array, then quantize original array with
  quantize_with_scale_zero_point.
  For static: identical to PTQ's logic except calibration step uses padded
  array when quant_stats are absent.
  """
  if not rule.act_static_scale:
    calibration = _calibrate_on_padded(array, how)
    scale, zp = qarray.compute_scale_zero_point(calibration, how.qtype)
    return qarray.quantize_with_scale_zero_point(array, how.qtype, scale, zp)

  # Static scale path (SRQ-like).
  quant_stat = flax_util.get_and_delete_variable('quant_stats', act_name)

  def init():
    if quant_stat is not None:
      aggregator = _ptq.averaging.SimpleMovingAverage()
      calibration = aggregator.get_calibration(quant_stat)
    else:
      calibration = _calibrate_on_padded(array, how)
      calibration = jax.tree.map(
          lambda x: x.mean(axis=rule.act_batch_axes, keepdims=True), calibration
      )
    nonlocal zp
    scale, zp = qarray.compute_scale_zero_point(calibration, how.qtype)
    return WithAux(scale, how)

  zp = None
  scale = flax_util.get_or_create_param(act_name + '_scale', init)
  if zp is not None:
    zp = flax_util.get_or_create_param(act_name + '_zero_point', lambda: zp)
  return qarray.quantize_with_scale_zero_point(array, how.qtype, scale.array, zp)


def create_quantized_param(
    name: str, value: jax.Array, how: qarray.HowToQuantize
) -> WithAux[qarray.QArray]:
  """Create a quantized param or array with padding-aware calibration.

  For (value, how): returns quantized QArray.

  For (name, value, how): returns WithAux containing the quantized QArray and HowToQuantize, and replaces the param in the module.
  """
  calibration = _calibrate_on_padded(value, how)
  scale, zp = qarray.compute_scale_zero_point(calibration, how.qtype)
  qarr = qarray.quantize_with_scale_zero_point(value, how.qtype, scale, zp, how.noise_fn)
  unboxed = WithAux(qarr, how)

  module = flax_util.get_current_module()
  if isinstance(module, nn.Module):
    if not module.is_initializing():
      raise ValueError(
          "It seems you're feeding an unquantized param to a quantized model."
      )
    param = module.get_variable('params', name)
    boxed = jax.tree.map(
        lambda value: flax_util.update_boxed(param, value=value), unboxed
    )
    module.put_variable('params', name, boxed)
  elif isinstance(module, nnx.Module):
    param = getattr(module, name)
    boxed = jax.tree.map(
        lambda value: flax_util.update_boxed(param, value=value), unboxed
    )
    setattr(module, name, boxed)
  return unboxed