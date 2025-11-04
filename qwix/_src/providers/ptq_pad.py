# Copyright 2024 Google LLC
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

from __future__ import annotations

import dataclasses
from typing import Callable

import flax
import jax
from jax import numpy as jnp
from qwix._src import flax_util
from qwix._src import qconfig
from qwix._src.core import dot_general as core_dot
from qwix._src.core import einsum as core_einsum
from qwix._src.core import qarray
from qwix._src.providers import ptq as _ptq
from qwix._src.providers.ptq import PtqProvider, WithAux


def _compute_pad_width(
    array: jax.Array, tiled_axes: dict[int, int | float]
) -> list[tuple[int, int]]:
  """End-padding widths so each tiled axis is divisible by tile size."""
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


def _maybe_pad(x: jax.Array, tiled_axes: dict[int, int | float]) -> jax.Array:
  """Pad `x` along `tiled_axes` if needed."""
  pw = _compute_pad_width(x, tiled_axes)
  if any(b or a for (b, a) in pw):
    x = jnp.pad(x, pw, constant_values=0)
  return x


def _pad_preserve_metadata(x, tiled_axes: dict[int, int | float]):
  """Pad while preserving linen/nnx metadata wrappers if present."""
  if isinstance(x, (flax.linen.meta.AxisMetadata, flax.nnx.Variable)):
    raw = flax_util.unbox(x)
    padded = _maybe_pad(raw, tiled_axes)
    return flax_util.update_boxed(x, value=padded)
  else:
    return _maybe_pad(x, tiled_axes)


class PtqPadProvider(PtqProvider):
  """Ensures tiling divisibility by padding pre-quantization."""

  def _prepare_and_pad(
      self,
      lhs: jax.Array,
      rhs: jax.Array | WithAux[qarray.QArray],
      get_how: Callable[[bool, object, str], qarray.HowToQuantize],
      rule: qconfig.QuantizationRule,
      op_id: str,
  ) -> tuple[jax.Array | qarray.QArray, jax.Array | qarray.QArray]:
    """Shared: quantize lhs/rhs as needed and pad to tile boundaries."""
    # rhs
    rhs_how = None
    if isinstance(rhs, WithAux):
      rhs_how = rhs.how
      rhs = rhs.array
    elif (weight_name := flax_util.find_param(rhs)) is not None:
      rhs_how = get_how(
          False, rule.weight_qtype, rule.weight_calibration_method
      )
      rhs = create_quantized_param(weight_name, rhs, rhs_how).array
    elif rule.act_qtype is not None:
      rhs_how = get_how(False, rule.act_qtype, rule.act_calibration_method)
      rhs = quantize_act(rhs, rhs_how, rule, op_id + '_rhs')

    # lhs (activation)
    lhs_how = None
    if rule.act_qtype is not None:
      lhs_how = get_how(True, rule.act_qtype, rule.act_calibration_method)
      lhs = quantize_act(lhs, lhs_how, rule, op_id + '_lhs')
    else:
      # No act quant; still derive tiling for padding using weight path.
      lhs_how = get_how(True, rule.weight_qtype, rule.weight_calibration_method)

    # Pad rhs
    if rhs_how and rhs_how.tiled_axes:
      if isinstance(rhs, qarray.QArray):
        padded_qvalue = _pad_preserve_metadata(
            rhs.qvalue, dict(rhs_how.tiled_axes)
        )
        rhs = dataclasses.replace(rhs, qvalue=padded_qvalue)
      else:
        rhs = _maybe_pad(rhs, dict(rhs_how.tiled_axes))

    # Pad lhs
    if lhs_how and lhs_how.tiled_axes:
      if isinstance(lhs, qarray.QArray):
        padded_qvalue = _pad_preserve_metadata(
            lhs.qvalue, dict(lhs_how.tiled_axes)
        )
        lhs = dataclasses.replace(lhs, qvalue=padded_qvalue)
      else:
        lhs = _maybe_pad(lhs, dict(lhs_how.tiled_axes))

    return lhs, rhs

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

    # Compute how-to-quantize.
    get_how = lambda for_lhs, qtype, calib: core_dot.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(
            len(lhs.shape),
            len((rhs.array if isinstance(rhs, WithAux) else rhs).shape),
        ),
        for_lhs=for_lhs,
        qtype=qtype,
        tile_size=rule.tile_size,
        calibration_method=calib,
    )
    lhs, rhs = self._prepare_and_pad(lhs, rhs, get_how, rule, op_id)

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

    # Compute how-to-quantize.
    get_how = lambda for_lhs, qtype, calib: core_einsum.get_how_to_quantize(
        einsum_str=einsum_str,
        ndims=(len(lhs.shape), len(rhs.shape)),
        for_lhs=for_lhs,
        qtype=qtype,
        tile_size=rule.tile_size,
        calibration_method=calib,
    )
    lhs, rhs = self._prepare_and_pad(lhs, rhs, get_how, rule, op_id)

    return core_einsum.einsum(einsum_str, lhs, rhs)

  def get_intercept_map(self):
    # Override maps for dot_general/einsum, keep others from PtqProvider.
    return super().get_intercept_map() | {
        'jax.lax.dot_general': self.dot_general,
        'jax.numpy.einsum': self.einsum,
    }


def quantize_act(
    array: jax.Array,
    how: qarray.HowToQuantize,
    rule: qconfig.QuantizationRule,
    act_name: str,
) -> qarray.QArray:
  """Pad then delegate to base PTQ activation quantization (no slicing)."""
  array_padded = _maybe_pad(array, how.tiled_axes)
  return _ptq.quantize_act(array_padded, how, rule, act_name)


def create_quantized_param(
    name: str, value: jax.Array, how: qarray.HowToQuantize
) -> WithAux[qarray.QArray]:
  """Pad then delegate to base PTQ param quantization (no slicing)."""
  value_padded = _maybe_pad(value, how.tiled_axes)
  return _ptq.create_quantized_param(name, value_padded, how)


def _pad_params_like_abstract(params, abstract_quantized_params):
  """Pad params along tiled axes according to abstract WithAux.how."""

  def get_value_from_path(obj, path: tuple[str, ...]):
    for key in path:
      obj = obj[key] if isinstance(obj, dict) else getattr(obj, key)
    return obj

  padded = {}
  for path, param in flax.traverse_util.flatten_dict(params).items():
    abs_param = get_value_from_path(abstract_quantized_params, path)
    if isinstance(abs_param, WithAux):
      param = _maybe_pad(param, abs_param.how.tiled_axes)
    padded[path] = param
  return flax.traverse_util.unflatten_dict(padded)


def quantize_params(
    params,
    abstract_quantized_params,
    quant_stats=flax.core.FrozenDict(),
):
  """Pad params along tiled axes then delegate to base quantize_params."""
  params_padded = _pad_params_like_abstract(params, abstract_quantized_params)
  return _ptq.quantize_params(
      params_padded, abstract_quantized_params, quant_stats
  )
