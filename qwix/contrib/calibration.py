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
"""Common calibration logic for quantization providers."""

import abc
import dataclasses
from typing import Any, Callable

import flax
import jax
from jax import numpy as jnp
from qwix._src import averaging
from qwix._src import flax_util
from qwix._src import qconfig
from qwix._src.providers import ptq


class CalibrationProvider(qconfig.QuantizationProvider, metaclass=abc.ABCMeta):
  """Base class for calibration providers that intercept dot_general/einsum.

  This provider handles the common boilerplate for all calibration providers:
  rule type checking, dimension validation, weight name lookup, and LHS
  reshaping. Subclasses implement `_collect_stats` to define what happens
  with the validated, reshaped activations.
  """

  @abc.abstractmethod
  def get_rule_type(self) -> type[qconfig.QuantizationRule]:
    """Returns the rule type that this provider handles."""

  @abc.abstractmethod
  def get_stats_suffix(self) -> str:
    """Returns the suffix for the stats variable name (e.g., '_gptq')."""

  @abc.abstractmethod
  def _collect_stats(self, lhs: jax.Array, weight_name: str) -> None:
    """Collects statistics from the reshaped input activations.

    Called after all validation passes. The LHS has already been reshaped
    to (contracting_dim, rest) format.

    Args:
      lhs: Input activations reshaped to (ca, rest) format.
      weight_name: The name of the weight parameter for this operation.
    """

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      *args,
      rule: qconfig.QuantizationRule | None = None,
      **kwargs,
  ) -> jax.Array:
    res = jax.lax.dot_general(lhs, rhs, dimension_numbers, *args, **kwargs)
    if rule is None:
      rule, _ = self._get_current_rule_and_op_id('dot_general')

    rule_type = self.get_rule_type()
    if not isinstance(rule, rule_type):
      return res

    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    if lhs_ba or rhs_ba or len(lhs_ca) != 1 or len(rhs_ca) != 1:
      # We only support standard dot_general with 1 contracting axis for now.
      # If the operation is not supported, we skip calibration for it.
      return res

    weight_name = flax_util.find_param(rhs)
    if weight_name is None:
      # If we cannot identify the weight parameter, we skip calibration.
      return res

    # Reorder lhs to (ca, rest) format.
    lhs = jnp.moveaxis(lhs, lhs_ca[0], 0)
    lhs = lhs.reshape(lhs.shape[0], -1)

    self._collect_stats(lhs, weight_name)

    return res

  def einsum(self, einsum_str, *operands, **kwargs):
    rule, _ = self._get_current_rule_and_op_id('einsum')
    rule_type = self.get_rule_type()
    if not isinstance(rule, rule_type):
      return jnp.einsum(einsum_str, *operands, **kwargs)

    if not isinstance(einsum_str, str) or len(operands) != 2:
      return jnp.einsum(einsum_str, *operands, **kwargs)

    def stats_dot_general(lhs, rhs, dimension_numbers, *args, **kwargs):
      return self.dot_general(
          lhs, rhs, dimension_numbers, *args, rule=rule, **kwargs
      )

    with jax.disable_jit():
      return jnp.einsum(
          einsum_str,
          *operands,
          _dot_general=stats_dot_general,
          **kwargs,
      )

  def get_intercept_map(self) -> dict[str, Callable[..., Any]]:
    return super().get_intercept_map() | {
        'jax.lax.dot_general': self.dot_general,
        'jax.numpy.einsum': self.einsum,
    }


class SinglePassCalibrationProvider(CalibrationProvider, metaclass=abc.ABCMeta):
  """Calibration provider that collects single-pass statistics.

  This provider implements the simple stats template: `compute_stats`
  produces a dict of arrays, which are accumulated into the `quant_stats`
  collection using `SimpleMovingAverage`.
  """

  @abc.abstractmethod
  def compute_stats(self, lhs: jax.Array) -> dict[str, Any]:
    """Computes statistics from the input array."""

  def _collect_stats(self, lhs: jax.Array, weight_name: str) -> None:
    stats = self.compute_stats(lhs)
    aggregator = averaging.SimpleMovingAverage()
    var_name = weight_name + self.get_stats_suffix()
    quant_stat = flax_util.get_or_create_variable(
        'quant_stats', var_name, lambda: aggregator.init(stats)
    )
    if flax_util.should_update_quant_stats():
      quant_stat.value = aggregator.update(quant_stat.value, stats)


class TwoPassCalibrationProvider(CalibrationProvider, metaclass=abc.ABCMeta):
  """Calibration provider that requires two forward passes per batch.

  This base class encapsulates the two-pass calibration protocol:

  1. Float pass: Cache float-precision activations (original model weights).
  2. Quantized pass: Compute stats using cached float and current quantized
     activations.

  Subclasses implement ``compute_stats`` to define algorithm-specific
  stat computation from the float/quantized activation pair.

  Two-pass calibration must run in eager mode (outside ``jax.jit``) because
  the float LHS cache is Python-side state that cannot be traced. Use
  ``calibrate_batch()`` to run both passes.
  """

  _FLOAT_MODE = 'float'
  _QUANTIZED_MODE = 'quantized'

  def __init__(self, rules):
    super().__init__(rules)
    self._mode: str | None = None
    self._float_lhs_cache: dict[str, jax.Array] = {}

  @abc.abstractmethod
  def compute_stats(
      self, quantized_lhs: jax.Array, float_lhs: jax.Array
  ) -> dict[str, Any]:
    """Computes statistics from quantized and float activation pair.

    Args:
      quantized_lhs: Quantized input activations, shape (ca, samples).
      float_lhs: Float input activations, shape (ca, samples).

    Returns:
      A dict of stat arrays to accumulate via SimpleMovingAverage.
    """

  def calibrate_batch(
      self,
      cal_model: Any,
      float_variables: Any,
      quant_variables: Any,
      *args,
      **kwargs,
  ) -> dict[str, Any]:
    """Run both float and quantized passes for one calibration batch.

    Args:
      cal_model: The model returned by qwix_model.quantize_model().
      float_variables: Variables dict with float params (original weights).
      quant_variables: Variables dict with dequantized PTQ params.
      *args: Positional args passed to cal_model.apply() (e.g., input batch).
      **kwargs: Keyword args passed to cal_model.apply().

    Returns:
      The new variables dict containing accumulated quant_stats from the
      quantized pass.
    """
    # Pass 1: Float forward pass to cache float-precision activations.
    self._mode = self._FLOAT_MODE
    self._float_lhs_cache.clear()
    cal_model.apply(float_variables, *args, mutable='quant_stats', **kwargs)

    # Validate that the float pass cached something.
    if not self._float_lhs_cache:
      raise ValueError(
          'No float activations cached during float pass. '
          'Ensure the model has matching rule layers.'
      )

    # Pass 2: Quantized forward pass to compute and accumulate stats.
    self._mode = self._QUANTIZED_MODE
    _, new_vars = cal_model.apply(
        quant_variables, *args, mutable='quant_stats', **kwargs
    )
    self._mode = None
    return new_vars

  def _collect_stats(self, lhs: jax.Array, weight_name: str) -> None:
    if self._mode is None:
      raise ValueError(
          'Must use calibrate_batch() to run two-pass calibration.'
      )

    var_name = weight_name + self.get_stats_suffix()

    # Cache key includes module path to distinguish layers with the same
    # parameter name (e.g., Dense_0/kernel vs Dense_1/kernel).
    module_path = '/'.join(map(str, flax_util.get_current_module_path()))
    cache_key = module_path + '/' + var_name

    if self._mode == self._FLOAT_MODE:
      self._float_lhs_cache[cache_key] = lhs
    elif self._mode == self._QUANTIZED_MODE:
      float_lhs = self._float_lhs_cache.get(cache_key)
      if float_lhs is None:
        raise ValueError(
            f'No cached float LHS for {cache_key}. Ensure float pass '
            'covers the same operations as quantized pass.'
        )
      stats = self.compute_stats(lhs, float_lhs)

      # Accumulate stats via SimpleMovingAverage.
      aggregator = averaging.SimpleMovingAverage()
      quant_stat = flax_util.get_or_create_variable(
          'quant_stats', var_name, lambda: aggregator.init(stats)
      )
      if flax_util.should_update_quant_stats():
        quant_stat.value = aggregator.update(quant_stat.value, stats)


def normalize_weight(
    x: jax.Array, contraction_axis: int
) -> tuple[jax.Array, Callable[..., Any]]:
  """Normalizes a weight tensor into (rows, columns) format.

  Reshapes a weight tensor of arbitrary rank into a 2D matrix where the
  contraction axis becomes the last dimension. Returns a restore function
  to undo the transformation.

  Args:
    x: Weight tensor of arbitrary shape.
    contraction_axis: The axis that will be contracted in the matrix multiply.

  Returns:
    A tuple of (normalized_weight, restore_shape):
      - normalized_weight: The weight reshaped to (ra, ca) format.
      - restore_shape: A function to restore the original shape.
  """
  x = jnp.moveaxis(x, contraction_axis, -1)
  before_shape = x.shape
  x = x.reshape(-1, x.shape[-1])

  def restore_shape(x):
    x = x.reshape(before_shape)
    return jax.tree.map(lambda x: jnp.moveaxis(x, -1, contraction_axis), x)

  return x, restore_shape


@dataclasses.dataclass(frozen=True)
class CalibratedQuantContext:
  """Context containing a weight, its calibration stats, and metadata for quantization.

  Attributes:
    weight: Normalized weight in (rows, columns) format.
    how: The HowToQuantize with axes adjusted for the normalized shape.
    calibration_stats: The averaged calibration statistics dict.
    abs_w: The original WithAux wrapper (for metadata access).
    contracting_axis: The original contracting axis before normalization.
    restore_shape: Function to restore the weight to its original shape.
    path: The flattened dict path for this weight.
  """

  weight: jax.Array
  how: Any
  calibration_stats: dict[str, jax.Array]
  abs_w: ptq.WithAux
  contracting_axis: int
  restore_shape: Callable[..., Any]
  path: tuple[str, ...]


def quantize_params_with_calibration(
    params: Any,
    abstract_quantized_params: Any,
    quant_stats: Any,
    stats_suffix: str,
    quantize_fn: Callable[[CalibratedQuantContext], Any],
    *,
    allow_extra_params: bool = False,
) -> Any:
  """Shared framework for calibration-based weight quantization.

  This function handles the common boilerplate for all calibration-based
  quantization algorithms (GPTQ, QEP, AWQ): parameter iteration, stats
  lookup, weight normalization, and PTQ fallback. The algorithm-specific
  logic is provided via `quantize_fn`.

  Args:
    params: The floating-point param tree to quantize.
    abstract_quantized_params: The param tree from PTQ model with WithAux
      wrappers containing HowToQuantize information.
    quant_stats: The quant_stats dict from a CalibrationProvider.
    stats_suffix: The suffix for looking up stats (e.g., '_gptq').
    quantize_fn: A callable that takes a PreparedWeight and returns the
      quantized result to store in the output tree.
    allow_extra_params: If True, allow extra parameters not in
      abstract_quantized_params.

  Returns:
    The quantized params tree.
  """
  quantized_params = {}
  not_quantized_params = {}
  for path, w in flax.traverse_util.flatten_dict(params).items():
    abs_w = ptq.get_value_from_path(abstract_quantized_params, path)
    stats_path = (*path[:-1], path[-1] + stats_suffix)
    stats = ptq.get_value_from_path(quant_stats, stats_path)

    if not isinstance(abs_w, ptq.WithAux) or stats is None:
      not_quantized_params[path] = w
      continue

    # Get the contracting axis by assuming that all non-contracting axes
    # are in channelwise_axes.
    contracting_axis = set(range(w.ndim)) - set(abs_w.how.channelwise_axes)
    if len(contracting_axis) != 1:
      # Fallback to PTQ if we can't identify a single contracting axis.
      not_quantized_params[path] = w
      continue
    contracting_axis = list(contracting_axis)[0]

    # Normalize the weight to (ra, ca) format.
    w_norm, restore_shape = normalize_weight(w, contracting_axis)
    how = dataclasses.replace(abs_w.how, channelwise_axes=[0])
    if contracting_axis in how.tiled_axes:
      how = dataclasses.replace(
          how, tiled_axes={1: how.tiled_axes[contracting_axis]}
      )

    # Get calibration stats.
    calibration_stats = averaging.SimpleMovingAverage().get_calibration(stats)

    # Delegate to algorithm-specific quantization.
    ctx = CalibratedQuantContext(
        weight=w_norm,
        how=how,
        calibration_stats=calibration_stats,
        abs_w=abs_w,
        contracting_axis=contracting_axis,
        restore_shape=restore_shape,
        path=path,
    )
    quantized_params[path] = quantize_fn(ctx)

  # PTQ fallback for non-quantized params.
  not_quantized_params = flax.traverse_util.unflatten_dict(not_quantized_params)
  ptq_quantized_params = ptq.quantize_params(
      not_quantized_params,
      abstract_quantized_params,
      allow_extra_params=allow_extra_params,
  )
  ptq_quantized_params = flax.traverse_util.flatten_dict(ptq_quantized_params)
  quantized_params.update(ptq_quantized_params)

  return flax.traverse_util.unflatten_dict(quantized_params)
