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

"""Integration of QEP (Quantization Error Propagation) into Qwix.

QEP extends PTQ methods like GPTQ by accounting for quantization noise in input
activations from previous layers. It requires a two-pass calibration (float +
quantized) to collect hessian_delta statistics, then applies weight correction
before standard PTQ quantization.

During inference, QEP uses the same PtqProvider as the PTQ equivalent. The only
difference is the calibration and weight quantization steps.

Please check the test for an example usage.
"""

import dataclasses
import enum
from typing import Any

import flax
import jax
from jax import numpy as jnp
from qwix._src import averaging
from qwix._src import flax_util
from qwix._src import qconfig
from qwix._src.providers import ptq
from qwix.contrib import calibration
from qwix.contrib import gptq
from qwix.contrib import gptq_core
from qwix.contrib import qep_core


@dataclasses.dataclass(frozen=True, kw_only=True)
class QepRule(gptq.GptqRule):
  """Use this rule to enable QEP (input-compensated GPTQ).

  QEP extends PTQ methods like GPTQ by accounting for quantization noise in
  input activations from previous layers.

  Attributes:
    correction_factor: Weight correction factor. 0.0 = no correction, 1.0 = full
      correction. Default 0.5 per QEP paper recommendations.
    dampening_factor: Dampening factor for QEP weight correction Hessian
      inversion. Default 0.01.
  """

  correction_factor: float = 0.5
  dampening_factor: float = 0.01


@enum.unique
class _CalibrationMode(enum.Enum):
  FLOAT = enum.auto()
  QUANTIZED = enum.auto()


class QepCalibrationProvider(calibration.StatsCalibrationProvider):
  """Calibration provider for QEP (input-compensated GPTQ).

  QEP extends PTQ methods like GPTQ by accounting for quantization noise in
  input activations.

  This provider operates in two modes per calibration batch:

  1. Float mode: Caches float-precision LHS activations.
  2. Quantized mode: Computes QEP stats using the cached float LHS and
     the current quantized LHS.

  IMPORTANT: QEP calibration must run in eager mode (outside jax.jit) because
  the float LHS cache is Python-side state that cannot be traced.

  Use calibrate_batch() to run both passes for a single batch:

  ```python

  qep_provider = QepCalibrationProvider(rules)
  cal_model = qwix_model.quantize_model(model, qep_provider)

  for batch in calibration_data:
    new_vars = qep_provider.calibrate_batch(
        cal_model, float_variables, quant_variables, batch
    )
    variables.update(new_vars)

  ```
  """

  def __init__(self, rules):
    super().__init__(rules)
    self._mode: _CalibrationMode | None = None
    self._float_lhs_cache: dict[str, jax.Array] = {}

  def calibrate_batch(
      self,
      cal_model: Any,
      float_variables: Any,
      quant_variables: Any,
      *args,
      **kwargs,
  ) -> dict[str, Any]:
    """Run both float and quantized passes for one calibration batch.

    This encapsulates the two-pass QEP protocol.

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
    self._mode = _CalibrationMode.FLOAT
    self._float_lhs_cache.clear()
    cal_model.apply(float_variables, *args, mutable='quant_stats', **kwargs)

    # Validate that the float pass cached something.
    if not self._float_lhs_cache:
      raise ValueError(
          'No float activations cached during float pass. '
          'Ensure the model has QepRule-matched layers.'
      )

    # Pass 2: Quantized forward pass to compute and accumulate QEP stats.
    self._mode = _CalibrationMode.QUANTIZED
    _, new_vars = cal_model.apply(
        quant_variables, *args, mutable='quant_stats', **kwargs
    )
    self._mode = None
    return new_vars

  def get_rule_type(self) -> type[qconfig.QuantizationRule]:
    return QepRule

  def compute_stats(self, lhs: jax.Array) -> dict[str, Any]:
    raise NotImplementedError(
        'QepCalibrationProvider overrides dot_general directly.'
    )

  def get_stats_suffix(self) -> str:
    return '_qep'

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

    if not isinstance(rule, QepRule):
      return res

    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    if lhs_ba or rhs_ba or len(lhs_ca) != 1 or len(rhs_ca) != 1:
      return res

    weight_name = flax_util.find_param(rhs)
    if weight_name is None:
      return res

    if self._mode is None:
      raise ValueError('Must use calibrate_batch() to run QEP calibration.')

    # Reshape LHS to (ca, rest) format.
    lhs_reshaped = jnp.moveaxis(lhs, lhs_ca[0], 0)
    lhs_reshaped = lhs_reshaped.reshape(lhs_reshaped.shape[0], -1)

    var_name = weight_name + self.get_stats_suffix()

    # Cache key includes module path to distinguish layers with the same
    # parameter name (e.g., Dense_0/kernel vs Dense_1/kernel).
    module_path = '/'.join(map(str, flax_util.get_current_module_path()))
    cache_key = module_path + '/' + var_name

    if self._mode == _CalibrationMode.FLOAT:
      # Cache the float LHS for later use in quantized mode.
      self._float_lhs_cache[cache_key] = lhs_reshaped
    elif self._mode == _CalibrationMode.QUANTIZED:
      # Compute QEP stats using cached float LHS and current quantized LHS.
      float_lhs = self._float_lhs_cache.get(cache_key)
      if float_lhs is None:
        raise ValueError(
            f'No cached float LHS for {cache_key}. Ensure float pass '
            'covers the same operations as quantized pass.'
        )
      stats = qep_core.compute_qep_stats(lhs_reshaped, float_lhs)

      # Accumulate stats via SimpleMovingAverage.
      aggregator = averaging.SimpleMovingAverage()
      quant_stat = flax_util.get_or_create_variable(
          'quant_stats', var_name, lambda: aggregator.init(stats)
      )
      if flax_util.should_update_quant_stats():
        quant_stat.value = aggregator.update(quant_stat.value, stats)

    return res


def quantize_params(
    params: Any,
    abstract_quantized_params: Any,
    qep_quant_stats: Any,
    *,
    allow_extra_params: bool = False,
    gptq_block_size: int = 128,
    gptq_damping_factor: float = 0.01,
    correction_factor: float = 0.5,
    dampening_factor: float = 0.01,
) -> Any:
  """Quantizes the params with QEP (weight correction + GPTQ).

  Args:
    params: See ptq.quantize_params.
    abstract_quantized_params: See ptq.quantize_params.
    qep_quant_stats: The quant_stats dict from QepCalibrationProvider. For
      params with no qep_quant_stats, they will be quantized with the default
      PTQ algorithm.
    allow_extra_params: See ptq.quantize_params.
    gptq_block_size: The block size of GPTQ.
    gptq_damping_factor: The damping factor of GPTQ.
    correction_factor: QEP weight correction factor. 0.0 = no correction, 1.0 =
      full correction. Default 0.5 per QEP paper.
    dampening_factor: QEP damping factor for Hessian inversion. Default 0.01.

  Returns:
    The quantized params consumable by PtqProvider.
  """
  quantized_params = {}
  not_quantized_params = {}
  for path, w in flax.traverse_util.flatten_dict(params).items():
    abs_w = ptq.get_value_from_path(abstract_quantized_params, path)
    qep_stats_path = (*path[:-1], path[-1] + '_qep')
    qep_stats = ptq.get_value_from_path(qep_quant_stats, qep_stats_path)

    if not isinstance(abs_w, ptq.WithAux) or qep_stats is None:
      # Not quantized by QEP.
      not_quantized_params[path] = w
      continue

    contracting_axis = set(range(w.ndim)) - set(abs_w.how.channelwise_axes)
    if len(contracting_axis) != 1:
      # Fallback to PTQ if we can't identify a single contracting axis.
      not_quantized_params[path] = w
      continue

    contracting_axis = list(contracting_axis)[0]

    # Normalize the weight to (ra, ca) format.
    w, restore_shape = gptq_core.normalize_weight(w, contracting_axis)
    how = dataclasses.replace(abs_w.how, channelwise_axes=[0])
    if contracting_axis in how.tiled_axes:
      # Update tiling configuration to match the reshaped weight matrix,
      # where the contracting axis is now axis 1.
      how = dataclasses.replace(
          how, tiled_axes={1: how.tiled_axes[contracting_axis]}
      )

    # Get the calibration stats.
    calibration_stats = averaging.SimpleMovingAverage().get_calibration(
        qep_stats
    )
    hessian = calibration_stats['hessian']
    assert hessian.shape[0] == w.shape[1] and hessian.shape[1] == w.shape[1]

    # QEP weight correction (applied before GPTQ quantization).
    hessian_delta = calibration_stats.get('hessian_delta')
    if hessian_delta is None:
      raise ValueError(
          f'hessian_delta not found in QEP stats for {path}. '
          'Ensure QepCalibrationProvider was used for calibration.'
      )
    w = qep_core.weight_correct(
        w,
        hessian,
        hessian_delta,
        correction_factor=correction_factor,
        dampening_factor=dampening_factor,
    )

    # Quantize the input-compensated weight with GPTQ.
    w = gptq_core.quantize_weight(
        w, hessian, how, blocksize=gptq_block_size, percdamp=gptq_damping_factor
    )[0]
    w = restore_shape(w)
    quantized_params[path] = abs_w.replace(array=w)

  # Quantize the non-QEP params with PTQ.
  not_quantized_params = flax.traverse_util.unflatten_dict(not_quantized_params)
  ptq_quantized_params = ptq.quantize_params(
      not_quantized_params,
      abstract_quantized_params,
      allow_extra_params=allow_extra_params,
  )
  ptq_quantized_params = flax.traverse_util.flatten_dict(ptq_quantized_params)
  quantized_params.update(ptq_quantized_params)

  return flax.traverse_util.unflatten_dict(quantized_params)
