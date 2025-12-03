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

"""Integration of GPTQ into Qwix.

During inference, GPTQ uses the same PtqProvider as PTQ. The only difference is
that GPTQ requires an extra calibration step to produce gptq_quant_stats, which
will then be consumed by the GPTQ's quantize_params function. After that, the
quantized params tree will look exactly the same as PTQ's.

Please check the test for an example usage.
"""

import dataclasses
from typing import Any, Callable

import flax
import jax
from jax import numpy as jnp
from qwix._src import averaging
from qwix._src import flax_util
from qwix._src import qconfig
from qwix._src.providers import ptq
from qwix.contrib import gptq_core


@dataclasses.dataclass(frozen=True, kw_only=True)
class GptqRule(qconfig.QuantizationRule):
  """Use this rule to enable GPTQ."""


class GptqCalibrationProvider(qconfig.QuantizationProvider):
  """Calibration for GPTQ.

  This provider is used to collect gptq_quant_stats, which will be used by the
  quantize_params function below. It does not perform the actual quantization of
  the model parameters, nor use any quantized ops.
  """

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      *args,
      **kwargs,
  ) -> jax.Array:
    res = jax.lax.dot_general(lhs, rhs, dimension_numbers, *args, **kwargs)
    rule, _ = self._get_current_rule_and_op_id('dot_general')
    if not isinstance(rule, GptqRule):
      return res

    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    if lhs_ba or rhs_ba or len(lhs_ca) != 1 or len(rhs_ca) != 1:
      raise NotImplementedError(f'Unsupported: {dimension_numbers}')

    weight_name = flax_util.find_param(rhs)
    assert weight_name is not None

    # Reorder lhs to (ca, rest).
    lhs = jnp.moveaxis(lhs, lhs_ca[0], 0)
    lhs = lhs.reshape(lhs.shape[0], -1)
    hessian = gptq_core.compute_hessian(lhs)

    # Collect the hessian.
    hessian = {'hessian': hessian}
    aggregator = averaging.SimpleMovingAverage()
    quant_stat = flax_util.get_or_create_variable(
        'quant_stats', weight_name + '_gptq', lambda: aggregator.init(hessian)
    )
    if flax_util.should_update_quant_stats():
      quant_stat.value = aggregator.update(quant_stat.value, hessian)

    return res

  def get_intercept_map(self) -> dict[str, Callable[..., Any]]:
    return super().get_intercept_map() | {
        'jax.lax.dot_general': self.dot_general
    }


def quantize_params(
    params: Any,
    abstract_quantized_params: Any,
    gptq_quant_stats: Any,
    *,
    allow_extra_params: bool = False,
    gptq_block_size: int = 128,
    gptq_damping_factor: float = 0.01,
) -> Any:
  """Quantizes the params with GPTQ.

  Args:
    params: See ptq.quantize_params.
    abstract_quantized_params: See ptq.quantize_params.
    gptq_quant_stats: The quant_stats dict from GptqCalibrationProvider. SRQ is
      not supported yet. For params with no gptq_quant_stats, they will be
      quantized with the default PTQ algorithm.
    allow_extra_params: See ptq.quantize_params.
    gptq_block_size: The block size of GPTQ.
    gptq_damping_factor: The damping factor of GPTQ.

  Returns:
    The quantized params consumable by PtqProvider.
  """
  quantized_params = {}
  not_quantized_params = {}
  for path, w in flax.traverse_util.flatten_dict(params).items():
    abs_w = ptq.get_value_from_path(abstract_quantized_params, path)
    gptq_stats_path = (*path[:-1], path[-1] + '_gptq')
    gptq_stats = ptq.get_value_from_path(gptq_quant_stats, gptq_stats_path)

    if not isinstance(abs_w, ptq.WithAux) or gptq_stats is None:
      # Not quantized by GPTQ.
      not_quantized_params[path] = w
      continue

    # HACK: get the contracting axis by assuming that all non-contracting axes
    # are in channelwise_axes.
    contracting_axis = set(range(w.ndim)) - set(abs_w.how.channelwise_axes)
    assert len(contracting_axis) == 1
    contracting_axis = list(contracting_axis)[0]

    # Normalize the weight to (ra, ca) format.
    w, restore_shape = gptq_core.normalize_weight(w, contracting_axis)
    how = dataclasses.replace(abs_w.how, channelwise_axes=[0])
    if contracting_axis in how.tiled_axes:
      how = dataclasses.replace(
          how, tiled_axes={1: how.tiled_axes[contracting_axis]}
      )

    # Get the hessian, which should be (ca, ca).
    calibration = averaging.SimpleMovingAverage().get_calibration(gptq_stats)
    hessian = calibration['hessian']
    assert hessian.shape[0] == w.shape[1] and hessian.shape[1] == w.shape[1]

    # Quantize the weight with GPTQ.
    w = gptq_core.quantize_weight(
        w, hessian, how, blocksize=gptq_block_size, percdamp=gptq_damping_factor
    )[0]
    w = restore_shape(w)
    quantized_params[path] = abs_w.replace(array=w)

  # Quantize the non-GPTQ params with PTQ.
  not_quantized_params = flax.traverse_util.unflatten_dict(not_quantized_params)
  ptq_quantized_params = ptq.quantize_params(
      not_quantized_params,
      abstract_quantized_params,
      allow_extra_params=allow_extra_params,
  )
  ptq_quantized_params = flax.traverse_util.flatten_dict(ptq_quantized_params)
  quantized_params.update(ptq_quantized_params)

  return flax.traverse_util.unflatten_dict(quantized_params)
