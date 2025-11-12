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
"""Model-level quantization config."""

from collections.abc import Callable, Collection, Sequence
import dataclasses
import re
from typing import Any

from absl import logging
from flax import nnx
import jax
from jax.experimental import pallas as pl
from qwix._src import aux_data
from qwix._src import flax_util
from qwix._src import interception


@dataclasses.dataclass(frozen=True, kw_only=True)
class QuantizationRule:
  """Quantization rules that match and configure the quantization behavior."""

  ########################################################
  ### "Keys" that are used to match modules.
  ########################################################

  # A regex for matching the module path.
  module_path: str = '.*'

  # The ops this rule applies to, which is a collection of op names, e.g.,
  # einsum. If not set, the rule applies to all ops.
  op_names: Collection[str] = ()

  ########################################################
  ### "Configs" that specify the quantization behavior.
  ########################################################

  # Quantized type for weights.
  weight_qtype: jax.typing.DTypeLike | None = None

  # If set, quantize the activations to the given type.
  act_qtype: jax.typing.DTypeLike | None = None

  # If set, enable subchannel for the contraction axis with the given tile size.
  # If it's a float, it must be "1 / tile_count" and the actual tile size will
  # be round(axis_size * tile_size).
  tile_size: int | float | None = None

  # If set, enable static-range quantization for input activations. Quantization
  # statistics for the input activations are collected and will appear in the
  # Flax collection "quant_stats". This requires act_qtype to be set.
  #
  # If None, the default behavior is to use static-range quantization in ODML
  # and dynamic-range quantization in XLA.
  act_static_scale: bool | None = None

  # The method to calibrate weights, in format of <method>[,<args>]. Supported
  # methods are:
  #   absmax[,<scale>]: symmetric quantization using maximum absolute value. A
  #     scale factor is optional, e.g. "absmax,0.8" clips the range to 0.8x.
  #   minmax[,<scale>]: asymmetric quantization using min and max values. A
  #     scale factor is optional.
  #   rms,<scale>: symmetric quantization using root mean square. A scale factor
  #     is required for this method.
  #   fixed,<min>,<max>: use a fixed range [min, max], e.g. "fixed,-10,10". The
  #     quantization is symmetric if min == -max.
  weight_calibration_method: str = 'absmax'

  # The method to calibrate activations. Supported methods are the same as
  # weight_calibration_method. If None, the default behavior is to use minmax
  # in ODML and absmax in XLA.
  act_calibration_method: str | None = None

  # Batch axes when calculating scales for activations. This only applies when
  # static-range quantization is enabled. Batch axes are treated differently
  # during calibration, i.e., the mean of quant stats is calculated along the
  # batch axes. In dynamic-range quantization, the batch axes will have
  # per-channel scales and this config is ignored.
  act_batch_axes: Collection[int] = (0,)


def get_current_rule(op_name: str) -> QuantizationRule | None:
  """Returns the current quantization rule if intercepted, or None otherwise."""
  del op_name
  return None


class QuantizationProvider:
  """Interface for model integration.

  A provider can be either explicitly called by model authors, or implicitly
  injected into the model by using interception.py.
  """

  def __init__(self, rules: Sequence[QuantizationRule]):
    """Initialize the provider.

    Args:
      rules: The quantization rules in the order of precedence.
    """
    self._rules = [self._init_rule(rule) for rule in rules]
    self._logged_ops = set()

  def _init_rule(self, rule: QuantizationRule) -> QuantizationRule:
    """Validate and set default values for the rule."""
    if rule.act_qtype is None and rule.act_static_scale is not None:
      raise ValueError(f'Invalid rule: {rule}.')
    if rule.act_static_scale is None:
      rule = dataclasses.replace(rule, act_static_scale=False)
    if rule.act_calibration_method is None:
      rule = dataclasses.replace(rule, act_calibration_method='absmax')
    return rule

  def get_intercept_map(self) -> dict[str, Callable[..., Any]]:
    """Returns the intercept map for interception.wrap_func_intercepted."""
    # Common functions that are intercepted by all quantization providers.
    intercept_map = {
        'qwix._src.qconfig.get_current_rule': (
            lambda op: self._get_current_rule_and_op_id(op, only_rule=True)[0]
        )
    }
    if interception.has_attribute('jax.experimental.pallas.pallas_call'):
      # Disable interception for ops in pallas_call.
      intercept_map['jax.experimental.pallas.pallas_call'] = (
          lambda *args, **kwargs: interception.disable_interceptions(
              pl.pallas_call(*args, **kwargs)
          )
      )
    return intercept_map

  def process_model_inputs(
      self, model: Any, model_args: Sequence[Any], model_kwargs: dict[str, Any]
  ) -> tuple[Any, Sequence[Any], dict[str, Any]]:
    """Process the model and its inputs before it is called."""
    if isinstance(model, nnx.Module):
      for _, node in model.iter_modules():
        # Clear the op_count which is used in _get_current_rule_and_op_id below.
        aux_data.clear(node)
    return model, model_args, model_kwargs

  def process_model_output(self, method_name: str, model_output: Any) -> Any:
    """Process the model output before it is returned."""
    del method_name
    return model_output

  def _get_current_rule_and_op_id(
      self,
      op_name: str,
      *,
      only_rule: bool = False,
      repeated_call: bool = False,
  ) -> tuple[QuantizationRule | None, str | None]:
    """Returns the quantization rule and a unique op id for given op.

    This function is intended to be called by subclasses.

    Args:
      op_name: The name of the op.
      only_rule: If True, only return the rule and not the op id.
      repeated_call: If True, use the previous op id. This is useful when this
        function is called multiple times for the same op.

    Returns:
      A tuple of the quantization rule and a unique op id for the given op.
    """
    # Lookup the rule.
    module_path = '/'.join(map(str, flax_util.get_current_module_path()))
    rule_idx = None
    for idx, rule in enumerate(self._rules):
      if re.fullmatch(rule.module_path, module_path) and (
          not rule.op_names or op_name in rule.op_names
      ):
        rule_idx = idx
        break
    rule = self._rules[rule_idx] if rule_idx is not None else None
    if only_rule:
      return rule, None

    # Always generate op_id regardless of whether a rule is found.
    module = flax_util.get_current_module()
    count = aux_data.get(module, op_name + '_count', 0)
    if repeated_call:
      count -= 1
    else:
      aux_data.set(module, op_name + '_count', count + 1)
    op_id = op_name + str(count)

    if (module_path, op_id) not in self._logged_ops:
      # Avoid logging the same message multiple times.
      self._logged_ops.add((module_path, op_id))
      logging.info(
          '[QWIX] module=%r op=%s rule=%s', module_path, op_id, rule_idx
      )
    return rule, op_id
