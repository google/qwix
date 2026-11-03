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
import contextlib
import dataclasses
import re
from typing import Any

from absl import logging
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from qwix._src import aux_data
from qwix._src import interception
from qwix._src.utils import flax_util


@dataclasses.dataclass(frozen=True, kw_only=True)
class QuantizationRule:
  """Quantization rules that match and configure the quantization behavior."""

  ########################################################
  ### "Keys" that are used to match modules.
  ########################################################

  # A regex for matching the module path.
  module_path: str = '.*'

  # Optional regex for excluding module paths that would otherwise match
  # module_path (e.g. '.*router.*|.*vision.*').
  exclude_module_path: str | None = None

  # The ops this rule applies to, which is a collection of op names, e.g.,
  # einsum. If not set, the rule applies to all ops.
  op_names: Collection[str] = ()

  # Step window [start_step, end_step) during which this rule is active when
  # evaluated inside flax_util.step_context(step). end_step=-1 means unbounded.
  start_step: int = 0
  end_step: int = -1

  # Optional collection of scan layer indices (0-indexed) to target when
  # evaluated inside flax_util.scan_index_context(scan_idx).
  target_scan_indices: Collection[int] | None = None

  ########################################################
  ### "Configs" that specify the quantization behavior.
  ########################################################

  # Quantized type for weights.
  weight_qtype: jax.typing.DTypeLike | None = None

  # If set, quantize the activations to the given type.
  act_qtype: jax.typing.DTypeLike | None = None

  # If set, enable subchannel for the innermost (last) contraction axis of both
  # operands with the given tile size; the axis must be a multiple of it.
  # If it's a float, it must be "1 / tile_count" and the actual tile size will
  # be round(axis_size * tile_size).
  # Microscaled formats (mxfp8, mxfp4, ...) have a fixed block size, which
  # tile_size must equal if set. In PTQ and QLoRA it can stay unset: each
  # microscaled operand uses its block size and the other operand is not
  # tiled, e.g. per-token activations with MXFP4 weights.
  # TODO(siyuez): Apply the same default in QT.
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


def is_rule_active_in_context(rule: QuantizationRule) -> jax.Array | bool:
  """Returns whether rule is active under current step and scan_index contexts."""
  active: jax.Array | bool = True
  step = flax_util.get_current_step()
  if step is not None and (rule.start_step > 0 or rule.end_step >= 0):
    if isinstance(step, int):
      if rule.start_step > 0 and step < rule.start_step:
        return False
      if rule.end_step >= 0 and step >= rule.end_step:
        return False
    else:
      step_arr = jnp.asarray(step)
      if rule.start_step > 0:
        active = active & (step_arr >= rule.start_step)
      if rule.end_step >= 0:
        active = active & (step_arr < rule.end_step)

  scan_idx = flax_util.get_current_scan_index()
  if scan_idx is not None and rule.target_scan_indices is not None:
    targets = tuple(rule.target_scan_indices)
    if not targets:
      return False
    if isinstance(scan_idx, int):
      if scan_idx not in targets:
        return False
    else:
      in_targets = jnp.isin(
          jnp.asarray(scan_idx), jnp.asarray(targets, dtype=jnp.int32)
      )
      active = active & in_targets
  return active


def get_current_rule(op_name: str) -> QuantizationRule | None:
  """Returns the current quantization rule if intercepted, or None otherwise."""
  del op_name
  return None


class QuantizationProvider:
  """Interface for model integration.

  A provider can be either explicitly called by model authors, or implicitly
  injected into the model by using interception.py.
  """

  def __init__(
      self, rules: Sequence[QuantizationRule], *, disable_jit: bool = False
  ):
    """Initialize the provider.

    Args:
      rules: The quantization rules in the order of precedence.
      disable_jit: Whether to disable JIT when wrapping methods.
    """
    self._rule_matches = [0] * len(rules)
    self._rules = [self._init_rule(rule) for rule in rules]
    self._logged_ops = set()
    self._initial_run_complete = False
    self.disable_jit = disable_jit

  def _init_rule(self, rule: QuantizationRule) -> QuantizationRule:
    """Validate and set default values for the rule."""
    if rule.act_qtype is None and rule.act_static_scale is not None:
      raise ValueError(
          f'Invalid rule: act_static_scale is set to {rule.act_static_scale} '
          'but act_qtype is None. Please either unset act_static_scale, '
          f'or set act_qtype to enable activation quantization. (Rule: {rule})'
      )
    if rule.act_static_scale is None:
      rule = dataclasses.replace(rule, act_static_scale=False)
    if rule.act_calibration_method is None:
      rule = dataclasses.replace(rule, act_calibration_method='absmax')
    if rule.target_scan_indices is not None:
      rule = dataclasses.replace(
          rule, target_scan_indices=tuple(rule.target_scan_indices)
      )
    return rule

  @contextlib.contextmanager
  def intercept(
      self,
      *,
      step: jax.Array | int | None = None,
      scan_index: jax.Array | int | None = None,
  ):
    """Context manager to activate this provider's interceptors and runtime context."""
    active_interceptors = []
    for get_interceptor in self.get_interceptors():
      interceptor = interception._preprocess_interceptor(  # pylint: disable=protected-access
          get_interceptor(), self.disable_jit
      )
      if not interception.interception_manager.is_active(interceptor):
        interception.interception_manager.activate_interceptor(interceptor)
        active_interceptors.append(interceptor)
    with contextlib.ExitStack() as stack:
      if step is not None:
        stack.enter_context(flax_util.step_context(step))
      if scan_index is not None:
        stack.enter_context(flax_util.scan_index_context(scan_index))
      try:
        yield self
      finally:
        for interceptor in reversed(active_interceptors):
          interception.interception_manager.deactivate_interceptor(interceptor)

  def get_intercept_map(self) -> dict[str, Callable[..., Any]]:
    """Returns the intercept map for interception.wrap_func_intercepted."""
    # Common functions that are intercepted by all quantization providers.
    intercept_map = {
        'qwix._src.qconfig.get_current_rule': (
            lambda op: self._get_current_rule_and_op_id(op, only_rule=True)[0]
        )
    }
    if interception.has_attribute('jax.experimental.pallas.pallas_call'):
      # 1. Disable interceptions during the pallas_call factory setup.
      # 2. Wrap the returned callable so that interceptions remain disabled
      #    during deferred execution.
      intercept_map['jax.experimental.pallas.pallas_call'] = (
          lambda *args, **kwargs: interception.disable_interceptions(
              interception.disable_interceptions(pl.pallas_call)(
                  *args, **kwargs
              )
          )
      )
    return intercept_map

  def get_interceptors(
      self,
  ) -> Sequence[Callable[[], interception.Interceptor]]:
    """Returns a list of interceptor factories.

    The default implementation returns a single interceptor that handles
    all ops. Subclasses can override this method to return multiple
    interceptors if needed (e.g. ODML providers).
    """
    return [
        lambda: interception.Interceptor(
            mapping=self.get_intercept_map(), id=id(self)
        )
    ]

  def process_model_inputs(
      self, model: Any, model_args: Sequence[Any], model_kwargs: dict[str, Any]
  ) -> tuple[Any, Sequence[Any], dict[str, Any]]:
    """Process the model and its inputs before it is called."""
    if isinstance(model, nnx.Module):
      for _, node in nnx.iter_modules(model):
        # Clear the op_count which is used in _get_current_rule_and_op_id below.
        aux_data.clear(node)
    return model, model_args, model_kwargs

  def process_model_output(self, method_name: str, model_output: Any) -> Any:
    """Process the model output before it is returned."""
    del method_name
    self._initial_run_complete = True
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
    try:
      module_path = '/'.join(map(str, flax_util.get_current_module_path()))
    except ValueError:
      return None, None
    rule_idx = None
    for idx, rule in enumerate(self._rules):
      if not re.fullmatch(rule.module_path, module_path):
        continue
      if rule.exclude_module_path and re.search(
          rule.exclude_module_path, module_path
      ):
        continue
      if not rule.op_names or op_name in rule.op_names:
        rule_idx = idx
        break
    rule = self._rules[rule_idx] if rule_idx is not None else None
    if rule_idx is not None:
      self._rule_matches[rule_idx] += 1
    if only_rule:
      return rule, None

    # Always generate op_id regardless of whether a rule is found.
    module = flax_util.get_current_module()
    tracking_key = module.scope if isinstance(module, nn.Module) else module
    count = aux_data.get(tracking_key, op_name + '_count', 0)
    if repeated_call:
      count -= 1
    else:
      aux_data.set(tracking_key, op_name + '_count', count + 1)
    op_id = op_name + str(count)

    if (module_path, op_id) not in self._logged_ops:
      # Avoid logging the same message multiple times.
      self._logged_ops.add((module_path, op_id))
      logging.debug(
          '[QWIX] module=%r op=%s rule=%s', module_path, op_id, rule_idx
      )
    return rule, op_id

  def get_unused_rules(self) -> Sequence[QuantizationRule]:
    """Returns the quantization rules that did not match any operations.

    This should be called after model quantization (e.g., `quantize_model`) to
    verify that all rules were applied as expected. A rule is considered unused
    if its `module_path` regex did not match any module's path, or if its
    `op_names` did not match any intercepted operation within a matching module.

    Returns:
      A sequence of unused quantization rules.
    """
    if not self._initial_run_complete:
      raise ValueError(
          'Quantization is not completed yet. Please call `quantize_model`'
          ' before calling `get_unused_rules`.'
      )
    return [
        self._rules[i]
        for i, rule_matches in enumerate(self._rule_matches)
        if rule_matches == 0
    ]
