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
"""Qwix for ODML."""

import dataclasses
import functools
from typing import Any, Sequence, Type

import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from qwix import aux_data
from qwix import averaging
from qwix import odml_ops
from qwix import qat
from qwix import qconfig
from qwix.core import qarray


class OdmlQatProvider(qat.QatProvider):
  """QAT provider for ODML.

  Compared with the regular QAT provider, this provider
    * Quantizes all ops more than just conv, einsum, and dot_general.
    * Quantizes output activations via a delayed fake_quant.
    * Supports limited per-channel quantization for weights.
    * Doesn't support subchannel quantization.
  """

  def __init__(
      self,
      rules: Sequence[qconfig.QuantizationRule],
      *,
      disable_per_channel_weights: bool = False,
      # TODO: Update these two args to fixed_range_for_{inputs,outputs}.
      static_calibration_for_input: dict[str, float] | None = None,
      static_calibration_for_output: dict[str, float] | None = None,
      strict: bool = True,
  ):
    """Constructor.

    Args:
      rules: The quantization rules.
      disable_per_channel_weights: Whether to disable per-channel quantization
        for weights.
      static_calibration_for_input: The static calibration for the model input,
        e.g. {'min': 0, 'max': 1}.
      static_calibration_for_output: The static calibration for the model
        output.
      strict: Whether to raise an error if an unknown op is discovered.
    """
    super().__init__(rules)
    self._fixed_range_for_inputs = None
    if static_calibration_for_input is not None:
      self._fixed_range_for_inputs = (
          static_calibration_for_input['min'],
          static_calibration_for_input['max'],
      )
    self._fixed_range_for_outputs = None
    if static_calibration_for_output is not None:
      self._fixed_range_for_outputs = (
          static_calibration_for_output['min'],
          static_calibration_for_output['max'],
      )
    self._strict = strict
    self._ops = odml_ops.get_all_ops()

    for name in [
        'jax.lax.conv_general_dilated',
        'jax.lax.dot_general',
        'jax.numpy.einsum',
    ]:
      self._ops[name] = functools.partial(
          self._ops[name],
          disable_per_channel_weights=disable_per_channel_weights,
          check_activation=strict,
      )

  def _init_rule(
      self, rule: qconfig.QuantizationRule
  ) -> qconfig.QuantizationRule:
    """Set ODML specific default values."""
    if rule.act_qtype is not None and rule.act_static_scale is None:
      rule = dataclasses.replace(rule, act_static_scale=True)
    if rule.act_calibration_method is None:
      rule = dataclasses.replace(rule, act_calibration_method='minmax')
    return super()._init_rule(rule)

  def get_intercept_map(self):
    """Used for interception."""
    # Only use the parent's nn_param. Others are handled by ourselves.
    intercept_map = {'flax.linen.Module.param': self.nn_param}
    # Compile the policy into the intercept map.
    for name, op in self._ops.items():
      op: Type[odml_ops.QuantizedOp]
      intercept_map[name] = op(
          op_full_name=name,
          get_rule_and_op_id_fn=self._get_current_rule_and_op_id,
          fake_quant_fn=self._fake_quant,
      )
    return intercept_map

  def process_model_inputs(
      self, model: Any, model_args: Any, model_kwargs: Any
  ) -> tuple[Any, Any, Any]:
    """Quantize the input of the model."""
    op = odml_ops.ModelInput(
        fixed_range_for_output=self._fixed_range_for_inputs,
        get_rule_and_op_id_fn=self._get_current_rule_and_op_id,
        fake_quant_fn=self._fake_quant,
    )
    return model, *jax.tree.map(op, (model_args, model_kwargs))

  def process_model_output(self, method_name: str, model_output: Any) -> Any:
    """Quantize the output of the model."""
    if method_name == '__call__':
      method_name = 'final'  # backwards compatibility.
    op = odml_ops.FinalOutput(
        op_full_name=method_name + '_output',
        fixed_range_for_output=self._fixed_range_for_outputs,
        get_rule_and_op_id_fn=self._get_current_rule_and_op_id,
        fake_quant_fn=self._fake_quant,
        check_activation=self._strict,
    )
    return jax.tree.map(op, model_output)


class OdmlConversionProvider(OdmlQatProvider):
  """Quantization provider for ODML conversion.

  This mode is similar to OdmlQatProvider, but all fake_quant ops are annotated
  by composites and the scales are computed statically in numpy.

  Supported modes:
    * Weight-only quantization.
    * Static-range quantization.

  Usage:
    # The params can be from QAT or the FP model.
    params = ...

    # If using static-range quantization, quant_stats are needed and can be
    # obtained by either 1) QAT or 2) calibrating.
    quant_stats = ...

    # Apply OdmlConversionProvider to the model.
    conversion_model = qwix.quantize_model(
        fp_model, qwix.OdmlConversionProvider(rules, params, quant_stats)
    )
    # Convert and get the ODML model, which is an ai_edge_jax.model.TfLiteModel.
    odml_model = ai_edge_jax.convert(
        conversion_model.apply, {'params': params}, (inputs,)
    )
    # The odml_model can be exported or directly run.
    odml_model.export('/tmp/odml_model.tflite')
    odml_model(inputs)
  """

  def __init__(
      self,
      rules: Sequence[qconfig.QuantizationRule],
      params,
      quant_stats,
      **kwargs,
  ):
    super().__init__(rules, **kwargs)
    # Store params and quant_stats statically so they won't become tracers.
    self._flatten_params = flax.traverse_util.flatten_dict(params)
    self._quant_stats = quant_stats

  def _fake_quant(
      self,
      array: jax.Array,
      how: qarray.HowToQuantize,
      quant_stat_name: str | None = None,
  ) -> jax.Array:
    assert not how.tiled_axes, 'Tiled axes are not supported in ODML.'
    how = dataclasses.replace(how, scale_transpose=None)

    # Make the scale and zero point statically computed.
    with jax.ensure_compile_time_eval():
      # Check if the array is a weight or an activation.
      weight_name = aux_data.get(array, 'weight_name', None)
      if weight_name is not None:  # Weights.
        mdl: nn.Module = nn.module._context.module_stack[-1]  # pylint: disable=protected-access
        weight = self._flatten_params[mdl.path + (weight_name,)]
        calibration = qarray.calibrate(weight, how)
        scale, zp = qarray.compute_scale_zero_point(calibration, how.qtype)
      elif quant_stat_name is not None:  # Static-range activations.
        scale, zp = self._compute_static_scale_zero_point(how, quant_stat_name)
      else:  # Dynamic-range activations.
        scale, zp = None, None

      attributes = self._get_attributes(
          scale=scale, zp=zp, dtype=how.qtype, is_weight=weight_name is not None
      )

    @functools.partial(jax.lax.composite, name='quant.fake_quant')
    def _fake_quant_op(x, **attributes):
      del attributes  # attributes are only for the composite op.
      return qarray.dequantize(
          qarray.quantize(x, how)
          if scale is None
          else qarray.quantize_with_scale_zero_point(x, how, scale, zp)
      )

    return _fake_quant_op(array, **attributes)

  def _compute_static_scale_zero_point(
      self, how: qarray.HowToQuantize, quant_stat_name: str
  ) -> tuple[jax.Array, jax.Array | None]:
    """Statically compute the scale and zero point for weights or activations."""
    # Look up the quant_stat for the activation.
    obj = self._quant_stats
    for key in nn.module._context.module_stack[-1].path:  # pylint: disable=protected-access
      obj = obj[key]
    quant_stat = obj[quant_stat_name]

    if 'count' not in quant_stat or quant_stat['count'] == 0:
      raise ValueError(f'quant_stats is not initialized for {quant_stat_name}')
    if any(jnp.isnan(v).any() for v in quant_stat.values()):
      raise ValueError(f'quant_stats has NaN for {quant_stat_name}')
    calibration = averaging.SimpleMovingAverage().get_calibration(quant_stat)
    return qarray.compute_scale_zero_point(calibration, how.qtype)

  def _get_attributes(
      self,
      *,
      scale: jax.Array | None,
      zp: jax.Array | None,
      dtype: jax.typing.DTypeLike,
      is_weight: bool,
  ) -> dict[str, Any]:
    """Return the attributes for the fake_quant composite."""
    # For dynamic-range quantization, the scale is an empty array.
    if scale is None:
      scale = np.array([], np.float32)
    # Flatten the scale because ODML wants a 1D array.
    quantization_dim = None
    for dim, length in enumerate(scale.shape):
      if length > 1:
        if quantization_dim is None:
          quantization_dim = dim
        else:
          raise ValueError(f'Cannot flatten scale with shape {scale.shape}.')
    match jnp.dtype(dtype):
      case jnp.int8:
        dtype = 'i8'
      case _:
        raise ValueError(f'Unsupported dtype {dtype} for ODML conversion.')
    attributes = {
        'scale': np.asarray(scale, np.float32).flatten(),
        'dtype': dtype,
        # narrow_range is an ODML-specific optimization that reduces the range
        # of int8 quantization from [-128, 127] to [-127, 127], such that the
        # int8 x int8 product can be represented in int16. LiteRT quantization
        # spec requires narrow_range to be True for weights.
        #
        # Since Qwix uses [-127.5, 127.5] in symmetric quantization, setting
        # it to True will only affect exact -127.5 and should have minimal
        # impact on the quantization result.
        'narrow_range': is_weight,
    }
    if zp is not None:
      # zero_point has to be int64 for ODML.
      attributes['zero_point'] = np.asarray(zp, np.int64).flatten()
    if quantization_dim is not None:
      attributes['quantization_dimension'] = quantization_dim
    return attributes
