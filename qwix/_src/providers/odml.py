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
from typing import Any, Callable, Sequence, Type

import flax
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from qwix._src import aux_data
from qwix._src import averaging
from qwix._src import flax_util
from qwix._src import qconfig
from qwix._src.core import qarray
from qwix._src.providers import odml_ops


class OdmlQatProvider(qconfig.QuantizationProvider):
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
      fixed_range_for_inputs: tuple[float, float] | None = None,
      fixed_range_for_outputs: tuple[float, float] | None = None,
      strict: bool = True,
  ):
    """Constructor.

    Args:
      rules: The quantization rules.
      disable_per_channel_weights: Whether to disable per-channel quantization
        for weights.
      fixed_range_for_inputs: Use a fixed range when quantizing the model
        inputs, e.g. (0, 1).
      fixed_range_for_outputs: Use a fixed range when quantizing the model
        outputs, e.g. (0, 1).
      strict: Whether to raise an error if an unknown op is discovered.
    """
    super().__init__(rules)
    self._fixed_range_for_inputs = fixed_range_for_inputs
    self._fixed_range_for_outputs = fixed_range_for_outputs
    self._strict = strict
    self._ops = odml_ops.get_all_ops()

    for name in [
        'jax.lax.conv_general_dilated',
        'jax.lax.dot_general',
        'jax.numpy.einsum',
        'jax.numpy.dot',
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

  def nn_param(
      self,
      module: nn.Module,
      name: str,
      init_fn: Callable[..., Any],
      *init_args,
      unbox: bool = True,
      **init_kwargs,
  ) -> jax.Array | nn.meta.AxisMetadata[jax.Array]:
    """Intercepts nn.Module.param to associate weight_name aux_data."""
    ret = nn.Module.param(
        module, name, init_fn, *init_args, unbox=unbox, **init_kwargs
    )
    # Clear the previous aux_data such as fq_array.
    aux_data.clear(ret if unbox else ret.unbox())
    # weight_name is used to distinguish weights from activations.
    aux_data.set(ret if unbox else ret.unbox(), 'weight_name', name)
    return ret

  def get_intercept_map(self):
    """Used for interception."""
    intercept_map = super().get_intercept_map()
    intercept_map['flax.linen.Module.param'] = self.nn_param
    # Add all the ops to the intercept map.
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
    # Set weight_name for nnx models. Linen models are handled in nn_param.
    if isinstance(model, nnx.Module):
      for path, node in nnx.iter_graph(model):
        if isinstance(node, nnx.Module):
          aux_data.clear(node)  # clear the op_count.
        elif isinstance(node, nnx.Param):
          # Clear the previous aux_data such as fq_array.
          aux_data.clear(node.value)
          # weight_name is used to distinguish weights from activations.
          aux_data.set(node.value, 'weight_name', path[-1])

    # Quantize the model inputs if needed.
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
    # Quantize the model output if needed.
    op = odml_ops.FinalOutput(
        op_full_name=method_name + '_output',
        fixed_range_for_output=self._fixed_range_for_outputs,
        get_rule_and_op_id_fn=self._get_current_rule_and_op_id,
        fake_quant_fn=self._fake_quant,
        check_activation=self._strict,
    )
    return jax.tree.map(op, model_output)

  def _fake_quant(
      self,
      array: jax.Array,
      how: qarray.HowToQuantize,
      quant_stat_name: str | None = None,
  ) -> jax.Array:
    """Apply fake quantization to array.

    This function can be used on both activations and weights. Gradient will be
    passed through.

    Args:
      array: The array to quantize.
      how: How to quantize the array.
      quant_stat_name: The name for the quantization statistics. If set, the
        quantization statistics will be collected and the scale will be computed
        from the statistics.

    Returns:
      The fake quantized array.
    """
    # Check and apply the fixed-range calibration asscociated with the array.
    fixed_range = aux_data.get(array, 'fixed_range', None)
    if fixed_range is not None:
      calibration_method = f'fixed,{fixed_range[0]},{fixed_range[1]}'
      how = dataclasses.replace(how, calibration_method=calibration_method)

    calibration = qarray.calibrate(array, how)
    if quant_stat_name is not None:
      is_fixed_range = how.calibration_method.startswith('fixed')
      calibration = self._collect_quant_stat(
          quant_stat_name, calibration, is_fixed_range
      )
    scale, zero_point = qarray.compute_scale_zero_point(calibration, how.qtype)
    q_array = qarray.quantize_with_scale_zero_point(
        array, how.qtype, scale, zero_point
    )
    dq_array = qarray.dequantize(q_array)
    # Use a straight through estimator as the gradient of the dq_array.
    ste_array = qarray.clip_to_calibration(array, calibration, how.tiled_axes)
    return ste_array + jax.lax.stop_gradient(dq_array - ste_array)

  def _collect_quant_stat(
      self,
      name: str,
      calibration: averaging.Calibration,
      calibration_is_fixed_range: bool,
  ) -> averaging.Calibration:
    """Collects the quantization statistics."""
    # For SRQ, only per-tensor scale is supported, so we don't need to check the
    # act_batch_axes at all.
    calibration = jax.tree.map(lambda x: x.mean(keepdims=True), calibration)

    aggregator = averaging.SimpleMovingAverage()
    quant_stat = flax_util.get_or_create_variable(
        'quant_stats', name, lambda: aggregator.init(calibration)
    )

    if flax_util.should_update_quant_stats():
      if calibration_is_fixed_range:
        # For fixed-range calibration, start from an empty quant_stat to avoid
        # floating-point accumulation error. Alternatively, we could skip
        # storing the quant_stat for fixed-range calibration.
        quant_stat.value = aggregator.init(calibration)
      quant_stat.value = aggregator.update(quant_stat.value, calibration)

    return aggregator.get_calibration(quant_stat.value, calibration)


class OdmlConversionProvider(OdmlQatProvider):
  """Quantization provider for ODML conversion.

  This mode is similar to OdmlQatProvider, but all fake_quant ops are annotated
  by composites and the scales are computed statically in numpy.

  Supported modes:

  * Weight-only quantization.
  * Static-range quantization.

  Usage::

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

  def get_intercept_map(self):
    intercept_map = super().get_intercept_map()
    # Override dot_general to flatten N-D weights to 2-D.
    intercept_map['jax.lax.dot_general'] = functools.partial(
        self._flatten_dot_general,
        _dot_general=intercept_map['jax.lax.dot_general'],
        _reshape=intercept_map['jax.lax.reshape'],
    )
    return intercept_map

  def _flatten_dot_general(self, *args, _dot_general, _reshape, **kwargs):
    """Flatten N-D weights to 2-D to support channelwise quantization."""
    # This special handling is needed because tflite doesn't support multiple
    # quantization_dimensions.
    if (
        aux_data.get(args[1], 'weight_name', None) is not None
        and args[1].ndim > 2
        and tuple(args[2][0][1]) == (0,)
    ):
      args = list(args)
      dout = args[1].shape[1:]
      args[1] = _reshape(args[1], (args[1].shape[0], np.prod(dout)))
      out = _dot_general(*args, **kwargs)
      return _reshape(out, out.shape[:-1] + dout)
    return _dot_general(*args, **kwargs)

  def _fake_quant(
      self,
      array: jax.Array,
      how: qarray.HowToQuantize,
      quant_stat_name: str | None = None,
  ) -> jax.Array:
    assert not how.tiled_axes, 'Tiled axes are not supported in ODML.'

    # Make the scale and zero point statically computed.
    with jax.ensure_compile_time_eval():
      # Check if the array is a weight or an activation.
      weight_name = aux_data.get(array, 'weight_name', None)
      if weight_name is not None:  # Weights.
        assert quant_stat_name is None
        mdl_path = flax_util.get_current_module_path()
        weight = self._flatten_params[mdl_path + (weight_name,)]
        if weight.shape != array.shape:  # when _flatten_dot_general is used.
          weight = weight.reshape(array.shape)
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
          else qarray.quantize_with_scale_zero_point(x, how.qtype, scale, zp)
      )

    return _fake_quant_op(array, **attributes)

  def _compute_static_scale_zero_point(
      self, how: qarray.HowToQuantize, quant_stat_name: str
  ) -> tuple[jax.Array, jax.Array | None]:
    """Statically compute the scale and zero point for weights or activations."""
    # Look up the quant_stat for the activation.
    obj = self._quant_stats
    for key in flax_util.get_current_module_path():
      obj = obj[key]
    quant_stat = obj[quant_stat_name]

    if 'count' not in quant_stat or quant_stat['count'] == 0:
      raise ValueError(f'quant_stats is not initialized for {quant_stat_name}')
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
    if jnp.isnan(scale).any() or jnp.isinf(scale).any() or (scale == 0).any():
      raise ValueError(f'Invalid scale: {scale}')
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
