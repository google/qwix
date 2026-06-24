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
from qwix._src import interception
from qwix._src import qconfig
from qwix._src.core import einsum_info
from qwix._src.core import qarray
from qwix._src.providers import odml_ops
from qwix._src.utils import flax_util


@dataclasses.dataclass(frozen=True)
class _FlattenedEinsumLayout:
  """Layout metadata for rewriting an einsum as ``(M, K) x (K, C)``.

  The supported einsum family has no shared batch dimensions. We group all
  LHS-only output axes into M, all contraction axes into K, and all RHS-only
  output axes into C. The fields below record the transposes, reshapes, and
  final output permutation needed to make that rewrite numerically equivalent
  to the original einsum.
  """

  lhs_perm: tuple[int, ...]
  lhs_ordered_shape: tuple[int, ...]
  lhs_flat_shape: tuple[int, int]
  rhs_perm: tuple[int, ...]
  rhs_flat_shape: tuple[int, int]
  output_shape: tuple[int, ...]
  output_perm: tuple[int, ...] | None


def _prod(shape: Sequence[int]) -> int:
  """Returns the product of a shape as a Python int, including empty shapes."""
  prod = 1
  for dim in shape:
    prod *= dim
  return prod


class OdmlQatProvider(qconfig.QuantizationProvider):
  """QAT provider for ODML.

  Compared with the regular QAT provider, this provider

  * Quantizes all ops more than just conv, einsum, and dot_general.
  * Quantizes output activations via a delayed fake_quant.
  * Supports limited per-channel quantization for weights.
  * Doesn't support subchannel quantization.

  ## Tensor-Centric Rules vs. Operation-Centric Rules
  In other Qwix providers (like PTQ), a rule's `act_qtype` defines how the
  inputs to the matched operation are quantized. In the ODML provider, the
  meaning is flipped to align with LiteRT's (TFLite) tensor-centric data model:
  *   **`weight_qtype`**: Applies immediately to the weights of the matched
      operation.
  *   **`act_qtype`**: Defines the quantization type for the **OUTPUT** of the
      operation matching the rule.

  In LiteRT, quantization parameters belong to Tensors (edges), not Operations
  (nodes). Tying quantization to the tensor provides two key benefits:
  *   **Hardware Execution Efficiency**: Edge hardware accelerators (NPUs/DSPs)
      operate directly on memory buffers. They expect self-contained tensor
      descriptors that include quantization parameters, allowing them to load
      and interpret the data statically without needing to inspect operation
      metadata.
  *   **Simpler Graph Optimizations**: It makes transformations like operator
      fusion much easier to implement. Operations can be fused without losing or
      complicating the quantization state of the remaining tensors. In Qwix
      ODML, we leverage this by attaching rules to tensors but delaying their
      application, keeping paths between fusible ops (like Conv and ReLU) clear
      of `FakeQuant` nodes.
  *   Please refer to tensorflow/compiler/mlir/lite/schema/schema.fbs for more.
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
    # For ODML interception, we always disable JIT. This is because ODML relies
    # on execution at the Python level to:
    # 1. Patch low-level structural primitives (e.g., Primitive.bind) to
    #    propagate metadata.
    # 2. Support bytecode patching and recursive PjitFunction interception.
    #    JAX's C++ dispatch bypasses the patched Python `__code__` when JIT
    #    is enabled, preventing us from catching inner function calls.
    super().__init__(rules, disable_jit=True)
    self._fixed_range_for_inputs = fixed_range_for_inputs
    self._fixed_range_for_outputs = fixed_range_for_outputs
    self._strict = strict
    self._ops = odml_ops.get_all_ops()

    # Only these contraction ops support toggling channelwise weight
    # quantization (standard for ODML). For other ops, per-channel weight
    # quantization is either not applicable or not supported.
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
    aux_data.set(
        ret if unbox else ret.unbox(), odml_ops.AuxDataKey.WEIGHT_NAME, name
    )
    aux_data.set(
        ret if unbox else ret.unbox(),
        odml_ops.AuxDataKey.IS_UNTRANSFORMED_WEIGHT,
        True,
    )
    return ret

  def get_interceptors(
      self,
  ) -> Sequence[Callable[[], interception.Interceptor]]:
    """Returns a list of interceptor factories.

    The interceptors are returned in the following order:
    1. Structural interceptor: Handles low-level primitives (e.g.
       `PrimitiveBindOp`) to propagate metadata.
    2. Numerical interceptor: Handles high-level ops (e.g. `dot_general`) to
       quantize them.
    """

    # Functional layer: handle primitives to propagate metadata.
    return [
        lambda: interception.Interceptor(
            mapping={
                interception.PRIMITIVE_BIND_KEY: odml_ops.PrimitiveBindOp()
            },
            id=hash((id(self), 0)),
        ),
        lambda: interception.Interceptor(
            mapping=self.get_intercept_map(),
            id=hash((id(self), 1)),
        ),
    ]

  def get_intercept_map(self):
    """Returns a map of function names to their intercepted implementations.

    This method instantiates operator classes from `odml_ops` as functors that
    bind to this provider's specific context (e.g., `_fake_quant`). JAX uses
    these instances' `__call__` methods to replace the original operations,
    allowing them to maintain operator-specific logic while accessing
    provider-level state.
    """
    intercept_map = super().get_intercept_map()
    intercept_map['flax.linen.Module.param'] = self.nn_param
    # Add all the ops to the intercept map.
    for name, op in self._ops.items():
      op: Type[odml_ops.QuantizedOp]
      intercept_map[name] = op(
          op_full_name=name,
          get_rule_and_op_id_fn=self._get_current_rule_and_op_id,  # pyrefly: ignore[bad-argument-type]
          fake_quant_fn=self._fake_quant,
      )
    return intercept_map

  def process_model_inputs(
      self, model: Any, model_args: Any, model_kwargs: Any
  ) -> tuple[Any, Any, Any]:
    """Prepares model activations for quantization metadata propagation.

    This method also handles weight tagging for NNX models as a special case.

    Args:
      model: The model to process.
      model_args: Positional arguments to the model.
      model_kwargs: Keyword arguments to the model.

    Returns:
      The processed model and arguments with appropriate auxiliary data.
    """
    # Weight Handling (NNX only): Eagerly iterate over the graph to clear stale
    # metadata and tag parameters with _WEIGHT_NAME. For Flax Linen models,
    # weights are handled lazily via `nn_param` interception.
    if isinstance(model, nnx.Module):
      for path, node in nnx.iter_graph(model):
        if isinstance(node, nnx.Module):
          aux_data.clear(node)  # clear the op_count.
        elif isinstance(node, nnx.Param):
          # Clear the previous aux_data such as fq_array.
          aux_data.clear(node.value)
          # weight_name is used to distinguish weights from activations.
          aux_data.set(node.value, odml_ops.AuxDataKey.WEIGHT_NAME, path[-1])
          aux_data.set(
              node.value, odml_ops.AuxDataKey.IS_UNTRANSFORMED_WEIGHT, True
          )

    # Activation Handling: Apply the `ModelInput` operator to all leaves of
    # `model_args` and `model_kwargs` (the actual arguments passed to the
    # model).
    # ModelInput behavior:
    # - For non-jax.Array objects (e.g., bool, int), it's a no-op.
    # - For jax.Array objects, it clears stale metadata, marks them as
    #   activations (_IS_ACTIVATION = True), and attaches fixed ranges if set.
    # This prepares the inputs as origin points for metadata tracking.
    op = odml_ops.ModelInput(
        fixed_range_for_output=self._fixed_range_for_inputs,
        get_rule_and_op_id_fn=self._get_current_rule_and_op_id,
        fake_quant_fn=self._fake_quant,
    )
    return model, *jax.tree.map(op, (model_args, model_kwargs))

  def process_model_output(self, method_name: str, model_output: Any) -> Any:
    """Quantize the output of the model."""
    self._initial_run_complete = True
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
    """Numerical operation used by intercepted model ops to fake-quantize tensors.

    This method is the core implementation passed as a callback to intercepted
    operators (e.g., in `odml_ops.py`). It is invoked by those operators to
    perform the actual numerical quantization tasks for both activations and
    weights during the model execution.

    It handles:
    1. Calibration (including fixed-range overrides from `aux_data`).
    2. Quantization statistics collection and moving-average updates.
    3. Scale and zero-point computation.
    4. Gradient pass-through via a straight-through estimator (STE).

    Args:
      array: The array to quantize.
      how: Parameters defining how to quantize the array (e.g., qtype).
      quant_stat_name: Unique name for collecting and averaging quantization
        statistics. If None, statistics are not collected.

    Returns:
      The fake quantized array.
    """
    # Check and apply the fixed-range calibration asscociated with the array.
    fixed_range = aux_data.get(array, odml_ops.AuxDataKey.FIXED_RANGE, None)
    if fixed_range is not None:
      calibration_method = f'fixed,{fixed_range[0]},{fixed_range[1]}'
      how = dataclasses.replace(how, calibration_method=calibration_method)

    calibration = qarray.calibrate(array, how)
    if quant_stat_name is not None:
      is_fixed_range = how.calibration_method.startswith('fixed')
      calibration = self._update_and_get_quant_stat(
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

  def _update_and_get_quant_stat(
      self,
      name: str,
      calibration: averaging.Calibration,
      calibration_is_fixed_range: bool,
  ) -> averaging.Calibration:
    """Updates the running quantization statistics and returns the average."""
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
    )
    intercept_map['jax.numpy.einsum'] = functools.partial(
        self._flatten_einsum,
        _einsum=intercept_map['jax.numpy.einsum'],
    )
    return intercept_map

  def _flatten_dot_general(self, *args, _dot_general, **kwargs):
    """Flatten N-D weights to 2-D to support channelwise quantization."""
    # This special handling is needed because tflite doesn't support multiple
    # quantization_dimensions.
    if (
        aux_data.get(args[1], odml_ops.AuxDataKey.WEIGHT_NAME, None) is not None
        and args[1].ndim > 2
        and tuple(args[2][0][1]) == (0,)
    ):
      args = list(args)
      dout = args[1].shape[1:]
      original_weight = args[1]
      args[1] = jax.lax.reshape(args[1], (args[1].shape[0], np.prod(dout)))
      odml_ops.forward_metadata(original_weight, args[1])
      out = _dot_general(*args, **kwargs)
      res = jax.lax.reshape(out, out.shape[:-1] + dout)
      odml_ops.forward_metadata(out, res)
      return res
    return _dot_general(*args, **kwargs)

  @staticmethod
  def _maybe_get_flattened_einsum_layout(
      info: einsum_info.EinsumInfo,
      lhs_shape: tuple[int, ...],
      rhs_shape: tuple[int, ...],
  ) -> _FlattenedEinsumLayout | None:
    """Returns a flattening layout for supported RHS-weight einsums.

    Supported pattern:
      lhs_free + contract, contract + rhs_free -> lhs_free + rhs_free

    The output may be any permutation of ``lhs_free + rhs_free``. We explicitly
    reject shared-batch labels (labels present in lhs, rhs, and output), because
    flattening those while keeping ODML-compatible scales requires a separate
    channelwise-axis policy decision.

    Args:
      info: Parsed Einstein summation notation information.
      lhs_shape: Shape tuple of the left-hand side input array.
      rhs_shape: Shape tuple of the right-hand side input array.

    Returns:
      A `_FlattenedEinsumLayout` capturing transposition permutations and
      shapes, or `None` if flattening is not applicable.
    """
    if info.batch_chars:
      return None

    # Preserve the operand order when building each logical axis group. This
    # keeps the deterministic flattened layout and simple output reshaping.
    contract_chars = tuple(c for c in info.lhs if c in info.contract_chars)
    lhs_free_chars = tuple(
        c for c in info.lhs if c in info.out and c not in contract_chars
    )
    rhs_free_chars = tuple(
        c for c in info.rhs if c in info.out and c not in contract_chars
    )

    lhs_supported_chars = set(lhs_free_chars) | set(contract_chars)
    rhs_supported_chars = set(rhs_free_chars) | set(contract_chars)
    # Reject reductions or side labels that are not part of the matmul-shaped
    # rewrite. Those stay on the original einsum path.
    if set(info.lhs) != lhs_supported_chars:
      return None
    if set(info.rhs) != rhs_supported_chars:
      return None

    expected_out_chars = ''.join(lhs_free_chars + rhs_free_chars)
    if len(expected_out_chars) != len(info.out) or set(
        expected_out_chars
    ) != set(info.out):
      return None

    lhs_axis = {c: i for i, c in enumerate(info.lhs)}
    rhs_axis = {c: i for i, c in enumerate(info.rhs)}

    for c in contract_chars:
      if lhs_shape[lhs_axis[c]] != rhs_shape[rhs_axis[c]]:
        return None

    lhs_free_axes = tuple(lhs_axis[c] for c in lhs_free_chars)
    lhs_contract_axes = tuple(lhs_axis[c] for c in contract_chars)
    rhs_contract_axes = tuple(rhs_axis[c] for c in contract_chars)
    rhs_free_axes = tuple(rhs_axis[c] for c in rhs_free_chars)

    lhs_free_shape = tuple(lhs_shape[i] for i in lhs_free_axes)
    lhs_contract_shape = tuple(lhs_shape[i] for i in lhs_contract_axes)
    rhs_contract_shape = tuple(rhs_shape[i] for i in rhs_contract_axes)
    rhs_free_shape = tuple(rhs_shape[i] for i in rhs_free_axes)

    # If the original RHS channelwise scale would have only one non-unit
    # dimension, ODML/TFLite can already express it as a scale vector. Flatten
    # only when multiple RHS output axes would otherwise create a multi-D scale.
    if sum(dim != 1 for dim in rhs_free_shape) <= 1:
      return None

    lhs_perm = lhs_free_axes + lhs_contract_axes
    rhs_perm = rhs_contract_axes + rhs_free_axes

    return _FlattenedEinsumLayout(
        lhs_perm=lhs_perm,
        lhs_ordered_shape=lhs_free_shape + lhs_contract_shape,
        lhs_flat_shape=(_prod(lhs_free_shape), _prod(lhs_contract_shape)),
        rhs_perm=rhs_perm,
        rhs_flat_shape=(_prod(rhs_contract_shape), _prod(rhs_free_shape)),
        output_shape=lhs_free_shape + rhs_free_shape,
        output_perm=info.output_perm,
    )

  @staticmethod
  def _flatten_einsum_lhs(
      lhs: jax.Array, layout: _FlattenedEinsumLayout
  ) -> jax.Array:
    """Converts the LHS from its original layout to ``(M, K)``."""
    lhs_ordered = jax.lax.transpose(lhs, layout.lhs_perm)
    flat_lhs = jax.lax.reshape(lhs_ordered, layout.lhs_flat_shape)
    odml_ops.forward_metadata(lhs, flat_lhs)
    return flat_lhs

  @staticmethod
  def _unflatten_einsum_lhs(
      flat_lhs: jax.Array, layout: _FlattenedEinsumLayout
  ) -> jax.Array:
    """Restores a cached flattened LHS fake-quant tracer to original layout."""
    lhs_ordered = jax.lax.reshape(flat_lhs, layout.lhs_ordered_shape)
    inverse_perm = tuple(int(i) for i in np.argsort(layout.lhs_perm))
    lhs = jax.lax.transpose(lhs_ordered, inverse_perm)
    odml_ops.forward_metadata(flat_lhs, lhs)
    return lhs

  @staticmethod
  def _flatten_einsum_rhs(
      rhs: jax.Array, layout: _FlattenedEinsumLayout
  ) -> jax.Array:
    """Converts the RHS weight to ``(K, C)`` and records its static layout."""
    rhs_ordered = jax.lax.transpose(rhs, layout.rhs_perm)
    flat_rhs = jax.lax.reshape(rhs_ordered, layout.rhs_flat_shape)
    odml_ops.forward_metadata(rhs, flat_rhs)
    # Conversion fake-quant later reads the original static param, so it needs
    # this permutation to replay the same RHS layout before computing scales.
    aux_data.set(
        flat_rhs,
        odml_ops.AuxDataKey.FLATTENED_EINSUM_PERM,
        layout.rhs_perm,
    )
    return flat_rhs

  def _flatten_einsum(self, *args, _einsum, **kwargs):
    """Flatten RHS einsum weights to 2-D for ODML per-axis quantization.

    TFLite can represent a scale vector along one quantized dimension. For
    binary einsums with an RHS weight and no shared batch axes, this rewrites:
      lhs_free + contract, contract + rhs_free -> output

    into:
      (M, K), (K, C) -> (M, C)

    The result is reshaped and, if needed, transposed back to the requested
    einsum output order. Unsupported einsums intentionally fall back to the
    existing ODML path, which may still reject a multi-D scale. This includes
    shared-batch einsums whose ODML scale policy is a separate
    accuracy/compatibility tradeoff.

    Args:
      *args: Positional arguments passed to `einsum`, typically (einsum_str,
        lhs, rhs).
      _einsum: Underlying einsum functional implementation to call.
      **kwargs: Keyword arguments passed to `einsum`.

    Returns:
      The resulting output array from the einsum execution.
    """

    if len(args) != 3:
      return _einsum(*args, **kwargs)

    einsum_str, lhs, rhs = args
    weight_name = aux_data.get(rhs, odml_ops.AuxDataKey.WEIGHT_NAME, None)
    is_original_weight = aux_data.get(
        rhs, odml_ops.AuxDataKey.IS_UNTRANSFORMED_WEIGHT, False
    )
    # Only conversion-time original RHS params need this workaround. Transformed
    # weight views may carry WEIGHT_NAME for existing ODML behavior, but they do
    # not match the stored static param layout used for scale calibration.
    if not isinstance(einsum_str, str) or weight_name is None:
      return _einsum(*args, **kwargs)
    if not is_original_weight:
      return _einsum(*args, **kwargs)

    try:
      info = einsum_info.EinsumInfo.parse(
          einsum_str, ndims=(lhs.ndim, rhs.ndim)
      )
    except (NotImplementedError, ValueError):
      return _einsum(*args, **kwargs)

    layout = self._maybe_get_flattened_einsum_layout(info, lhs.shape, rhs.shape)
    if layout is None:
      return _einsum(*args, **kwargs)

    args = list(args)
    args[1] = self._flatten_einsum_lhs(lhs, layout)

    # Forward Propagation: if lhs already has a cached fake-quantized sibling,
    # reshape that sibling into the same flattened layout.
    fq_lhs = aux_data.get(lhs, odml_ops.AuxDataKey.FQ_ARRAY, None)
    if fq_lhs is not None:
      if isinstance(fq_lhs, str) and fq_lhs == 'self':
        aux_data.set(args[1], odml_ops.AuxDataKey.FQ_ARRAY, 'self')
      else:
        fq_flat_lhs = self._flatten_einsum_lhs(fq_lhs, layout)
        aux_data.set(args[1], odml_ops.AuxDataKey.FQ_ARRAY, fq_flat_lhs)

    args[2] = self._flatten_einsum_rhs(rhs, layout)
    out = _einsum('ab,bc->ac', args[1], args[2], **kwargs)

    # Backward Propagation: if this branch fake-quantized lhs first, unflatten
    # its cached tracer so sibling branches see the original lhs layout.
    if fq_lhs is None:
      fq_flat_lhs = aux_data.get(args[1], odml_ops.AuxDataKey.FQ_ARRAY, None)
      if fq_flat_lhs is not None:
        if isinstance(fq_flat_lhs, str) and fq_flat_lhs == 'self':
          aux_data.set(lhs, odml_ops.AuxDataKey.FQ_ARRAY, 'self')
        else:
          fq_original_lhs = self._unflatten_einsum_lhs(fq_flat_lhs, layout)
          aux_data.set(lhs, odml_ops.AuxDataKey.FQ_ARRAY, fq_original_lhs)

    res = jax.lax.reshape(out, layout.output_shape)
    if layout.output_perm is not None:
      res = jax.lax.transpose(res, layout.output_perm)
    odml_ops.forward_metadata(out, res)
    return res

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
      weight_name = aux_data.get(array, odml_ops.AuxDataKey.WEIGHT_NAME, None)
      if weight_name is not None:  # Weights.
        assert quant_stat_name is None
        mdl_path = flax_util.get_current_module_path()
        weight = self._flatten_params[mdl_path + (weight_name,)]
        flattened_perm = aux_data.get(
            array, odml_ops.AuxDataKey.FLATTENED_EINSUM_PERM, None
        )
        if flattened_perm is not None:
          # The runtime RHS has already been flattened to (K, C), but params
          # still store the original high-rank weight. Recreate the same layout
          # before calibration so the exported scale is one-dimensional and
          # aligned with the flattened composite operand.
          if len(flattened_perm) != weight.ndim:
            raise ValueError(
                'Cannot replay flattened einsum layout on static weight with '
                f'shape {weight.shape} and permutation {flattened_perm}.'
            )
          weight = jax.lax.reshape(
              jax.lax.transpose(weight, flattened_perm), array.shape
          )
        elif weight.shape != array.shape:  # when _flatten_dot_general is used.
          weight = weight.reshape(array.shape)
        calibration = qarray.calibrate(weight, how)
        scale, zp = qarray.compute_scale_zero_point(calibration, how.qtype)
      elif quant_stat_name is not None:  # Static-range activations.
        scale, zp = self._compute_static_scale_zero_point(how, quant_stat_name)
        # Match scale/zp rank to the activation flattened by _flatten_einsum.
        if scale.ndim != array.ndim and scale.size == 1:
          scale = scale.reshape((1,) * array.ndim)
        if zp is not None and zp.ndim != array.ndim and zp.size == 1:
          zp = zp.reshape((1,) * array.ndim)
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
      scale = np.array([], np.float32)  # pyrefly: ignore[bad-assignment]
    if jnp.isnan(scale).any() or jnp.isinf(scale).any() or (scale == 0).any():  # pyrefly: ignore[bad-argument-type, missing-attribute]
      raise ValueError(f'Invalid scale: {scale}')
    # Flatten the scale because ODML wants a 1D array.
    quantization_dim = None
    for dim, length in enumerate(scale.shape):  # pyrefly: ignore[missing-attribute]
      if length > 1:
        if quantization_dim is None:
          quantization_dim = dim
        else:
          raise ValueError(f'Cannot flatten scale with shape {scale.shape}.')  # pyrefly: ignore[missing-attribute]
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
      attributes['zero_point'] = np.asarray(zp, np.int64).flatten()  # pyrefly: ignore
    if quantization_dim is not None:
      attributes['quantization_dimension'] = quantization_dim  # pyrefly: ignore
    return attributes
