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
"""ODML ops for QAT."""

import dataclasses
import functools
import sys
from typing import Any, Callable, Sequence

import jax
from jax import numpy as jnp
from qwix._src import aux_data
from qwix._src import qconfig
from qwix._src.core import conv_general
from qwix._src.core import dot_general
from qwix._src.core import einsum
from qwix._src.core import qarray


def get_all_ops():
  """Get all the functions to intercept and the corresponding ops."""

  # Quantization ops can generally be categorized into 3 categories:
  # https://ai.google.dev/edge/litert/models/quantization_spec
  #
  #   1. Ops that allow different scales for inputs and output. For example,
  #      add, mean, etc. These ops have FQ on all inputs and output. They are
  #      handled either by the generic QuantizedOp class (e.g. add), or by
  #      custom classes inheriting from QuantizedOp (e.g. dot_general).
  #   2. Ops that only allow the same scale for inputs and output. For example,
  #      reshape, avg_pool, split, etc. These ops have FQ only on the input or
  #      the output, and the converter is able to propagate the scale.
  #      1. FQ on the input: avg_pool, resize, etc. Their output values depend
  #         on the complete input values.
  #      2. FQ on the output: clip, pad, relu, etc. Their output values don't
  #         depend on the unused part of the input values, so it's more accurate
  #         to use the scale from the output.
  #      3. Some "transparent" ops such as reshape, transpose, etc. don't change
  #         the actual value of the array. They cannot be selected by the
  #         quantization rules. Instead, their behavior depends on whether the
  #         previous op is quantized.
  #   3. Ops that don't have a corresponding quantized op, e.g. sin, cos. If
  #      the previous op wants to quantize the output, the op will fake quantize
  #      its input. Otherwise, the op will insert no FQ.

  # l2_norm is not listed here because there's no standard implementation.
  # Usually people implement it as
  #
  #   def l2_norm(x, eps=1e-6):
  #     return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)
  #
  # To support this op, register it manually in the provider (both QAT and
  # conversion) using the tanh handler, e.g.
  #
  #   l2_norm_full_name = l2_norm.__module__ + '.' + l2_norm.__name__
  #   provider._ops[l2_norm_full_name] = provider._ops['jax.numpy.tanh']

  partial = functools.partial
  quantize = lambda *a, **k: functools.partial(QuantizedOp, input_idx=a, **k)

  return {
      # go/keep-sorted start
      'flax.linen.BatchNorm.__call__': BatchNorm,
      'flax.linen.Dropout.__call__': partial(TransparentOp, input_idx=[1]),
      'flax.linen.GroupNorm.__call__': quantize(1, op_name='norm_op'),
      'flax.linen.LayerNorm.__call__': quantize(1, op_name='norm_op'),
      'flax.linen.avg_pool': OnlyInputOp,
      'flax.linen.max_pool': OnlyInputOp,
      'jax._src.numpy.indexing.rewriting_take': Take,  # a.__getitem__
      'jax.custom_jvp.__call__': CustomJvpCall,  # handles relu and relu6.
      'jax.image.resize': OnlyInputOp,
      'jax.lax.broadcast_in_dim': TransparentOp,
      'jax.lax.conv_general_dilated': DotEinsumConv,
      'jax.lax.dot_general': DotEinsumConv,
      'jax.lax.reshape': TransparentOp,
      'jax.lax.split': Split,
      'jax.lax.squeeze': TransparentOp,
      'jax.lax.stop_gradient': TransparentOp,
      'jax.lax.transpose': TransparentOp,
      'jax.lax.with_sharding_constraint': TransparentOp,
      'jax.nn.gelu': quantize(0),
      'jax.nn.leaky_relu': quantize(0),
      'jax.nn.silu': Silu,
      'jax.nn.softmax': Softmax,
      'jax.numpy.array': TransparentOp,
      'jax.numpy.astype': TransparentOp,
      'jax.numpy.clip': OnlyOutputOp,
      'jax.numpy.concatenate': Concatenate,
      'jax.numpy.cos': NoQuantOp,
      'jax.numpy.dot': DotEinsumConv,
      'jax.numpy.einsum': DotEinsumConv,
      'jax.numpy.equal': functools.partial(NoQuantOp, input_idx=[0, 1]),
      'jax.numpy.floor': OnlyOutputOp,
      'jax.numpy.mean': quantize(0),
      'jax.numpy.pad': OnlyOutputOp,
      'jax.numpy.repeat': quantize(0),  # not fully supported by the converter.
      'jax.numpy.sin': NoQuantOp,
      'jax.numpy.squeeze': TransparentOp,
      'jax.numpy.sum': quantize(0),
      'jax.numpy.take': Take,
      'jax.numpy.tanh': Tanh,
      'jax.numpy.true_divide': quantize(0, 1, op_name='truediv'),
      'jax.numpy.ufunc.__call__': UfuncCall,  # handles add, sub, mul, pow.
      # go/keep-sorted end
  }


NotAnActivationError = ValueError(
    "The array is expected to be an activation, but it's not. This is usually"
    " because your model has ops that Qwix doesn't support, or your model is"
    ' using those ops in a way Qwix failed to intercept. To disable this check,'
    ' pass `strict=False` to the provider.'
)


### Possible auxiliary data associated with an array

# Whether an array should be fake quantized by the next op and what rule to use.
#
# For the output of an op, it's not fake-quantized immediately because the next
# op may choose to delay the FQ, e.g. dot_general + add + relu can be fused and
# no FQ should be inserted in between.
_FQ_RULE = 'fq_rule'  # QuantizationRule

# Whether the (unquantized) array is already fake-quantized in another code path
# and what the fake-quantized array is. This avoids the same array being
# fake-quantized multiple times.
_FQ_ARRAY = 'fq_array'  # array

# Whether the previous op allows to fuse arithmetic ops or batch norm after it.
_ALLOW_FUSION = 'allow_fusion'  # bool

# Whether the array is an activation. An array can be either an activation,
# a weight, or a constant.
_IS_ACTIVATION = 'is_activation'  # bool

# Whether the array is a weight and what is its name. Weights don't need to have
# quantization statistics collected because they are statically quantized.
# The name is useful in the conversion provider to find the static weight.
_WEIGHT_NAME = 'weight_name'  # str

# Fixed range for logistic functions whose output ranges are known, e.g.
# softmax.
_FIXED_RANGE = 'fixed_range'  # tuple[float, float]


GetRuleAndOpIdFn = Callable[[str], tuple[qconfig.QuantizationRule, str]]
FakeQuantFn = Callable[[jax.Array, qarray.HowToQuantize, str | None], jax.Array]


class QuantizedOp:
  """A generic quantized op that allows different scales for inputs and output.

  Each op is initialized once and invoked multiple times. Thus the rule and the
  op id are given as a function.
  """

  # Which args are considered as the inputs of the op.
  input_idx: Sequence[int] = ()

  # Fixed range for the op output.
  fixed_range_for_output: tuple[float, float] | None = None

  def __init__(
      self,
      *,
      op_full_name: str,
      get_rule_and_op_id_fn: GetRuleAndOpIdFn,
      fake_quant_fn: FakeQuantFn,
      op_name: str | None = None,
      **kwargs,
  ):
    """Create a generic op.

    Arguments here are provided during runtime.

    Args:
      op_full_name: The full name of the op, e.g. jax.numpy.add.
      get_rule_and_op_id_fn: A function that returns the quantization rule and
        the op id for the op.
      fake_quant_fn: A function that performs fake quantization on an array.
      op_name: The name of the op, e.g. add. If not given, it will be inferred
        from op_full_name.
      **kwargs: Additional keyword arguments to override class attributes.
    """
    # Those are attributes that are given during runtime.
    self._op_full_name = op_full_name
    self._get_rule_and_op_id_fn = get_rule_and_op_id_fn
    self._fake_quant_fn = fake_quant_fn
    if op_name is None:
      op_name = op_full_name.split('.')[-1].replace('__', '')
    self._op_name = op_name

    # Use kwargs to override class attributes.
    for key, value in kwargs.items():
      if not hasattr(type(self), key):
        raise AttributeError(f'Unsupported keyword argument {key}.')
      setattr(self, key, value)

  def __call__(self, *args, **kwargs):
    """Quantize the op."""
    # If there's no activation in the args, skip the quantization because
    # they are all weights or constants that can be constant-folded.
    if not self._inputs_have_activations(args):
      return self._call_original_op(*args, **kwargs)

    rule, op_id = self._get_rule_and_op_id_fn(self._op_name)
    args = self._fake_quant_inputs(args, rule, op_id)
    out = self._call_original_op(*args, **kwargs)
    return self._fake_quant_output(out, rule)

  def _inputs_have_activations(self, args: Sequence[Any]) -> bool:
    """Check if there's any activation in the inputs."""
    if not self.input_idx:
      raise ValueError(f'input_idx is not set for op {self._op_name}.')
    for idx in self.input_idx:
      if isinstance(args[idx], jax.Array) and aux_data.get(
          args[idx], _IS_ACTIVATION, False
      ):
        return True
    return False

  def _call_original_op(self, *args, **kwargs) -> Any:
    """Get the original function from op_full_name."""
    name_parts = self._op_full_name.split('.')
    obj = sys.modules[name_parts[0]]
    for attr in name_parts[1:]:
      obj = getattr(obj, attr)
    return obj(*args, **kwargs)

  def _fake_quant_inputs(
      self,
      args: Sequence[Any],
      rule: qconfig.QuantizationRule | None,
      op_id: str,
  ) -> Sequence[Any]:
    """Fake quantize the inputs of the op."""
    args = list(args)
    if len(self.input_idx) == 1:
      idx = self.input_idx[0]
      args[idx] = self._maybe_fake_quant(args[idx], rule, op_id)
    elif len(self.input_idx) == 2:
      lhs, rhs = tuple(self.input_idx)  # pylint: disable=unbalanced-tuple-unpacking
      # Binary ops could have non-array args, e.g. x + 1.
      if isinstance(args[lhs], jax.Array):
        args[lhs] = self._maybe_fake_quant(args[lhs], rule, op_id + '_lhs')
      if isinstance(args[rhs], jax.Array):
        args[rhs] = self._maybe_fake_quant(args[rhs], rule, op_id + '_rhs')
    else:
      raise ValueError(
          f'Unsupported num of inputs {self.input_idx} for op {self._op_name}.'
      )
    return args

  def _maybe_fake_quant(
      self,
      array: jax.Array,
      rule: qconfig.QuantizationRule | None,
      quant_stat_name: str,
  ) -> jax.Array:
    """Fake quantize the array based on the given rule.

    This function assumes the array is an activation, unless it has weight_name
    aux_data, e.g., in jnp.take.

    Args:
      array: The array to quantize.
      rule: The quantization rule for the array. If None, the array will not be
        quantized.
      quant_stat_name: The name for the quantization statistics.

    Returns:
      The fake quantized array.
    """
    # Check if the array is already quantized in another code path.
    fq_array = aux_data.get(array, _FQ_ARRAY, None)
    if fq_array is not None:
      return array if fq_array == 'self' else fq_array

    # Only quantize float arrays.
    if array.dtype not in (jnp.float32, jnp.bfloat16):
      return array

    # Check if the array is a weight.
    if aux_data.get(array, _WEIGHT_NAME, None) is not None:
      if rule and rule.weight_qtype:
        how = qarray.HowToQuantize(
            qtype=rule.weight_qtype,
            channelwise_axes=(),
            tiled_axes={},
            # Use act_calibration_method because it is more like an activation,
            # i.e., asymmetric rather than symmetric.
            calibration_method=rule.act_calibration_method,
        )
        fq_array = self._fake_quant_fn(array, how, None)
        aux_data.set(array, _FQ_ARRAY, fq_array)
        return fq_array
      return array

    # Check if the array should be quantized as the output of the previous op.
    previous_rule = aux_data.get(array, _FQ_RULE, None)
    if previous_rule is not None:
      rule = previous_rule

    if rule is None or rule.act_qtype is None:
      return array
    if not rule.act_static_scale:
      # DRQ is only supported in DotEinsumConv and they should call
      # _fake_quant_fn directly.
      return array

    how = qarray.HowToQuantize(
        qtype=rule.act_qtype,
        tiled_axes={},
        # Use per-channel scales for batch axes, which will be reduced later
        # in _collect_quant_stat.
        channelwise_axes=rule.act_batch_axes,
        calibration_method=rule.act_calibration_method,
    )

    fq_array = self._fake_quant_fn(array, how, quant_stat_name)
    aux_data.set(array, _FQ_ARRAY, fq_array)
    return fq_array

  def _fake_quant_output(
      self, array: jax.Array, rule: qconfig.QuantizationRule | None
  ) -> jax.Array:
    """Fake quantize an output activation, which is delayed to the next op."""
    aux_data.set(array, _IS_ACTIVATION, True)
    if self.fixed_range_for_output is not None:
      aux_data.set(array, _FIXED_RANGE, self.fixed_range_for_output)
    # Output is only quantized in SRQ.
    if rule and rule.act_qtype and rule.act_static_scale:
      aux_data.set(array, _FQ_RULE, rule)
    return array


class OnlyInputOp(QuantizedOp):
  """An op that only fake quantizes the input."""

  input_idx = [0]  # default to a unary op.

  def __call__(self, *args, **kwargs):
    if not self._inputs_have_activations(args):
      return self._call_original_op(*args, **kwargs)
    rule, op_id = self._get_rule_and_op_id_fn(self._op_name)
    # Only fake quantize the input.
    args = self._fake_quant_inputs(args, rule, op_id)
    out = self._call_original_op(*args, **kwargs)
    if rule and rule.act_qtype:
      # Mark the output as already quantized.
      aux_data.set(out, _FQ_ARRAY, 'self')
    return self._fake_quant_output(out, rule)


class OnlyOutputOp(QuantizedOp):
  """An op that only fake quantizes the output."""

  input_idx = [0]  # default to a unary op.

  def __call__(self, *args, **kwargs):
    if not self._inputs_have_activations(args):
      return self._call_original_op(*args, **kwargs)
    rule, _ = self._get_rule_and_op_id_fn(self._op_name)
    if rule is None or rule.act_qtype is None:
      rule = aux_data.get(args[self.input_idx[0]], _FQ_RULE, None)
    # No quantization on the input.
    out = self._call_original_op(*args, **kwargs)
    return self._fake_quant_output(out, rule)


class TransparentOp(QuantizedOp):
  """A transparent op doesn't quantize anything. It only forwards aux data.

  This is mainly used for ops that doesn't change the actual value of the
  inputs, e.g. reshape, transpose, etc. This is similar to OnlyOutputOp,
  but it reuses the previous rule and forwards more aux data.
  """

  input_idx = [0]  # default to a unary op.
  forwarded_aux_data = (
      _IS_ACTIVATION,
      _WEIGHT_NAME,
      _FQ_RULE,
      _FIXED_RANGE,
      _ALLOW_FUSION,
  )

  def __call__(self, *args, **kwargs):
    if len(self.input_idx) > 1:
      raise ValueError(
          f'Unsupported num of inputs {self.input_idx} for op {self._op_name}.'
      )
    out = self._call_original_op(*args, **kwargs)

    def forward(out, arg):
      for key in self.forwarded_aux_data:
        value = aux_data.get(arg, key, None)
        if value is not None:
          aux_data.set(out, key, value)
      # Also forward the FQ_ARRAY if it's used to skip the quantization.
      fq_array = aux_data.get(arg, _FQ_ARRAY, None)
      if fq_array == 'self':
        aux_data.set(out, _FQ_ARRAY, fq_array)

    jax.tree.map(forward, out, args[self.input_idx[0]])
    return out


class NoQuantOp(QuantizedOp):
  """An fp op doesn't have a corresponding quantized op."""

  input_idx = [0]  # default to a unary op.

  def __call__(self, *args, **kwargs):
    if not self._inputs_have_activations(args):
      return self._call_original_op(*args, **kwargs)

    # Only fake quantize the input when the previous op wants.
    _, op_id = self._get_rule_and_op_id_fn(self._op_name)
    args = self._fake_quant_inputs(args, None, op_id)
    out = self._call_original_op(*args, **kwargs)

    # and don't quantize the output.
    return self._fake_quant_output(out, None)


class ModelInput(QuantizedOp):
  """A synthetic op for the model input."""

  def __init__(self, **kwargs):
    # The op_full_name doesn't matter here.
    super().__init__(op_full_name='model_input', **kwargs)

  def __call__(self, x: Any) -> Any:
    aux_data.clear(x)  # such as FQ_ARRAY
    if isinstance(x, jax.Array):
      self._fake_quant_output(x, None)
    return x


class FinalOutput(QuantizedOp):
  """A synthetic op for the model output."""

  # Whether to check if the output is an activation.
  check_activation: bool = True

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def __call__(self, x: Any) -> Any:
    if self.check_activation and not aux_data.get(x, _IS_ACTIVATION, False):
      raise NotAnActivationError
    _, op_id = self._get_rule_and_op_id_fn(self._op_name)
    if self.fixed_range_for_output is not None:
      aux_data.set(x, _FIXED_RANGE, self.fixed_range_for_output)
    # Only FQ the output if the previous op wants.
    return self._maybe_fake_quant(x, None, op_id)


class BatchNorm(QuantizedOp):
  """BatchNorm op, which can be fused into previous op completely."""

  def __call__(self, norm, x: jax.Array, *args, **kwargs) -> jax.Array:
    if not aux_data.get(x, _IS_ACTIVATION, False):
      return norm(x, *args, **kwargs)
    if aux_data.get(x, _ALLOW_FUSION, False):
      rule = aux_data.get(x, _FQ_RULE, None)
      out = norm(x, *args, **kwargs)
      aux_data.set(out, _ALLOW_FUSION, True)
    else:
      rule, op_id = self._get_rule_and_op_id_fn('batch_norm_op')
      x = self._maybe_fake_quant(x, rule, op_id)
      out = norm(x, *args, **kwargs)
    return self._fake_quant_output(out, rule)


class Softmax(QuantizedOp):
  """Softmax op."""

  input_idx = [0]
  # The converter requires (scale, zero_point) = (1.0 / 256.0, -128). Qwix uses
  # [-128, 127], which maps to [0, 255 / 256] with the above scale/zp.
  fixed_range_for_output = (0.0, 255 / 256)


class Tanh(QuantizedOp):
  """tanh op."""

  input_idx = [0]
  # The converter requires (scale, zero_point) = (1.0 / 128.0, 0). Qwix uses
  # [-128, 127], which maps to [-1, 127 / 128] with the above scale/zp.
  fixed_range_for_output = (-1.0, 127 / 128)


class UfuncCall(QuantizedOp):
  """ufunc call op."""

  def __call__(self, *args: Any, **kwargs: Any) -> jax.Array:
    # Assume single threaded.
    self._op_name = args[0].__name__
    self._output_allow_fusion = False
    self.input_idx = [1] if self._op_name == 'negative' else [1, 2]
    return super().__call__(*args, **kwargs)

  def _fake_quant_inputs(
      self,
      args: Sequence[Any],
      rule: qconfig.QuantizationRule | None,
      op_id: str,
  ) -> Sequence[Any]:
    """Fake quantize the inputs of the op."""
    if (
        self._op_name in ('add', 'sub', 'mul', 'truediv')
        and aux_data.get(args[1], _ALLOW_FUSION, False)
        and not aux_data.get(args[2], _IS_ACTIVATION, False)
    ):
      # The previous op allows to fuse adding a constant.
      self._output_allow_fusion = True
      return args
    return super()._fake_quant_inputs(args, rule, op_id)

  def _fake_quant_output(
      self, array: jax.Array, rule: qconfig.QuantizationRule | None
  ) -> jax.Array:
    if self._output_allow_fusion:
      aux_data.set(array, _ALLOW_FUSION, True)
    return super()._fake_quant_output(array, rule)


class Concatenate(QuantizedOp):
  """Concatenate op."""

  def __call__(self, arrays: Sequence[jax.Array], *args, **kwargs) -> jax.Array:
    """QAT concatenate."""
    if not any(aux_data.get(x, _IS_ACTIVATION, False) for x in arrays):
      return self._call_original_op(arrays, *args, **kwargs)

    # Forward the fixed_range if all inputs have the same.
    fixed_range = aux_data.get(arrays[0], _FIXED_RANGE, None)
    if any(aux_data.get(x, _FIXED_RANGE, None) != fixed_range for x in arrays):
      fixed_range = None

    # If ourselves is not quantized, fake quantize the inputs if needed.
    # Otherwise, don't insert FQ for the inputs.
    rule, op_id = self._get_rule_and_op_id_fn(self._op_name)
    if not rule or rule.act_qtype is None:
      arrays = [
          self._maybe_fake_quant(x, rule, op_id + f'_input{i}')
          for i, x in enumerate(arrays)
      ]

    out = jnp.concatenate(arrays, *args, **kwargs)

    if fixed_range is not None:
      aux_data.set(out, _FIXED_RANGE, fixed_range)
    return self._fake_quant_output(out, rule)


class Take(OnlyInputOp):
  """jax.numpy.take or rewriting_take."""

  input_idx = [0, 1]

  def __call__(self, *args, **kwargs) -> jax.Array:
    if not self._inputs_have_activations(args):
      return self._call_original_op(*args, **kwargs)

    # Only x needs to be fake quantized.
    args = list(args)
    rule, op_id = self._get_rule_and_op_id_fn(self._op_name)
    args[0] = self._maybe_fake_quant(args[0], rule, op_id)

    if rule and rule.act_qtype:
      # Provide a default fill_value for take. Otherwise, it will be nan
      # and cause the conversion to fail.
      kwargs.setdefault('fill_value', 0)

    out = self._call_original_op(*args, **kwargs)
    if rule and rule.act_qtype:
      # Output doesn't need more FQ.
      aux_data.set(out, _FQ_ARRAY, 'self')
    return self._fake_quant_output(out, rule)


class Split(OnlyInputOp):
  """jax.lax.split."""

  def __call__(
      self, x: jax.Array, sizes: Sequence[int], axis: int = 0
  ) -> jax.Array:
    if not aux_data.get(x, _IS_ACTIVATION, False):
      return self._call_original_op(x, sizes, axis)

    rule, op_id = self._get_rule_and_op_id_fn(self._op_name)
    x = self._maybe_fake_quant(x, rule, op_id)

    outputs = self._call_original_op(x, sizes, axis)
    # Output doesn't need more FQ.
    for output in outputs:
      aux_data.set(output, _FQ_ARRAY, 'self')
      self._fake_quant_output(output, rule)
    return outputs


class Silu(QuantizedOp):
  """jax.nn.silu."""

  def __call__(self, x: jax.Array) -> jax.Array:
    """QAT silu."""
    if not aux_data.get(x, _IS_ACTIVATION, False):
      return self._call_original_op(x)
    rule, op_id = self._get_rule_and_op_id_fn(self._op_name)
    x = self._maybe_fake_quant(x, rule, op_id)
    y = jax.nn.sigmoid(x)
    aux_data.set(y, _FIXED_RANGE, Softmax.fixed_range_for_output)
    y = self._maybe_fake_quant(y, rule, op_id + '_sigmoid')
    return self._fake_quant_output(x * y, rule)


class DotEinsumConv(QuantizedOp):
  """dot_general/einsum/conv_general_dilated."""

  # Whether to check if the lhs is an activation.
  check_activation: bool = True

  # Whether to disable per-channel quantization for weights.
  disable_per_channel_weights: bool = False

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    if self._op_name == 'einsum':
      self.input_idx = [1, 2]
    else:
      self.input_idx = [0, 1]

  def _get_how_to_quantize(
      self,
      for_lhs: bool,
      qtype: jax.typing.DTypeLike,
      calibration_method: str,
      args: Sequence[Any],
      kwargs: dict[str, Any],
  ) -> qarray.HowToQuantize:
    """Get the HowToQuantize for the given op and arguments."""
    match self._op_name:
      case 'dot_general' | 'dot':
        if self._op_name == 'dot':
          assert args[1].ndim <= 2
          d_num = ((args[0].ndim - 1,), (0,)), ((), ())
        else:
          d_num = args[2] if len(args) > 2 else kwargs['dimension_numbers']
        return dot_general.get_how_to_quantize(
            dimension_numbers=d_num,
            ndims=(len(args[0].shape), len(args[1].shape)),
            for_lhs=for_lhs,
            qtype=qtype,
            tile_size=None,  # Tiling is not supported in ODML.
            calibration_method=calibration_method,
        )
      case 'einsum':
        return einsum.get_how_to_quantize(
            einsum_str=args[0],
            ndims=(len(args[1].shape), len(args[2].shape)),
            for_lhs=for_lhs,
            qtype=qtype,
            tile_size=None,  # Tiling is not supported in ODML.
            calibration_method=calibration_method,
        )
      case 'conv_general_dilated':
        d_num = args[6] if len(args) > 6 else kwargs['dimension_numbers']
        return conv_general.get_how_to_quantize(
            dimension_numbers=d_num,
            for_lhs=for_lhs,
            qtype=qtype,
            calibration_method=calibration_method,
        )
      case _:
        raise ValueError(f'Unsupported op {self._op_name}.')

  def __call__(self, *args, **kwargs) -> jax.Array:
    lhs_idx, rhs_idx = self.input_idx
    rule, op_id = self._get_rule_and_op_id_fn(self._op_name)
    args = list(args)

    lhs_is_activation = aux_data.get(args[lhs_idx], _IS_ACTIVATION, False)
    lhs_is_weight = aux_data.get(args[lhs_idx], _WEIGHT_NAME, None) is not None
    rhs_is_activation = aux_data.get(args[rhs_idx], _IS_ACTIVATION, False)
    rhs_is_weight = aux_data.get(args[rhs_idx], _WEIGHT_NAME, None) is not None
    assert lhs_is_activation + lhs_is_weight <= 1
    assert rhs_is_activation + rhs_is_weight <= 1

    # Check for strict mode.
    if self.check_activation and not lhs_is_activation and not lhs_is_weight:
      raise NotAnActivationError

    # Fake quantize lhs.
    if (rule and rule.act_qtype and not rule.act_static_scale) and (
        lhs_is_activation and rhs_is_weight  # DRQ only supports act x weight.
    ):
      # Handle DRQ, which allows per-channel quantization for activations.
      lhs_how = self._get_how_to_quantize(
          for_lhs=True,
          qtype=rule.act_qtype,
          calibration_method=rule.act_calibration_method,
          args=args,
          kwargs=kwargs,
      )
      args[lhs_idx] = self._fake_quant_fn(args[lhs_idx], lhs_how, None)
    else:
      # Assume lhs is an activation or a weight but not a constant.
      args[lhs_idx] = self._maybe_fake_quant(
          args[lhs_idx], rule, op_id + '_lhs'
      )

    # Fake quantize rhs.
    if rule and rule.weight_qtype and rhs_is_weight:
      # Weights on RHS can be per-channel quantized.
      rhs_how = self._get_how_to_quantize(
          for_lhs=False,
          qtype=rule.weight_qtype,
          calibration_method=rule.weight_calibration_method,
          args=args,
          kwargs=kwargs,
      )
      if self.disable_per_channel_weights:
        rhs_how = dataclasses.replace(rhs_how, channelwise_axes=())
      args[rhs_idx] = self._fake_quant_fn(args[rhs_idx], rhs_how, None)
    elif rhs_is_activation:
      args[rhs_idx] = self._maybe_fake_quant(
          args[rhs_idx], rule, op_id + '_rhs'
      )

    out = self._call_original_op(*args, **kwargs)
    aux_data.set(out, _ALLOW_FUSION, True)
    return self._fake_quant_output(out, rule)


class CustomJvpCall(OnlyOutputOp):
  """This is only for intercepting relu and relu6.

  It's possible to add generic jax.custom_jvp support to interception.py, but we
  only have relu that needs to be intercepted in this way, so let's keep it
  simple for now.
  """

  input_idx = [1]

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    if args[0] in (jax.nn.relu, jax.nn.relu6):
      return super().__call__(*args, **kwargs)
    return self._call_original_op(*args, **kwargs)
