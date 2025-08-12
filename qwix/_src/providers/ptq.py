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
"""Post-training quantization (PTQ)."""

import functools
from typing import Any, Callable, Generic, Sequence, TypeVar

from flax import linen as nn
from flax import nnx
import flax.linen.dtypes
import jax
import numpy as np
from qwix._src import aux_data
from qwix._src import averaging
from qwix._src import flax_util
from qwix._src import qconfig
from qwix._src.core import conv_general
from qwix._src.core import dot_general
from qwix._src.core import einsum
from qwix._src.core import qarray


ArrayTypeVar = TypeVar('ArrayTypeVar', jax.Array, qarray.QArray)


@flax.struct.dataclass
class WithAux(Generic[ArrayTypeVar]):
  """An array/QArray with auxiliary information.

  The main purpose of this class is to embed the how to quantize information
  into the param tree, such that the quantize_params() function can quantize
  params without knowing the model structure.

  Attributes:
    array: The underlying array.
    how: How the array is quantized, which is used by quantize_params so that it
      knows how to quantize the original weights.
    weight_name: The name of the weight, which is used by LoraProvider to create
      the LoRA weights.
    value: Satisfies the nnx.Variable interface.
  """

  array: ArrayTypeVar
  how: qarray.HowToQuantize = flax.struct.field(pytree_node=False)
  weight_name: str = flax.struct.field(pytree_node=False)

  # This allows us to appear like nnx.Variable.
  value = property(flax_util.unbox)
  shape = property(lambda self: self.array.shape)


class PtqProvider(qconfig.QuantizationProvider):
  """Quantization provider for PTQ.

  In PTQ mode, weights needs to be pre-quantized. However, Qwix doesn't know
  about how to quantize them until the actual ops get called. To solve this,
  we still initialize the original weights when the model is initialized, but
  we replace them with the quantized weights when the ops are called.

  * It should be invisible to users in Flax linen because `module.init` will
    call both the setup() and __call__() methods.
  * If memory usage is a concern, wrapping `module.init` with jit or eval_shape
    should avoid materializing the original weights.
  * NNX can use the same trick so we don't need to intercept nnx.Param.
  * This approach allows the original weights to be supplied during `apply`,
    and will actually quantize them correctly. This can be an alternative to
    `quantize_params` if partial param quantization is not needed.
  """

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

    get_how_to_quantize = functools.partial(
        dot_general.get_how_to_quantize,
        dimension_numbers=dimension_numbers,
        ndims=(len(lhs.shape), len(rhs.shape)),
    )

    # Prepare rhs.
    if isinstance(rhs, WithAux):  # weight, already quantized
      rhs = rhs.array
    elif aux_data.get(rhs, 'weight_name', None):  # weight, not quantized
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.weight_qtype,
          tile_size=rule.tile_size,
          calibration_method=rule.weight_calibration_method,
      )
      rhs = _create_quantized_param(rhs, rhs_how).array
    elif rule.act_qtype is not None:  # act
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.act_qtype,
          tile_size=rule.tile_size,
          calibration_method=rule.act_calibration_method,
      )
      rhs = _quantize_act(rhs, rhs_how, rule, op_id + '_rhs')

    # Prepare lhs.
    if rule.act_qtype is not None:
      lhs_how = get_how_to_quantize(
          for_lhs=True,
          qtype=rule.act_qtype,
          tile_size=rule.tile_size,
          calibration_method=rule.act_calibration_method,
      )
      lhs = _quantize_act(lhs, lhs_how, rule, op_id + '_lhs')
    return dot_general.dot_general(
        lhs, rhs, dimension_numbers, out_sharding=out_sharding
    )

  def einsum(
      self,
      einsum_str: str,
      *operands: jax.Array,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      _dot_general: Callable[..., jax.Array] = jax.lax.dot_general,  # pylint: disable=invalid-name
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
    get_how_to_quantize = functools.partial(
        einsum.get_how_to_quantize,
        einsum_str=einsum_str,
        ndims=(len(lhs.shape), len(rhs.shape)),
    )

    # Prepare rhs.
    if isinstance(rhs, WithAux):  # weight, already quantized
      rhs = rhs.array
    elif aux_data.get(rhs, 'weight_name', None):  # weight, not quantized
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.weight_qtype,
          tile_size=rule.tile_size,
          calibration_method=rule.weight_calibration_method,
      )
      rhs = _create_quantized_param(rhs, rhs_how).array
    elif rule.act_qtype is not None:  # act
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.act_qtype,
          tile_size=rule.tile_size,
          calibration_method=rule.act_calibration_method,
      )
      rhs = _quantize_act(rhs, rhs_how, rule, op_id + '_rhs')

    # Prepare lhs.
    if rule.act_qtype is not None:
      lhs_how = get_how_to_quantize(
          for_lhs=True,
          qtype=rule.act_qtype,
          tile_size=rule.tile_size,
          calibration_method=rule.act_calibration_method,
      )
      lhs = _quantize_act(lhs, lhs_how, rule, op_id + '_lhs')
    return einsum.einsum(einsum_str, lhs, rhs)

  def conv_general_dilated(
      self,
      lhs: jax.Array,
      rhs: jax.Array | WithAux[qarray.QArray],
      window_strides: Sequence[int],
      padding: str | Sequence[tuple[int, int]],
      lhs_dilation: Sequence[int] | None = None,
      rhs_dilation: Sequence[int] | None = None,
      dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
      feature_group_count: int = 1,
      batch_group_count: int = 1,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
  ) -> jax.Array:
    rule, op_id = self._get_current_rule_and_op_id('conv_general_dilated')
    if rule is None or rule.weight_qtype is None:
      return jax.lax.conv_general_dilated(
          lhs,
          rhs,
          window_strides,
          padding,
          lhs_dilation=lhs_dilation,
          rhs_dilation=rhs_dilation,
          dimension_numbers=dimension_numbers,
          feature_group_count=feature_group_count,
          batch_group_count=batch_group_count,
          precision=precision,
          preferred_element_type=preferred_element_type,
      )
    dimension_numbers = jax.lax.conv_dimension_numbers(
        lhs.shape, rhs.shape, dimension_numbers
    )

    # Prepare rhs.
    if isinstance(rhs, WithAux):  # weight, already quantized
      rhs = rhs.array
    else:
      assert aux_data.get(rhs, 'weight_name', None)
      rhs_how = conv_general.get_how_to_quantize(
          dimension_numbers=dimension_numbers,
          for_lhs=False,
          qtype=rule.weight_qtype,
          calibration_method=rule.weight_calibration_method,
      )
      rhs = _create_quantized_param(rhs, rhs_how).array

    # Prepare lhs.
    if rule.act_qtype != rule.weight_qtype:
      raise ValueError(
          'conv_general_dilated requires same act_qtype and weight_qtype. Got:'
          f' {rule.act_qtype=} {rule.weight_qtype=}'
      )
    lhs_how = conv_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        for_lhs=True,
        qtype=rule.act_qtype,
        calibration_method=rule.act_calibration_method,
    )
    lhs = _quantize_act(lhs, lhs_how, rule, op_id + '_lhs')
    return conv_general.conv_general_dilated(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
    )

  def nn_param(self, module: nn.Module, name: str, *args, **kwargs):
    """Intercepts nn.Module.param to associate weight_name aux_data."""
    # Don't check the shape if the param is already quantized.
    existing_param = module.get_variable('params', name)
    if isinstance(existing_param, WithAux):
      return nn.unbox(existing_param)
    ret = module.param(name, *args, **kwargs)
    aux_data.set(ret, 'weight_name', name)
    return ret

  def promote_dtype(self, *args, **kwargs):
    """Intercepts flax.{linen,nnx.nn}.dtypes.promote_dtype."""
    if len(args) == 1 and isinstance(args[0], Sequence):
      args = args[0]  # nnx version
    # Skip WithAux.
    array_args = [x if isinstance(x, jax.Array) else None for x in args]
    array_args = flax.linen.dtypes.promote_dtype(*array_args, **kwargs)
    # Forward weight_name aux_data.
    for x, y in zip(array_args, args):
      if x is not None and (name := aux_data.get(y, 'weight_name', None)):
        aux_data.set(x, 'weight_name', name)
    return [x if x is not None else y for x, y in zip(array_args, args)]

  def dot(
      self,
      a,
      b,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      out_sharding=None,
  ):
    """Intercepts jax.numpy.dot."""
    a_ndim = np.ndim(a)
    if isinstance(b, WithAux):
      b_ndim = b.array.qvalue.ndim
    else:
      b_ndim = np.ndim(b)
    if a_ndim == 0 or b_ndim == 0:
      contract_dims = ((), ())
    else:
      if b_ndim == 1:
        contract_dims = ((a_ndim - 1,), (0,))
      else:
        contract_dims = ((a_ndim - 1,), (b_ndim - 2,))
    return self.dot_general(
        a,
        b,
        dimension_numbers=(contract_dims, ((), ())),
        precision=precision,
        preferred_element_type=preferred_element_type,
        out_sharding=out_sharding,
    )

  def get_intercept_map(self):
    """Used for interception."""
    return super().get_intercept_map() | {
        'jax.lax.conv_general_dilated': self.conv_general_dilated,
        'jax.lax.dot_general': self.dot_general,
        'jax.numpy.dot': self.dot,
        'jax.numpy.einsum': self.einsum,
        'flax.linen.Module.param': self.nn_param,
        'flax.linen.dtypes.promote_dtype': self.promote_dtype,
        'flax.nnx.nn.dtypes.promote_dtype': self.promote_dtype,
    }

  def process_model_inputs(
      self, model: nn.Module | nnx.Module, model_args: Any, model_kwargs: Any
  ) -> tuple[nn.Module | nnx.Module, Any, Any]:
    """Processes the nnx.Module instance before it is called."""
    # Set weight_name for nnx models. Linen models are handled in nn_param.
    if isinstance(model, nnx.Module):
      for path, node in nnx.iter_graph(model):
        if isinstance(node, nnx.Module):
          aux_data.clear(node)  # clear the op_count.
        elif isinstance(node, nnx.Param):
          # weight_name is used to distinguish weights from activations.
          aux_data.set(node.value, 'weight_name', path[-1])
    return model, model_args, model_kwargs


def _quantize_act(
    array: jax.Array,
    how: qarray.HowToQuantize,
    rule: qconfig.QuantizationRule,
    act_name: str,
) -> qarray.QArray:
  """Quantizes the input activation with support for static scale."""
  if not rule.act_static_scale:
    return qarray.quantize(array, how)

  # Construct the scale and zero_point from the quant stats, if available.
  # This is useful in NNX when a PTQ model is converted from a QAT model.
  # We delete the quant_stat after the first forward pass so that the PTQ
  # model appears the same as a regular one.
  quant_stat = flax_util.get_and_delete_variable('quant_stats', act_name)

  def init():
    if quant_stat is not None:
      aggregator = averaging.SimpleMovingAverage()
      calibration = aggregator.get_calibration(quant_stat)
    else:
      calibration = qarray.calibrate(array, how)
      # Apply act_batch_axes for static scale.
      calibration = jax.tree.map(
          lambda x: x.mean(axis=rule.act_batch_axes, keepdims=True), calibration
      )
    nonlocal zp
    scale, zp = qarray.compute_scale_zero_point(calibration, how.qtype)
    # Wrap scale in WithAux so that quantize_params can work.
    return WithAux(scale, how, act_name + '_scale')

  zp = None
  scale = flax_util.get_or_create_param(act_name + '_scale', init)
  if zp is not None:
    zp = flax_util.get_or_create_param(act_name + '_zero_point', lambda: zp)
  return qarray.quantize_with_scale_zero_point(
      array, how.qtype, scale.array, zp
  )


def _create_quantized_param(
    value: jax.Array, how: qarray.HowToQuantize
) -> WithAux[qarray.QArray]:
  """Creates the quantized param and replaces the original param in the module.

  Args:
    value: The unquantized jax.Array with weight_name aux_data.
    how: How to quantize the param.

  Returns:
    An unboxed WithAux.
  """
  weight_name = aux_data.get(value, 'weight_name')
  unboxed = WithAux(qarray.quantize(value, how), how, weight_name)

  # The following code is about replacing the saved param with WithAux, with
  # correct metadata.

  module = flax_util.get_current_module()
  if isinstance(module, nn.Module):
    if not module.is_initializing():
      raise ValueError(
          "It seems you're feeding an unquantized param to a quantized model."
      )
    param = module.get_variable('params', weight_name)
    boxed = jax.tree.map(
        lambda value: flax_util.update_boxed(param, value=value), unboxed
    )
    module.put_variable('params', weight_name, boxed)
  elif isinstance(module, nnx.Module):
    param = getattr(module, weight_name)
    boxed = jax.tree.map(
        lambda value: flax_util.update_boxed(param, value=value), unboxed
    )
    setattr(module, weight_name, boxed)

  return unboxed


def quantize_params(
    params: Any,
    abstract_quantized_params: Any,
    quant_stats: Any = flax.core.FrozenDict(),
) -> Any:
  """Quantize the param tree for PTQ.

  This function provides advanced param quantization for PTQ. It is useful when
  the original params are too large to fit in the HBM.

  Args:
    params: The floating-point param tree to quantize, which is usually
      generated by the original or QAT model. The tree doesn't need to be
      complete and can be a subtree of the whole param tree. In NN, the tree
      needs to be unboxed, i.e. nn.unbox(). In NNX, the tree needs to be a pure
      dict, i.e. nnx.to_pure_dict().
    abstract_quantized_params: The param tree generated by the PTQ model, which
      can be abstract with jax.ShapeDtypeStruct as leaves instead of jax.Array.
      In NN, the tree may contain AxisMetadata. In NNX, this should be the PTQ
      model itself, possibly abstract.
    quant_stats: The quantization statistics. This is only used in SRQ.

  Returns:
    The quantized param tree, which has the same structure as the input params
    but with quantized leaves.
  """

  def get_value_from_path(obj, path: tuple[str, ...]):
    for key in path:
      obj = obj[key] if isinstance(obj, dict) else getattr(obj, key)
    return obj

  quantized_params = {}
  for path, param in flax.traverse_util.flatten_dict(params).items():
    if not isinstance(param, jax.Array):
      raise TypeError(f'params is not a pure dict of jax.Array: {type(param)}')
    abs_param = get_value_from_path(abstract_quantized_params, path)
    if isinstance(abs_param, WithAux):
      param = abs_param.replace(array=qarray.quantize(param, abs_param.how))
    quantized_params[path] = param

  # SRQ only: compute scale and zero_point from the quant_stats.
  all_quant_stats_paths = {
      path[:-1] for path in flax.traverse_util.flatten_dict(quant_stats)
  }
  for path in all_quant_stats_paths:
    quant_stat = get_value_from_path(quant_stats, path)
    if quant_stat['count'] == 0:
      raise ValueError(f'quant_stats is not initialized for {path}.')

    # Get the act_qtype from the scale, which is a WithAux[jax.Array].
    scale_path = (*path[:-1], path[-1] + '_scale')
    abs_scale = get_value_from_path(abstract_quantized_params, scale_path)
    act_qtype = abs_scale.how.qtype

    calibration = averaging.SimpleMovingAverage().get_calibration(quant_stat)
    scale, zero_point = qarray.compute_scale_zero_point(calibration, act_qtype)
    quantized_params[scale_path] = abs_scale.replace(array=scale)
    if zero_point is not None:
      quantized_params[(*path[:-1], path[-1] + '_zero_point')] = zero_point

  if isinstance(abstract_quantized_params, nnx.Module):
    # Convert WithAux to a pure dict so that nnx.update() can work.
    quantized_params = nnx.to_pure_dict(nnx.state(quantized_params))

  return flax.traverse_util.unflatten_dict(quantized_params)
