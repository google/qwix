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
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

from absl import logging
from flax import linen as nn
from flax import nnx
import flax.linen.dtypes
import jax
from jax import numpy as jnp
from qwix._src import averaging
from qwix._src import flax_util
from qwix._src import qconfig
from qwix._src.core import conv_general
from qwix._src.core import dot
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
    value: Satisfies the nnx.Variable interface.
  """

  array: ArrayTypeVar
  how: qarray.HowToQuantize = flax.struct.field(pytree_node=False)

  # This allows us to appear like nnx.Variable.
  value = property(flax_util.unbox)
  shape = property(lambda self: flax_util.unbox(self.array).shape)
  ndim = property(lambda self: flax_util.unbox(self.array).ndim)
  __getitem__ = lambda self, key: jax.tree.map(lambda x: x[key], self.value)

  def reshape(self, *shape):
    if len(shape) == 1:
      try:
        shape = tuple(shape[0])
      except TypeError:
        pass
    if tuple(self.shape) != tuple(shape):
      raise ValueError(
          'PTQ weights should already have the target shape. Got'
          f' {self.shape=} but {shape=} is requested.'
      )
    return self


# Register as NNX data to allow JAX arrays in Module attributes.
nnx.register_data_type(WithAux)


_PREQUANTIZED_ARRAY_LEAF_NAMES = frozenset(('qvalue', 'scale', 'zero_point'))


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

  def __init__(
      self,
      rules: Sequence[qconfig.QuantizationRule],
      *,
      disable_jit: bool = False,
      _qarray_module=qarray,
      _dot_general_fn=dot_general.dot_general,
      _einsum_fn=einsum.einsum,
      _conv_general_dilated_fn=conv_general.conv_general_dilated,
  ):
    """Initializes the PTQ provider."""
    super().__init__(rules, disable_jit=disable_jit)
    self._qarray_module = _qarray_module
    self._dot_general_fn = _dot_general_fn
    self._einsum_fn = _einsum_fn
    self._conv_general_dilated_fn = _conv_general_dilated_fn

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
        tile_size=rule.tile_size,
    )

    # Prepare rhs.
    if isinstance(rhs, WithAux):  # weight, already quantized
      rhs = rhs.array
    elif weight_name := flax_util.find_param(rhs):  # weight, not quantized
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.weight_qtype,
          calibration_method=rule.weight_calibration_method,
      )
      rhs = create_quantized_param(
          weight_name, rhs, rhs_how, _qarray_module=self._qarray_module
      ).array
    elif rule.act_qtype is not None:  # act
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.act_qtype,
          calibration_method=rule.act_calibration_method,
      )
      rhs = quantize_act(
          rhs, rhs_how, rule, op_id + '_rhs', _qarray_module=self._qarray_module
      )

    # Prepare lhs.
    if rule.act_qtype is not None:
      lhs_how = get_how_to_quantize(
          for_lhs=True,
          qtype=rule.act_qtype,
          calibration_method=rule.act_calibration_method,
      )
      lhs = quantize_act(
          lhs, lhs_how, rule, op_id + '_lhs', _qarray_module=self._qarray_module
      )
    return self._dot_general_fn(
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
    if len(operands) != 2:
      # TODO(jiwonshin): Support N-ary einsum if there is a need in the future.
      raise ValueError(f'Unsupported einsum format: {einsum_str=} {operands=}')

    lhs, rhs = operands
    get_how_to_quantize = functools.partial(
        einsum.get_how_to_quantize,
        einsum_str=einsum_str,
        ndims=(len(lhs.shape), len(rhs.shape)),
        tile_size=rule.tile_size,
    )

    # Prepare rhs.
    if isinstance(rhs, WithAux):  # weight, already quantized
      rhs = rhs.array
    elif weight_name := flax_util.find_param(rhs):  # weight, not quantized
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.weight_qtype,
          calibration_method=rule.weight_calibration_method,
      )
      rhs = create_quantized_param(
          weight_name, rhs, rhs_how, _qarray_module=self._qarray_module
      ).array
    elif rule.act_qtype is not None:  # act
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.act_qtype,
          calibration_method=rule.act_calibration_method,
      )
      rhs = quantize_act(
          rhs, rhs_how, rule, op_id + '_rhs', _qarray_module=self._qarray_module
      )

    # Prepare lhs.
    if rule.act_qtype is not None:
      lhs_how = get_how_to_quantize(
          for_lhs=True,
          qtype=rule.act_qtype,
          calibration_method=rule.act_calibration_method,
      )
      lhs = quantize_act(
          lhs, lhs_how, rule, op_id + '_lhs', _qarray_module=self._qarray_module
      )
    return self._einsum_fn(einsum_str, lhs, rhs)

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
      out_sharding=None,
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
          out_sharding=out_sharding,
      )
    dimension_numbers = jax.lax.conv_dimension_numbers(
        lhs.shape, rhs.shape, dimension_numbers
    )

    # Prepare rhs.
    if isinstance(rhs, WithAux):  # weight, already quantized
      rhs = rhs.array
    else:
      weight_name = flax_util.find_param(rhs)
      rhs_how = conv_general.get_how_to_quantize(
          dimension_numbers=dimension_numbers,
          for_lhs=False,
          qtype=rule.weight_qtype,
          calibration_method=rule.weight_calibration_method,
      )
      rhs = create_quantized_param(
          weight_name, rhs, rhs_how, _qarray_module=self._qarray_module
      ).array

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
    lhs = quantize_act(
        lhs, lhs_how, rule, op_id + '_lhs', _qarray_module=self._qarray_module
    )
    return self._conv_general_dilated_fn(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        out_sharding=out_sharding,
    )

  def nn_param(self, module: nn.Module, name: str, *args, **kwargs):
    """Intercepts nn.Module.param to handle quantized params."""
    # Don't check the shape if the param is already quantized.
    existing_param = module.get_variable('params', name)
    if isinstance(existing_param, WithAux):
      return nn.unbox(existing_param)
    return module.param(name, *args, **kwargs)

  def promote_dtype(self, *args, **kwargs):
    """Intercepts promote_dtype to handle quantized params."""
    if len(args) == 1 and isinstance(args[0], Sequence):
      args = args[0]  # nnx version
    # Skip WithAux.
    array_args = [x if isinstance(x, jax.Array) else None for x in args]
    array_args = flax.linen.dtypes.promote_dtype(*array_args, **kwargs)
    return [x if x is not None else y for x, y in zip(array_args, args)]

  def dot(
      self,
      a: jax.Array,
      b: jax.Array | WithAux[qarray.QArray],
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      out_sharding=None,
  ):
    """Intercepts jax.numpy.dot."""
    return dot.dot(
        a,
        b,
        precision=precision,
        preferred_element_type=preferred_element_type,
        out_sharding=out_sharding,
        _qwix_dot_general=self.dot_general,
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


def quantize_act(
    array: jax.Array,
    how: qarray.HowToQuantize,
    rule: qconfig.QuantizationRule,
    act_name: str,
    *,
    _qarray_module=qarray,
) -> qarray.QArray:
  """Quantizes the input activation with support for static scale."""
  if not rule.act_static_scale:
    return _qarray_module.quantize(array, how)

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
      calibration = _qarray_module.calibrate(array, how)
      # Apply act_batch_axes for static scale.
      calibration = jax.tree.map(
          lambda x: x.mean(axis=rule.act_batch_axes, keepdims=True), calibration
      )
    nonlocal zp
    scale, zp = _qarray_module.compute_scale_zero_point(calibration, how.qtype)
    # Wrap scale in WithAux because quantize_params needs to know the qtype.
    return WithAux(scale, how)

  zp = None
  scale = flax_util.get_or_create_param(act_name + '_scale', init)
  if zp is not None:
    zp = flax_util.get_or_create_param(act_name + '_zero_point', lambda: zp)
  return _qarray_module.quantize_with_scale_zero_point(
      array, how.qtype, scale.array, zp
  )


def create_quantized_param(
    name: str,
    value: jax.Array,
    how: qarray.HowToQuantize,
    *,
    _qarray_module=qarray,
) -> WithAux[qarray.QArray]:
  """Creates the quantized param and replaces the original param in the module.

  Args:
    name: The name of the param in the module.
    value: The unquantized jax.Array.
    how: How to quantize the param.
    _qarray_module: The qarray module to use. Useful for extending.

  Returns:
    An unboxed WithAux.
  """
  unboxed = WithAux(_qarray_module.quantize(value, how), how)

  # The following code is about replacing the saved param with WithAux, with
  # correct metadata.

  module = flax_util.get_current_module()
  if isinstance(module, nn.Module):
    if not module.is_initializing():
      raise ValueError(
          "It seems you're feeding an unquantized param to a quantized model."
      )
    param = module.get_variable('params', name)
    boxed = jax.tree.map(
        lambda value: flax_util.update_boxed(param, value=value), unboxed
    )
    module.put_variable('params', name, boxed)
  elif isinstance(module, nnx.Module):
    param = getattr(module, name)
    boxed = jax.tree.map(
        lambda value: flax_util.update_boxed(param, value=value), unboxed
    )
    setattr(module, name, boxed)

  return unboxed


def quantize_params(
    params: Any,
    abstract_quantized_params: Any,
    quant_stats: Any = flax.core.FrozenDict(),
    *,
    allow_extra_params: bool = False,
    _qarray_module=qarray,
) -> Any:
  """Quantize the param tree for PTQ.

  This function quantizes the param tree (weights) for PTQ. It doesn't need to
  run the model and is useful when the original params are too large to fit in
  the HBM.

  Args:
    params: The floating-point param tree to quantize, which is usually
      generated by the original or QAT model. The tree doesn't need to be
      complete and can be a subtree of the whole param tree. In NN, the tree
      needs to be unboxed, i.e. nn.unbox(). In NNX, the tree needs to be a pure
      dict, i.e. nnx.to_pure_dict().
    abstract_quantized_params: The param tree generated by the PTQ model, which
      can be abstract with jax.ShapeDtypeStruct as leaves instead of jax.Array.
      This includes the information of how to quantize each param. In NN, the
      tree may contain AxisMetadata. In NNX, this should be the PTQ model
      itself, possibly abstract.
    quant_stats: The quantization statistics, which needs to be a pure dict of
      unboxed values. This is only used in SRQ.
    allow_extra_params: If True, allow the params to contain extra parameters
      that are not present in the abstract_quantized_params, e.g., params for
      loss computation that are not needed in PTQ.
    _qarray_module: The qarray module to use. Useful for extending.

  Returns:
    The quantized param tree, which has the same structure as the input params
    but with quantized leaves.
  """
  quantized_params = {}
  for path, param in flax.traverse_util.flatten_dict(params).items():
    if not isinstance(param, jax.Array):
      raise TypeError(f'params is not a pure dict of jax.Array: {type(param)}')
    abs_param = get_value_from_path(abstract_quantized_params, path)
    if abs_param is None:
      if allow_extra_params:
        continue
      raise ValueError(f'{path} is not found in the abstract_quantized_params.')
    if isinstance(abs_param, WithAux):
      # The param might not be in the shape needed for compute, in case the
      # module reshapes before compute. Abstract param has the compute shape.
      param = param.reshape(abs_param.shape)
      param = abs_param.replace(
          array=_qarray_module.quantize(param, abs_param.how)
      )
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
    assert isinstance(abs_scale, WithAux)
    act_qtype = abs_scale.how.qtype

    calibration = averaging.SimpleMovingAverage().get_calibration(quant_stat)
    scale, zero_point = _qarray_module.compute_scale_zero_point(
        calibration, act_qtype
    )
    quantized_params[scale_path] = abs_scale.replace(array=scale)
    if zero_point is not None:
      quantized_params[(*path[:-1], path[-1] + '_zero_point')] = zero_point

  if isinstance(abstract_quantized_params, nnx.Module):
    # Convert WithAux to a pure dict so that nnx.update() can work.
    quantized_params = nnx.to_pure_dict(nnx.state(quantized_params))

  return flax.traverse_util.unflatten_dict(quantized_params)


def process_prequantized_params(
    checkpoint_params: Any,
    template_params: Any,
    *,
    allow_extra_params: bool = False,
    use_checkpoint_sharding: bool = False,
) -> Any:
  """Converts external pre-quantized params into an `nnx.update`-friendly pure dict.

  Args:
    checkpoint_params: A nested dict matching NNX state paths (without `.value`
      suffixes). Leaves must be either a `jax.Array` or a dict containing
      `'qvalue'`, `'scale'`, and optional `'zero_point'`.
    template_params: An NNX PTQ model, possibly abstract (e.g., from
      `nnx.eval_shape`).
    allow_extra_params: If True, ignore payload entries not present in
      `template_params`.
    use_checkpoint_sharding: If True, use sharding from checkpoint_params
      instead of template_params.

  Returns:
    A nested pure dict consumable by `nnx.update`. The nested key paths will
    include `'value'` at the end, pointing to either a dict or a JAX array.
  """
  if not isinstance(template_params, nnx.Module):
    raise TypeError(
        'process_prequantized_params only supports NNX PTQ models. Got'
        f' {type(template_params)}.'
    )

  def _is_leaf(path, x):
    """Checks if x is a pre-quantized leaf.

    Stop at dicts that contain the 'qvalue' key. This identifies them
    as pre-quantized weight payloads.

    Args:
      path: The parameter path. Unused.
      x: The value to check.

    Returns:
      True if x is a pre-quantized leaf, False otherwise.
    """
    del path
    return isinstance(x, dict) and 'qvalue' in x

  # Value: WithAux(Case 1), QArray(Case 2), or a JAX array(Case 3).
  flat_processed = {}
  # Value: dict containing 'qvalue'(quantized) or a JAX array(fp).
  flat_checkpoint = flax.traverse_util.flatten_dict(
      checkpoint_params, is_leaf=_is_leaf
  )
  # tuple path example: ('layers', 0, 'mlp', 'experts_gate_up_proj_weight').
  for path, checkpoint_param in flat_checkpoint.items():
    # Value: WithAux(quantized) or jax.ShapeDtypeStruct / jax.Array (fp)
    template_param = get_value_from_path(template_params, path)
    if template_param is None:
      if not allow_extra_params:
        raise ValueError(
            'Found extra parameters in prequantized_params not present in'
            f' template: {path}'
        )
      logging.info('Skipping parameter not in template: %s', path)
      continue

    # Case 1: checkpoint_param is prequantized (dict), template_param is WithAux
    if isinstance(checkpoint_param, dict) and isinstance(
        template_param, WithAux
    ):
      processed = _process_quantized_param(
          checkpoint_param,
          template_param,
          path,
          use_checkpoint_sharding=use_checkpoint_sharding,
      )

    # Case 2: checkpoint_param is prequantized (dict), template_param is fp.
    elif isinstance(checkpoint_param, dict) and not isinstance(
        template_param, WithAux
    ):
      processed = _create_qarray_from_checkpoint(
          checkpoint_param,
          path,
      )

    # Case 3: checkpoint_param is fp, template_param is fp.
    elif not isinstance(checkpoint_param, dict) and not isinstance(
        template_param, WithAux
    ):
      processed = _apply_sharding_and_dtype(
          checkpoint_param,
          template_param,
          path,
          use_checkpoint_sharding=use_checkpoint_sharding,
      )

    # Handle invalid combinations (e.g., checkpoint is full precision, but
    # template expects WithAux).
    else:
      raise ValueError(
          f'Unhandled or invalid parameter combination for {path}. '
          f'checkpoint_param is {type(checkpoint_param)}, '
          f'template_param is {type(template_param)}.'
      )

    flat_processed[path] = processed

  # Key: nested key path.
  # Value: WithAux(Case 1), QArray(Case 2), or a JAX array(Case 3).
  nested_processed = flax.traverse_util.unflatten_dict(flat_processed)
  # Key: nested key path and ['value'].
  # Value: Dict(Case 1 and 2) or a JAX array(Case 3).
  #   - Case 1: dict {"array": {"qvalue": ..., "scale": ...}}
  #   - Case 2: dict {"qvalue": ..., "scale": ...}
  #   - Case 3: <jax.Array>
  return nnx.to_pure_dict(nnx.state(nested_processed))


def _validate_prequantized_dict(
    checkpoint_param: Any,
    path: tuple[str, ...],
) -> None:
  """Validates the flat quantized parameter dictionary format.

  Checks that all the keys in the input dictionary are in
  _PREQUANTIZED_ARRAY_LEAF_NAMES.

  Args:
    checkpoint_param: A dict containing quantized leaves (`qvalue`, `scale`, and
      optional `zero_point`).
    path: The parameter path for error messages.
  """
  if not isinstance(checkpoint_param, dict):
    raise ValueError(
        f'{path} is quantized in the template_params. Expected a'
        ' dict containing qvalue/scale leaves.'
    )

  unsupported_keys = set(checkpoint_param) - _PREQUANTIZED_ARRAY_LEAF_NAMES
  if unsupported_keys:
    raise ValueError(
        f'{path} has unsupported quantized leaves'
        f' {sorted(unsupported_keys)}. Expected only'
        f' {sorted(_PREQUANTIZED_ARRAY_LEAF_NAMES)}.'
    )

  if 'qvalue' not in checkpoint_param or 'scale' not in checkpoint_param:
    param_summary = {
        k: (type(v), getattr(v, 'shape', None))
        for k, v in checkpoint_param.items()
    }
    raise ValueError(
        f'{path} is missing required "qvalue" or "scale" in pre-quantized'
        f' dictionary. Found keys: {list(checkpoint_param.keys())}, summary:'
        f' {param_summary}'
    )


def _apply_sharding_and_dtype(
    checkpoint_value: Any,
    template_value: Any,
    path: tuple[str, ...],
    allow_broadcast: bool = False,
    use_checkpoint_sharding: bool = False,
) -> jax.Array:
  """Converts a host/device array-like value into the template's array shape."""
  template_value = flax_util.unbox(template_value)
  if not isinstance(template_value, (jax.Array, jax.ShapeDtypeStruct)):
    raise TypeError(
        f'{path} does not resolve to a JAX array or ShapeDtypeStruct in the'
        ' template_params.'
    )

  target_dtype = template_value.dtype
  template_shape = template_value.shape
  if use_checkpoint_sharding:
    sharding = _get_sharding(getattr(checkpoint_value, 'sharding', None), path)
  else:
    sharding = _get_sharding(getattr(template_value, 'sharding', None), path)

  # Handle sharding.
  if sharding is not None:
    checkpoint_value = jax.device_put(checkpoint_value, sharding)
  else:
    checkpoint_value = jnp.asarray(checkpoint_value)

  # Handle dtype promotion to match template dtype.
  if checkpoint_value.dtype != target_dtype:
    checkpoint_value = checkpoint_value.astype(target_dtype)

  # Handle shape promotion.
  if checkpoint_value.shape != template_shape:
    # For scales and zero_point(allow_broadcast), broadcast 2d blocksize
    # scales/zero_points into 1d.
    if allow_broadcast:
      try:
        checkpoint_value = qarray.broadcast_to(checkpoint_value, template_shape)
      except Exception as e:
        raise ValueError(
            f'{path} has shape {checkpoint_value.shape}, expected'
            f' {template_shape}.'
        ) from e
    # For qvalue (not allow_broadcast), it should match the template shape.
    else:
      raise ValueError(
          f'{path} has shape {checkpoint_value.shape}, expected'
          f' {template_shape}.'
      )

  return checkpoint_value


def _get_sharding(
    sharding: Any,
    path: tuple[str, ...],
) -> jax.sharding.Sharding | None:
  """Resolves abstract mesh shardings into concrete device shardings.

  Args:
    sharding: The abstract or concrete sharding definition. If it is a
      NamedSharding with an AbstractMesh, it will be resolved to use the active
      mesh.
    path: The parameter path for error messages.

  Returns:
    A concrete `jax.sharding.Sharding` object, or None if the input was None.
  """
  if sharding is None:
    return None
  if isinstance(sharding, jax.sharding.NamedSharding) and isinstance(
      sharding.mesh, jax.sharding.AbstractMesh
  ):
    concrete_mesh = jax.sharding.get_mesh()
    if concrete_mesh.empty:
      raise ValueError(
          f'{path} requires an active mesh to place pre-quantized'
          ' arrays. Run process_prequantized_params inside the same'
          ' jax.set_mesh(...) context used for the sharded PTQ model.'
      )
    return jax.sharding.NamedSharding(concrete_mesh, sharding.spec)
  return sharding


def _process_quantized_param(
    checkpoint_param: Mapping[str, Any],
    template_param: WithAux[qarray.QArray],
    path: tuple[str, ...],
    *,
    use_checkpoint_sharding: bool,
) -> WithAux[qarray.QArray]:
  """Builds a WithAux[QArray] leaf from a quantized parameter dictionary and a quantized template.

  Args:
    checkpoint_param: A dictionary containing quantized leaves (`qvalue`,
      `scale`, and optional `zero_point`).
    template_param: An abstract template parameter containing the target shapes
      and types for coercion.
    path: The parameter path for error messages.
    use_checkpoint_sharding: Whether to use sharding from the checkpoint.

  Returns:
    A new `WithAux[QArray]` leaf whose contents have been coerced to the correct
    device placement, shape, and dtype.
  """
  _validate_prequantized_dict(checkpoint_param, path)

  template_array = template_param.array
  qvalue = _apply_sharding_and_dtype(
      checkpoint_param['qvalue'],
      template_array.qvalue,
      path,
      use_checkpoint_sharding=use_checkpoint_sharding,
  )
  scale = _apply_sharding_and_dtype(
      checkpoint_param['scale'],
      template_array.scale,
      path,
      allow_broadcast=True,
      use_checkpoint_sharding=use_checkpoint_sharding,
  )

  template_zero_point = template_array.zero_point
  zero_point = checkpoint_param.get('zero_point')
  if template_zero_point is None and zero_point is not None:
    raise ValueError(
        f'{path} provided an unexpected "zero_point" for a symmetric'
        ' quantized param.'
    )
  if template_zero_point is not None and zero_point is None:
    raise ValueError(f'{path} is missing required quantized leaf "zero_point".')
  if template_zero_point is not None:
    zero_point = _apply_sharding_and_dtype(
        zero_point,
        template_zero_point,
        path,
        allow_broadcast=True,
        use_checkpoint_sharding=use_checkpoint_sharding,
    )

  qarray_leaf = qarray.QArray(
      qvalue=qvalue,
      scale=scale,
      zero_point=zero_point,
      qtype=template_param.how.qtype,
  )
  qarray.validate_qarray(qarray_leaf)
  return template_param.replace(array=qarray_leaf)


def _create_qarray_from_checkpoint(
    checkpoint_param: Mapping[str, Any],
    path: tuple[str, ...],
) -> qarray.QArray:
  """Builds a QArray directly from a quantized parameter dictionary.

  This function assumes the checkpoint parameters are already in the correct
  shape and dtype.

  Args:
    checkpoint_param: A dictionary containing quantized leaves (`qvalue`,
      `scale`, and optional `zero_point`).
    path: The parameter path for error messages.

  Returns:
    A new `QArray` constructed from the checkpoint parameters.
  """
  _validate_prequantized_dict(checkpoint_param, path)

  # TODO(jiwonshin): Add support for using template_param as source of truth for
  # sharding, shape and dtype conversions in the future.
  qvalue = checkpoint_param['qvalue']
  scale = checkpoint_param['scale']
  zero_point = checkpoint_param.get('zero_point')
  qarray_leaf = qarray.QArray(qvalue=qvalue, scale=scale, zero_point=zero_point)
  qarray.validate_qarray(qarray_leaf)
  return qarray_leaf


def get_value_from_path(obj: Any, path: tuple[str | int, ...]) -> Any:
  """Helper that returns the value from the path in the object.

  Args:
    obj: The object to traverse (e.g., an NNX Module, dict, or list).
    path: A tuple of keys to traverse. Keys can be strings (for dict keys or
      object attributes) or integers (for list indices).

  Returns:
    The value found at the specified path, or None if not found.

  Example:
    If path is ('layers', 0, 'mlp', 'weight'), this function will return:
    obj.layers[0].mlp.weight
  """
  for key in path:
    if obj is None:
      return None
    if isinstance(obj, dict):
      obj = obj.get(key)
    elif isinstance(obj, (list, nnx.List)) and isinstance(key, int):
      obj = obj[key] if 0 <= key < len(obj) else None
    else:
      obj = getattr(obj, key, None)
  return obj
