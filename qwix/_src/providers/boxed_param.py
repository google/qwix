# Copyright 2026 Google LLC
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
"""Shared provider support for boxed quantized parameters."""

import functools
from typing import Callable, Generic, Sequence, TypeVar

import flax
from flax import linen as nn
from flax import nnx
import flax.linen.dtypes
import jax
from jax import numpy as jnp
from qwix._src import averaging
from qwix._src import qconfig
from qwix._src.core import conv_general
from qwix._src.core import dot
from qwix._src.core import dot_general
from qwix._src.core import einsum
from qwix._src.core import qarray
from qwix._src.utils import flax_util

ArrayTypeVar = TypeVar('ArrayTypeVar', jax.Array, qarray.QArray)


@flax.struct.dataclass
class WithAux(Generic[ArrayTypeVar]):
  """An array/QArray with auxiliary information.

  The main purpose of this class is to embed the how to quantize information
  into the param tree, such that quantize_params() can quantize params without
  knowing the model structure.

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
  dtype = property(lambda self: flax_util.unbox(self.array).dtype)

  def astype(self, dtype):
    new_value = flax_util.unbox(self.array).astype(dtype)
    return self.replace(array=flax_util.update_boxed(self.array, value=new_value))  # pyrefly: ignore

  def reshape(self, *shape):
    if len(shape) == 1:
      try:
        shape = tuple(shape[0])
      except TypeError:
        pass
    if tuple(self.shape) != tuple(shape):
      raise ValueError(
          'Boxed weights should already have the target shape. Got'
          f' {self.shape=} but {shape=} is requested.'
      )
    return self


# Register as NNX data to allow JAX arrays in Module attributes.
nnx.register_data_type(WithAux)


class BoxedParamProvider(qconfig.QuantizationProvider):
  """Base provider for runtime ops that substitute boxed parameters.

  This provider owns the shared WithAux/QArray handling used by PTQ inference
  and QLoRA training. Subclasses add their domain-specific behavior on top of
  these intercepted JAX operations.
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
    """Initializes the boxed-parameter provider."""
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
          rhs,  # pyrefly: ignore[bad-argument-type]
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
      rhs = rhs.array  # pyrefly: ignore[bad-assignment]
    elif weight_name := flax_util.find_param(rhs):  # weight, not quantized
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.weight_qtype,
          calibration_method=rule.weight_calibration_method,
      )
      rhs = create_quantized_param(  # pyrefly: ignore[bad-assignment]
          weight_name, rhs, rhs_how, _qarray_module=self._qarray_module
      ).array
    elif rule.act_qtype is not None:  # act
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.act_qtype,
          calibration_method=rule.act_calibration_method,
      )
      rhs = quantize_act(  # pyrefly: ignore[bad-assignment]
          rhs, rhs_how, rule, op_id + '_rhs', _qarray_module=self._qarray_module  # pyrefly: ignore[unsupported-operation]
      )

    # Prepare lhs.
    if rule.act_qtype is not None:
      lhs_how = get_how_to_quantize(
          for_lhs=True,
          qtype=rule.act_qtype,
          calibration_method=rule.act_calibration_method,
      )
      lhs = quantize_act(  # pyrefly: ignore[bad-assignment]
          lhs, lhs_how, rule, op_id + '_lhs', _qarray_module=self._qarray_module  # pyrefly: ignore[unsupported-operation]
      )
    return self._dot_general_fn(
        lhs, rhs, dimension_numbers, out_sharding=out_sharding  # pyrefly: ignore[bad-argument-type]
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
          rhs, rhs_how, rule, op_id + '_rhs', _qarray_module=self._qarray_module  # pyrefly: ignore[unsupported-operation]
      )

    # Prepare lhs.
    if rule.act_qtype is not None:
      lhs_how = get_how_to_quantize(
          for_lhs=True,
          qtype=rule.act_qtype,
          calibration_method=rule.act_calibration_method,
      )
      lhs = quantize_act(
          lhs, lhs_how, rule, op_id + '_lhs', _qarray_module=self._qarray_module  # pyrefly: ignore[unsupported-operation]
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
          rhs,  # pyrefly: ignore[bad-argument-type]
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
      rhs = rhs.array  # pyrefly: ignore[bad-assignment]
    else:
      weight_name = flax_util.find_param(rhs)
      rhs_how = conv_general.get_how_to_quantize(
          dimension_numbers=dimension_numbers,
          for_lhs=False,
          qtype=rule.weight_qtype,
          calibration_method=rule.weight_calibration_method,
      )
      rhs = create_quantized_param(  # pyrefly: ignore[bad-assignment]
          weight_name, rhs, rhs_how, _qarray_module=self._qarray_module  # pyrefly: ignore[bad-argument-type]
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
    lhs = quantize_act(  # pyrefly: ignore[bad-assignment]
        lhs, lhs_how, rule, op_id + '_lhs', _qarray_module=self._qarray_module  # pyrefly: ignore[unsupported-operation]
    )
    return self._conv_general_dilated_fn(
        lhs,
        rhs,  # pyrefly: ignore[bad-argument-type]
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
    """Intercepts nn.Module.param to handle boxed params."""
    existing_param = module.get_variable('params', name)
    if isinstance(existing_param, WithAux):
      return nn.unbox(existing_param)
    return module.param(name, *args, **kwargs)

  def promote_dtype(self, *args, **kwargs):
    """Intercepts promote_dtype to handle boxed params."""
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
        b,  # pyrefly: ignore[bad-argument-type]
        precision=precision,
        preferred_element_type=preferred_element_type,
        out_sharding=out_sharding,
        _qwix_dot_general=self.dot_general,
    )

  def asarray(self, a, dtype=None, order=None, **kwargs):
    """Intercepts jax.numpy.asarray to correctly handle WithAux and QArray.

    Without this interception, calling `jax.numpy.asarray` on these custom
    PyTree structures would result in a crash, as JAX cannot natively convert
    them into flat arrays.

    This function covers the following cases:
      - nnx.State containing QArray components (unboxes into QArray).
      - WithAux (propagates to the underlying array).
      - QArray (preserves QArray, casting dtype if requested).

    For all other types, including linen Variables, this falls back to calling
    `jax.numpy.asarray` on the unboxed value.

    Args:
      a: The input array or structure to convert.
      dtype: The desired data-type for the array.
      order: The memory layout of the array (not used).
      **kwargs: Additional keyword arguments to pass to `jnp.asarray`.

    Returns:
      The correctly unwrapped and converted array.
    """
    # 1. Unbox early to handle nnx.Param, Linen variables, etc.
    a = flax_util.unbox(a)

    # 2. Handle nnx.State reconstruction
    if isinstance(a, nnx.State) and 'array' in a:
      a = a['array']
      if isinstance(a, nnx.State) and 'qvalue' in a and 'scale' in a:
        # Since we already unboxed, the values inside the state are no longer
        # nnx.Variable.
        qkwargs = {'qvalue': a['qvalue'], 'scale': a['scale']}
        if 'zero_point' in a:
          qkwargs['zero_point'] = a['zero_point']
        a = qarray.QArray(**qkwargs)  # pyrefly: ignore

    # 3. Handle custom types
    if isinstance(a, WithAux):
      return a.replace(  # pyrefly: ignore[missing-attribute]
          array=self.asarray(a.array, dtype=dtype, order=order, **kwargs)
      )

    if isinstance(a, qarray.QArray):
      if dtype is not None and a.dtype != dtype:
        return a.astype(dtype)
      return a

    # 4. Fallback for standard JAX arrays
    return jnp.asarray(a, dtype=dtype, order=order, **kwargs)

  def get_intercept_map(self):
    """Used for interception."""
    return super().get_intercept_map() | {
        'jax.lax.conv_general_dilated': self.conv_general_dilated,
        'jax.lax.dot_general': self.dot_general,
        'jax.numpy.asarray': self.asarray,
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
  # This is useful in NNX when a boxed-param model is converted from a QAT
  # model. We delete the quant_stat after the first forward pass so that the
  # converted model appears the same as a regular one.
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
