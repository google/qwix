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
"""Low-Rank Adapation (LoRA) support."""
import dataclasses
import string
from typing import Any, Callable, Collection, Sequence
import warnings

from flax import linen as nn
from flax import nnx
from flax import typing
import jax
from jax.nn import initializers
from qwix._src import flax_util
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.core import einsum
from qwix._src.core import qarray
from qwix._src.providers import ptq


def apply_lora_to_model(
    model: qwix_model.ModelType,
    provider: qconfig.QuantizationProvider,
    *model_inputs: Any,
    methods: Collection[str] = ('__call__',),
    **model_inputs_kwargs: Any,
) -> qwix_model.ModelType:
  """Applies LoRA to a model."""
  # RNG is always needed for LoRA, so we eagerly check it here.
  if isinstance(model, nnx.Module) and 'rngs' not in model_inputs_kwargs:
    warnings.warn(
        'rngs must be provided for NNX models to initialize LoRA weights. '
        'Please specify rngs=nnx.Rngs(...) in apply_lora_to_model.'
    )
    model_inputs_kwargs['rngs'] = nnx.Rngs(10003)

  # apply_lora_to_model is just an alias for quantize_model.
  return qwix_model.quantize_model(
      model,
      provider,
      *model_inputs,
      methods=methods,
      **model_inputs_kwargs,
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class LoraRule(qconfig.QuantizationRule):
  """LoRA rules that match and configure the LoRA behavior."""

  ########################################################
  ### "Configs" that specify the LoRA behavior.
  ########################################################

  # The rank of the LoRA.
  rank: int

  # The alpha scaling parameter of the LoRA. It controls how we update the
  # original weights with LoRA weights
  alpha: float

  # The dropout rate for the LoRA.
  dropout: float = 0.0

  # The initializers for the LoRA A (fan-in) weight, default as he_uniform().
  lora_a_initializer: Callable[..., jax.Array] = initializers.he_uniform()

  # The initializer for the LoRA B (fan-out) weight, default as zeros.
  lora_b_initializer: Callable[..., jax.Array] = initializers.zeros


def _find_lora_dim_char(all_dims: set[str]):
  if 'r' not in all_dims:
    return 'r'
  return sorted(set(string.ascii_letters) - all_dims)[0]


def _parse_einsum_str_for_lora(
    lhs_shape: typing.Shape,
    rhs_shape: typing.Shape,
    einsum_str: str,
    lora_rank: int,
) -> tuple[
    typing.Shape,  # a_shape
    typing.Shape,  # b_shape
    str,  # lora_einsum_str
    Sequence[int | None],  # a_sharding_transpose
    Sequence[int | None],  # b_sharding_transpose
]:
  """Returns lora param shapes and einsum string for LoRA."""
  einsum_info = einsum.get_einsum_info(
      einsum_str, (len(lhs_shape), len(rhs_shape))
  )
  lora_dim_char = _find_lora_dim_char(
      set(einsum_info.lhs) | set(einsum_info.rhs)
  )

  a_shape, b_shape = (), (lora_rank,)
  a_str, b_str = '', lora_dim_char
  a_sharding_transpose, b_sharding_transpose = (), (None,)
  assert len(einsum_info.rhs) == len(rhs_shape)
  for i, (c, dim) in enumerate(zip(einsum_info.rhs, rhs_shape)):
    if c in set(einsum_info.lhs) & set(einsum_info.rhs):  # batch or contracting
      a_str += c
      a_shape += (dim,)
      a_sharding_transpose += (i,)
    else:
      b_str += c
      b_shape += (dim,)
      b_sharding_transpose += (i,)
  a_str += lora_dim_char
  a_shape += (lora_rank,)
  a_sharding_transpose += (None,)

  return (
      a_shape,
      b_shape,
      ','.join([einsum_info.lhs, a_str, b_str]) + '->' + einsum_info.out,
      a_sharding_transpose,
      b_sharding_transpose,
  )


class LoraProvider(ptq.PtqProvider):
  """Provider for (Q)LoRA.

  LoraProvider inherits from PtqProvider, because the base model is frozen
  during LoRA training.
  """

  def __init__(self, rules=None, **kwargs):
    """Initializes the LoraProvider.

    Usage:
      LoraProvider(module_path='module_path', rank=4, alpha=0.5)
    or
      LoraProvider([
          LoraRule(module_path='module_path', rank=4, alpha=0.5)
      ])

    Args:
      rules: A list of quantization rules.
      **kwargs: The keyword arguments to create a rule. Only one of rules and
        kwargs should be provided.
    """
    if rules is None:
      rules = [LoraRule(**kwargs)]
    elif kwargs:
      raise ValueError('Only one of rules and kwargs should be provided.')
    super().__init__(rules=rules)

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array | ptq.WithAux[qarray.QArray],
      dimension_numbers: jax.lax.DotDimensionNumbers,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      out_sharding: jax.sharding.NamedSharding | None = None,
  ) -> jax.Array:
    """LoRA dot_general."""
    res = super().dot_general(
        lhs,
        rhs,
        dimension_numbers,
        precision,
        preferred_element_type,
        out_sharding=out_sharding,
    )

    rule, _ = self._get_current_rule_and_op_id(
        'dot_general', repeated_call=True
    )
    if not isinstance(rule, LoraRule):
      return res

    weight_name = flax_util.find_param(rhs, ptq.WithAux)
    if weight_name is None:  # rhs is not a weight.
      return res

    # We only support ...a,ab->...b for now.
    assert (
        len(rhs.shape) == 2
        and tuple(dimension_numbers[0][1]) == (0,)
        and not dimension_numbers[1][1]
    ), f'Unsupported: {rhs.shape=} {dimension_numbers=}'

    lora_a, lora_b = _get_or_create_lora_params(
        name=weight_name,
        rule=rule,
        a_shape=(rhs.shape[0], rule.rank),
        b_shape=(rule.rank, rhs.shape[1]),
        a_sharding_transpose=(0, None),
        b_sharding_transpose=(None, 1),
    )

    if rule.dropout > 0:
      # This also works for linen.
      lhs = nnx.Dropout(rule.dropout)(lhs, rngs=flax_util.make_rng('dropout'))

    return res + lhs @ lora_a @ lora_b * (rule.alpha / rule.rank)

  def einsum(
      self,
      einsum_str: str,
      *operands: jax.Array | ptq.WithAux[qarray.QArray],
      **kwargs,
  ) -> jax.Array:
    """LoRA einsum."""
    res = super().einsum(einsum_str, *operands, **kwargs)

    rule, _ = self._get_current_rule_and_op_id('einsum', repeated_call=True)
    if not isinstance(rule, LoraRule):
      return res

    if not isinstance(einsum_str, str) or len(operands) != 2:
      raise ValueError(f'Unsupported einsum format: {einsum_str=} {operands=}')
    lhs, rhs = operands

    weight_name = flax_util.find_param(rhs, ptq.WithAux)
    if weight_name is None:  # rhs is not a weight.
      return res

    (
        a_shape,
        b_shape,
        lora_einsum_str,
        a_sharding_transpose,
        b_sharding_transpose,
    ) = _parse_einsum_str_for_lora(lhs.shape, rhs.shape, einsum_str, rule.rank)

    # Store the lora_einsum_str for debugging.
    module = flax_util.get_current_module()
    setattr(module, weight_name + '_lora_einsum_str', lora_einsum_str)

    lora_a, lora_b = _get_or_create_lora_params(
        name=weight_name,
        rule=rule,
        a_shape=a_shape,
        b_shape=b_shape,
        a_sharding_transpose=a_sharding_transpose,
        b_sharding_transpose=b_sharding_transpose,
    )

    if rule.dropout > 0:
      # This also works for linen.
      lhs = nnx.Dropout(rule.dropout)(lhs, rngs=flax_util.make_rng('dropout'))

    return res + (
        jax.numpy.einsum(lora_einsum_str, lhs, lora_a, lora_b, **kwargs)
        * (rule.alpha / rule.rank)
    )

  def conv_general_dilated(
      self,
      lhs: jax.Array,
      rhs: jax.Array | ptq.WithAux[qarray.QArray],
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
    """LoRA conv_general_dilated."""
    res = super().conv_general_dilated(
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

    rule, _ = self._get_current_rule_and_op_id(
        'conv_general_dilated', repeated_call=True
    )
    if not isinstance(rule, LoraRule):
      return res

    weight_name = flax_util.find_param(rhs, ptq.WithAux)
    assert weight_name is not None, 'rhs must be a weight.'

    dimension_numbers = jax.lax.conv_dimension_numbers(
        lhs.shape, rhs.shape, dimension_numbers
    )
    # Assert that the out feature is the last dimension of rhs and out.
    assert (
        dimension_numbers.rhs_spec[0] == len(rhs.shape) - 1
        and dimension_numbers.out_spec[1] == len(lhs.shape) - 1
    ), f'Unsupported: {dimension_numbers=}'

    lora_a, lora_b = _get_or_create_lora_params(
        name=weight_name,
        rule=rule,
        a_shape=(*rhs.shape[:-1], rule.rank),
        b_shape=(rule.rank, rhs.shape[-1]),
        a_sharding_transpose=(*range(len(rhs.shape) - 1), None),
        b_sharding_transpose=(None, len(rhs.shape) - 1),
    )

    if rule.dropout > 0:
      # This also works for linen.
      lhs = nnx.Dropout(rule.dropout)(lhs, rngs=flax_util.make_rng('dropout'))

    return res + jax.lax.conv_general_dilated(
        lhs,
        lora_a,
        window_strides,
        padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        precision=precision,
        preferred_element_type=preferred_element_type,
    ) @ lora_b * (rule.alpha / rule.rank)


def _get_or_create_lora_params(
    *,
    name: str,
    rule: LoraRule,
    a_shape: typing.Shape,
    b_shape: typing.Shape,
    a_sharding_transpose: Sequence[int | None],
    b_sharding_transpose: Sequence[int | None],
) -> tuple[jax.Array, jax.Array]:
  """Get or create LoRA params.

  Args:
    name: The prefix of the LoRA param.
    rule: The LoRA rule.
    a_shape: The shape of the lora_a param.
    b_shape: The shape of the lora_b param.
    a_sharding_transpose: The transpose to derive the sharding for lora_a.
    b_sharding_transpose: The transpose to derive the sharding for lora_b.

  Returns:
    A tuple of LoRA weights (lora_a, lora_b).
  """
  # Get the boxed param so that we can access the metadata.
  module = flax_util.get_current_module()
  if isinstance(module, nn.Module):
    param = module.get_variable('params', name)
    lora_a = module.get_variable('params', name + '_lora_a')
    lora_b = module.get_variable('params', name + '_lora_b')
  else:  # isinstance(module, nnx.Module)
    param = getattr(module, name)
    lora_a = getattr(module, name + '_lora_a', None)
    lora_b = getattr(module, name + '_lora_b', None)

  if lora_a is not None and lora_b is not None:
    return flax_util.unbox(lora_a), flax_util.unbox(lora_b)

  def get_canonical_pspec(x):
    """Returns the canonical sharding.spec if x contains a concrete array."""
    x = flax_util.unbox(x)
    sharding = getattr(x, 'sharding', None)
    if not isinstance(sharding, jax.sharding.NamedSharding):
      return None
    # The sharding.spec may be shorter than the ndim.
    padded_pspec = sharding.spec + (None,) * (x.ndim - len(sharding.spec))
    return sharding.update(spec=padded_pspec)

  # Get the dtype, boxed param, and (optional) sharding from the original param.
  if isinstance(param, ptq.WithAux):
    lora_dtype = flax_util.unbox(param.array.scale).dtype
    boxed = param.array.qvalue
    sharding = get_canonical_pspec(boxed)
  else:  # base model is not quantized.
    lora_dtype = flax_util.unbox(param).dtype
    boxed = param
    sharding = get_canonical_pspec(boxed)

  def init_with_sharding(initializer, rng, shape, transpose):
    value = initializer(rng, shape, lora_dtype)
    if sharding is not None:
      lora_pspec = flax_util.update_sharding(sharding.spec, transpose=transpose)
      value = jax.device_put(value, sharding.update(spec=lora_pspec))
    value = flax_util.update_boxed(boxed, value=value, transpose=transpose)
    if isinstance(value, nnx.Variable):
      return nnx.VariableMetadata(value.value, metadata=value.get_metadata())
    return value

  lora_a = flax_util.get_or_create_param(
      name + '_lora_a',
      lambda rng: init_with_sharding(
          rule.lora_a_initializer, rng, a_shape, a_sharding_transpose
      ),
      nnx_param_type=nnx.LoRAParam,
      need_rng=True,
  )
  lora_b = flax_util.get_or_create_param(
      name + '_lora_b',
      lambda rng: init_with_sharding(
          rule.lora_b_initializer, rng, b_shape, b_sharding_transpose
      ),
      nnx_param_type=nnx.LoRAParam,
      need_rng=True,
  )
  return lora_a, lora_b
