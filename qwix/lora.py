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
from typing import Callable, Sequence

from flax import linen as nn
from flax import nnx
from flax import typing
import jax
from jax.nn import initializers
from qwix import aux_data
from qwix import flax_util
from qwix import model as qwix_model
from qwix import ptq
from qwix import qconfig
from qwix.core import einsum
from qwix.core import qarray


# apply_lora_to_model is just an alias for quantize_model.
apply_lora_to_model = qwix_model.quantize_model


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

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array | ptq.WithAux[qarray.QArray],
      dimension_numbers: jax.lax.DotDimensionNumbers,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
  ) -> jax.Array:
    """QAT dot_general."""
    res = super().dot_general(
        lhs, rhs, dimension_numbers, precision, preferred_element_type
    )

    rule, _ = self._get_current_rule_and_op_id(
        'dot_general', repeated_call=True
    )
    if not isinstance(rule, LoraRule):
      return res

    if isinstance(rhs, ptq.WithAux):
      weight_name = rhs.weight_name
      rhs_shape = qarray.get_original_shape(rhs.array)
    else:
      weight_name = aux_data.get(rhs, 'weight_name', None)
      rhs_shape = rhs.shape

    if weight_name is None:
      return res

    # TODO: support arbitrary dimension numbers.
    assert (
        len(rhs_shape) == 2
        and list(dimension_numbers[0][1]) == [0]
        and not dimension_numbers[1][1]
    ), f'Unsupported: {rhs_shape=} {dimension_numbers=}'

    rhs_lora_a, rhs_lora_b, dropout_layer = _get_or_create_lora_params(
        name=weight_name,
        rule=rule,
        a_shape=(rhs_shape[0], rule.rank),
        b_shape=(rule.rank, rhs_shape[1]),
        a_sharding_transpose=(0, None),
        b_sharding_transpose=(None, 1),
    )

    if dropout_layer is not None:
      # TODO: Use nnx.Rngs(0) for now. Need to check `deterministic`
      # to decide whether to provide a rng.
      # NOTE: this is wrong because it always uses the same rng.
      lhs = dropout_layer(lhs, rngs=nnx.Rngs(0))

    return res + lhs @ rhs_lora_a @ rhs_lora_b * rule.alpha

  def einsum(
      self,
      einsum_str: str,
      *operands: jax.Array | ptq.WithAux[qarray.QArray],
      **kwargs,
  ) -> jax.Array:
    """QAT einsum."""
    res = super().einsum(einsum_str, *operands, **kwargs)

    rule, _ = self._get_current_rule_and_op_id('einsum', repeated_call=True)
    if not isinstance(rule, LoraRule):
      return res

    if not isinstance(einsum_str, str) or len(operands) != 2:
      raise ValueError(f'Unsupported einsum format: {einsum_str=} {operands=}')
    lhs, rhs = operands

    if isinstance(rhs, ptq.WithAux):
      weight_name = rhs.weight_name
      rhs_shape = qarray.get_original_shape(rhs.array)
    else:
      weight_name = aux_data.get(rhs, 'weight_name', None)
      rhs_shape = rhs.shape

    if weight_name is None:
      return res

    (
        a_shape,
        b_shape,
        lora_einsum_str,
        a_sharding_transpose,
        b_sharding_transpose,
    ) = _parse_einsum_str_for_lora(lhs.shape, rhs_shape, einsum_str, rule.rank)

    # Store the lora_einsum_str for debugging.
    module = flax_util.get_current_module()
    setattr(module, weight_name + '_lora_einsum_str', lora_einsum_str)

    rhs_lora_a, rhs_lora_b, dropout_layer = _get_or_create_lora_params(
        name=weight_name,
        rule=rule,
        a_shape=a_shape,
        b_shape=b_shape,
        a_sharding_transpose=a_sharding_transpose,
        b_sharding_transpose=b_sharding_transpose,
    )

    if dropout_layer is not None:
      # TODO: Use nnx.Rngs(0) for now. Need to check `deterministic`
      # to decide whether to provide a rng.
      # NOTE: this is wrong because it always uses the same rng.
      lhs = dropout_layer(lhs, rngs=nnx.Rngs(0))

    return res + (
        jax.numpy.einsum(lora_einsum_str, lhs, rhs_lora_a, rhs_lora_b, **kwargs)
        * rule.alpha
    )


def _get_or_create_lora_params(
    *,
    name: str,
    rule: LoraRule,
    a_shape: typing.Shape,
    b_shape: typing.Shape,
    a_sharding_transpose: Sequence[int | None],
    b_sharding_transpose: Sequence[int | None],
) -> tuple[jax.Array, jax.Array, nnx.Dropout | None]:
  """Get or create LoRA params.

  Args:
    name: The prefix of the LoRA param.
    rule: The LoRA rule.
    a_shape: The shape of the lora_a param.
    b_shape: The shape of the lora_b param.
    a_sharding_transpose: The transpose to derive the sharding for lora_a.
    b_sharding_transpose: The transpose to derive the sharding for lora_b.

  Returns:
    A tuple of (lora_a, lora_b, dropout_layer).
  """
  dropout_layer = None
  if rule.dropout > 0:
    dropout_layer = nnx.Dropout(rule.dropout)

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
    return flax_util.unbox(lora_a), flax_util.unbox(lora_b), dropout_layer

  if isinstance(param, ptq.WithAux):
    lora_dtype = flax_util.unbox(param.array.scale).dtype
    boxed = param.array.qvalue
    if isinstance(param.array, qarray.TransposedQArray):
      # Restore to the original sharding.
      tiled_axes = qarray.get_tiled_axes(flax_util.unbox(param.array))
      boxed = flax_util.update_boxed(boxed, merge=set(tiled_axes))
  else:  # base model is not quantized.
    lora_dtype = flax_util.unbox(param).dtype
    boxed = param

  def init(initializer, shape, transpose):
    # TODO: use the actual rng.
    value = initializer(jax.random.key(0), shape, lora_dtype)
    value = flax_util.update_boxed(boxed, value=value, transpose=transpose)
    if isinstance(value, nnx.Variable):
      return nnx.VariableMetadata(value.value, metadata=value.get_metadata())
    return value

  lora_a = flax_util.get_or_create_param(
      name + '_lora_a',
      lambda: init(rule.lora_a_initializer, a_shape, a_sharding_transpose),
      nnx.LoRAParam,
  )
  lora_b = flax_util.get_or_create_param(
      name + '_lora_b',
      lambda: init(rule.lora_b_initializer, b_shape, b_sharding_transpose),
      nnx.LoRAParam,
  )
  return lora_a, lora_b, dropout_layer
