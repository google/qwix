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
"""Quantized training (QT) support."""

from typing import Any, Callable, Sequence

from flax import linen as nn
from flax import nnx
import flax.linen.dtypes
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from qwix import aux_data
from qwix import interception
from qwix import qconfig
from qwix.core import dot_general_qt


class QtProvider(qconfig.QuantizationProvider):
  """Quantization provider for Quantized Training(QT)."""

  def __init__(
      self, rules, disable_channelwise_axes=True, use_original_residuals=False
  ):
    """Initializes the QtProvider.

    Args:
      rules: The quantization rules.
      disable_channelwise_axes: Whether to disable channelwise axes.
      use_original_residuals: Whether to use original residuals instead of
        quantized residuals.
    """
    super().__init__(rules=rules)
    self._disable_channelwise_axes = disable_channelwise_axes
    self._use_original_residuals = use_original_residuals

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      *,
      out_sharding=None,
  ) -> jax.Array:
    """QT dot_general."""
    rule, _ = self._get_current_rule_and_op_id('dot_general')
    if rule is None or rule.weight_qtype is None:
      return jax.lax.dot_general(
          lhs,
          rhs,
          dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
          out_sharding=out_sharding,
      )
    config = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype=rule.act_qtype,
        rhs_qtype=rule.weight_qtype
        if aux_data.get(rhs, 'weight_name', None)
        else rule.act_qtype,
        bwd_qtype=rule.bwd_qtype,
        tile_size=rule.tile_size,
        # TODO(jiwonshin): use separate calibration methods for lhs and rhs.
        calibration_method='absmax',
        disable_channelwise_axes=self._disable_channelwise_axes,
        use_original_residuals=self._use_original_residuals,
    )
    return dot_general_qt.dot_general_qt(lhs, rhs, dimension_numbers, config)

  def einsum(
      self,
      einsum_str: str,
      *operands: jax.Array,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      _dot_general: Callable[..., jax.Array] = jax.lax.dot_general,  # pylint: disable=invalid-name
      out_sharding=None,
  ) -> jax.Array:
    """QT einsum."""
    rule, _ = self._get_current_rule_and_op_id('einsum')
    if rule is None or rule.weight_qtype is None:
      return jnp.einsum(
          einsum_str,
          *operands,
          precision=precision,
          preferred_element_type=preferred_element_type,
          _dot_general=_dot_general,
          out_sharding=out_sharding,
      )
    if not isinstance(einsum_str, str) or len(operands) != 2:
      raise ValueError(f'Unsupported einsum format: {einsum_str=} {operands=}')
    _, rhs = operands

    config = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype=rule.act_qtype,
        rhs_qtype=rule.weight_qtype
        if aux_data.get(rhs, 'weight_name', None)
        else rule.act_qtype,
        bwd_qtype=rule.bwd_qtype,
        tile_size=rule.tile_size,
        # TODO(jiwonshin): use separate calibration methods for lhs and rhs.
        calibration_method='absmax',
        disable_channelwise_axes=self._disable_channelwise_axes,
        use_original_residuals=self._use_original_residuals,
    )
    custom_dot_general = lambda *args, **kwargs: dot_general_qt.dot_general_qt(
        *args[:3], config
    )
    return jnp.einsum(
        einsum_str,
        *operands,
        precision=precision,
        preferred_element_type=preferred_element_type,
        _dot_general=custom_dot_general,
        out_sharding=out_sharding,
    )

  def nn_param(
      self,
      module: nn.Module,
      name: str,
      init_fn: Callable[..., Any],
      *init_args,
      unbox: bool = True,
      **init_kwargs,
  ) -> jax.Array | nn.meta.AxisMetadata[jax.Array]:
    """Intercepts nn.Module.param."""
    ret = nn.Module.param(
        module, name, init_fn, *init_args, unbox=unbox, **init_kwargs
    )
    aux_data.clear(ret if unbox else ret.unbox())
    aux_data.set(ret if unbox else ret.unbox(), 'weight_name', name)
    return ret

  def promote_dtype(self, *args, **kwargs):
    """Intercepts flax.{linen,nnx.nn}.dtypes.promote_dtype."""
    if len(args) == 1 and isinstance(args[0], Sequence):
      args = args[0]  # nnx version
    ret = flax.linen.dtypes.promote_dtype(*args, **kwargs)
    # Forward weight_name aux_data.
    for x, y in zip(ret, args):
      if y is not None and (name := aux_data.get(y, 'weight_name', None)):
        aux_data.set(x, 'weight_name', name)
    return ret

  def get_intercept_map(self):
    """Used for interception."""
    return {
        # TODO(jiwonshin): add support for quantized conv_general_dilated.
        'jax.lax.dot_general': self.dot_general,
        'jax.numpy.einsum': self.einsum,
        # Disable interception for ops in pallas_call.
        'jax.experimental.pallas.pallas_call': (
            lambda *args, **kwargs: interception.disable_interceptions(
                pl.pallas_call(*args, **kwargs)
            )
        ),
        'flax.linen.Module.param': self.nn_param,
        'flax.linen.dtypes.promote_dtype': self.promote_dtype,
        'flax.nnx.nn.dtypes.promote_dtype': self.promote_dtype,
    }

  def process_model_inputs(
      self, model: nn.Module | nnx.Module, model_args: Any, model_kwargs: Any
  ) -> tuple[nn.Module | nnx.Module, Any, Any]:
    """Processes the nnx.Module instance before it is called."""
    if isinstance(model, nnx.Module):
      for path, node in nnx.iter_graph(model):
        if isinstance(node, nnx.Module):
          aux_data.clear(node)  # clear the op_count.
        elif isinstance(node, nnx.Param):
          # weight_name is used to distinguish weights from activations.
          aux_data.clear(node.value)
          aux_data.set(node.value, 'weight_name', path[-1])
    return model, model_args, model_kwargs
