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
"""Common calibration logic for quantization providers."""

import abc
from typing import Any, Callable

import jax
from jax import numpy as jnp
from qwix._src import averaging
from qwix._src import flax_util
from qwix._src import qconfig


class StatsCalibrationProvider(
    qconfig.QuantizationProvider, metaclass=abc.ABCMeta
):
  """Base class for calibration providers that collect statistics.

  This provider intercepts `dot_general` and `einsum` operations to compute and
  collect statistics on the input activations or gradients. The statistics are
  stored in a `quant_stats` collection using `SimpleMovingAverage`.
  """

  @abc.abstractmethod
  def get_rule_type(self) -> type[qconfig.QuantizationRule]:
    """Returns the rule type that this provider handles."""

  @abc.abstractmethod
  def compute_stats(self, lhs: jax.Array) -> dict[str, Any]:
    """Computes statistics from the input array."""

  @abc.abstractmethod
  def get_stats_suffix(self) -> str:
    """Returns the suffix for the stats variable name."""

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      *args,
      rule: qconfig.QuantizationRule | None = None,
      **kwargs,
  ) -> jax.Array:
    res = jax.lax.dot_general(lhs, rhs, dimension_numbers, *args, **kwargs)
    if rule is None:
      rule, _ = self._get_current_rule_and_op_id('dot_general')

    rule_type = self.get_rule_type()
    if not isinstance(rule, rule_type):
      return res

    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    if lhs_ba or rhs_ba or len(lhs_ca) != 1 or len(rhs_ca) != 1:
      # We only support standard dot_general with 1 contracting axis for now.
      # If the operation is not supported, we skip calibration for it.
      return res

    weight_name = flax_util.find_param(rhs)
    if weight_name is None:
      # If we cannot identify the weight parameter, we skip calibration.
      return res

    # Reorder lhs to (ca, rest) and compute stats.
    lhs = jnp.moveaxis(lhs, lhs_ca[0], 0)
    lhs = lhs.reshape(lhs.shape[0], -1)

    # Collect the stats.
    stats = self.compute_stats(lhs)
    aggregator = averaging.SimpleMovingAverage()
    var_name = weight_name + self.get_stats_suffix()
    quant_stat = flax_util.get_or_create_variable(
        'quant_stats', var_name, lambda: aggregator.init(stats)
    )
    if flax_util.should_update_quant_stats():
      quant_stat.value = aggregator.update(quant_stat.value, stats)

    return res

  def einsum(self, einsum_str, *operands, **kwargs):
    rule, _ = self._get_current_rule_and_op_id('einsum')
    rule_type = self.get_rule_type()
    if not isinstance(rule, rule_type):
      return jnp.einsum(einsum_str, *operands, **kwargs)

    if not isinstance(einsum_str, str) or len(operands) != 2:
      return jnp.einsum(einsum_str, *operands, **kwargs)

    def stats_dot_general(lhs, rhs, dimension_numbers, *args, **kwargs):
      return self.dot_general(
          lhs, rhs, dimension_numbers, *args, rule=rule, **kwargs
      )

    with jax.disable_jit():
      return jnp.einsum(
          einsum_str,
          *operands,
          _dot_general=stats_dot_general,
          **kwargs,
      )

  def get_intercept_map(self) -> dict[str, Callable[..., Any]]:
    return super().get_intercept_map() | {
        'jax.lax.dot_general': self.dot_general,
        'jax.numpy.einsum': self.einsum,
    }
