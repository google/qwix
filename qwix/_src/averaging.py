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
"""Averaging of calibration into quant stats."""

from typing import TypeAlias

import jax
import jax.numpy as jnp

QuantStat: TypeAlias = dict[str, jax.Array]
Calibration: TypeAlias = dict[str, jax.Array]


class SimpleMovingAverage:
  """Simple moving average maintains a count and sums of the calibration."""

  def __init__(self, bootstrap_steps: int = 0):
    """Initializes the simple moving average.

    Args:
      bootstrap_steps: The number of steps to bootstrap the quantization
        statistics.
    """
    self._bootstrap_steps = bootstrap_steps

  def init(self, calibration: Calibration) -> QuantStat:
    """Initializes the quantization statistics from the calibration."""
    quant_stat = {'count': jnp.zeros((), jnp.int32)}
    for key, value in calibration.items():
      # Always use float32 for the sum, because bf16 only has 7 bits of
      # precision, which will cause accumulation becomes a no-op after a few
      # hundreds of steps.
      quant_stat[f'sum_of_{key}'] = jnp.zeros_like(value, dtype=jnp.float32)
    return quant_stat

  def update(
      self, quant_stat: QuantStat, calibration: Calibration
  ) -> QuantStat:
    """Updates the quantization statistics."""
    quant_stat = quant_stat.copy()
    quant_stat['count'] += 1
    for key, value in calibration.items():
      # Disallow automatic shape broadcasting which usually indicates a config
      # error.
      if quant_stat[f'sum_of_{key}'].shape != value.shape:
        raise ValueError(
            f'Shape mismatch between {quant_stat} and {calibration}'
        )
      quant_stat[f'sum_of_{key}'] += value
    return quant_stat

  def get_calibration(
      self,
      quant_stat: QuantStat,
      default_calibration: Calibration | None = None,
  ) -> Calibration:
    """Returns the average calibration.

    If a default calibration is provided, it will be used when the number of
    samples is less than the bootstrap steps.

    Args:
      quant_stat: The quantization statistics.
      default_calibration: The default calibration to use when the number of
        samples is less than the bootstrap steps.

    Returns:
      The average calibration.
    """
    calibration = {}
    for key, value in quant_stat.items():
      if key.startswith('sum_of_'):
        calibration[key.removeprefix('sum_of_')] = value / quant_stat['count']

    if default_calibration is not None:
      calibration = jax.tree.map(
          lambda x, y: x.astype(y.dtype), calibration, default_calibration
      )
      return jax.lax.cond(
          quant_stat['count'] > self._bootstrap_steps,
          lambda: calibration,
          lambda: default_calibration,
      )

    return calibration
