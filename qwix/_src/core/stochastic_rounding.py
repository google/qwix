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
"""Stochastic rounding utilities."""

from typing import Sequence
import jax


def uniform_noise(
    shape: tuple[int, ...],
    *,
    key: jax.Array,
    channelwise_noise_axes: Sequence[int] = (0,),
) -> jax.Array:
  """Uniform noise."""

  # Keep shape dimensions only for channelwise_noise_axes.
  noise_shape = tuple(
      dim if axis in channelwise_noise_axes else 1
      for axis, dim in enumerate(shape)
  )
  return jax.random.uniform(key, noise_shape) - 0.5
