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
"""Core sparsity quantized training support."""

from flax import nnx
import jax
import jax.numpy as jnp
from qwix._src.core import sparsity


class SparsityModule(nnx.Module):
  """A stateful module for managing and applying structured sparsity in Flax NNX.

  This module tracks the training step and maintains a persistent sparsity mask
  as `nnx.BatchStat` variables (effectively part of the model's batch stats,
  not trainable parameters). It can be used to apply structured N:M sparsity
  to activations and/or weights.

  For weight sparsity, it periodically updates a cached boolean mask based on
  the `SparsityRule` and applied it to the weights. For activation sparsity,
  it computes and applies the mask dynamically on each call if enabled.

  Attributes:
    step: An `nnx.BatchStat` tracking the number of update steps.
    mask: An `nnx.BatchStat` holding the persistent boolean mask for weights.
    sparsity_rule: The `SparsityRule` configuration.
  """

  step: nnx.BatchStat
  mask: nnx.BatchStat

  def __init__(
      self,
      shape: tuple[int, ...],
      sharding_axes: tuple[jax.sharding.PartitionSpec | None, ...],
      sparsity_rule: sparsity.SparsityRule | None = None,
  ):
    self.sparsity_rule = sparsity_rule
    self.step = nnx.BatchStat(jnp.zeros([], jnp.int32))
    self.mask = nnx.BatchStat(
        jnp.ones(shape, jnp.bool_), sharding=sharding_axes
    )

  def _maybe_update_mask(
      self,
      weight: jax.Array,
      step: jax.Array,
  ) -> jax.Array:
    """Updates the sparsity mask based on the current step and config."""
    mask_val = self.mask.value
    if mask_val.shape != weight.shape:
      mask_val = mask_val[tuple(slice(0, s) for s in weight.shape)]

    def mask_update(w: jax.Array, mask_val: jax.Array) -> jax.Array:  # pylint: disable=unused-argument
      if self.sparsity_rule is None:
        return mask_val
      return sparsity.get_sparsity_mask(
          w,
          n_sparsity=self.sparsity_rule.weight_sparsity_n,
          m_sparsity=self.sparsity_rule.weight_sparsity_m,
          order=self.sparsity_rule.weight_sparsity_order,
          block_size=self.sparsity_rule.weight_sparsity_block_size,
          offset=self.sparsity_rule.weight_sparsity_offset,
      )

    def no_mask_update(w, mask_val):  # pylint: disable=unused-argument
      return mask_val

    def should_update_mask(step: jax.Array):
      if self.sparsity_rule is None:
        return False
      in_update_window = jnp.greater_equal(
          step, self.sparsity_rule.weight_sparsity_start_step
      )
      is_update_step = jnp.equal(
          (step - self.sparsity_rule.weight_sparsity_start_step)
          % self.sparsity_rule.weight_sparsity_update_step,
          0,
      )
      should_update = jnp.logical_and(in_update_window, is_update_step)
      return should_update

    new_mask_val = jax.lax.cond(
        should_update_mask(step),
        mask_update,
        no_mask_update,
        weight,
        mask_val,
    )
    return new_mask_val

  def __call__(
      self, inputs: jax.Array, weight: jax.Array
  ) -> tuple[jax.Array, jax.Array]:

    if self.sparsity_rule is None:
      return inputs, weight

    if self.sparsity_rule.activation_sparsity_m != 0:
      input_mask = sparsity.get_sparsity_mask(
          inputs,
          n_sparsity=self.sparsity_rule.activation_sparsity_n,
          m_sparsity=self.sparsity_rule.activation_sparsity_m,
          order=self.sparsity_rule.activation_sparsity_order,
          block_size=self.sparsity_rule.activation_sparsity_block_size,
          offset=self.sparsity_rule.activation_sparsity_offset,
      )
      inputs = jnp.where(
          input_mask, inputs, jnp.zeros(inputs.shape, inputs.dtype)
      )
    if self.sparsity_rule.weight_sparsity_m != 0:
      if self.mask is None:
        self.mask = nnx.BatchStat(jnp.ones(weight.shape, jnp.bool_))

      # Only update if not in eval mode
      if not self.sparsity_rule.eval_mode:
        new_mask = self._maybe_update_mask(weight=weight, step=self.step.value)
        self.mask.value = new_mask
        self.step.value = self.step.value + 1

      weight = jnp.where(
          self.mask.value, weight, jnp.zeros(weight.shape, weight.dtype)
      )

    return inputs, weight
