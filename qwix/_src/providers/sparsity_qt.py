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
"""Core sparsity quantized training support."""

import flax.linen as nn
import jax
import jax.numpy as jnp
from qwix._src import flax_util
from qwix._src.core import sparsity


class SparsityModule(nn.Module):
  """Sparsity module for Flax."""

  sparsity_rule: sparsity.SparsityRule | None = None

  def _maybe_update_mask(
      self,
      weight: jax.Array,
      step: jax.Array,
  ) -> jax.Array:
    """Updates the sparsity mask based on the current step and config."""

    mask_val = flax_util.get_or_create_variable(
        'compression', 'mask', lambda: jnp.ones(weight.shape, jnp.bool_)
    )
    # NOTE: Reshape if mask and wesight have shape mismatch.
    if mask_val.shape != weight.shape:
      mask_val = jnp.reshape(mask_val, weight.shape)

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
      return jnp.logical_and(in_update_window, is_update_step)

    new_mask_val = jax.lax.cond(
        should_update_mask(step),
        mask_update,
        no_mask_update,
        weight,
        mask_val,
    )
    return new_mask_val

  @nn.compact
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

      step = flax_util.get_or_create_variable(
          'compression', 'step', lambda: jnp.zeros([], jnp.int32)
      )

      mask = flax_util.get_or_create_variable(
          'compression', 'mask', lambda: jnp.ones(weight.shape, jnp.bool_)
      )

      if not self.is_initializing() and self.has_variable(
          'compression', 'mask'
      ):
        # Do not update mask for eval.
        if not self.sparsity_rule.eval_mode:
          new_mask = self._maybe_update_mask(weight=weight, step=step.value)
          mask.value = new_mask
          step.value = step.value + 1

        # Unless updated mask is all ones, so we apply mask irrespective of
        # start_step

        weight = jnp.where(
            mask.value, weight, jnp.zeros(weight.shape, weight.dtype)
        )

    return inputs, weight
