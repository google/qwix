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
"""Einsum info and broadcasting logic."""

import dataclasses
import re
from typing import Any, Sequence

import jax
from jax import numpy as jnp
import opt_einsum
from qwix._src.core import qarray


@dataclasses.dataclass(slots=True)
class EinsumInfo:
  """Info needed to perform a binary einsum using dot_general.

  This class parses binary einsum strings and provides information needed to
  execute the operation using `jax.lax.dot_general`.

  Example:
    For `abc,bcd->acd`:
    - lhs: `abc` (a: free, b: contract, c: batch)
    - rhs: `bcd` (b: contract, c: batch, d: free)
    - out: `acd`
  """

  lhs: str
  rhs: str
  out: str

  @classmethod
  def parse(
      cls, einsum_str: str, ndims: tuple[int, int] | None = None
  ) -> 'EinsumInfo':
    """Parses a binary einsum string into an EinsumInfo object.

    Args:
      einsum_str: The einsum string, e.g., 'abc,bcd->acd'.
      ndims: The number of dimensions of the lhs and rhs array. If provided, it
        parses the einsum string using `opt_einsum` first, which allows implicit
        subscripts and ellipsis.

    Returns:
      An EinsumInfo object containing the parsed lhs, rhs, and out strings.

    Raises:
      NotImplementedError: If the einsum string does not match the expected
        binary format (lhs,rhs->out) or contains repeated indices within a
        single term.

    Example:
      >>> EinsumInfo.parse('abc,bcd->acd')
      EinsumInfo(lhs='abc', rhs='bcd', out='acd')
    """
    if ndims is not None:
      input_subs, out, _ = opt_einsum.parser.parse_einsum_input(
          (einsum_str, jnp.zeros((1,) * ndims[0]), jnp.zeros((1,) * ndims[1]))
      )
      einsum_str = f'{input_subs}->{out}'

    einsum_str = einsum_str.replace(' ', '')
    cls._validate_binary_einsum(einsum_str)

    input_subs, out = einsum_str.split('->')
    lhs, rhs = input_subs.split(',')
    return cls(lhs, rhs, out)

  @staticmethod
  def _validate_binary_einsum(einsum_str: str) -> None:
    """Validates that the einsum string is a supported binary operation."""
    if not re.match(r'^[a-zA-Z]*,[a-zA-Z]*->[a-zA-Z]*$', einsum_str):
      raise NotImplementedError(
          f'Unsupported einsum string: "{einsum_str}". '
          'EinsumInfo only supports binary einsums with explicit output '
          'subscripts (e.g., "abc,bcd->acd").'
      )

    input_subs, out = einsum_str.split('->')
    lhs, rhs = input_subs.split(',')

    for name, subs in zip(['lhs', 'rhs', 'out'], [lhs, rhs, out]):
      if len(set(subs)) != len(subs):
        raise NotImplementedError(
            f'Repeated indices in {name} ("{subs}") are not supported. '
            f'Einsum string: "{einsum_str}".'
        )

  @property
  def batch_chars(self) -> list[str]:
    """Returns the list of batch characters (present in lhs, rhs, and out).

    Example:
      For `abc,bcd->acd`, batch characters are `['c']`.
    """
    return sorted(set(self.lhs) & set(self.rhs) & set(self.out))

  @property
  def contract_chars(self) -> list[str]:
    """Returns the list of contraction characters (in lhs, rhs but not out).

    Example:
      For `abc,bcd->acd`, contraction characters are `['b']`.
    """
    return sorted(set(self.lhs) & set(self.rhs) - set(self.out))

  @property
  def dimension_numbers(
      self,
  ) -> jax.lax.DotDimensionNumbers:
    """Returns dimension_numbers for dot_general compatibility.

    Returns:
      A tuple of ((lhs_contract_dims, rhs_contract_dims),
                  (lhs_batch_dims, rhs_batch_dims)).

    Example:
      For `abc,bcd->acd`:
      - LHS: `abc` -> `b` (dim 1) is contract, `c` (dim 2) is batch.
      - RHS: `bcd` -> `b` (dim 0) is contract, `c` (dim 1) is batch.
      - Result: `(((1,), (0,)), ((2,), (1,)))`
    """
    lhs_map = {c: i for i, c in enumerate(self.lhs)}
    rhs_map = {c: i for i, c in enumerate(self.rhs)}
    lhs_contract = tuple(lhs_map[c] for c in self.contract_chars)
    rhs_contract = tuple(rhs_map[c] for c in self.contract_chars)
    lhs_batch = tuple(lhs_map[c] for c in self.batch_chars)
    rhs_batch = tuple(rhs_map[c] for c in self.batch_chars)
    return ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))

  @property
  def output_perm(self) -> tuple[int, ...] | None:
    """Returns the output permutation if needed, or None.

    `jax.lax.dot_general` produces output dimensions in the order:
    [batch_dims, lhs_free_dims, rhs_free_dims]. If the desired output
    (specified in the einsum string) differs from this order, a permutation
    is required.

    Returns:
      A tuple representing the permutation, or None if no permutation is
      needed.

    Example:
      For `abc,bcd->acd`:
      - `dot_general` output order: `c` (batch), `a` (lhs free), `d` (rhs free)
      -> `cad`.
      - Desired output: `acd`.
      - Permutation required to go from `cad` to `acd`: `(1, 0, 2)`.
    """
    lhs_remaining = [
        c for c in self.lhs if c in self.out and c not in self.batch_chars
    ]
    rhs_remaining = [
        c for c in self.rhs if c in self.out and c not in self.batch_chars
    ]
    current_out_chars = self.batch_chars + lhs_remaining + rhs_remaining
    current_pos_map = {c: i for i, c in enumerate(current_out_chars)}
    perm = tuple(current_pos_map[c] for c in self.out)
    if perm == tuple(range(len(perm))):
      return None
    return perm


def sanitize_shape(shape: Sequence[int | Any]) -> tuple[int, ...]:
  """Replaces non-concrete integer dimensions with 1.

  This is valid because `opt_einsum` uses dimension sizes primarily for cost
  estimation (FLOPs/memory) to find the optimal path. The correctness of the
  contraction sequence depends on the einsum string indices, not the dimension
  sizes. Using 1 as a placeholder allows `opt_einsum` to proceed with a
  potentially sub-optimal but valid path. Note that `jnp.einsum` performs
  similar sanitization by calling `opt_einsum.contract_path` with placeholder
  values (e.g. 8) for non-constant dimensions.

  Args:
    shape: The shape to sanitize.

  Returns:
    The sanitized shape.
  """
  return tuple(s if isinstance(s, int) else 1 for s in shape)


def broadcast_operands(
    operands: Sequence[qarray.MaybeQArray], operand_subs_list: Sequence[str]
) -> list[qarray.MaybeQArray]:
  """Broadcasts operands to matching shapes for shared dimensions.

  This function ensures that all operands have compatible shapes for the
  implying einsum operation. It handles tiling (repeating dimensions) if needed
  for QArray compatibility.

  Args:
    operands: The list of operands (arrays or QArrays).
    operand_subs_list: The list of subscript strings for each operand.

  Returns:
    A list of broadcasted operands.
  """
  char_to_size = {}
  for operand, subs in zip(operands, operand_subs_list):
    for i, char in enumerate(subs):
      size = operand.shape[i]
      if char not in char_to_size:
        # First time we see this character, just record the size.
        char_to_size[char] = size
      else:
        current_max = char_to_size[char]
        if (
            isinstance(size, int)
            and isinstance(current_max, int)
            and size > current_max
            and size % current_max == 0
        ):
          # If both are concrete integers, we can broadcast to the new size,
          # provided that the new size is a multiple of the current size.
          char_to_size[char] = size
        elif isinstance(current_max, int) and current_max == 1:
          # If current max is 1, we can always broadcast to the new size,
          # even if it is symbolic.
          char_to_size[char] = size

  broadcasted_operands = []
  for operand, subs in zip(operands, operand_subs_list):
    target_shape = tuple(char_to_size[c] for c in subs)
    operand = qarray.broadcast_to(operand, target_shape)
    broadcasted_operands.append(operand)
  return broadcasted_operands
