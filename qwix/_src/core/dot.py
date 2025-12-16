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
"""Quantized jax.numpy.dot with subchannel support."""
# pylint: disable=line-too-long

from typing import Callable

import jax
from qwix._src.core import dot_general
from qwix._src.core import qarray


def dot(
    a: qarray.MaybeQArray,
    b: qarray.MaybeQArray,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    out_sharding=None,
    *,
    _qwix_dot_general: Callable[..., jax.Array] = dot_general.dot_general,
):
  """Computes the dot product with support for ``QArray`` inputs.

  This function serves as a drop-in replacement for
  `jax.numpy.dot <https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dot.html>`_.

  It automatically dispatches to a quantized implementation if inputs are
  compatible ``QArray``s. It also supports subchannel quantization where
  applicable.

  Args:
    a: The left-hand side, either a jax.Array or QArray.
    b: The right-hand side, either a jax.Array or QArray.
    precision: The numerical precision configuration for the computation.
    preferred_element_type: The element type for the accumulation. Unlike
      standard ``jax.numpy.dot``, this function exposes this argument to allow
      control over quantized accumulation precision.
    out_sharding: Optional sharding spec for the output array.
    _qwix_dot_general: Internal argument for dependency injection of the
      underlying ``dot_general`` implementation. Defaults to
      ``qwix.dot_general``.

  Returns:
    The dot product of ``a`` and ``b``.
  """
  if a.ndim == 0 or b.ndim == 0:
    contract_dims = ((), ())
  elif b.ndim == 1:
    contract_dims = ((a.ndim - 1,), (0,))
  else:
    contract_dims = ((a.ndim - 1,), (b.ndim - 2,))
  return _qwix_dot_general(
      a,
      b,
      dimension_numbers=(contract_dims, ((), ())),
      precision=precision,
      preferred_element_type=preferred_element_type,
      out_sharding=out_sharding,
  )
