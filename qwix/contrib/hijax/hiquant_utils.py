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
"""Utility functions for quantization."""

import jax.numpy as jnp

_INTEGER_DTYPES = set([
    jnp.int8,
    jnp.int16,
    jnp.int32,
    jnp.int64,
    jnp.uint8,
    jnp.uint16,
    jnp.uint32,
    jnp.uint64,
    jnp.dtype("int8"),
    jnp.dtype("int16"),
    jnp.dtype("int32"),
    jnp.dtype("int64"),
    jnp.dtype("uint8"),
    jnp.dtype("uint16"),
    jnp.dtype("uint32"),
    jnp.dtype("uint64"),
])
_FLOAT_DTYPES = set([
    jnp.bfloat16,
    jnp.float16,
    jnp.float32,
    jnp.float64,
    jnp.dtype("bfloat16"),
    jnp.dtype("float16"),
    jnp.dtype("float32"),
    jnp.dtype("float64"),
])


def get_bitwidth(dtype: jnp.dtype) -> int:
  if dtype in _INTEGER_DTYPES:
    return jnp.iinfo(dtype).bits
  elif dtype in _FLOAT_DTYPES:
    return jnp.finfo(dtype).bits
  else:
    raise ValueError(f"Unsupported dtype: {dtype}")


def get_accumulation_dtype(dtype: jnp.dtype) -> jnp.dtype:
  if dtype in _INTEGER_DTYPES:
    return jnp.int32
  elif dtype in _FLOAT_DTYPES:
    return jnp.float32
  else:
    raise ValueError(f"Unsupported dtype: {dtype}")


def is_integer_dtype(dtype: jnp.dtype) -> bool:
  return dtype in _INTEGER_DTYPES


def is_float_dtype(dtype: jnp.dtype) -> bool:
  return dtype in _FLOAT_DTYPES
