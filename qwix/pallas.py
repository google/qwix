# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwix APIs for quantizing Pallas kernels.

This module is usually imported as `import qwix.pallas as qpl` in user code.

To add quantization support to a Pallas kernel, generally users need to

1) Quantize the inputs outside of the kernel via `qpl.quantize`.
2) Instead of calling `pl.pallas_call`, call `qpl.pallas_call`, which supports
   taking QArrays as inputs. `qpl.pallas_call` takes the same block specs as
   `pl.pallas_call` and adjusts them automatically for QArrays.
3) When inside the kernel, use `qpl.einsum` / `qpl.dot_general` / `qpl.dot`
   which take QArrays as inputs and return jax.Array. Alternatively, call
   `dequantize` on the QArray to obtain the dequantized jax.Array.

Note: dot_general / einsum / dot in this module should only be called inside
pallas kernels, as they can be inefficient when called outside of the kernel.
"""

import functools

from qwix._src import qconfig
from qwix._src.core import dot as qwix_dot
from qwix._src.core import dot_general as qwix_dot_general
from qwix._src.core import einsum as qwix_einsum
from qwix._src.core import pallas
from qwix._src.core import qarray

__all__ = [
    'QArray',
    'quantize',
    'dequantize',
    'get_current_rule',
    'pallas_call',
    'dot_general',
    'dot',
    'einsum',
]

# APIs for generic QArray operations. These are deprecated in favor of the
# qwix.* APIs.
QArray = qarray.QArray
quantize = qarray.quantize_api
dequantize = qarray.dequantize

# APIs for quantizing Pallas kernels.
get_current_rule = qconfig.get_current_rule
pallas_call = pallas.pallas_call

# Use loop-based implementations for Pallas kernels.
dot_general = qwix_dot_general.loop_dot_general
dot = functools.partial(qwix_dot.dot, _qwix_dot_general=dot_general)
einsum = functools.partial(qwix_einsum.einsum, _qwix_dot_general=dot_general)
