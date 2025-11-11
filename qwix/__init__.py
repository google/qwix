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

"""Qwix: A Jax quantization library."""

# pylint: disable=g-multiple-import, g-importing-member

from qwix._src.core.conv_general import conv_general_dilated
from qwix._src.core.dot import dot
from qwix._src.core.dot_general import dot_general
from qwix._src.core.einsum import einsum
from qwix._src.core.qarray import QArray, dequantize, quantize_api as quantize
from qwix._src.core.ragged_dot import ragged_dot, ragged_dot_general
from qwix._src.flax_util import QuantStat
from qwix._src.model import quantize_model
from qwix._src.providers.lora import LoraProvider, LoraRule, apply_lora_to_model
from qwix._src.providers.odml import OdmlConversionProvider, OdmlQatProvider
from qwix._src.providers.ptq import PtqProvider, quantize_params
from qwix._src.providers.qt import QtProvider, QtRule
from qwix._src.qconfig import QuantizationProvider, QuantizationRule
