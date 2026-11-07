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

"""Hadamard rotation and VMEM-fused row-wise quantization kernels."""

# pylint: disable=g-importing-member

from qwix.contrib.kernels.hadamard.kernel import Config
from qwix.contrib.kernels.hadamard.kernel import get_heuristics_config
from qwix.contrib.kernels.hadamard.kernel import hadamard_pallas_supported
from qwix.contrib.kernels.hadamard.op import hadamard_dot_general_qt
from qwix.contrib.kernels.hadamard.op import hadamard_rotate
from qwix.contrib.kernels.hadamard.op import hadamard_rotate_quantize
from qwix.contrib.kernels.hadamard.reference import hadamard_rotate_quantize_reference
from qwix.contrib.kernels.hadamard.reference import hadamard_rotate_reference
from qwix.contrib.kernels.hadamard.reference import random_signs
