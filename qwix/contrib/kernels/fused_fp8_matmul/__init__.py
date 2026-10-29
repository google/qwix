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

"""Fused FP8 matmul: `lhs @ rhs` with in-kernel row quantization of `lhs`."""

# pylint: disable=g-importing-member

# TODO(stellasyan): Export the kernels (`forward_kernel.fused_fp8_matmul_pallas`
# and the backward ones) here once there is a caller that wants explicit tiles;
# until then they are importable from their modules.
from qwix.contrib.kernels.fused_fp8_matmul.forward_kernel import Config
from qwix.contrib.kernels.fused_fp8_matmul.forward_kernel import fp8_matmul_supported_on
from qwix.contrib.kernels.fused_fp8_matmul.forward_kernel import get_heuristics_config
from qwix.contrib.kernels.fused_fp8_matmul.op import fused_fp8_matmul
from qwix.contrib.kernels.fused_fp8_matmul.reference import fused_fp8_matmul_reference
