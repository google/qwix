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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import qarray
from qwix._src.core import qarray_qt


class QArrayQtTest(parameterized.TestCase):

  def test_qarray_with_gradient(self):
    x = jnp.ones((3, 3), jnp.float32)

    def fake_quant_sum(x):
      how = qarray.HowToQuantize(qtype=jnp.int8)
      x = qarray_qt.quantize_with_calibration(
          x, how.qtype, qarray.calibrate(x, how)
      )
      x = qarray_qt.dequantize(x)
      return jnp.sum(x)

    self.assertTrue((jax.grad(fake_quant_sum)(x) == x).all())


if __name__ == '__main__':
  absltest.main()
