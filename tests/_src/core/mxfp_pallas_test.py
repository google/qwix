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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from qwix._src.core import mxfp_pallas
from qwix._src.core import qarray


def _quantize(key, m, k, n):
  """Returns FP8 activations and MXFP4 weights with an exact f32 reference."""
  n_shape = (n,) if isinstance(n, int) else n
  x = jax.random.normal(key, (m, k), jnp.bfloat16)
  w = jax.random.normal(jax.random.fold_in(key, 1), (k, *n_shape), jnp.bfloat16)
  x_q = qarray.quantize(
      x, qarray.HowToQuantize(qtype=jnp.float8_e4m3fn, channelwise_axes=[0])
  )
  w_q = qarray.quantize(
      w,
      qarray.HowToQuantize(
          qtype='mxfp4',
          channelwise_axes=list(range(1, 1 + len(n_shape))),
          tiled_axes={0: 32},
      ),
  )
  return x_q, w_q, _reference(x_q, w_q)


def _reference(x_q, w_q):
  """Exact f32 reference; dequantize() and a default TPU dot round to bf16."""

  def as_f32(q):
    return qarray.dequantize(q.replace(scale=q.scale.astype(jnp.float32)))

  return jnp.tensordot(
      as_f32(x_q), as_f32(w_q), axes=1, precision=jax.lax.Precision.HIGHEST
  )


class MxfpPallasTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.interpret = jax.devices()[0].platform != 'tpu'

  def assert_close(self, out, ref):
    if self.interpret and out.dtype == jnp.float32:
      np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-3)
    else:
      out = out.astype(jnp.float32)
      np.testing.assert_allclose(out, ref, atol=2e-2 * np.max(np.abs(ref)))

  def _matmul(self, x_q, w_q, out_dtype=jnp.float32):
    return mxfp_pallas.mxfp4_matmul(
        x_q.qvalue,
        x_q.scale,
        w_q.qvalue,
        w_q.scale,
        out_dtype=out_dtype,
        interpret=self.interpret,
    )

  @parameterized.named_parameters(
      dict(testcase_name='decode', m=16, k=256, n=128),
      dict(testcase_name='padded_m', m=1100, k=256, n=128),
      dict(testcase_name='tiled_n', m=8, k=256, n=2048),
      dict(testcase_name='3d_weight', m=16, k=256, n=(16, 512)),
      dict(
          testcase_name='bf16_out', m=32, k=256, n=128, out_dtype=jnp.bfloat16
      ),
  )
  def test_matches_dequantized_matmul(self, m, k, n, out_dtype=jnp.float32):
    x_q, w_q, ref = _quantize(jax.random.key(0), m, k, n)
    out = jax.jit(lambda x, w: self._matmul(x, w, out_dtype))(x_q, w_q)
    self.assertEqual(out.shape, (m, *w_q.shape[1:]))
    self.assertEqual(out.dtype, out_dtype)
    self.assert_close(out, ref)

  def test_multi_step_cache(self):
    # Both shapes pick (tm=16, tk=2048, tn=128) with num_steps=2 and 4.
    for k in (4096, 8192):
      x_q, w_q, ref = _quantize(jax.random.key(k), 16, k, 128)
      out = jax.jit(self._matmul)(x_q, w_q)
      self.assert_close(out, ref)

  def test_zero_channel(self):
    x_q, w_q, _ = _quantize(jax.random.key(2), 16, 256, 128)
    w_q = w_q.replace(scale=w_q.scale.at[:, 0].set(0))
    ref = _reference(x_q, w_q)
    out = jax.jit(self._matmul)(x_q, w_q)
    self.assert_close(out, ref)


if __name__ == '__main__':
  absltest.main()
