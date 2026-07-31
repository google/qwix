from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix._src.core import qarray
from qwix.contrib.hijax import convert
from qwix.contrib.hijax import hiqarray


class HiqarrayTest(parameterized.TestCase):

  def test_as_hiqarray(self):
    x = jnp.ones((16, 32, 64))
    scale = jnp.ones((1, 1, 64))
    zero_point = jnp.zeros((1, 1, 64))

    xq = convert.as_hiqarray(x, scale, zero_point)

    self.assertIsInstance(xq, hiqarray.HiQArray)

  def test_to_hiqarray(self):
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    x = jax.random.normal(key1, (16, 32))
    scale = jax.random.normal(key2, (16, 1))

    def quantize_fn(data, scale, zero_point, qtype):
      return qarray.quantize_with_scale_zero_point(
          data, qtype, scale, zero_point
      ).qvalue

    xq = convert.to_hiqarray(
        x, scale, None, quantize_fn=quantize_fn, qtype=jnp.int8
    )

    self.assertIsInstance(xq, hiqarray.HiQArray)

  def test_from_hiqarray(self):
    x = jnp.ones((16, 32))
    scale = jnp.ones((16, 1))

    xq = convert.as_hiqarray(x, scale, None)

    def dequantize_fn(data, scale, zero_point):
      del zero_point  # unused
      return qarray.call_with_generic_broadcast(jnp.multiply, data, scale)

    y = convert.from_hiqarray(xq, dequantize_fn=dequantize_fn)

    self.assertIsInstance(y, jax.Array)
    self.assertEqual(y.shape, x.shape)
    self.assertEqual(y.dtype, x.dtype)


if __name__ == "__main__":
  absltest.main()
