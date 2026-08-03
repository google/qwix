import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix._src.core import qarray
from qwix.contrib.hijax import convert
from qwix.contrib.hijax import hiqarray


class HiqarrayTest(parameterized.TestCase):

  def test_as_hiqarray(self):
    x = jnp.ones((16, 32))
    scale = jnp.ones((16, 1))

    xq = convert.as_hiqarray(x, scale, None)

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

  def test_to_hiqarray_bwd(self):
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    x = jax.random.normal(key1, (16, 32))
    scale = jax.random.normal(key2, (16, 1))

    def quantize_fn(data, scale, zero_point, qtype):
      return qarray.quantize_with_scale_zero_point(
          data, qtype, scale, zero_point
      ).qvalue

    fn = functools.partial(
        convert.to_hiqarray, quantize_fn=quantize_fn, qtype=jnp.int8
    )

    _, vjp_fn = jax.vjp(fn, x, scale, None)
    g = jnp.ones_like(x)
    bwd = vjp_fn(g)

    self.assertTrue(jnp.allclose(bwd[0], g))
    self.assertLess(jnp.max(jnp.abs(bwd[1])), 1e-6)
    self.assertIsNone(bwd[2])

  def test_from_hiqarray_bwd(self):
    x = jnp.ones((16, 32))
    scale = jnp.ones((16, 1))

    xq = convert.as_hiqarray(x, scale, None)

    def dequantize_fn(data, scale, zero_point):
      del zero_point  # unused
      return qarray.call_with_generic_broadcast(jnp.multiply, data, scale)

    fn = functools.partial(convert.from_hiqarray, dequantize_fn=dequantize_fn)

    _, vjp_fn = jax.vjp(fn, xq)
    g = jnp.ones_like(xq.qvalue)
    bwd = vjp_fn(g)

    self.assertTrue(jnp.allclose(bwd[0], g))

  def test_permute_dims(self):
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    x = jax.random.normal(key1, (16, 32))
    scale = jax.random.normal(key2, (16, 1))

    xq = convert.as_hiqarray(x, scale, None)

    yq = convert.permute_dims(xq, (1, 0))
    zq = convert.transpose(xq)

    self.assertTrue(jnp.allclose(jnp.transpose(x), yq.qvalue))
    self.assertTrue(jnp.allclose(jnp.transpose(scale), yq.scale))
    self.assertTrue(jnp.allclose(yq.qvalue, zq.qvalue))
    self.assertTrue(jnp.allclose(yq.scale, zq.scale))

  def test_permute_dims_bwd(self):
    key = jax.random.key(0)
    key1, key2, key3 = jax.random.split(key, 3)
    x = jax.random.normal(key1, (16, 32))
    scale = jax.random.normal(key2, (16, 1))

    xq = convert.as_hiqarray(x, scale, None)

    _, vjp_fn = jax.vjp(convert.transpose, xq)
    g = jax.random.normal(key3, (32, 16))
    bwd = vjp_fn(g)

    self.assertTrue(jnp.allclose(bwd[0], jnp.transpose(g)))


if __name__ == "__main__":
  absltest.main()
