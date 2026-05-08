from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from qwix.contrib.kernels import quantized_matmul


class QuantizedMatmulTest(parameterized.TestCase):

  @staticmethod
  def jax_answer(x: jax.Array, sx: jax.Array, y: jax.Array, sy: jax.Array):
    m, k = x.shape
    _, n = y.shape
    sm, sk = sx.shape
    _, sn = sy.shape
    x = x.reshape(sm, m // sm, sk, k // sk)
    y = y.reshape(sk, k // sk, sn, n // sn)

    # int matmul
    xy = jnp.einsum("abcd,cdef->abcef", x, y, preferred_element_type=jnp.int32)

    # multiply scales with the matmul result.
    xys = jnp.einsum(
        "abcef,ac,ce->abef", xy, sx, sy, preferred_element_type=jnp.float32
    )
    return xys.reshape(m, n)

  @staticmethod
  def generate_quantized_arrays(
      data_shape, scale_shape, dtype, *, key: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    k1, k2 = jax.random.split(key)
    x = jax.random.randint(
        k1, data_shape, minval=-120, maxval=120, dtype=jnp.int8
    )
    sx = jax.random.normal(k2, scale_shape, dtype=dtype)
    return x, sx

  # TODO(chapmanjames): Add bfloat16 tests when implemented.
  @parameterized.parameters(
      (1024, 1024, 1024, 8, 8, 8, 256, 512, 1024, jnp.float32),
      (256, 512, 1024, 2, 4, 8, 128, 128, 128, jnp.float32),
      (256, 512, 1024, 1, 1, 1, 256, 512, 1024, jnp.float32),
  )
  def test_kernel_dot_general(self, m, k, n, sm, sk, sn, bm, bk, bn, dtype):
    key1, key2 = jax.random.split(jax.random.key(0))
    lhs = QuantizedMatmulTest.generate_quantized_arrays(
        (m, k), (sm, sk), dtype, key=key1
    )
    rhs = QuantizedMatmulTest.generate_quantized_arrays(
        (k, n), (sk, sn), dtype, key=key2
    )

    kernel_answer = quantized_matmul.quantized_matmul(
        lhs[0],
        lhs[1],
        rhs[0],
        rhs[1],
        bm=bm,
        bk=bk,
        bn=bn,
        dtype=dtype,
    )

    qwix_answer = QuantizedMatmulTest.jax_answer(lhs[0], lhs[1], rhs[0], rhs[1])
    np.testing.assert_allclose(kernel_answer, qwix_answer, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
  absltest.main()
