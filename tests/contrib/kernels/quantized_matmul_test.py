from absl.testing import parameterized
import jax.numpy as jnp
from qwix._src.core import dot_general
from qwix._src.core import qarray
from qwix.contrib.kernels import quantized_matmul
from google3.testing.pybase import googletest


class QuantizedMatmulTest(parameterized.TestCase, googletest.TestCase):

  @parameterized.parameters(
      ((4, 8), (8, 16), 4, 16, 1, 1),
      ((8, 16), (16, 32), 8, 32, 2, 2),
      ((4, 4), (4, 4), 2, 2, 2, 2),
  )
  def test_kernel_dot_general(self, lhs_shape, rhs_shape, bm, bn, bk, tile_k):
    lhs = jnp.ones(lhs_shape, jnp.float32)
    rhs = jnp.ones(rhs_shape, jnp.float32)

    # Channelwise on axis 1 (contracting)
    lhs_how = qarray.HowToQuantize(
        qtype=jnp.int8,
        tiled_axes={1: tile_k},
    )
    # Channelwise on axis 0 (contracting)
    rhs_how = qarray.HowToQuantize(
        qtype=jnp.int8,
        tiled_axes={0: tile_k},
    )

    q_lhs = qarray.quantize(lhs, lhs_how)
    q_rhs = qarray.quantize(rhs, rhs_how)

    kernel_answer = quantized_matmul.q_matmul(
        q_lhs.qvalue,
        q_lhs.scale,
        q_rhs.qvalue,
        q_rhs.scale,
        bm=bm,
        bn=bn,
        bk=bk,
    )

    qwix_answer = dot_general.dot_general(
        q_lhs,
        q_rhs,
        (([1], [0]), ([], [])),
    )
    self.assertTrue(jnp.allclose(kernel_answer, qwix_answer))


if __name__ == "__main__":
  googletest.main()
