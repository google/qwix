import jax.numpy as jnp
from qwix._src.core import dot_general
from qwix._src.core import qarray
from qwix.contrib.kernels import quantized_matmul

from google3.testing.pybase import googletest


class QuantizedMatmulTest(googletest.TestCase):

  def test_kernel_dot_general(self):
    lhs = jnp.ones((4, 8), jnp.float32)
    rhs = jnp.ones((8, 16), jnp.float32)

    # Channelwise on axis 1 (contracting)
    lhs_how = qarray.HowToQuantize(
        qtype=jnp.int8,
        tiled_axes={1: 1},
    )
    # Channelwise on axis 0 (contracting)
    rhs_how = qarray.HowToQuantize(
        qtype=jnp.int8,
        tiled_axes={0: 1},
    )

    q_lhs = qarray.quantize(lhs, lhs_how)
    q_rhs = qarray.quantize(rhs, rhs_how)

    kernel_answer = quantized_matmul.q_matmul(
        q_lhs.qvalue, q_lhs.scale, q_rhs.qvalue, q_rhs.scale, bm=4, bn=16, bk=1
    )

    qwix_answer = dot_general.dot_general(
        q_lhs,
        q_rhs,
        (([1], [0]), ([], [])),
    )
    self.assertTrue(jnp.allclose(kernel_answer, qwix_answer))


if __name__ == "__main__":
  googletest.main()
