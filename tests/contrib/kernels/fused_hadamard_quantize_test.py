from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from qwix.contrib import hadamard_rot
from qwix.contrib.kernels import fused_hadamard_quantize as fhq


class LHSFusedHadamardQuantizeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != "tpu":
      self.skipTest(
          "Fused Hadamard quantize is a TPU-only Pallas kernel; it requires "
          "TensorCore mesh setup (pltpu.create_tensorcore_mesh needs "
          "device.num_cores), which is unavailable on CPU/GPU."
      )

  @parameterized.parameters(
      (1024, 1024, 8, 8, 1024, 1024),
      (1024, 1024, 8, 8, 512, 512),
  )
  def test_kernel_dot_general(self, m, n, sm, sn, bm, bn):
    hadamard_power = 7
    key = jax.random.PRNGKey(0)
    lhs = jax.random.normal(key, (m, n), dtype=jnp.float32)

    kernel_answer_xq, kernel_answer_sx = fhq.fused_hadamard_quantize(
        lhs,
        bm=bm,
        bn=bn,
        sm=sm,
        sn=sn,
        hadamard_power=hadamard_power,
    )

    had_mat = hadamard_rot._create_hadamard_matrix(  # pylint: disable=protected-access
        hadamard_power,
        None,
        row_sign_flip=False,
        col_sign_flip=False,
        dtype=jnp.int8,
    )[
        0
    ]
    # Apply Hadamard transform to the LHS
    lhs_jax = jnp.matmul(lhs.reshape(sm, m // sm, sn, n // sn), had_mat)

    # Quantize the LHS
    vqfn = jax.vmap(fhq.quantize_a_tile, in_axes=0, out_axes=(0, 0))
    vvqfn = jax.vmap(vqfn, in_axes=2, out_axes=(2, 2))
    xq, sx = vvqfn(lhs_jax)
    xq = xq.reshape(m, n)
    sx = sx.reshape(sm, sn)

    # Check numerics
    np.testing.assert_allclose(kernel_answer_xq, xq, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(kernel_answer_sx, sx, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  absltest.main()
