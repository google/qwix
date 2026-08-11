import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from qwix.contrib.kernels import fused_hadamard_quantized_matmul as had_kernel


def _absmax_quantize(
    x,
    scale_shape,
    dtype=jnp.int8,
):
  """Quantizes x using the absmax method."""
  logical_shape = x.shape
  tmp_shape = []
  for a, b in zip(x.shape, scale_shape):
    tmp_shape.append(b)
    tmp_shape.append(a // b)
    if a % b != 0:
      raise ValueError(
          f"Dimension {a} is not divisible by block size {b}. Cannot quantize."
      )

  # Reduce along the odd indices of tmp_shape
  xr = x.reshape(tmp_shape)
  max_val = jnp.iinfo(dtype).max + 0.5
  scale = (
      jnp.max(
          jnp.abs(xr),
          axis=tuple(range(1, len(tmp_shape), 2)),
          keepdims=True,
      )
      / max_val
  )

  x = xr / scale
  x = jnp.round(x)
  x = x.astype(dtype)
  x = x.reshape(logical_shape)
  return x, scale.reshape(scale_shape)


class FusedHadamardQuantizedMatmulTest(parameterized.TestCase):

  @parameterized.parameters(
      (128, 128, 128, 128, 1, 1, 1),
      (128, 128, 128, 128, 128, 1, 1),
      (4096, 1024, 512, 4096, 32, 32, 32),
  )
  def test_identity(self, size, bm, bk, bn, sm, sk, sn):
    had_power = 7
    dtype = jnp.bfloat16
    qtype = jnp.int8
    key = jax.random.key(0)

    x = jnp.identity(size, dtype=dtype)
    w = jnp.identity(size, dtype=dtype)
    w = had_kernel.hadamard_transform_rhs(w, had_power, key)
    wq, sw = _absmax_quantize(w, (sk, sn), dtype=qtype)
    sw = sw.astype(jnp.float32)

    fn = functools.partial(
        had_kernel.hadamard_quantize_multiply,
        bm=bm,
        bk=bk,
        bn=bn,
        sm_global=sm,
        hadamard_power=had_power,
        accum_dtype=jnp.float32,
    )

    kernel_ans = jax.jit(fn)(x, wq, sw, key)

    np.testing.assert_allclose(
        kernel_ans,
        jnp.identity(size, dtype=dtype),
        rtol=1e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
  absltest.main()
