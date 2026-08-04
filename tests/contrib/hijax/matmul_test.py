from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix._src.core import qarray
from qwix.contrib.hijax import convert
from qwix.contrib.hijax import matmul


class MatmulTest(parameterized.TestCase):

  @staticmethod
  def _matmul_fn(x, sx, xzp, y, sy, yzp, **kwargs):
    del xzp, yzp  # unused
    # quantized matmul with non-trivial reduction axis.
    m, k = x.shape
    _, n = y.shape
    sm, sk = sx.shape
    _, sn = sy.shape
    assert y.shape[0] == k
    assert sy.shape[0] == sk
    xr = x.reshape(sm, m // sm, sk, k // sk)
    yr = y.reshape(sk, k // sk, sn, n // sn)

    # multiply qvalues
    xy = jnp.einsum("abcd,cdef->abcef", xr, yr, **kwargs)
    assert xy.shape == (sm, m // sm, sk, sn, n // sn)

    # multiply scales
    s = sx[:, :, None] * sy[None, :, :]
    assert s.shape == (sm, sk, sn)

    # Compute final output
    out = jnp.sum(xy * s[:, None, :, :, None], axis=2)
    return out.reshape(m, n)

  @staticmethod
  def _quantize_fn(data, scale, zero_point, qtype):
    return qarray.quantize_with_scale_zero_point(
        data, qtype, scale, zero_point
    ).qvalue

  @staticmethod
  def _dequantize_fn(data, scale, zero_point):
    del zero_point  # unused
    return qarray.call_with_generic_broadcast(jnp.multiply, data, scale)

  @staticmethod
  def _scale_zp_fn(data):
    return jnp.ones_like(data), None

  def _create_bwd_config(self, mode_str, **kwargs):
    if mode_str == "dequantize":
      config = matmul.OpConfig(
          op=jnp.matmul,
          kwargs=dict(preferred_element_type=jnp.float32),
      )
      mode = matmul.BwdDequantizeMatmulConfig(
          dequantize_fn=self._dequantize_fn,
          dequantize_kwargs=dict(),
      )
    elif mode_str == "static_quantize":
      config = matmul.OpConfig(
          op=self._matmul_fn,
          kwargs=dict(preferred_element_type=jnp.int32),
      )
      mode = matmul.BwdStaticQuantizeMatmulConfig(
          quantize_fn=self._quantize_fn,
          quantize_kwargs=dict(qtype=jnp.int8),
          grad_scale=kwargs["grad_scale"],
          grad_zero_point=kwargs.get("grad_zero_point", None),
      )
    elif mode_str == "dynamic_quantize":
      config = matmul.OpConfig(
          op=self._matmul_fn,
          kwargs=dict(preferred_element_type=jnp.int32),
      )
      mode = matmul.BwdDynamicQuantizeMatmulConfig(
          quantize_fn=self._quantize_fn,
          quantize_kwargs=dict(qtype=jnp.int8),
          scale_zp_fn=self._scale_zp_fn,
      )
    else:
      raise NotImplementedError(f"Unsupported dlhs mode: {mode_str.value}")
    return mode, config

  def test_matmul_fwd(self):
    # Create lossless quantized arrays
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    xy_kwargs = dict(minval=-100, maxval=100, dtype=jnp.int8)
    x = jax.random.randint(key1, (16, 32), **xy_kwargs)
    y = jax.random.randint(key2, (32, 64), **xy_kwargs)

    sx = jnp.ones((16, 1), dtype=jnp.float32)
    sy = jnp.ones((1, 64), dtype=jnp.float32)

    # Convert to HiQArray
    xq = convert.as_hiqarray(x, sx, None)
    yq = convert.as_hiqarray(y, sy, None)

    # Perform quantized matmul
    fwd_config = matmul.OpConfig(
        op=self._matmul_fn,
        kwargs=dict(preferred_element_type=jnp.int32),
    )
    config = matmul.MatmulFwdBwdConfig(
        fwd_config=fwd_config,
        dlhs_config=matmul.OpConfig.null_op(),
        drhs_config=matmul.OpConfig.null_op(),
    )
    z = matmul.matmul(xq, yq, config=config)

    # Compare with regular matmul
    self.assertTrue(
        jnp.allclose(
            z,
            jnp.matmul(x, y, preferred_element_type=jnp.int32).astype(
                jnp.float32
            ),
        )
    )

  @parameterized.parameters("dequantize", "static_quantize", "dynamic_quantize")
  def test_matmul_bwd(self, mode_str):
    # Create lossless quantized arrays
    key = jax.random.key(0)
    key1, key2, key3 = jax.random.split(key, 3)
    xy_kwargs = dict(minval=-100, maxval=100, dtype=jnp.int8)
    x = jax.random.randint(key1, (16, 32), **xy_kwargs)
    y = jax.random.randint(key2, (32, 64), **xy_kwargs)

    sx = jnp.ones((16, 1), dtype=jnp.float32)
    sy = jnp.ones((1, 64), dtype=jnp.float32)

    # Convert to HiQArray
    xq = convert.as_hiqarray(x, sx, None)
    yq = convert.as_hiqarray(y, sy, None)

    # Create information for backward pass
    g_int = jax.random.randint(key3, (16, 64), **xy_kwargs)
    g = g_int.astype(jnp.float32)
    sg = jnp.ones((16, 64), dtype=jnp.float32)

    # Setup fwd config
    fwd_config = matmul.OpConfig(
        op=self._matmul_fn,
        kwargs=dict(preferred_element_type=jnp.int32),
    )

    # Setup backward pass configs
    dlhs_mode, dlhs_config = self._create_bwd_config(mode_str, grad_scale=sg)
    drhs_mode, drhs_config = self._create_bwd_config(mode_str, grad_scale=sg)

    # Create config
    config = matmul.MatmulFwdBwdConfig(
        fwd_config=fwd_config,
        dlhs_mode=dlhs_mode,
        dlhs_config=dlhs_config,
        drhs_mode=drhs_mode,
        drhs_config=drhs_config,
    )

    # Compute vjp for the quantized matmul
    _, vjp_fn = jax.vjp(
        lambda a, b: matmul.matmul(
            a,
            b,
            config=config,
        ),
        xq,
        yq,
    )
    hijax_dlhs, hijax_drhs = vjp_fn(g)

    # Compute true backward pass
    true_dlhs = jnp.matmul(g, y.T, **config.dlhs_config.kwargs).astype(
        jnp.float32
    )
    true_drhs = jnp.matmul(x.T, g, **config.drhs_config.kwargs).astype(
        jnp.float32
    )

    # Compare dlhs and drhs to true values
    self.assertTrue(jnp.allclose(hijax_dlhs, true_dlhs))
    self.assertTrue(jnp.allclose(hijax_drhs, true_drhs))


if __name__ == "__main__":
  absltest.main()
