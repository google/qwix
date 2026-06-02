from absl.testing import absltest
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
from qwix.contrib.hijax import test_utils
import qwix.contrib.hijax.hiqarray as hq
import qwix.contrib.hijax.hiqarray_common as hqc


class HiQArrayTest(absltest.TestCase):

  def test_qarray_exists(self):
    qarray = test_utils.create_random_qarray((16, 16), (0, 1), (16, 16))
    self.assertIsInstance(qarray, hq.HiQArray)

  # Test other transforms and basic functionality here
  def test_ref(self):
    qarray = test_utils.create_random_qarray((16, 16), (0, 1), (16, 16))
    _ = jax.new_ref(qarray)

  def test_ref_get(self):
    qarray = test_utils.create_random_qarray((16, 16), (0, 1), (1, 1))

    @jax.jit
    def f(q):
      ref = jax.new_ref(q)
      return ref[:, 0:2]

    o = f(qarray)
    assert jnp.all(o.qvalue == qarray.qvalue[:, 0:2])
    assert jnp.all(o.scale == qarray.scale[:, 0:2])

    assert o.shape == (16, 2)
    assert o.metadata.data_shape == (16, 2)
    assert o.metadata.quant_shape == (16, 2)

  def test_ref_swap(self):
    qarray1 = test_utils.create_random_qarray((16, 16), (0, 1), (16, 16))
    qarray2 = test_utils.create_random_qarray((16, 16), (0, 1), (16, 16))

    @jax.jit
    def f(q1, q2):
      ref = jax.new_ref(q1)
      ref[:, :] = q2
      return ref.get()

    o = f(qarray1, qarray2)
    assert jnp.all(o.qvalue == qarray2.qvalue)
    assert jnp.all(o.scale == qarray2.scale)

  def test_ref_swap_2(self):
    qarray1 = test_utils.create_random_qarray((8, 8), (0, 1), (1, 1))
    qarray2 = test_utils.create_random_qarray((16, 16), (0, 1), (1, 1))

    @jax.jit
    def f(q1, q2):
      ref = jax.new_ref(q1)
      ref2 = jax.new_ref(q2)
      ref[:, :] = ref2[:8, :8]
      return ref.get()

    o = f(qarray1, qarray2)
    assert jnp.all(o.qvalue == qarray2.qvalue[:8, :8])
    assert jnp.all(o.scale == qarray2.scale[:8, :8])

    assert o.shape == (8, 8)
    assert o.metadata.data_shape == (8, 8)
    assert o.metadata.quant_shape == (8, 8)

  def test_simple_kernel(self):
    qarray = test_utils.create_random_qarray((16, 16), (0, 1), (16, 16))

    def fn(q):
      def q_kernel(q_ref, o_ref):
        o_ref[...] = hq.from_hiqarray(q_ref[...])

      return pl.pallas_call(
          q_kernel,
          out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
          interpret=pltpu.InterpretParams(),
      )(q)

    fn(qarray)

  def test_bad_block_spec(self):
    qarray = test_utils.create_random_qarray((16, 16), (0, 1), (16, 16))

    block_spec = pl.BlockSpec((2, 2), lambda i, j: (i, j))
    ty = jax.typeof(qarray)
    with self.assertRaises(ValueError):
      ty.lower_block_spec(block_spec)

  def test_pallas_call(self):
    qarray = test_utils.create_random_qarray((16, 16), (0, 1), (1, 1))

    def fn(q, *, bm=8, bn=8):
      m, n = q.shape

      def q_kernel(q_ref, o_ref):
        assert q_ref.shape == (bm, bn)
        out_arr = hq.from_hiqarray(q_ref[...])
        o_ref[...] = out_arr

      out_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
      out_spec = pl.BlockSpec((bm, bn), lambda i, j: (i, j))
      grid = (m // bm, n // bn)
      return pl.pallas_call(
          q_kernel,
          out_shape=out_shape,
          out_specs=out_spec,
          grid=grid,
          in_specs=[pl.BlockSpec((bm, bn), lambda i, j: (i, j))],
          interpret=pltpu.InterpretParams(),
      )(q)

    fn(qarray)


class ToHiQArrayTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    orig_shape = (16, 16)
    quant_info = {0: 2, 1: 2}
    self.qvalue = jax.random.normal(k1, orig_shape)
    self.metadata = hqc.QuantizationMetadata.init(
        orig_shape, quant_info, jnp.float32, jnp.int8
    )
    self.scale = jax.random.normal(
        k2, self.metadata.quant_shape, dtype=jnp.float32
    )
    self.zero_point = jax.random.normal(
        k3, self.metadata.quant_shape, dtype=jnp.float32
    )

  def test_to_qarray_with_zero_point(self):
    # Checks that to_qarray works
    qarray = hq.to_hiqarray(
        self.qvalue,
        self.scale,
        self.zero_point,
        self.metadata,
        key=jax.random.key(0),
    )
    self.assertIsInstance(qarray, hq.HiQArray)

  def test_to_qarray_without_zero_point(self):
    qarray = hq.to_hiqarray(
        self.qvalue, self.scale, None, self.metadata, key=jax.random.key(0)
    )
    self.assertIsInstance(qarray, hq.HiQArray)

  def test_to_qarray_backward_with_zero_point(self):
    # Set up simple examples for autograd testing
    scale = jnp.ones_like(self.scale)
    zp = jnp.zeros_like(self.zero_point)

    fn = lambda data, scale, zero_point: hq.to_hiqarray(
        data, scale, zero_point, self.metadata, key=jax.random.key(0)
    )
    qarray = hq.to_hiqarray(
        self.qvalue, scale, zp, self.metadata, key=jax.random.key(0)
    )

    primals_out, bwd_fn = jax.vjp(fn, self.qvalue, scale, zp)

    # Test that primals are correct
    np.testing.assert_allclose(primals_out.qvalue, qarray.qvalue)

    # Compute cotangents
    cotangent = jax.random.normal(
        jax.random.key(0), self.qvalue.shape, dtype=self.qvalue.dtype
    )
    next_cotangent = bwd_fn(cotangent)

    # Check that cotangents are correct
    np.testing.assert_allclose(next_cotangent[0], cotangent)

  def test_to_qarray_backward_without_zero_point(self):
    # Set up simple examples for autograd testing
    scale = jnp.ones_like(self.scale)

    fn = lambda data, scale: hq.to_hiqarray(
        data, scale, None, self.metadata, key=jax.random.key(0)
    )
    qarray = hq.to_hiqarray(
        self.qvalue, scale, None, self.metadata, key=jax.random.key(0)
    )

    primals_out, bwd_fn = jax.vjp(fn, self.qvalue, scale)

    # Test that primals are correct
    np.testing.assert_allclose(primals_out.qvalue, qarray.qvalue)

    # Compute cotangents
    cotangent = jax.random.normal(
        jax.random.key(0), self.qvalue.shape, dtype=self.qvalue.dtype
    )
    next_cotangent = bwd_fn(cotangent)

    # Check that cotangents are correct
    np.testing.assert_allclose(next_cotangent[0], cotangent)


class FromHiQArrayTest(absltest.TestCase):
  # Checks that from_qarray works

  def setUp(self):
    super().setUp()
    orig_shape = (16, 16)
    quant_axes = (0, 1)
    group_sizes = (2, 2)
    self.qarray = test_utils.create_random_qarray(
        orig_shape, quant_axes, group_sizes, jax.random.key(0)
    )

  def test_from_qarray(self):
    array = hq.from_hiqarray(self.qarray)
    self.assertIsInstance(array, jax.Array)

  def test_from_qarray_backward_with_zero_point(self):
    # Set up simple examples for autograd testing
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    data = jax.random.normal(
        k1, self.qarray.shape, dtype=self.qarray.metadata.dtype
    )
    scale = jnp.ones_like(self.qarray.scale)
    zp = jnp.zeros_like(self.qarray.zero_point)
    qarray = hq.to_hiqarray(data, scale, zp, self.qarray.metadata, key=k2)

    arr = hq.from_hiqarray(qarray)
    primals_out, bwd_fn = jax.vjp(hq.from_hiqarray, qarray)

    # Test that primals are correct
    np.testing.assert_allclose(arr, primals_out)

    # Compute cotangents
    cotangent = jax.random.normal(k3, arr.shape, dtype=arr.dtype)
    next_cotangent = bwd_fn(cotangent)

    # Check that cotangents are correct
    np.testing.assert_allclose(next_cotangent[0], cotangent)


class HiQArrayPermuteAxesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    orig_shape = (16, 16)
    quant_axes = (0, 1)
    group_sizes = (2, 2)
    self.qarray = test_utils.create_random_qarray(
        orig_shape, quant_axes, group_sizes, jax.random.key(0)
    )

  def test_permute_axes(self):
    axes = (1, 0)
    permuted_qarray = hq.permute_axes(self.qarray, axes)
    self.assertIsInstance(permuted_qarray, hq.HiQArray)
    np.testing.assert_allclose(
        permuted_qarray.qvalue, jnp.permute_dims(self.qarray.qvalue, axes)
    )
    np.testing.assert_allclose(
        permuted_qarray.scale, jnp.permute_dims(self.qarray.scale, axes)
    )
    np.testing.assert_allclose(
        permuted_qarray.zero_point,
        jnp.permute_dims(self.qarray.zero_point, axes),
    )

  def test_permute_axes_noop(self):
    permuted_qarray = hq.permute_axes(self.qarray, (0, 1))
    self.assertIsInstance(permuted_qarray, hq.HiQArray)
    np.testing.assert_allclose(permuted_qarray.qvalue, self.qarray.qvalue)
    np.testing.assert_allclose(permuted_qarray.scale, self.qarray.scale)
    np.testing.assert_allclose(
        permuted_qarray.zero_point, self.qarray.zero_point
    )
    self.assertEqual(permuted_qarray.metadata, self.qarray.metadata)

  def test_permute_axes_involution(self):
    permuted_qarray = hq.permute_axes(self.qarray, (1, 0))
    permuted_qarray2 = hq.permute_axes(permuted_qarray, (1, 0))
    np.testing.assert_allclose(permuted_qarray2.qvalue, self.qarray.qvalue)
    np.testing.assert_allclose(permuted_qarray2.scale, self.qarray.scale)
    np.testing.assert_allclose(
        permuted_qarray2.zero_point, self.qarray.zero_point
    )
    self.assertEqual(permuted_qarray2.metadata, self.qarray.metadata)

  def test_permute_axes_autograd(self):
    cotangent = jax.random.normal(jax.random.key(0), self.qarray.shape)
    primals_out, bwd_fn = jax.vjp(hq.permute_axes, self.qarray, (1, 0))
    del primals_out  # Unused.
    next_cotangent = bwd_fn(cotangent)
    np.testing.assert_allclose(
        next_cotangent[0], jnp.permute_dims(cotangent, (1, 0))
    )


class HiQArrayIntegrationTest(absltest.TestCase):

  def test_to_from_qarray(self):
    # Checks that round trip works for integer types and unit scale/zero point
    orig_shape = (16, 16)
    quant_info = {0: 2, 1: 2}
    data = jax.random.uniform(
        jax.random.key(0), orig_shape, minval=-128, maxval=128
    ).astype(jnp.int32)
    metadata = hqc.QuantizationMetadata.init(
        orig_shape, quant_info, jnp.float32, jnp.int32
    )
    scale = jnp.ones(metadata.quant_shape, dtype=jnp.int32)
    zero_point = jnp.zeros(metadata.quant_shape, dtype=jnp.int32)
    qarray = hq.to_hiqarray(
        data, scale, zero_point, metadata, key=jax.random.key(0)
    )
    array = hq.from_hiqarray(qarray)
    qarray2 = hq.to_hiqarray(
        array, scale, zero_point, metadata, key=jax.random.key(0)
    )
    np.testing.assert_allclose(qarray.qvalue, qarray2.qvalue)

  def test_inspect(self):
    orig_shape = (16, 16)
    quant_info = {0: 2, 1: 2}
    data = jax.random.uniform(
        jax.random.key(0), orig_shape, minval=-128, maxval=128
    ).astype(jnp.int32)
    metadata = hqc.QuantizationMetadata.init(
        orig_shape, quant_info, jnp.float32, jnp.int32
    )
    scale = jnp.ones(metadata.quant_shape, dtype=jnp.int32)
    zero_point = jnp.zeros(metadata.quant_shape, dtype=jnp.int32)
    fn = jax.make_jaxpr(hq.to_hiqarray, static_argnums=(3,))(
        data, scale, zero_point, metadata, key=jax.random.key(0)
    )
    print(fn)


if __name__ == "__main__":
  absltest.main()
