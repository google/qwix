"""Tests for ADMM-Q in Qwix, modeled on qwix/tests/contrib/gptq_test.py."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from qwix._src import model as qwix_model
from qwix._src.core import qarray
from qwix._src.providers import ptq
from qwix._src.utils import flax_util
from qwix.contrib import gptq_core

from qwix_admmq import admmq
from qwix_admmq import admmq_core


def _layer(seed, d_out=64, d_in=256, n=2048, common=0.0):
  """A random weight and the mean X X^T of correlated, uneven inputs."""
  rng = np.random.default_rng(seed)
  x = rng.standard_normal((n, d_in)).astype(np.float32)
  if common:
    x += common * rng.standard_normal((n, 1)).astype(np.float32)
  x *= rng.uniform(0.2, 3.0, size=d_in).astype(np.float32)
  w = (rng.standard_normal((d_out, d_in)) * 0.02).astype(np.float32)
  return jnp.asarray(w), jnp.asarray(x.T @ x / n)


def _loss(w_q, w, h):
  diff = w_q - w
  return float(jnp.sum((diff @ h) * diff))


class AdmmqCoreTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('sym_per_channel', 'absmax', {}, 0.0),
      ('sym_g128', 'absmax', {1: 128}, 0.0),
      ('asym_per_channel', 'minmax', {}, 0.0),
      ('asym_g128', 'minmax', {1: 128}, 0.0),
      ('sym_ill_conditioned', 'absmax', {}, 40.0),
  )
  def test_quantize_weight(self, calibration_method, tiled_axes, common):
    w, h = _layer(0, common=common)
    how = qarray.HowToQuantize(
        qtype=jnp.int4,
        channelwise_axes=[0],
        tiled_axes=tiled_axes,
        calibration_method=calibration_method,
    )
    qw, loss = admmq_core.quantize_weight(w, h, how)

    # A valid Qwix QArray on the int4 grid, with the reported loss.
    q = np.asarray(qw.qvalue).astype(int)
    self.assertGreaterEqual(q.min(), -8)
    self.assertLessEqual(q.max(), 7)
    self.assertEqual(qw.scale.shape, qarray.get_scale_shape(w.shape, how))
    self.assertEqual(qw.zero_point is None, calibration_method == 'absmax')
    w_q = qarray.dequantize(qw)
    self.assertAlmostEqual(_loss(w_q, w, h), float(loss), delta=1e-4 * loss)

    # Better than plain PTQ (round to nearest).
    ptq_loss = _loss(qarray.dequantize(qarray.quantize(w, how)), w, h)
    self.assertLess(float(loss), ptq_loss)

  @parameterized.named_parameters(
      ('per_channel', {}, 0.0),
      ('g128', {1: 128}, 0.0),
      ('ill_conditioned', {}, 40.0),
  )
  def test_better_than_gptq(self, tiled_axes, common):
    # Symmetric only: Qwix's GPTQ currently fails on asymmetric (minmax) rules.
    w, h = _layer(1, common=common)
    how = qarray.HowToQuantize(
        qtype=jnp.int4, channelwise_axes=[0], tiled_axes=tiled_axes
    )
    _, loss = admmq_core.quantize_weight(w, h, how)
    gptq_w = qarray.dequantize(gptq_core.quantize_weight(w, h, how)[0])
    self.assertLess(float(loss), _loss(gptq_w, w, h))

  def test_unsupported_qtypes(self):
    for qtype in (jnp.float8_e4m3fn, 'nf4'):
      how = qarray.HowToQuantize(qtype=qtype, channelwise_axes=[0])
      self.assertFalse(admmq_core.is_supported(how))
    how = qarray.HowToQuantize(
        qtype='mxint4', channelwise_axes=[0], tiled_axes={1: 32}
    )
    self.assertFalse(admmq_core.is_supported(how))
    for qtype in (jnp.int4, jnp.int8, 'int3'):
      how = qarray.HowToQuantize(qtype=qtype, channelwise_axes=[0])
      self.assertTrue(admmq_core.is_supported(how))


class AdmmqTest(parameterized.TestCase):
  """End-to-end calibrate -> quantize_params -> PtqProvider, as for GPTQ."""

  def _calibration_input(self):
    return jax.random.normal(jax.random.key(0), (256, 32))

  def test_dense_model_linen(self):
    class DenseModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(64)(x)
        return x

    model = DenseModel()
    x = self._calibration_input()
    variables = model.init(jax.random.key(1), x)

    # 1. Calibration.
    rules = [admmq.AdmmqRule(module_path='Dense_0', weight_qtype=jnp.int4)]
    cal_model = qwix_model.quantize_model(
        model, admmq.AdmmqCalibrationProvider(rules)
    )
    _, new_variables = cal_model.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)
    stats = variables['quant_stats']['Dense_0']['kernel_admmq']
    self.assertEqual(stats['count'], 1)
    self.assertEqual(stats['sum_of_hessian'].shape, (32, 32))
    fp_y, new_variables = cal_model.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)
    self.assertEqual(
        variables['quant_stats']['Dense_0']['kernel_admmq']['count'], 2
    )

    # 2. Model preparation for inference.
    ptq_model = qwix_model.quantize_model(model, ptq.PtqProvider(rules))
    abs_variables = jax.eval_shape(ptq_model.init, jax.random.key(2), x)

    # 3. Weight transformation with ADMM-Q vs plain PTQ.
    admmq_params = admmq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )
    admmq_y = ptq_model.apply({'params': admmq_params}, x)
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )
    ptq_y = ptq_model.apply({'params': ptq_params}, x)

    mae = lambda a, b: jnp.mean(jnp.abs(a - b))
    self.assertLess(mae(fp_y, admmq_y), mae(fp_y, ptq_y))

  def test_dense_model_nnx(self):
    class Model(nnx.Module):

      def __init__(self, *, rngs=nnx.Rngs(0)):
        self.dense1 = nnx.Linear(32, 128, rngs=rngs)
        self.dense2 = nnx.Linear(128, 64, rngs=rngs)

      def __call__(self, x):
        return self.dense2(nn.gelu(self.dense1(x)))

    model = Model()
    x = self._calibration_input()

    rules = [admmq.AdmmqRule(module_path='dense1', weight_qtype=jnp.int4)]
    cal_model = qwix_model.quantize_model(
        model, admmq.AdmmqCalibrationProvider(rules), x
    )
    fp_y = cal_model(x)
    stats = cal_model.dense1.kernel_admmq
    self.assertEqual(stats['count'], 1)
    self.assertEqual(stats['sum_of_hessian'].shape, (32, 32))

    ptq_model = qwix_model.quantize_model(model, ptq.PtqProvider(rules), x)
    ptq_y = ptq_model(x)

    state = nnx.to_pure_dict(nnx.state(cal_model, nnx.Param))
    quant_stats = nnx.to_pure_dict(nnx.state(cal_model, flax_util.QuantStat))
    admmq_params = admmq.quantize_params(state, ptq_model, quant_stats)
    nnx.update(ptq_model, admmq_params)
    admmq_y = ptq_model(x)

    mae = lambda a, b: jnp.mean(jnp.abs(a - b))
    self.assertLess(mae(fp_y, admmq_y), mae(fp_y, ptq_y))

  def test_einsum_model_linen(self):
    class EinsumModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Einsum(shape=(x.shape[-1], 128), einsum_str='bi,io->bo')(x)
        x = nn.gelu(x)
        x = nn.Einsum(shape=(128, 64), einsum_str='bi,io->bo')(x)
        return x

    model = EinsumModel()
    x = self._calibration_input()
    variables = model.init(jax.random.key(1), x)

    rules = [admmq.AdmmqRule(module_path='Einsum_0', weight_qtype=jnp.int4)]
    provider = admmq.AdmmqCalibrationProvider(rules)

    @jax.jit
    def calibrate_step(v, x):
      m = qwix_model.quantize_model(model, provider)
      return m.apply(v, x, mutable='quant_stats')

    fp_y, new_variables = calibrate_step(variables, x)
    variables.update(new_variables)
    stats = variables['quant_stats']['Einsum_0']['kernel_admmq']
    self.assertEqual(stats['sum_of_hessian'].shape, (32, 32))

    ptq_model = qwix_model.quantize_model(model, ptq.PtqProvider(rules))
    abs_variables = jax.eval_shape(ptq_model.init, jax.random.key(2), x)
    admmq_params = admmq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )
    admmq_y = ptq_model.apply({'params': admmq_params}, x)
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )
    ptq_y = ptq_model.apply({'params': ptq_params}, x)

    mae = lambda a, b: jnp.mean(jnp.abs(a - b))
    self.assertLess(mae(fp_y, admmq_y), mae(fp_y, ptq_y))

  def test_unsupported_qtype_falls_back_to_ptq(self):
    model = nnx.Linear(32, 64, rngs=nnx.Rngs(0))
    x = self._calibration_input()
    rules = [
        admmq.AdmmqRule(module_path='.*', weight_qtype=jnp.float8_e4m3fn)
    ]
    cal_model = qwix_model.quantize_model(
        model, admmq.AdmmqCalibrationProvider(rules), x
    )
    _ = cal_model(x)
    ptq_model = qwix_model.quantize_model(model, ptq.PtqProvider(rules), x)
    ptq_y = ptq_model(x)

    state = nnx.to_pure_dict(nnx.state(cal_model, nnx.Param))
    quant_stats = nnx.to_pure_dict(nnx.state(cal_model, flax_util.QuantStat))
    admmq_params = admmq.quantize_params(state, ptq_model, quant_stats)
    nnx.update(ptq_model, admmq_params)
    np.testing.assert_allclose(ptq_model(x), ptq_y, rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
