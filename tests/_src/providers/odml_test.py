# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test the ODML providers for op coverage."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from qwix._src import aux_data
from qwix._src import flax_util
from qwix._src import interception
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.providers import odml
from qwix._src.providers import odml_ops


class NamedParamModule(nn.Module):
  features: int
  param_name: str

  @nn.compact
  def __call__(self, x):
    w = self.param(
        self.param_name, nn.initializers.normal(), (x.shape[-1], self.features)
    )
    return jnp.dot(x, w)


class OdmlTest(parameterized.TestCase):

  def _run_linen_einsum_conversion(self, rules):
    class EinsumModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        return nn.Einsum(
            shape=(4, x.shape[-1], 8),
            einsum_str='BTD,NDH->BTNH',
            use_bias=False,
        )(x)

    model = EinsumModel()
    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    model_input = jnp.arange(2 * 3 * 16, dtype=jnp.float32).reshape(2, 3, 16)
    model_input = model_input / jnp.max(model_input)
    qat_vars = qat_model.init(jax.random.key(0), model_input)
    qat_res, new_vars = qat_model.apply(qat_vars, model_input, mutable=True)
    qat_vars.update(new_vars)

    conversion_provider = odml.OdmlConversionProvider(
        rules, qat_vars['params'], qat_vars['quant_stats']
    )
    conversion_model = qwix_model.quantize_model(model, conversion_provider)
    conversion_res = conversion_model.apply(qat_vars, model_input)
    return qat_vars, qat_res, conversion_res

  def _run_nnx_einsum_conversion(self, rules):
    class NnxEinsumModel(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.einsum = nnx.Einsum(
            'BTD,NDH->BTNH',
            (4, 16, 8),
            rngs=rngs,
        )

      def __call__(self, x):
        return self.einsum(x)

    model = NnxEinsumModel(nnx.Rngs(0))
    qat_provider = odml.OdmlQatProvider(rules)
    model_input = jnp.arange(2 * 3 * 16, dtype=jnp.float32).reshape(2, 3, 16)
    model_input = model_input / jnp.max(model_input)
    qat_model = qwix_model.quantize_model(model, qat_provider, model_input)
    qat_res = qat_model(model_input)

    quant_stats = nnx.to_pure_dict(nnx.state(qat_model, flax_util.QuantStat))
    params = nnx.to_pure_dict(nnx.state(qat_model, nnx.Param))

    conversion_provider = odml.OdmlConversionProvider(
        rules, params, quant_stats
    )
    conversion_model = qwix_model.quantize_model(
        model, conversion_provider, model_input
    )
    conversion_res = conversion_model(model_input)
    return quant_stats, qat_res, conversion_res

  def test_linen(self):
    class LinenModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)
        return x

    model = LinenModel()
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]
    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    model_input = jnp.ones((1, 16), dtype=jnp.float32)
    qat_vars = qat_model.init(jax.random.key(0), model_input)
    qat_res, new_vars = qat_model.apply(qat_vars, model_input, mutable=True)
    qat_vars.update(new_vars)

    self.assertEqual(
        {
            '/'.join(k[:-1])
            for k in flax.traverse_util.flatten_dict(qat_vars['quant_stats'])
        },
        {
            'Dense_0/dot_general0_lhs',
            'Dense_1/dot_general0_lhs',
            'final_output0',
        },
    )

    conversion_provider = odml.OdmlConversionProvider(
        rules,
        qat_vars['params'],
        qat_vars['quant_stats'],
    )
    conversion_model = qwix_model.quantize_model(model, conversion_provider)
    conversion_res = conversion_model.apply(qat_vars, model_input)
    self.assertTrue(jnp.allclose(qat_res, conversion_res))

  def test_linen_shared_scope(self):
    """Test that shared scopes do not cause naming collisions."""

    class SharedScopeModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        m1 = NamedParamModule(features=8, param_name='weight1')
        m2 = NamedParamModule(features=8, param_name='weight2')

        nn.share_scope(self, m1)
        nn.share_scope(self, m2)

        x = m1(x)
        x = m2(x)
        return x

    model = SharedScopeModel()
    rules = [
        qconfig.QuantizationRule(
            module_path='.*', weight_qtype=jnp.int8, act_qtype=jnp.int8
        )
    ]
    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    qat_vars = qat_model.init(jax.random.key(0), jnp.ones((1, 16)))

    quant_stats = qat_vars['quant_stats']
    self.assertIn('dot0_lhs', quant_stats)
    self.assertIn('dot1_lhs', quant_stats)

  def test_linen_no_shared_scope(self):
    """Test that standard submodules have separate counters and paths."""

    class NoSharedScopeModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        # Standard Flax behavior: each module gets its own namespace
        x = NamedParamModule(features=8, param_name='weight', name='m1')(x)
        x = NamedParamModule(features=8, param_name='weight', name='m2')(x)
        return x

    model = NoSharedScopeModel()
    rules = [
        qconfig.QuantizationRule(
            module_path='.*', weight_qtype=jnp.int8, act_qtype=jnp.int8
        )
    ]
    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    qat_vars = qat_model.init(jax.random.key(0), jnp.ones((1, 16)))

    self.assertEqual(
        {
            '/'.join(k[:-1])
            for k in flax.traverse_util.flatten_dict(qat_vars['quant_stats'])
        },
        {
            'm1/dot0_lhs',
            'm2/dot0_lhs',
            'final_output0',
        },
    )

  def test_linen_einsum_multi_axis_weight_conversion(self):
    """Tests 3D einsum weights with a middle contracting dimension."""
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]

    qat_vars, qat_res, conversion_res = self._run_linen_einsum_conversion(rules)
    self.assertIn('einsum0_lhs', qat_vars['quant_stats']['Einsum_0'])
    self.assertEqual(conversion_res.shape, (2, 3, 4, 8))
    self.assertTrue(jnp.allclose(qat_res, conversion_res))

  def test_linen_einsum_flattened_activation_static_scale_conversion(self):
    """Tests static activation scales survive ODML's einsum flattening."""
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            act_qtype=jnp.int8,
        ),
    ]
    qat_vars, qat_res, conversion_res = self._run_linen_einsum_conversion(rules)

    # QAT stores static activation stats in the original BTD rank. Conversion
    # flattens the activation to 2-D before fake quantizing it.
    self.assertEqual(
        qat_vars['quant_stats']['Einsum_0']['einsum0_lhs']['sum_of_max'].shape,
        (1, 1, 1),
    )
    self.assertEqual(conversion_res.shape, (2, 3, 4, 8))
    self.assertTrue(jnp.allclose(qat_res, conversion_res))

  def test_linen_einsum_repeated_labels_fall_back(self):
    """Tests repeated-label einsums use the default einsum path."""

    class RepeatedLabelEinsumModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        return nn.Einsum(
            shape=(4, x.shape[-1], 8),
            einsum_str='DD,NDH->DNH',
            use_bias=False,
        )(x)

    model = RepeatedLabelEinsumModel()
    model_input = jnp.arange(16 * 16, dtype=jnp.float32).reshape(16, 16)
    model_input = model_input / jnp.max(model_input)
    variables = model.init(jax.random.key(0), model_input)
    fp_res = model.apply(variables, model_input)

    conversion_provider = odml.OdmlConversionProvider(
        [], variables['params'], {}
    )
    conversion_model = qwix_model.quantize_model(model, conversion_provider)
    conversion_res = conversion_model.apply(variables, model_input)

    self.assertEqual(conversion_res.shape, (16, 4, 8))
    self.assertTrue(jnp.allclose(fp_res, conversion_res))

  def test_nnx(self):
    class NnxModel(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(16, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 16, rngs=rngs)

      def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x

    model = NnxModel(nnx.Rngs(0))
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]
    qat_provider = odml.OdmlQatProvider(rules)
    model_input = jnp.ones((1, 16), dtype=jnp.float32)
    qat_model = qwix_model.quantize_model(model, qat_provider, model_input)
    qat_res = qat_model(model_input)

    quant_stats = nnx.to_pure_dict(nnx.state(qat_model, flax_util.QuantStat))
    self.assertEqual(
        {
            '/'.join(k[:-1])
            for k in flax.traverse_util.flatten_dict(quant_stats)
        },
        {
            'linear2/dot_general0_lhs',
            'linear1/dot_general0_lhs',
            'final_output0',
        },
    )

    conversion_provider = odml.OdmlConversionProvider(
        rules,
        nnx.to_pure_dict(nnx.state(qat_model, nnx.Param)),
        quant_stats,
    )
    conversion_model = qwix_model.quantize_model(
        model, conversion_provider, model_input
    )
    conversion_res = conversion_model(model_input)
    self.assertTrue(jnp.allclose(qat_res, conversion_res))

  def test_nnx_einsum_multi_axis_weight_conversion(self):
    """Tests 3D einsum weights with a middle contracting dimension."""
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]

    quant_stats, qat_res, conversion_res = self._run_nnx_einsum_conversion(
        rules
    )
    self.assertIn('einsum0_lhs', quant_stats['einsum'])
    self.assertEqual(conversion_res.shape, (2, 3, 4, 8))
    self.assertTrue(jnp.allclose(qat_res, conversion_res))

  def test_nnx_einsum_flattened_activation_static_scale_conversion(self):
    """Tests static activation scales survive ODML's einsum flattening for NNX."""
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            act_qtype=jnp.int8,
        ),
    ]
    quant_stats, qat_res, conversion_res = self._run_nnx_einsum_conversion(
        rules
    )

    # QAT stores static activation stats in the original BTD rank. Conversion
    # flattens the activation to 2-D before fake quantizing it.
    self.assertEqual(
        quant_stats['einsum']['einsum0_lhs']['sum_of_max'].shape,
        (1, 1, 1),
    )
    self.assertEqual(conversion_res.shape, (2, 3, 4, 8))
    self.assertTrue(jnp.allclose(qat_res, conversion_res))

  def test_odml_interception_stack(self):
    """Verifies that ODML providers return interceptors in the correct order."""
    rules = [qconfig.QuantizationRule(module_path='.*')]
    provider = odml.OdmlQatProvider(rules)

    factories = provider.get_interceptors()
    self.assertLen(factories, 2)

    # 1. Structural Layer (Primitives)
    structural_interceptor = factories[0]()
    self.assertIn(
        interception.PRIMITIVE_BIND_KEY, structural_interceptor.mapping
    )
    self.assertIsInstance(
        structural_interceptor.mapping[interception.PRIMITIVE_BIND_KEY],
        odml.odml_ops.PrimitiveBindOp,
    )

    # 2. Numerical Layer (High-level Ops)
    numerical_interceptor = factories[1]()
    self.assertIn('jax.lax.dot_general', numerical_interceptor.mapping)
    # Ensure no primitive bind in numerical layer to avoid double wrapping
    self.assertNotIn(
        interception.PRIMITIVE_BIND_KEY, numerical_interceptor.mapping
    )

  def test_mixed_tags_at_boundary(self):
    class BranchModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x1 = nn.Dense(features=8, name='quant_dense')(x)
        x2 = nn.Dense(features=8, name='float_dense')(x)
        return jnp.multiply(x1, x2)

    model = BranchModel()
    rules = [
        qconfig.QuantizationRule(
            module_path='.*quant_dense.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]
    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    model_input = jnp.ones((1, 8), dtype=jnp.float32)
    qat_vars = qat_model.init(jax.random.key(0), model_input)

    flat_stats = flax.traverse_util.flatten_dict(qat_vars['quant_stats'])
    stat_keys = {'/'.join(k[:-1]) for k in flat_stats}

    # quant_dense should have stats collected
    self.assertIn('quant_dense/dot_general0_lhs', stat_keys)

    # float_dense should NOT have stats collected
    self.assertNotIn('float_dense/dot_general0_lhs', stat_keys)

    # multiply should NOT have stats collected because of mixed tags handling
    self.assertNotIn('multiply0_lhs', stat_keys)
    self.assertNotIn('multiply0_rhs', stat_keys)

  def test_flatten_einsum_aux_data_propagation(self):
    """Verifies aux_data on lhs and rhs propagates in _flatten_einsum."""
    provider = odml.OdmlConversionProvider(rules=[], params={}, quant_stats={})
    lhs = jnp.ones((2, 3, 16))
    rhs = jnp.ones((4, 16, 8))  # N=4, D=16, H=8

    # 1. Attach various metadata keys to the original lhs input
    aux_data.set(lhs, odml_ops.AuxDataKey.IS_ACTIVATION, True)
    aux_data.set(lhs, odml_ops.AuxDataKey.FIXED_RANGE, (0.0, 1.0))
    rule = qconfig.QuantizationRule(act_qtype=jnp.int8)
    aux_data.set(lhs, odml_ops.AuxDataKey.FQ_RULE, rule)
    aux_data.set(lhs, odml_ops.AuxDataKey.ALLOW_FUSION, True)
    aux_data.set(lhs, odml_ops.AuxDataKey.FQ_ARRAY, 'self')

    # 2. Tag rhs as a weight and attach metadata to test _flatten_einsum
    aux_data.set(rhs, odml_ops.AuxDataKey.WEIGHT_NAME, 'test_weight')
    aux_data.set(rhs, odml_ops.AuxDataKey.FIXED_RANGE, (-1.0, 1.0))
    rhs_rule = qconfig.QuantizationRule(weight_qtype=jnp.int8)
    aux_data.set(rhs, odml_ops.AuxDataKey.FQ_RULE, rhs_rule)
    aux_data.set(rhs, odml_ops.AuxDataKey.ALLOW_FUSION, True)

    def mock_einsum(einsum_str, flat_lhs, flat_rhs, **_kwargs):
      del einsum_str  # Unused in mock
      self.assertEqual(flat_lhs.ndim, 2)
      self.assertTrue(
          aux_data.get(flat_lhs, odml_ops.AuxDataKey.IS_ACTIVATION, False)
      )
      self.assertEqual(
          aux_data.get(flat_lhs, odml_ops.AuxDataKey.FIXED_RANGE, None),
          (0.0, 1.0),
      )
      self.assertEqual(
          aux_data.get(flat_lhs, odml_ops.AuxDataKey.FQ_RULE, None), rule
      )
      self.assertTrue(
          aux_data.get(flat_lhs, odml_ops.AuxDataKey.ALLOW_FUSION, False)
      )
      self.assertEqual(
          aux_data.get(flat_lhs, odml_ops.AuxDataKey.FQ_ARRAY, None), 'self'
      )

      # Check rhs metadata propagation
      self.assertEqual(flat_rhs.ndim, 2)
      self.assertEqual(
          aux_data.get(flat_rhs, odml_ops.AuxDataKey.WEIGHT_NAME, None),
          'test_weight',
      )
      self.assertTrue(
          aux_data.get(flat_rhs, odml_ops.AuxDataKey.REWRITE_TAG, {}).get(
              'flattened_einsum_weight', False
          )
      )
      self.assertEqual(
          aux_data.get(flat_rhs, odml_ops.AuxDataKey.FIXED_RANGE, None),
          (-1.0, 1.0),
      )
      self.assertEqual(
          aux_data.get(flat_rhs, odml_ops.AuxDataKey.FQ_RULE, None), rhs_rule
      )
      self.assertTrue(
          aux_data.get(flat_rhs, odml_ops.AuxDataKey.ALLOW_FUSION, False)
      )
      out = jnp.ones((flat_lhs.shape[0], flat_rhs.shape[1]))
      aux_data.set(out, odml_ops.AuxDataKey.IS_ACTIVATION, True)
      aux_data.set(out, odml_ops.AuxDataKey.ALLOW_FUSION, True)
      aux_data.set(out, odml_ops.AuxDataKey.FQ_RULE, rule)
      return out

    # 4. Trigger the _flatten_einsum interception logic directly
    res = provider._flatten_einsum(
        'BTD,NDH->BTNH', lhs, rhs, _einsum=mock_einsum
    )
    self.assertEqual(res.shape, (2, 3, 4, 8))
    self.assertTrue(aux_data.get(res, odml_ops.AuxDataKey.IS_ACTIVATION, False))
    self.assertTrue(aux_data.get(res, odml_ops.AuxDataKey.ALLOW_FUSION, False))
    self.assertEqual(aux_data.get(res, odml_ops.AuxDataKey.FQ_RULE, None), rule)

  def test_flatten_dot_general_aux_data_propagation(self):
    """Verifies aux_data on rhs weight and output propagates."""
    provider = odml.OdmlConversionProvider(rules=[], params={}, quant_stats={})
    lhs = jnp.ones((2, 8, 16))
    rhs = jnp.ones((16, 4, 4))
    dimension_numbers = (((2,), (0,)), ((), ()))

    # 1. Attach metadata to rhs weight to verify propagation after flattening
    aux_data.set(rhs, odml_ops.AuxDataKey.WEIGHT_NAME, 'test_weight')
    aux_data.set(rhs, odml_ops.AuxDataKey.FIXED_RANGE, (-1.0, 1.0))
    rhs_rule = qconfig.QuantizationRule(weight_qtype=jnp.int8)
    aux_data.set(rhs, odml_ops.AuxDataKey.FQ_RULE, rhs_rule)
    aux_data.set(rhs, odml_ops.AuxDataKey.ALLOW_FUSION, True)

    rule = qconfig.QuantizationRule(act_qtype=jnp.int8)

    # 2. Define a mock inner dot_general to check weight and output metadata
    def mock_dot_general(unused_lhs, flat_rhs, unused_dim_nums, **_kwargs):
      self.assertEqual(flat_rhs.ndim, 2)
      self.assertEqual(
          aux_data.get(flat_rhs, odml_ops.AuxDataKey.WEIGHT_NAME, None),
          'test_weight',
      )
      self.assertEqual(
          aux_data.get(flat_rhs, odml_ops.AuxDataKey.FIXED_RANGE, None),
          (-1.0, 1.0),
      )
      self.assertEqual(
          aux_data.get(flat_rhs, odml_ops.AuxDataKey.FQ_RULE, None), rhs_rule
      )
      self.assertTrue(
          aux_data.get(flat_rhs, odml_ops.AuxDataKey.ALLOW_FUSION, False)
      )

      # Mock output carrying metadata
      out = jnp.ones((2, 8, 16))
      aux_data.set(out, odml_ops.AuxDataKey.IS_ACTIVATION, True)
      aux_data.set(out, odml_ops.AuxDataKey.ALLOW_FUSION, True)
      aux_data.set(out, odml_ops.AuxDataKey.FQ_RULE, rule)
      return out

    # 3. Trigger _flatten_dot_general and verify output metadata propagation
    res = provider._flatten_dot_general(
        lhs, rhs, dimension_numbers, _dot_general=mock_dot_general
    )
    self.assertEqual(res.shape, (2, 8, 4, 4))
    self.assertTrue(aux_data.get(res, odml_ops.AuxDataKey.IS_ACTIVATION, False))
    self.assertTrue(aux_data.get(res, odml_ops.AuxDataKey.ALLOW_FUSION, False))
    self.assertEqual(aux_data.get(res, odml_ops.AuxDataKey.FQ_RULE, None), rule)


if __name__ == '__main__':
  absltest.main()
