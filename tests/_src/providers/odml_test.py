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
from qwix._src import interception
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.providers import odml
from qwix._src.providers import odml_ops
from qwix._src.utils import flax_util


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

  def test_linen_partitioned_conversion_parameter(self):
    model = nn.Dense(
        4,
        use_bias=False,
        kernel_init=nn.with_partitioning(nn.ones_init(), ('input', 'output')),
    )
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
        ),
    ]
    model_input = jnp.ones((1, 3), dtype=jnp.float32)
    qat_model = qwix_model.quantize_model(model, odml.OdmlQatProvider(rules))
    variables = qat_model.init(jax.random.key(0), model_input)
    qat_result = qat_model.apply(variables, model_input)

    self.assertIsInstance(variables['params']['kernel'], nn.Partitioned)

    conversion_model = qwix_model.quantize_model(
        model,
        odml.OdmlConversionProvider(rules, variables['params'], {}),
    )
    conversion_result = conversion_model.apply(variables, model_input)

    self.assertTrue(jnp.allclose(qat_result, conversion_result))

  def test_conversion_parameter_must_unbox_to_array(self):
    with self.assertRaisesRegex(TypeError, 'parameter kernel has type object'):
      odml.OdmlConversionProvider([], {'kernel': object()}, {})

  def test_linen_shared_scope(self):
    """Test that shared scopes do not cause naming collisions."""

    class SharedScopeModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        m1 = NamedParamModule(features=8, param_name='weight1')
        m2 = NamedParamModule(features=8, param_name='weight2')

        # Reproduce Gemma3 pattern: multiple objects sharing one registry
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

  def test_odml_original_weight_marker_is_not_forwarded(self):
    """Tests a real structural interceptor invalidates layout identity."""
    rhs = jnp.ones((4, 5, 7), dtype=jnp.float32)
    aux_data.set(rhs, odml_ops.AuxDataKey.WEIGHT_NAME, 'kernel')
    aux_data.set(rhs, odml_ops.AuxDataKey.IS_ORIGINAL_WEIGHT, True)

    provider = odml.OdmlQatProvider([])
    transpose = interception.wrap_func_intercepted(
        lambda x: jnp.transpose(x, (0, 2, 1)),
        provider.get_interceptors()[0],
        disable_jit=False,
    )
    rhs_transposed = transpose(rhs)

    self.assertEqual(
        aux_data.get(rhs_transposed, odml_ops.AuxDataKey.WEIGHT_NAME, None),
        'kernel',
    )
    self.assertFalse(
        aux_data.get(
            rhs_transposed,
            odml_ops.AuxDataKey.IS_ORIGINAL_WEIGHT,
            False,
        )
    )

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

  @parameterized.parameters(False, True)
  def test_linen_original_weight_marker_is_initialized(self, use_axis_metadata):
    """Tests Linen tags one direct logical array for boxed and raw params."""
    observed_metadata = []

    class LinenModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        init = nn.initializers.ones
        if use_axis_metadata:
          init = nn.with_partitioning(init, ('input', 'output'))
        param = self.param(
            'kernel', init, (x.shape[-1], 4), unbox=not use_axis_metadata
        )
        weight = flax_util.unbox(param)
        observed_metadata.append((
            aux_data.get(weight, odml_ops.AuxDataKey.WEIGHT_NAME, None),
            aux_data.get(
                weight,
                odml_ops.AuxDataKey.IS_ORIGINAL_WEIGHT,
                False,
            ),
        ))
        return jnp.dot(x, weight)

    model = LinenModel()
    qat_model = qwix_model.quantize_model(model, odml.OdmlQatProvider([]))
    model_input = jnp.ones((1, 3), dtype=jnp.float32)
    variables = qat_model.init(jax.random.key(0), model_input)

    observed_metadata.clear()
    qat_model.apply(variables, model_input)

    self.assertEqual(observed_metadata, [('kernel', True)])

  def test_nnx_original_weight_marker_is_initialized(self):
    """Tests NNX model preprocessing tags the direct parameter state."""

    class NnxModel(nnx.Module):

      def __init__(self):
        self.kernel = nnx.Param(jnp.ones((3, 4), dtype=jnp.float32))

      def __call__(self, x):
        return jnp.dot(x, self.kernel)

    model_input = jnp.ones((1, 3), dtype=jnp.float32)
    qat_model = qwix_model.quantize_model(
        NnxModel(), odml.OdmlQatProvider([]), model_input
    )
    weight = flax_util.unbox(qat_model.kernel)

    self.assertEqual(
        aux_data.get(weight, odml_ops.AuxDataKey.WEIGHT_NAME, None),
        'kernel',
    )
    self.assertTrue(
        aux_data.get(
            weight,
            odml_ops.AuxDataKey.IS_ORIGINAL_WEIGHT,
            False,
        )
    )

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

  def test_matched_siblings_sharing_stats(self):
    """Test that matched sibling branches correctly share the tracer and resolve scales."""

    class SiblingModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        # Preceding quantized layer to set FQ_RULE and enable delayed sharing
        x = nn.Dense(features=8, name='pre_dense')(x)
        x1 = nn.Dense(features=8, name='sibling1')(x)
        x2 = nn.Dense(features=8, name='sibling2')(x)
        return jnp.multiply(x1, x2)

    model = SiblingModel()
    # Both are quantized under the same int8 rule.
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]

    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    model_input = jnp.ones((1, 8), dtype=jnp.float32)
    qat_vars = qat_model.init(jax.random.key(0), model_input)

    # Run calibration to accumulate stats
    _, new_vars = qat_model.apply(qat_vars, model_input, mutable=True)
    qat_vars.update(new_vars)

    # Since they are matched, they must have shared the FQ_ARRAY cache.
    # This means only sibling1 (which ran first) registered the stats.
    flat_stats = flax.traverse_util.flatten_dict(qat_vars['quant_stats'])
    stat_keys = {'/'.join(k[:-1]) for k in flat_stats}

    self.assertIn('sibling1/dot_general0_lhs', stat_keys)
    # sibling2 should NOT have registered stats because it reused sibling1's
    # tracer
    self.assertNotIn('sibling2/dot_general0_lhs', stat_keys)

    # Conversion: both sibling1 and sibling2 should convert successfully without
    # KeyErrors, with sibling2 resolving its static scale using sibling1's
    # registered path.
    conversion_provider = odml.OdmlConversionProvider(
        rules,
        qat_vars['params'],
        qat_vars['quant_stats'],
    )
    conversion_model = qwix_model.quantize_model(model, conversion_provider)
    # This apply should succeed cleanly (no KeyError!)
    conversion_res = conversion_model.apply(qat_vars, model_input)
    self.assertIsNotNone(conversion_res)

  def test_mismatched_sibling_quantized_vs_float(self):
    """Test that mismatched sibling branches (quantized vs float) are isolated."""

    class MixedSiblingModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(features=8, name='pre_dense')(x)
        x1 = nn.Dense(features=8, name='quant_sibling')(x)
        x2 = nn.Dense(features=8, name='float_sibling')(x)
        return jnp.multiply(x1, x2)

    model = MixedSiblingModel()
    # Only quantize the first sibling.
    rules = [
        qconfig.QuantizationRule(
            module_path='.*pre_dense.*|.*quant_sibling.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]

    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    model_input = jnp.ones((1, 8), dtype=jnp.float32)
    qat_vars = qat_model.init(jax.random.key(0), model_input)

    # Run calibration to accumulate stats
    _, new_vars = qat_model.apply(qat_vars, model_input, mutable=True)
    qat_vars.update(new_vars)

    # Stats verification
    flat_stats = flax.traverse_util.flatten_dict(qat_vars['quant_stats'])
    stat_keys = {'/'.join(k[:-1]) for k in flat_stats}

    self.assertIn('quant_sibling/dot_general0_lhs', stat_keys)
    self.assertNotIn('float_sibling/dot_general0_lhs', stat_keys)

    # Conversion runs successfully without any leaks or KeyErrors
    conversion_provider = odml.OdmlConversionProvider(
        rules,
        qat_vars['params'],
        qat_vars['quant_stats'],
    )
    conversion_model = qwix_model.quantize_model(model, conversion_provider)
    conversion_res = conversion_model.apply(qat_vars, model_input)
    self.assertIsNotNone(conversion_res)

  def test_mismatched_sibling_parameter_isolation(self):
    """Test that sibling branches with different quantized parameters are strictly isolated."""

    class MultiQuantSiblingModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        # Immediate quantization (no pre_dense) to allow different consumer
        # rules
        x1 = nn.Dense(features=8, name='sibling_minmax')(x)
        x2 = nn.Dense(features=8, name='sibling_absmax')(x)
        return jnp.multiply(x1, x2)

    model = MultiQuantSiblingModel()
    # Different rules for different siblings.
    rules = [
        qconfig.QuantizationRule(
            module_path='.*sibling_minmax.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            act_calibration_method='minmax',
        ),
        qconfig.QuantizationRule(
            module_path='.*sibling_absmax.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            act_calibration_method='absmax',
        ),
    ]

    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    model_input = jnp.ones((1, 8), dtype=jnp.float32)
    qat_vars = qat_model.init(jax.random.key(0), model_input)
    _, new_vars = qat_model.apply(qat_vars, model_input, mutable=True)
    qat_vars.update(new_vars)

    # Since their rules differ, they must have isolated (copied).
    # Therefore, BOTH siblings must have registered their own unique stats.
    flat_stats = flax.traverse_util.flatten_dict(qat_vars['quant_stats'])
    stat_keys = {'/'.join(k[:-1]) for k in flat_stats}

    self.assertIn('sibling_minmax/dot_general0_lhs', stat_keys)
    self.assertIn('sibling_absmax/dot_general0_lhs', stat_keys)

    # Both convert successfully using their own isolated scales
    conversion_provider = odml.OdmlConversionProvider(
        rules,
        qat_vars['params'],
        qat_vars['quant_stats'],
    )
    conversion_model = qwix_model.quantize_model(model, conversion_provider)
    conversion_res = conversion_model.apply(qat_vars, model_input)
    self.assertIsNotNone(conversion_res)

  def test_matched_siblings_with_reshape_sharing(self):
    """Test that sibling branches separated by reshape calibrate and convert separately."""

    class SiblingReshapeModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(features=8, name='pre_dense')(x)
        x1 = nn.Dense(features=8, name='sibling1')(x)
        # Reshape to different shape breaks FQ_ARRAY tracer sharing,
        # forcing sibling2 to run _fake_quant.
        x_reshaped = jnp.reshape(x, (8,))
        x2 = nn.Dense(features=8, name='sibling2')(x_reshaped)
        return jnp.multiply(x1, x2)

    model = SiblingReshapeModel()
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]

    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    model_input = jnp.ones((1, 8), dtype=jnp.float32)
    qat_vars = qat_model.init(jax.random.key(0), model_input)

    # Run calibration to accumulate stats
    _, new_vars = qat_model.apply(qat_vars, model_input, mutable=True)
    qat_vars.update(new_vars)

    flat_stats = flax.traverse_util.flatten_dict(qat_vars['quant_stats'])
    stat_keys = {'/'.join(k[:-1]) for k in flat_stats}

    self.assertIn('sibling1/dot_general0_lhs', stat_keys)
    # Sibling 2 collects its own stats in QAT because jnp.reshape
    # breaks FQ_ARRAY tracer-level cache sharing.
    self.assertIn('sibling2/dot_general0_lhs', stat_keys)

    conversion_provider = odml.OdmlConversionProvider(
        rules,
        qat_vars['params'],
        qat_vars['quant_stats'],
    )
    conversion_model = qwix_model.quantize_model(model, conversion_provider)
    # Both convert successfully using their own respective stats!
    conversion_res = conversion_model.apply(qat_vars, model_input)
    self.assertIsNotNone(conversion_res)

  def test_immediate_matched_siblings_sharing_stats(self):
    """Test that matched sibling branches using immediate quantization share the tracer and stats."""

    class SiblingModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        # Immediate quantization (no preceding quantized layer)
        x1 = nn.Dense(features=8, name='sibling1')(x)
        x2 = nn.Dense(features=8, name='sibling2')(x)
        return jnp.multiply(x1, x2)

    model = SiblingModel()
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]

    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    model_input = jnp.ones((1, 8), dtype=jnp.float32)
    qat_vars = qat_model.init(jax.random.key(0), model_input)

    # Run calibration to accumulate stats
    _, new_vars = qat_model.apply(qat_vars, model_input, mutable=True)
    qat_vars.update(new_vars)

    # Verify that stats are shared!
    flat_stats = flax.traverse_util.flatten_dict(qat_vars['quant_stats'])
    stat_keys = {'/'.join(k[:-1]) for k in flat_stats}

    self.assertIn('sibling1/dot_general0_lhs', stat_keys)
    # sibling2 should NOT have registered stats because it reused sibling1's
    # tracer (sharing!)
    self.assertNotIn('sibling2/dot_general0_lhs', stat_keys)

    # Conversion runs successfully
    conversion_provider = odml.OdmlConversionProvider(
        rules,
        qat_vars['params'],
        qat_vars['quant_stats'],
    )
    conversion_model = qwix_model.quantize_model(model, conversion_provider)
    conversion_res = conversion_model.apply(qat_vars, model_input)
    self.assertIsNotNone(conversion_res)

  def test_metadata_propagation_linear_arithmetic(self):
    """Test that elementwise linear arithmetic propagates quantization rules and ALLOW_FUSION."""

    class LinearPropagationModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        # Preceding quantized layer to set FQ_RULE
        x = nn.Dense(features=8, name='pre_dense')(x)

        # Elementwise linear scaling and shifting (mul, add, neg)
        # These should propagate the FQ_RULE and ALLOW_FUSION metadata cleanly.
        x = x * 2.5
        x = x + 1.2
        x = -x

        # Matching siblings consuming the scaled/shifted tracer
        x1 = nn.Dense(features=8, name='sibling1')(x)
        x2 = nn.Dense(features=8, name='sibling2')(x)
        return jnp.multiply(x1, x2)

    model = LinearPropagationModel()
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]

    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    model_input = jnp.ones((1, 8), dtype=jnp.float32)
    qat_vars = qat_model.init(jax.random.key(0), model_input)

    # Run calibration
    _, new_vars = qat_model.apply(qat_vars, model_input, mutable=True)
    qat_vars.update(new_vars)

    flat_stats = flax.traverse_util.flatten_dict(qat_vars['quant_stats'])
    stat_keys = {'/'.join(k[:-1]) for k in flat_stats}

    # If propagation succeeded, sibling1 and sibling2 successfully share the
    # delayed FQ_ARRAY cache, meaning only sibling1 registers stats.
    self.assertIn('sibling1/dot_general0_lhs', stat_keys)
    self.assertNotIn('sibling2/dot_general0_lhs', stat_keys)

    # Conversion runs cleanly without KeyErrors
    conversion_provider = odml.OdmlConversionProvider(
        rules,
        qat_vars['params'],
        qat_vars['quant_stats'],
    )
    conversion_model = qwix_model.quantize_model(model, conversion_provider)
    conversion_res = conversion_model.apply(qat_vars, model_input)
    self.assertIsNotNone(conversion_res)

  def test_metadata_propagation_reciprocal_division(self):
    """Test that activation / const propagates metadata but const / activation does not."""

    class ReciprocalDivisionModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        # Preceding quantized layer to set FQ_RULE
        x = nn.Dense(features=8, name='pre_dense')(x)

        # Linear division (activation / const) -> correctly propagates FQ_RULE
        x_linear = x / 2.0

        # Reciprocal division (const / activation) -> correctly strips FQ_RULE
        x_reciprocal = jax.lax.div(1.0, x)

        l1 = nn.Dense(features=8, name='linear_sibling1')(x_linear)
        l2 = nn.Dense(features=8, name='linear_sibling2')(x_linear)

        r1 = nn.Dense(features=8, name='reciprocal_sibling1')(x_reciprocal)
        r2 = nn.Dense(features=8, name='reciprocal_sibling2')(x_reciprocal)
        return (
            jnp.multiply(l1, l2),
            jnp.multiply(r1, r2),
            x_linear,
            x_reciprocal,
        )

    model = ReciprocalDivisionModel()
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]

    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    model_input = jnp.ones((1, 8), dtype=jnp.float32)
    qat_vars = qat_model.init(jax.random.key(0), model_input)

    # Run calibration
    res, new_vars = qat_model.apply(qat_vars, model_input, mutable=True)
    qat_vars.update(new_vars)

    _, _, x_linear, x_reciprocal = res

    # Linear division successfully propagated FQ_RULE
    self.assertIsNotNone(
        odml.odml_ops.aux_data.get(
            x_linear, odml.odml_ops.AuxDataKey.FQ_RULE, None
        )
    )
    # Reciprocal division correctly stripped FQ_RULE
    self.assertIsNone(
        odml.odml_ops.aux_data.get(
            x_reciprocal, odml.odml_ops.AuxDataKey.FQ_RULE, None
        )
    )


if __name__ == '__main__':
  absltest.main()
