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

import os

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import ai_edge_jax
import flax
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from qwix._src import flax_util
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.providers import odml


jax.config.update('jax_threefry_partitionable', False)


srq_test_cases = []
drq_test_cases = []


def srq_test_case(cls):
  srq_test_cases.append({'testcase_name': cls.__name__, 'model': cls()})
  return cls


def drq_test_case(cls):
  drq_test_cases.append({'testcase_name': cls.__name__, 'model': cls()})
  return cls


@srq_test_case
@drq_test_case
class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x

  def create_input(self):
    return jax.random.uniform(jax.random.key(0), (1, 28, 28, 1), jnp.float32)

  expected_quant_stats_keys = {
      'Conv_0/conv_general_dilated0_lhs',
      'avg_pool0',
      # No Conv_1/conv_general_dilated0_lhs because it's the output of avg_pool.
      'avg_pool1',
      # No Dense_0/dot_general0_lhs because it's the output of avg_pool.
      'Dense_1/dot_general0_lhs',
      'final_output0',
  }

  expected_ops_summary = {
      'quantize_op_count': 1,
      'dequantize_op_count': 1,
      'fp_op_count': 0,
      'int_op_count': 7,
  }

  drq_expected_ops_summary = {
      'quantize_op_count': 0,
      'dequantize_op_count': 0,
  }


@srq_test_case
@drq_test_case
class Transformer(nn.Module):
  """A simple Transformer model."""

  num_layers: int = 2
  num_heads: int = 4
  qkv_features: int = 8
  hidden_dim: int = 32
  vocab_size: int = 100
  embedding_dim: int = 16
  use_bias: bool = True

  @nn.compact
  def __call__(self, input_ids):
    x = nn.Embed(self.vocab_size, self.embedding_dim)(input_ids)
    for _ in range(self.num_layers):
      x += nn.MultiHeadAttention(
          self.num_heads,
          qkv_features=self.qkv_features,
          use_bias=self.use_bias,
      )(x)
      residual = x
      x = nn.Dense(self.hidden_dim, use_bias=self.use_bias)(x)
      x = nn.relu(x)
      x = nn.Dense(self.embedding_dim, use_bias=self.use_bias)(x)
      x = residual + x
    x = x.mean(axis=1)
    x = nn.Dense(self.vocab_size, use_bias=self.use_bias)(x)
    return x

  def create_input(self):
    return jax.random.randint(
        jax.random.key(0),
        (1, 10),
        minval=0,
        maxval=self.vocab_size,
        dtype=jnp.int32,
    )

  expected_quant_stats_keys = {
      # There's no MultiHeadAttention_0/{query,key,value}/dot_general0_lhs
      # because they are quantized in the embedder.
      'MultiHeadAttention_0/truediv0_lhs',  # query / jnp.sqrt(depth)
      'MultiHeadAttention_0/truediv0_rhs',
      'MultiHeadAttention_0/einsum0_lhs',
      'MultiHeadAttention_0/einsum0_rhs',
      'MultiHeadAttention_0/softmax0',  # dot_product_attention_weights
      'MultiHeadAttention_0/einsum1_lhs',
      'MultiHeadAttention_0/einsum1_rhs',
      'MultiHeadAttention_0/out/dot_general0_lhs',
      'add0_rhs',
      'Dense_0/dot_general0_lhs',
      'Dense_1/dot_general0_lhs',
      'add1_rhs',
      'MultiHeadAttention_1/query/dot_general0_lhs',  # also include key/value.
      'MultiHeadAttention_1/truediv0_lhs',
      'MultiHeadAttention_1/truediv0_rhs',
      'MultiHeadAttention_1/einsum0_lhs',
      'MultiHeadAttention_1/einsum0_rhs',
      'MultiHeadAttention_1/softmax0',
      'MultiHeadAttention_1/einsum1_lhs',
      'MultiHeadAttention_1/einsum1_rhs',
      'MultiHeadAttention_1/out/dot_general0_lhs',
      'add2_rhs',
      'Dense_2/dot_general0_lhs',
      'Dense_3/dot_general0_lhs',
      'add3_rhs',
      'mean0',
      'Dense_4/dot_general0_lhs',
      'final_output0',
  }

  # x.mean(axis=1) is not quantized correctly and produces
  # "dq -> fp_sum -> fp_div -> q".
  expected_ops_summary = {
      'quantize_op_count': 1,
      'dequantize_op_count': 2,  # mean and final_output.
      'fp_op_count': 2,  # fp_sum and fp_div in mean.
  }

  drq_expected_ops_summary = {
      'quantize_op_count': 0,
      'dequantize_op_count': 1,  # dequantize the embedding output.
  }


@srq_test_case
class GroupNormSilu(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.GroupNorm(num_groups=2)(x)
    x = nn.silu(x)
    return x

  def create_input(self):
    return jax.random.uniform(jax.random.key(0), (1, 128), jnp.float32)

  expected_quant_stats_keys = {
      'norm_op0',
      'silu0',
      'silu0_sigmoid',
      'final_output0',
  }

  # The group norm is converted as dq + group_norm composite + q.
  expected_ops_summary = {
      'quantize_op_count': 2,
      'dequantize_op_count': 2,
      'fp_op_count': 1,
      'int_op_count': 2,
  }


@srq_test_case
class DenseConcatenateResidual(nn.Module):

  @nn.compact
  def __call__(self, x):
    residual = x
    x = nn.Dense(features=10)(x)
    x = nn.relu(x)
    x = jnp.concatenate([x, residual], axis=-1)
    x = nn.Dense(features=residual.shape[-1])(x)
    x = residual + x
    return x

  def create_input(self):
    return jax.random.uniform(jax.random.key(0), (2, 5), jnp.float32)

  expected_quant_stats_keys = {
      'Dense_0/dot_general0_lhs',
      # No Dense_0/add0_input0 because add is fused.
      'Dense_1/dot_general0_lhs',
      # No add0_lhs because it's quantized as Dense_0/dot_general0_lhs.
      'add0_rhs',
      'final_output0',
  }

  expected_ops_summary = {
      'quantize_op_count': 1,
      'dequantize_op_count': 1,
      'fp_op_count': 0,
      'int_op_count': 4,
  }


@srq_test_case
class UNet(nn.Module):
  """A simple UNet model."""

  encoder_filters_sequence = (4, 8, 16)
  decoder_filters_sequence = (16, 8, 4)

  @nn.compact
  def __call__(self, x: list[jax.Array]) -> jnp.ndarray:
    x = jnp.concatenate(x, axis=-1)
    pre_downsample_features = []
    for encoder_filters in self.encoder_filters_sequence:
      x = nn.Conv(
          encoder_filters,
          kernel_size=(3, 3),
          strides=(1, 1),
      )(x)
      x = nn.relu(x)
      pre_downsample_features.append(x)
      x = nn.Conv(
          encoder_filters,
          kernel_size=(3, 3),
          strides=(2, 2),
      )(x)
      x = nn.relu(x)

    for decoder_filters in self.decoder_filters_sequence:
      x = nn.Conv(
          decoder_filters,
          kernel_size=(3, 3),
          strides=(1, 1),
      )(x)
      x = nn.relu(x)
      x = jax.image.resize(
          x,
          (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]),
          'bilinear',
      )
      x = jnp.concatenate([x, pre_downsample_features[-1]], axis=-1)
      pre_downsample_features.pop()

    x = nn.Conv(3, kernel_size=(3, 3), strides=(1, 1))(x)
    return x

  def create_input(self):
    return [
        jax.random.uniform(jax.random.key(0), (1, 128, 128, 3)) - 0.1
        for _ in range(2)
    ]

  expected_quant_stats_keys = {
      'Conv_0/conv_general_dilated0_lhs',
      'Conv_1/conv_general_dilated0_lhs',
      'Conv_2/conv_general_dilated0_lhs',
      'Conv_3/conv_general_dilated0_lhs',
      'Conv_4/conv_general_dilated0_lhs',
      'Conv_5/conv_general_dilated0_lhs',
      'Conv_6/conv_general_dilated0_lhs',
      'resize0',
      'Conv_7/conv_general_dilated0_lhs',
      'resize1',
      'Conv_8/conv_general_dilated0_lhs',
      'resize2',
      'Conv_9/conv_general_dilated0_lhs',
      'final_output0',
  }

  additional_provider_args = dict(
      fixed_range_for_inputs=(0, 1),
      fixed_range_for_outputs=(0, 1),
  )

  expected_quant_stats_values = {
      'Conv_0/conv_general_dilated0_lhs/count': 1,
      'Conv_0/conv_general_dilated0_lhs/sum_of_min': 0,
      'Conv_0/conv_general_dilated0_lhs/sum_of_max': 1,
      'final_output0/count': 1,
      'final_output0/sum_of_min': 0,
      'final_output0/sum_of_max': 1,
  }

  expected_ops_summary = {
      'quantize_op_count': 5,  # 2 input, 3 resize.
      'dequantize_op_count': 1,
      'fp_op_count': 0,
      'int_op_count': 17,
  }


@srq_test_case
class ConvBnRelu(nn.Module):
  """A single Conv + BN + Relu block."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(
        features=8,
        kernel_size=(3, 1),
        strides=(1, 1),
        use_bias=True,
    )(x)
    x = nn.BatchNorm(use_running_average=True)(x)
    x = nn.relu6(x)
    return x

  def create_input(self):
    return jax.random.uniform(jax.random.key(0), (1, 100, 1, 16), jnp.float32)

  expected_quant_stats_keys = {
      'Conv_0/conv_general_dilated0_lhs',
      'final_output0',
  }

  # Everything can be fused into a single op.
  expected_ops_summary = {
      'quantize_op_count': 1,
      'dequantize_op_count': 1,
      'fp_op_count': 0,
      'int_op_count': 1,
  }


@srq_test_case
class RepeatNegative(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=2, use_bias=False)(x)
    return -nn.Dense(features=5, use_bias=False)(x.repeat(3, axis=-1))

  def create_input(self):
    return jax.random.uniform(jax.random.key(0), (16, 10), jnp.float32, -1)


class VAE(nn.Module):

  def setup(self):
    self.encoder = nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1))
    self.decoder = nn.Conv(features=3, kernel_size=(3, 3), strides=(1, 1))

  def encode(self, x):
    return self.encoder(x)

  def decode(self, x):
    return self.decoder(x)

  def __call__(self, x):
    return self.decode(self.encode(x))

  def create_input(self):
    return jax.random.uniform(jax.random.key(0), (1, 100, 100, 3), jnp.float32)


class OdmlTest(parameterized.TestCase):

  @parameterized.named_parameters(*srq_test_cases)
  def test_srq(self, model: nn.Module):
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]
    additional_provider_args = getattr(model, 'additional_provider_args', {})
    qat_provider = odml.OdmlQatProvider(rules, **additional_provider_args)
    model_input = model.create_input()
    qat_model = qwix_model.quantize_model(model, qat_provider)
    qat_variables = qat_model.init(jax.random.key(0), model_input)

    # Make biases non-zero.
    qat_variables['params'] = jax.tree.map(
        lambda x: x + jax.random.uniform(jax.random.key(1), x.shape) * 0.001,
        qat_variables['params'],
    )
    # Run the forward pass once to get the quant_stats and batch stats.
    _, new_vars = qat_model.apply(
        qat_variables, model_input, mutable=['quant_stats', 'batch_stats']
    )
    qat_variables.update(new_vars)
    logging.info('quant_stats: %s', qat_variables['quant_stats'])

    if hasattr(model, 'expected_quant_stats_keys'):
      with self.subTest('quant_stats_keys'):
        quant_stats_keys = flax.traverse_util.flatten_dict(
            qat_variables['quant_stats']
        )
        self.assertEqual(
            set('/'.join(kp[:-1]) for kp in quant_stats_keys),
            set(model.expected_quant_stats_keys),
        )

    if hasattr(model, 'expected_quant_stats_values'):
      with self.subTest('quant_stats_values'):
        quant_stats_values = flax.traverse_util.flatten_dict(
            qat_variables['quant_stats']
        )
        quant_stats_values = {
            '/'.join(kp): v for kp, v in quant_stats_values.items()
        }
        for key, value in model.expected_quant_stats_values.items():
          self.assertIn(key, quant_stats_values)
          self.assertAlmostEqual(value, quant_stats_values[key], msg=key)

    conversion_provider = odml.OdmlConversionProvider(
        rules,
        qat_variables['params'],
        qat_variables['quant_stats'],
        **additional_provider_args,
    )
    odml_model = qwix_model.quantize_model(model, conversion_provider)

    # Compare the results between ODML and QAT.
    qat_result = jax.jit(qat_model.apply)(qat_variables, model_input)
    odml_result = jax.jit(odml_model.apply)(qat_variables, model_input)
    print_diff('qat vs odml', qat_result, odml_result)

    # Convert, compare and export the ODML model.
    edge_model = ai_edge_jax.convert(
        odml_model.apply,
        qat_variables,
        (model_input,),
        _litert_converter_flags={
            '_experimental_strict_qdq': True,
        },
    )
    self._save_edge_model(edge_model)

    if hasattr(model, 'expected_ops_summary'):
      with self.subTest('ops_summary'):
        interpreter = edge_model._interpreter_builder()
        ops_summary = self._summarize_ops_details(
            interpreter._get_ops_details()
        )
        self.assertDictContainsSubset(model.expected_ops_summary, ops_summary)

    edge_result = edge_model(model_input)

    print_diff('odml vs edge', odml_result, edge_result)
    print_diff('qat vs edge', qat_result, edge_result)

  def _save_edge_model(self, edge_model):
    """Save the edge model to test outputs."""
    output_dir = absltest.get_default_test_tmpdir()
    output_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', output_dir)
    export_path = f'{output_dir}/{self._testMethodName}.tflite'
    logging.info('Exporting to %s', export_path)
    edge_model.export(export_path)

  def _summarize_ops_details(self, ops_details):
    """Summarize the ops details."""
    summary = {
        'quantize_op_count': 0,
        'dequantize_op_count': 0,
        'fp_op_count': 0,  # op with any fp input/output
        'int_op_count': 0,
    }
    for op in ops_details:
      # op detail looks like {
      #     'index': 3,
      #     'op_name': 'CONV_2D',
      #     'inputs': array([32, 33, 31], dtype = int32),
      #     'outputs': array([34], dtype = int32),
      #     'operand_types': [np.int8, np.int8, np.int32],
      #     'result_types': [np.int8]
      # }
      if op['op_name'] == 'DELEGATE':
        continue
      elif op['op_name'] == 'QUANTIZE':
        summary['quantize_op_count'] += 1
      elif op['op_name'] == 'DEQUANTIZE':
        summary['dequantize_op_count'] += 1
      elif np.float32 in set(op['operand_types'] + op['result_types']):
        summary['fp_op_count'] += 1
      else:
        summary['int_op_count'] += 1
    return summary

  @parameterized.named_parameters(
      dict(
          testcase_name='CNN',
          model=CNN(),
      ),
      # TODO(b/441761069): this triggers an undefined-behavior error.
      # dict(
      #     testcase_name='Transformer',
      #     model=Transformer(),
      # ),
      dict(
          testcase_name='UNet',
          model=UNet(),
      ),
  )
  def test_weight_only(self, model: nn.Module):
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
        ),
    ]
    inputs = model.create_input()
    variables = model.init(jax.random.key(0), inputs)
    # Make biases non-zero.
    variables['params'] = jax.tree.map(
        lambda x: x + jax.random.uniform(jax.random.key(1), x.shape) * 0.001,
        variables['params'],
    )
    # Make batch stats non-zero.
    if 'batch_stats' in variables:
      variables['batch_stats'] = jax.tree.map(
          lambda x: x + jax.random.uniform(jax.random.key(2), x.shape),
          variables['batch_stats'],
      )

    additional_provider_args = getattr(model, 'additional_provider_args', {})
    conversion_provider = odml.OdmlConversionProvider(
        rules, variables['params'], {}, **additional_provider_args
    )
    odml_model = qwix_model.quantize_model(model, conversion_provider)

    # Compare the results between FP and QAT.
    qat_result = jax.jit(model.apply)(variables, inputs)
    odml_result = jax.jit(odml_model.apply)(variables, inputs)
    print_diff('fp vs odml', qat_result, odml_result)

    # Convert, compare and export the ODML model.
    edge_model = ai_edge_jax.convert(
        odml_model.apply,
        variables,
        (inputs,),
        _litert_converter_flags={'_experimental_strict_qdq': True},
    )
    self._save_edge_model(edge_model)

    edge_result = edge_model(inputs)
    print_diff('odml vs edge', odml_result, edge_result)
    print_diff('qat vs edge', qat_result, edge_result)

  def test_partial_quantization(self):
    rules = [
        # Don't quantize silu.
        qconfig.QuantizationRule(
            op_names=['silu'],
            act_qtype=None,
        ),
        # Quantize everything else.
        qconfig.QuantizationRule(
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]
    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(GroupNormSilu(), qat_provider)
    model_input = qat_model.create_input()
    qat_variables = qat_model.init(jax.random.key(0), model_input)
    self.assertEqual(
        set(qat_variables['quant_stats'].keys()),
        {
            'norm_op0',
            # silu0 is here because norm_op0 needs to quantize its output.
            'silu0',
            # but there's no silu0_sigmoid or final_output0.
        },
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='first_encoder_conv',
          module_path='Conv_0',
      ),
      dict(
          testcase_name='last_model_conv',
          module_path='Conv_9',
      ),
      dict(
          testcase_name='first_and_last_convs',
          module_path='Conv_0|Conv_9',
      ),
  )
  def test_partial_quantization_unet_succeeds(self, module_path: str):
    rules = [
        qconfig.QuantizationRule(
            module_path=module_path,
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]
    model = UNet()
    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    model_input = model.create_input()
    qat_variables = qat_model.init(jax.random.key(0), model_input)

    # Run the forward pass once to get the quant_stats and batch stats.
    _, new_vars = qat_model.apply(
        qat_variables, model_input, mutable=['quant_stats', 'batch_stats']
    )
    qat_variables.update(new_vars)
    logging.info(
        'quant_stats for partial quantization unet: %s',
        qat_variables['quant_stats'],
    )

    conversion_provider = odml.OdmlConversionProvider(
        rules,
        qat_variables['params'],
        qat_variables['quant_stats'],
    )
    odml_model = qwix_model.quantize_model(model, conversion_provider)
    ai_edge_jax.convert(
        odml_model.apply,
        qat_variables,
        (model_input,),
        _litert_converter_flags={
            '_experimental_strict_qdq': True,
        },
    )

  @parameterized.named_parameters(*drq_test_cases)
  def test_drq(self, model: nn.Module):
    rules = [
        qconfig.QuantizationRule(
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            # DRQ is only supported in dot_general, conv, and einsum and we
            # shouldn't need to specify op_names.
            act_static_scale=False,
        ),
    ]
    qat_provider = odml.OdmlQatProvider(rules)
    model_input = model.create_input()
    qat_model = qwix_model.quantize_model(model, qat_provider)
    variables = qat_model.init(jax.random.key(0), model_input)
    self.assertNotIn('quant_stats', variables)

    conversion_provider = odml.OdmlConversionProvider(
        rules, variables['params'], {}
    )
    odml_model = qwix_model.quantize_model(model, conversion_provider)

    # Convert, compare and export the ODML model.
    edge_model = ai_edge_jax.convert(
        odml_model.apply,
        variables,
        (model_input,),
        _litert_converter_flags={'_experimental_strict_qdq': True},
    )
    self._save_edge_model(edge_model)

    if hasattr(model, 'drq_expected_ops_summary'):
      with self.subTest('ops_summary'):
        interpreter = edge_model._interpreter_builder()
        ops_summary = self._summarize_ops_details(
            interpreter._get_ops_details()
        )
        self.assertDictContainsSubset(
            model.drq_expected_ops_summary, ops_summary
        )

    if self._testMethodName == 'test_drq_Transformer':
      return  # TODO(b/441761069): this triggers an XNNPack RuntimeError.

    edge_result = edge_model(model_input)
    fp_result = jax.jit(model.apply)(variables, model_input)
    qat_result = jax.jit(qat_model.apply)(variables, model_input)

    print_diff('fp vs qat', fp_result, qat_result)
    fp_mae, _ = print_diff('edge vs fp', edge_result, fp_result)
    qat_mae, _ = print_diff('edge vs qat', edge_result, qat_result)
    self.assertLess(qat_mae, fp_mae)
    self.assertLess(qat_mae, 0.0003)

  def test_vae_separate_export(self):
    rules = [
        qconfig.QuantizationRule(
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]
    model = VAE()
    qat_model = qwix_model.quantize_model(
        model, odml.OdmlQatProvider(rules), methods=['encode', 'decode']
    )
    # During training, __call__ method is invoked.
    model_input = model.create_input()
    variables = qat_model.init(jax.random.key(0), model_input)
    _, variables = qat_model.apply(variables, model_input, mutable=True)
    self.assertIn('encode_output0', variables['quant_stats'])
    self.assertIn('decode_output0', variables['quant_stats'])

    # During export, encode and decode methods are exported separately.
    conversion_provider = odml.OdmlConversionProvider(
        rules, variables['params'], variables['quant_stats']
    )
    odml_model = qwix_model.quantize_model(
        model, conversion_provider, methods=['encode', 'decode']
    )
    # Ensure that they can be called separately.
    encoded = odml_model.apply(variables, model_input, method='encode')
    odml_model.apply(variables, encoded, method='decode')

  def test_nnx(self):
    class NnxModel(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(16, 64, use_bias=False, rngs=rngs)
        self.linear2 = nnx.Linear(64, 16, use_bias=False, rngs=rngs)

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

    conversion_provider = odml.OdmlConversionProvider(
        rules,
        nnx.to_pure_dict(nnx.state(qat_model, nnx.Param)),
        nnx.to_pure_dict(nnx.state(qat_model, flax_util.QuantStat)),
    )
    conversion_model = qwix_model.quantize_model(
        model, conversion_provider, model_input
    )
    conversion_res = conversion_model(model_input)
    self.assertTrue(jnp.allclose(qat_res, conversion_res))

    graphdef, state = nnx.split(conversion_model)
    edge_model = ai_edge_jax.convert(
        lambda state, *args: nnx.merge(graphdef, state)(*args),
        state,
        (model_input,),
        _litert_converter_flags={'_experimental_strict_qdq': True},
    )
    self._save_edge_model(edge_model)

    edge_res = edge_model(model_input)
    print_diff('qat vs edge', qat_res, edge_res)


def print_diff(name, x, y):
  abs_diff = jnp.abs(x - y)
  rel_diff = jnp.nan_to_num(abs_diff / jnp.maximum(jnp.abs(x), jnp.abs(y)))
  abs_diff = abs_diff.mean()
  rel_diff = rel_diff.mean()
  logging.info('%s: %f, %f', name, abs_diff, rel_diff)
  return abs_diff, rel_diff


if __name__ == '__main__':
  absltest.main()
