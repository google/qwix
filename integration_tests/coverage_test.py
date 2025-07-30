# Copyright 2025 Google LLC
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

"""Test the basic functionality across multiple models and providers."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
from jax import numpy as jnp
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.providers import ptq
from qwix._src.providers import qt


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

  def create_inputs(self):
    return jax.random.uniform(jax.random.key(0), (1, 28, 28, 1), jnp.float32)


class Transformer(nn.Module):
  """A simple Transformer model."""

  num_layers: int = 2
  num_heads: int = 4
  qkv_features: int = 8
  hidden_dim: int = 32
  vocab_size: int = 100
  embedding_dim: int = 16

  @nn.compact
  def __call__(self, input_ids):
    x = nn.Embed(self.vocab_size, self.embedding_dim)(input_ids)
    for _ in range(self.num_layers):
      x += nn.MultiHeadAttention(
          self.num_heads, qkv_features=self.qkv_features
      )(x)
      residual = x
      x = nn.Dense(self.hidden_dim)(x)
      x = nn.relu(x)
      x = nn.Dense(self.embedding_dim)(x)
      x = residual + x
    x = x.mean(axis=1)
    x = nn.Dense(self.vocab_size)(x)
    return x

  def create_inputs(self):
    return jax.random.randint(
        jax.random.key(0), (1, 100), minval=0, maxval=100, dtype=jnp.int32
    )


class CoverageTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # TODO(jiwonshin): Re-enable once conv_general is implemented.
      # dict(
      #     testcase_name='cnn_srq',
      #     model=CNN(),
      #     rule=qconfig.QuantizationRule(
      #         module_path=r'.*',
      #         weight_qtype=jnp.int8,
      #         act_qtype=jnp.int8,
      #         act_static_scale=True,
      #     ),
      # ),
      dict(
          testcase_name='transformer_weight_only',
          model=Transformer(),
          rule=qconfig.QuantizationRule(
              module_path=r'.*',
              weight_qtype=jnp.int8,
          ),
      ),
  )
  def test_coverage(
      self, model: Transformer | CNN, rule: qconfig.QuantizationRule
  ):
    q_rules = [rule]
    inputs = model.create_inputs()

    # Randomly initialize the params.
    qt_model = qwix_model.quantize_model(model, qt.QtProvider(q_rules))
    qt_variables = qt_model.init(jax.random.key(0), inputs)
    # Run the forward pass once to get the quant_stats.
    _, new_vars = qt_model.apply(qt_variables, inputs, mutable='quant_stats')
    params = qt_variables['params']
    # quant_stats is not present in weight-only mode
    quant_stats = new_vars.get('quant_stats', {})

    ptq_model = qwix_model.quantize_model(model, ptq.PtqProvider(q_rules))
    ptq_abstract_params = jax.eval_shape(
        ptq_model.init, jax.random.key(0), inputs
    )['params']
    ptq_params = ptq.quantize_params(params, ptq_abstract_params, quant_stats)
    ptq_model.apply({'params': ptq_params}, inputs)


if __name__ == '__main__':
  absltest.main()
