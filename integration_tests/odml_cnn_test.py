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
"""Test Qwix ODML on Flax MNIST example."""

import functools
import os
from typing import Any, Mapping

from absl import logging
from absl.testing import absltest
import ai_edge_jax
import flax
from flax import linen as nn
from flax.training import train_state
import jax
from jax import numpy as jnp
import ml_collections
import numpy as np
import optax
from qwix import model as qwix_model
from qwix import odml
from qwix import qconfig
import tensorflow_datasets as tfds


class TrainStateWithQuantStats(train_state.TrainState):
  quant_stats: Any = flax.core.FrozenDict()


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


@jax.jit
def apply_model(state: TrainStateWithQuantStats, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(params):
    logits, new_vars = state.apply_fn(
        {'params': params, 'quant_stats': state.quant_stats},
        images,
        mutable='quant_stats',
    )
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, (logits, new_vars['quant_stats'])

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (logits, quant_stats)), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, quant_stats, loss, accuracy


@jax.jit
def update_model(
    state: TrainStateWithQuantStats, grads, quant_stats
) -> TrainStateWithQuantStats:
  return state.apply_gradients(grads=grads).replace(quant_stats=quant_stats)


def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []

  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]
    grads, quant_stats, loss, accuracy = apply_model(
        state, batch_images, batch_labels
    )
    state = update_model(state, grads, quant_stats)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy


@functools.cache
def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
  return train_ds, test_ds


def create_train_state(cnn: CNN, rng, config) -> TrainStateWithQuantStats:
  """Creates initial `TrainState`."""
  variables = cnn.init(rng, jnp.ones([1, 28, 28, 1]))
  tx = optax.sgd(config.learning_rate, config.momentum)
  return TrainStateWithQuantStats.create(apply_fn=cnn.apply, tx=tx, **variables)


def train_and_evaluate(
    cnn: CNN, qat_cnn: CNN, config: ml_collections.ConfigDict
) -> tuple[float, TrainStateWithQuantStats]:
  """Execute model training and evaluation loop."""
  train_ds, test_ds = get_datasets()
  rng = jax.random.key(0)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(qat_cnn, init_rng, config)

  train_loss = 0.0
  for epoch in range(1, config.num_epochs + 1):
    if epoch < config.qat_after_epoch:
      state = state.replace(apply_fn=cnn.apply)
    else:
      state = state.replace(apply_fn=qat_cnn.apply)
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(
        state, train_ds, config.batch_size, input_rng
    )
    _, _, test_loss, test_accuracy = apply_model(
        state, test_ds['image'], test_ds['label']
    )

    logging.info(
        'epoch:% 3d, train_loss: %.6f, train_accuracy: %.4f, test_loss: %.6f,'
        ' test_accuracy: %.4f',
        epoch,
        train_loss,
        train_accuracy * 100,
        test_loss,
        test_accuracy * 100,
    )

  return train_loss, state


def evaluate(cnn: CNN, variables: Mapping[str, Any]):
  """Only evaluate the model."""
  test_ds = get_datasets()[1]
  logits = cnn.apply(variables, test_ds['image'])
  accuracy = jnp.mean(jnp.argmax(logits, -1) == test_ds['label'])
  logging.info('evaluation accuracy: %.4f', accuracy * 100)
  return accuracy


class OdmlCnnTest(absltest.TestCase):

  def test_cnn_srq(self):
    cnn = CNN()
    q_rules = [
        qconfig.QuantizationRule(
            module_path=r'.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]
    qat_cnn = qwix_model.quantize_model(cnn, odml.OdmlQatProvider(q_rules))

    config = ml_collections.ConfigDict()
    config.learning_rate = 0.1
    config.momentum = 0.9
    config.batch_size = 128
    config.num_epochs = 10
    config.qat_after_epoch = 5

    # QAT with SRQ.
    _, qat_state = train_and_evaluate(cnn, qat_cnn, config)
    qat_test_accurary = evaluate(
        qat_cnn,
        {'params': qat_state.params, 'quant_stats': qat_state.quant_stats},
    )

    # ODML conversion.
    odml_conversion_provider = odml.OdmlConversionProvider(
        q_rules, qat_state.params, qat_state.quant_stats
    )
    conversion_cnn = qwix_model.quantize_model(cnn, odml_conversion_provider)
    conversion_test_accuracy = evaluate(
        conversion_cnn, {'params': qat_state.params}
    )

    # Convert and evaluate the ODML model.
    test_ds = get_datasets()[1]
    odml_model = ai_edge_jax.convert(
        conversion_cnn.apply,
        {'params': qat_state.params},
        (test_ds['image'],),
        _litert_converter_flags={'_experimental_strict_qdq': True},
    )
    output_dir = absltest.get_default_test_tmpdir()
    output_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', output_dir)
    odml_model.export(output_dir + '/qwix_cnn.tflite')
    logits = odml_model(test_ds['image'])
    odml_accuracy = jnp.mean(jnp.argmax(logits, -1) == test_ds['label'])
    logging.info('odml accuracy: %.4f', odml_accuracy * 100)

    self.assertAlmostEqual(
        qat_test_accurary, conversion_test_accuracy, places=3
    )
    self.assertAlmostEqual(qat_test_accurary, odml_accuracy, places=3)
    self.assertGreater(odml_accuracy, 0.98)


if __name__ == '__main__':
  absltest.main()
