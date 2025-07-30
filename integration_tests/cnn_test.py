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
"""Test QT on Flax MNIST example.

The majority of this file is copied from flax/examples/mnist/train.py.
"""

import functools
from typing import Any, Mapping
import unittest

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import linen as nn
from flax.training import train_state
import jax
from jax import numpy as jnp
import ml_collections
import numpy as np
import optax
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.providers import ptq
from qwix._src.providers import qt
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
    cnn: CNN, config: ml_collections.ConfigDict
) -> tuple[float, TrainStateWithQuantStats]:
  """Execute model training and evaluation loop."""
  train_ds, test_ds = get_datasets()
  rng = jax.random.key(0)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(cnn, init_rng, config)

  train_loss = 0.0
  for epoch in range(1, config.num_epochs + 1):
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


@unittest.skip('Disabling these tests until we add quantized convolution.')
class CnnTest(parameterized.TestCase):

  def test_drq(self):
    cnn = CNN()
    q_rules = [
        qconfig.QuantizationRule(
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
        ),
    ]
    qtcnn = qwix_model.quantize_model(cnn, qt.QtProvider(q_rules))

    config = ml_collections.ConfigDict()
    config.learning_rate = 0.1
    config.momentum = 0.9
    config.batch_size = 128
    config.num_epochs = 10

    # QT should generate slightly worse model compared to FP.
    fp_train_loss, fp_state = train_and_evaluate(cnn, config)
    qttrain_loss, qtstate = train_and_evaluate(qtcnn, config)
    self.assertLess(fp_train_loss, qttrain_loss)
    self.assertLess(qttrain_loss, 0.02)
    jax.tree.map(
        lambda x, y: self.assertEqual((x.shape, x.dtype), (y.shape, y.dtype)),
        fp_state.params,
        qtstate.params,
    )

    # PTQ should produce same result as qt.
    ptq_cnn = qwix_model.quantize_model(cnn, ptq.PtqProvider(q_rules))
    qttest_accuracy = evaluate(qtcnn, {'params': qtstate.params})
    ptq_abstract_params = jax.eval_shape(
        ptq_cnn.init, jax.random.key(0), jnp.ones([1, 28, 28, 1])
    )['params']
    ptq_params = ptq.quantize_params(qtstate.params, ptq_abstract_params)
    ptq_test_accuracy = evaluate(ptq_cnn, {'params': ptq_params})
    self.assertAlmostEqual(qttest_accuracy, ptq_test_accuracy, places=3)

    # PTQ can also be used directly on FP model.
    fp_ptq_params = ptq.quantize_params(fp_state.params, ptq_abstract_params)
    fp_ptq_test_accuracy = evaluate(ptq_cnn, {'params': fp_ptq_params})
    # It should produce a worse result.
    self.assertLess(fp_ptq_test_accuracy, ptq_test_accuracy)
    # ..but still relatively good.
    self.assertGreater(fp_ptq_test_accuracy, 0.98)

  def test_srq(self):
    cnn = CNN()
    q_rules = [
        qconfig.QuantizationRule(
            module_path=r'Conv_\d+',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            act_static_scale=True,
        ),
        qconfig.QuantizationRule(
            module_path=r'Dense_\d+',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            act_static_scale=True,
        ),
    ]
    qtcnn = qwix_model.quantize_model(cnn, qt.QtProvider(q_rules))

    config = ml_collections.ConfigDict()
    config.learning_rate = 0.1
    config.momentum = 0.9
    config.batch_size = 128
    config.num_epochs = 10

    # QT with SRQ.
    _, qtstate = train_and_evaluate(qtcnn, config)
    qttest_accurary = evaluate(
        qtcnn,
        {'params': qtstate.params, 'quant_stats': qtstate.quant_stats},
    )

    # PTQ with SRQ.
    ptq_cnn = qwix_model.quantize_model(cnn, ptq.PtqProvider(q_rules))
    ptq_abstract_params = jax.eval_shape(
        ptq_cnn.init, jax.random.key(0), jnp.ones([1, 28, 28, 1])
    )['params']
    ptq_params = ptq.quantize_params(
        qtstate.params, ptq_abstract_params, qtstate.quant_stats
    )
    ptq_test_accuracy = evaluate(ptq_cnn, {'params': ptq_params})

    self.assertAlmostEqual(qttest_accurary, ptq_test_accuracy, places=3)
    self.assertGreater(ptq_test_accuracy, 0.98)


if __name__ == '__main__':
  absltest.main()
