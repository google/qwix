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

import functools
from typing import Sequence

from absl import logging
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from qwix import flax_util
from qwix import model as qwix_model
from qwix import ptq
from qwix import qat
from qwix import qconfig
import tensorflow_datasets as tfds


class Loss(nnx.Variable):
  pass


class Encoder(nnx.Module):

  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
    self.linear_mean = nnx.Linear(dmid, dout, rngs=rngs)
    self.linear_std = nnx.Linear(dmid, dout, rngs=rngs)
    self.rngs = rngs

  def __call__(self, x: jax.Array) -> jax.Array:
    x = x.reshape((x.shape[0], -1))  # flatten
    x = self.linear1(x)
    x = jax.nn.relu(x)

    mean = self.linear_mean(x)
    std = jnp.exp(self.linear_std(x))

    self.kl_loss = Loss(
        jnp.mean(
            0.5 * jnp.mean(-jnp.log(std**2) - 1.0 + std**2 + mean**2, axis=-1)
        )
    )
    key = self.rngs.noise()
    z = mean + std * jax.random.normal(key, mean.shape)
    return z


class Decoder(nnx.Module):

  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
    self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, z: jax.Array) -> jax.Array:
    z = self.linear1(z)
    z = jax.nn.relu(z)
    logits = self.linear2(z)
    return logits


class VAE(nnx.Module):

  def __init__(
      self,
      din: int,
      hidden_size: int,
      latent_size: int,
      output_shape: Sequence[int],
      *,
      rngs: nnx.Rngs,
  ):
    self.output_shape = output_shape
    self.encoder = Encoder(din, hidden_size, latent_size, rngs=rngs)
    self.decoder = Decoder(
        latent_size, hidden_size, int(np.prod(output_shape)), rngs=rngs
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    z = self.encoder(x)
    logits = self.decoder(z)
    logits = jnp.reshape(logits, (-1, *self.output_shape))
    return logits

  def generate(self, z):
    logits = self.decoder(z)
    logits = jnp.reshape(logits, (-1, *self.output_shape))
    return nnx.sigmoid(logits)


def loss_fn(model: VAE, x: jax.Array):
  logits = model(x)
  losses = nnx.pop(model, Loss)
  kl_loss = sum(jax.tree_util.tree_leaves(losses), 0.0)
  reconstruction_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, x))
  loss = reconstruction_loss + 0.1 * kl_loss
  return loss


@nnx.jit
def train_step(model: VAE, optimizer: nnx.Optimizer, x: jax.Array):
  loss, grads = nnx.value_and_grad(loss_fn)(model, x)
  optimizer.update(grads)

  return loss


@nnx.jit
def eval_step(model: VAE, x: jax.Array):
  loss = loss_fn(model, x)
  return loss


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


def train_and_evaluate(model: VAE, config: ml_collections.ConfigDict):
  np.random.seed(42)

  train_ds, test_ds = get_datasets()
  # binarise data and remove the channel axis.
  x_train = (train_ds['image'][..., 0] > 0.98).astype(jnp.float32)
  x_test = (test_ds['image'][..., 0] > 0.98).astype(jnp.float32)

  logging.info('X_train: %s %s', x_train.shape, x_train.dtype)
  logging.info('X_test: %s %s', x_test.shape, x_test.dtype)

  optimizer = nnx.Optimizer(model, optax.adam(1e-3))
  train_loss = None
  for epoch in range(config.epochs):
    # Enables quant_stats update for training.
    model.set_attributes(
        disable_quant_stats_update=False, raise_if_not_found=False
    )
    losses = []
    for _ in range(config.steps_per_epoch):
      idxs = np.random.randint(0, len(x_train), size=(config.batch_size,))
      x_batch = x_train[idxs]

      loss = train_step(model, optimizer, x_batch)
      losses.append(np.asarray(loss))
    train_loss = np.mean(losses)

    # Disables quant_stats update for eval.
    model.set_attributes(
        disable_quant_stats_update=True, raise_if_not_found=False
    )
    eval_loss = eval_step(model, x_test)
    logging.info(
        'Epoch %d, train_loss: %f, eval_loss: %f', epoch, train_loss, eval_loss
    )

  return train_loss


def evaluate(model: VAE):
  test_ds = get_datasets()[1]
  x_test = (test_ds['image'][..., 0] > 0.98).astype(jnp.float32)
  # Disables quant_stats update for eval.
  model.set_attributes(
      disable_quant_stats_update=True, raise_if_not_found=False
  )
  eval_loss = eval_step(model, x_test)
  logging.info('eval_loss: %f', eval_loss)
  return eval_loss


class VaeQatTest(absltest.TestCase):

  def test_drq(self):
    batch_size = 64
    latent_size = 32
    image_shape: Sequence[int] = (28, 28)
    vae = VAE(
        din=int(np.prod(image_shape)),
        hidden_size=256,
        latent_size=latent_size,
        output_shape=image_shape,
        rngs=nnx.Rngs(0, noise=1),
    )
    q_rules = [
        qconfig.QuantizationRule(
            module_path=r'encoder.*', weight_qtype=jnp.int8, act_qtype=jnp.int8
        ),
        qconfig.QuantizationRule(
            module_path=r'decoder.*', weight_qtype=jnp.int4, act_qtype=jnp.int4
        ),
    ]
    model_input = jnp.zeros((batch_size, *image_shape))
    qat_vae = qwix_model.quantize_model(
        vae, qat.QatProvider(q_rules), model_input
    )

    config = ml_collections.ConfigDict()
    config.epochs = 20
    config.steps_per_epoch = 100
    config.batch_size = batch_size

    # QAT should generate slightly worse model compared to FP.
    fp_loss = train_and_evaluate(vae, config)
    qat_loss = train_and_evaluate(qat_vae, config)

    fp_params = nnx.variables(vae, nnx.Param)
    qat_params = nnx.variables(qat_vae, nnx.Param)

    self.assertLess(fp_loss, qat_loss)

    jax.tree.map(
        lambda x, y: self.assertEqual((x.shape, x.dtype), (y.shape, y.dtype)),
        fp_params,
        qat_params,
    )

    # PTQ should produce the same result as QAT.
    ptq_vae = qwix_model.quantize_model(
        qat_vae, ptq.PtqProvider(q_rules), model_input
    )
    ptq_loss = evaluate(ptq_vae)
    self.assertAlmostEqual(qat_loss, ptq_loss, delta=0.001)

    # PTQ without QAT should produce worse result.
    fp_ptq_vae = qwix_model.quantize_model(
        vae, ptq.PtqProvider(q_rules), model_input
    )
    fp_ptq_loss = evaluate(fp_ptq_vae)
    self.assertGreater(fp_ptq_loss, ptq_loss)

  def test_srq(self):
    batch_size = 64
    latent_size = 32
    image_shape: Sequence[int] = (28, 28)
    vae = VAE(
        din=int(np.prod(image_shape)),
        hidden_size=256,
        latent_size=latent_size,
        output_shape=image_shape,
        rngs=nnx.Rngs(0, noise=1),
    )
    q_rules = [
        qconfig.QuantizationRule(
            module_path=r'encoder.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            act_static_scale=True,
        ),
        qconfig.QuantizationRule(
            module_path=r'decoder.*',
            weight_qtype=jnp.int4,
            act_qtype=jnp.int4,
            act_static_scale=True,
        ),
    ]
    model_input = jnp.zeros((batch_size, *image_shape))
    qat_vae = qwix_model.quantize_model(
        vae,
        qat.QatProvider(q_rules),
        model_input,
    )

    config = ml_collections.ConfigDict()
    config.epochs = 20
    config.steps_per_epoch = 100
    config.batch_size = batch_size

    # QAT with SRQ.
    fp_loss = train_and_evaluate(vae, config)
    qat_loss = train_and_evaluate(qat_vae, config)

    fp_params = nnx.variables(vae, nnx.Param)
    qat_quant_stats = nnx.variables(qat_vae, flax_util.QuantStat)
    qat_params = nnx.variables(qat_vae, nnx.Param)

    self.assertLess(fp_loss, qat_loss)
    self.assertNotEmpty(qat_quant_stats)
    jax.tree.map(
        lambda x, y: self.assertEqual((x.shape, x.dtype), (y.shape, y.dtype)),
        fp_params,
        qat_params,
    )

    # PTQ with SRQ. Note that quantize_model also converts the params if called
    # with a QAT model.
    ptq_vae = qwix_model.quantize_model(
        qat_vae, ptq.PtqProvider(q_rules), model_input
    )
    ptq_loss = evaluate(ptq_vae)
    self.assertAlmostEqual(qat_loss, ptq_loss, delta=0.001)


if __name__ == '__main__':
  absltest.main()
