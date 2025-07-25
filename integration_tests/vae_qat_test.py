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
import numpy as np
import optax
from qwix import flax_util
from qwix import model as qwix_model
from qwix import ptq
from qwix import qconfig
from qwix import qt
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
def train_step(model: VAE, optimizer: nnx.ModelAndOptimizer, x: jax.Array):
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


def train_and_evaluate(
    model: VAE, *, epochs: int, batch_size: int, rng: jax.Array
):
  train_ds, test_ds = get_datasets()
  # binarise data and remove the channel axis.
  x_train = (train_ds['image'][..., 0] > 0.98).astype(jnp.float32)
  x_test = (test_ds['image'][..., 0] > 0.98).astype(jnp.float32)

  logging.info('X_train: %s %s', x_train.shape, x_train.dtype)
  logging.info('X_test: %s %s', x_test.shape, x_test.dtype)

  steps_per_epoch = x_train.shape[0] // batch_size
  optimizer = nnx.ModelAndOptimizer(model, optax.adam(1e-3))
  train_loss = None
  for epoch in range(epochs):
    rng, input_rng = jax.random.split(rng)
    perms = jax.random.permutation(input_rng, x_train.shape[0])
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Enables quant_stats update for training.
    model.set_attributes(
        disable_quant_stats_update=False, raise_if_not_found=False
    )
    losses = []
    for perm in perms:
      x_batch = x_train[perm, ...]

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
    qt_vae = qwix_model.quantize_model(vae, qt.QtProvider(q_rules), model_input)

    # QT should generate slightly worse model compared to FP.
    fp_loss = train_and_evaluate(
        vae, epochs=5, batch_size=batch_size, rng=jax.random.key(0)
    )
    qt_loss = train_and_evaluate(
        qt_vae, epochs=5, batch_size=batch_size, rng=jax.random.key(0)
    )

    fp_params = nnx.variables(vae, nnx.Param)
    qt_params = nnx.variables(qt_vae, nnx.Param)

    self.assertLess(fp_loss, qt_loss)

    jax.tree.map(
        lambda x, y: self.assertEqual((x.shape, x.dtype), (y.shape, y.dtype)),
        fp_params,
        qt_params,
    )

    # PTQ should produce the same result as qt.
    ptq_vae = qwix_model.quantize_model(
        qt_vae, ptq.PtqProvider(q_rules), model_input
    )
    ptq_loss = evaluate(ptq_vae)
    self.assertAlmostEqual(qt_loss, ptq_loss, delta=0.001)

    # PTQ without QT should produce worse result.
    # Not necessarily always true after only 5 epochs.
    # fp_ptq_vae = qwix_model.quantize_model(
    #     vae, ptq.PtqProvider(q_rules), model_input
    # )
    # fp_ptq_loss = evaluate(fp_ptq_vae)
    # self.assertGreater(fp_ptq_loss, ptq_loss)

  def test_srq(self):
    self.skipTest('Reenable once SRQ is implemented.')
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
    qt_vae = qwix_model.quantize_model(
        vae,
        qt.QtProvider(q_rules),
        model_input,
    )

    # QT with SRQ.
    fp_loss = train_and_evaluate(
        vae, epochs=5, batch_size=batch_size, rng=jax.random.key(0)
    )
    qt_loss = train_and_evaluate(
        qt_vae, epochs=5, batch_size=batch_size, rng=jax.random.key(0)
    )

    fp_params = nnx.variables(vae, nnx.Param)
    qt_quant_stats = nnx.variables(qt_vae, flax_util.QuantStat)
    qt_params = nnx.variables(qt_vae, nnx.Param)

    self.assertLess(fp_loss, qt_loss)
    self.assertNotEmpty(qt_quant_stats)
    jax.tree.map(
        lambda x, y: self.assertEqual((x.shape, x.dtype), (y.shape, y.dtype)),
        fp_params,
        qt_params,
    )

    # PTQ with SRQ. Note that quantize_model also converts the params if called
    # with a QT model.
    ptq_vae = qwix_model.quantize_model(
        qt_vae, ptq.PtqProvider(q_rules), model_input
    )
    ptq_loss = evaluate(ptq_vae)
    self.assertAlmostEqual(qt_loss, ptq_loss, delta=0.001)


if __name__ == '__main__':
  absltest.main()
