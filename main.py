import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

import numpy as np
from tqdm import tqdm

import tensorflow_datasets as tfds

class dotdict(dict):
	"""
	Dot notation to access dictionary attributes
	"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)

        return x

@jax.jit
def apply_model(state, images, labels):
	"""Computes gradients, loss and accuracy for a single batch."""
	def loss_fn(params):
		logits = state.apply_fn({'params': params}, images)
		one_hot = jax.nn.one_hot(labels, 10)
		loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
		return loss, logits

	grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
	(loss, logits), grads = grad_fn(state.params)
	accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
	return grads, loss, accuracy

@jax.jit
def update_model(state, grads):
	return state.apply_gradients(grads=grads)

@jax.jit
def update_model(state, grads):
	return state.apply_gradients(grads=grads)

def get_datasets():
	"""Load MNIST train and test datasets into memory."""
	ds_builder = tfds.builder('mnist')
	ds_builder.download_and_prepare()
	train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
	test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
	train_ds['image'] = jnp.float32(train_ds['image']) / 255.
	test_ds['image'] = jnp.float32(test_ds['image']) / 255.
	return train_ds, test_ds

def train_epoch(state, train_ds, batch_size, rng):
	"""Train for a single epoch."""
	train_ds_size = len(train_ds['image'])
	steps_per_epoch = train_ds_size // batch_size

	perms = jax.random.permutation(rng, len(train_ds['image']))
	perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
	perms = perms.reshape((steps_per_epoch, batch_size))

	epoch_loss = []
	epoch_accuracy = []

	for perm in perms:
		batch_images = train_ds['image'][perm, ...]
		batch_labels = train_ds['label'][perm, ...]
		grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
		state = update_model(state, grads)
		epoch_loss.append(loss)
		epoch_accuracy.append(accuracy)
	train_loss = np.mean(epoch_loss)
	train_accuracy = np.mean(epoch_accuracy)
	return state, train_loss, train_accuracy

def create_train_state(rng, config):
	"""Creates initial `TrainState`."""
	model = Model()
	params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params']
	tx = optax.sgd(config.learning_rate, config.momentum)
	return train_state.TrainState.create(
		apply_fn=model.apply, params=params, tx=tx)
    
def train_and_evaluate(config: dotdict) -> train_state.TrainState:
	"""Execute model training and evaluation loop.
	Args:
	config: Hyperparameter configuration for training and evaluation.
	workdir: Directory where the tensorboard summaries are written to.
	Returns:
	The train state (which includes the `.params`).
	"""
	train_ds, test_ds = get_datasets()
	rng = jax.random.PRNGKey(0)

	rng, init_rng = jax.random.split(rng)
	state = create_train_state(init_rng, config)

	with tqdm(range(1, config.num_epochs + 1), desc='Training') as epochs:
		for epoch in epochs:
			rng, input_rng = jax.random.split(rng)
			state, train_loss, train_accuracy = train_epoch(state, train_ds,
															config.batch_size,
															input_rng)
			_, test_loss, test_accuracy = apply_model(state, test_ds['image'],
														test_ds['label'])

			epochs.set_postfix(
				train_loss = train_loss,
				train_acc = train_accuracy,
				test_loss = test_loss,
				test_acc = test_accuracy
			)
	return state

def main():
	train_and_evaluate(config = dotdict({
		"num_epochs" : 5,
		"learning_rate" : 1e-4,
		"momentum" : 0.95,
		'batch_size' : 64
	}))

if __name__ == '__main__':
	main()