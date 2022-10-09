import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

class Model(nn.Module):
    @nn.compact

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        x = nn.softmax(x)

        return x

