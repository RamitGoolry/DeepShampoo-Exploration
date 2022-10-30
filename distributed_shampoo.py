from typing import Any, Union, Optional, Callable

import jax
import jax.numpy as jnp
import optax

from jax import jit

from optax._src import base
from icecream import ic

ScalarOrSchedule = Union[float, base.Schedule]

def inv_4th_root(M):
    M = jax.device_get(M)
    sqrtm_cpu = jax.jit(lambda m : jax.scipy.linalg.sqrtm(m), backend='cpu')
    M = sqrtm_cpu(M)
    M = sqrtm_cpu(M)
    # M = jax.device_put(M, gpus[0])
    # M = jnp.linalg.matrix_power(M, -1) # -1/4

    return M

inv_4th_root = jit(inv_4th_root, backend='cpu')

def distributed_shampoo(learning_rate : jnp.float32, beta1 : jnp.float32 = 0.9, beta2 : jnp.float32 = 0.99,
                        eps : jnp.float32 = 0.0001):
    """
    Optax implementation of Distributed Shampoo Optimizer

    Pseudocode:
        parameters learning_rate, beta_1, beta_2 (at timestep t)

        Recieve stochastic gradients G_t for each layer

        if t % tau_2 == 0:
            if beta_2 < 1:
                L_t = beta_2 * L_{t-tau_2} + (1 - beta_2) * G_t * G_t.T
                R_t = beta_2 * R_{t-tau_2} + (1 - beta_2) * G_t.T * G_t
            else:
                L_t = L_{t - tau_2} + G_t * G_t.T
                R_t = R_{t - tau_2} + G_t.T * G_t

        D_t = D_{t-1} + G_t \cdot G_t      (\cdot is TODO)
        M_t = beta_1 * M_{t-1} + (1 - beta_1) * D_t^{\odot - 1/2} \cdot G_t (\odot is TODO)

        if t % tau_1 == 0:
            Gather preconditioners L_{t - tau_1}^{-1/4} and R_{t - tau_1}^{-1/4} from CPUs
            Send L_t and R_t to CPU host to compute L_t^{-1/4} and R_t^{-1/4}

        if t > tau_1:
            P_t = beta_1 * P_{t-1} + (1 - beta_1) * L_t^{-1/4} G_t R_t^{-1/4}
            lr_t = learning_rate * ||M_t||_F / ||P_t||_F   (||p||_F is the frobenius norm)
            W_{t+1} = W_t - lr_t * P_t
        else:
            lr_t = learning_rate
            W_{t+1} = W_t - lr_t * M_t
    """

    # Starting with simpler update rule that does not work with tau_1 tau_2, etc
    # Update rule is simple:
    #  W_{t+1} = W_t - learning_rate * L_t^{-1/4} G_t R_t^{-1/4}

    def init_fn(params):
        L = jax.tree_map(
            lambda p : eps * jnp.ones(shape = (p.shape[0], p.shape[0])), 
        params)
        R = jax.tree_map(
            lambda p : eps * jnp.ones(shape = (p.shape[1], p.shape[1])) if len(p.shape) > 1 else eps * jnp.ones(shape=(1, 1))
        , params)

        return {'L' : L, 'R' : R} # Optimizer state

    def update_fn(grads, state, params=None):
        L, R = state['L'], state['R']

        # Update preconditioner matrices
        L = jax.tree_map(lambda l, g: jnp.asarray(l + jnp.dot(g, g.T)).astype(l.dtype), L, grads)
        R = jax.tree_map(lambda r, g: jnp.asarray(r + jnp.dot(g.T, g)).astype(r.dtype), R, grads)

        new_state = { 'L' : L, 'R' : R }

        cpus = jax.devices("cpu")
        gpus = jax.devices("gpu")

        ic(cpus)
        ic(gpus)

        L_inv4 = jax.tree_map(lambda l : inv_4th_root(l), L)
        R_inv4 = jax.tree_map(lambda r : inv_4th_root(r), R)

        @jax.jit
        def update_mul(l, g, r):
            if len(g.shape) == 1:
                # Expand dimensions
                g = jnp.expand_dims(g, 1)

            return jnp.dot(l, jnp.dot(g, r))

        updates = jax.tree_map(lambda l, g, r: -1 * learning_rate * update_mul(l, g, r), L_inv4, grads, R_inv4)

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
