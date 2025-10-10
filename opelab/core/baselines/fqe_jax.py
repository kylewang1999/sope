import numpy as np
from typing import Sequence
from tqdm import tqdm

import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit
import optax

from opelab.core.baseline import Baseline
from opelab.core.data import DataType, to_numpy
from opelab.core.mlp import MLP
from opelab.core.policy import Policy

class FQE(Baseline):
    def __init__(self, lr: float = 3e-3, tau=0.0005, layers: Sequence[int] = (500, 500), epochs: int = 100, batch_size: int = 256, verbose: int = 0, seed: int = 0):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.seed = seed
        self.tau = tau
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr))
        self.model = MLP(list(layers) + [1], nn.relu, output_activation=lambda s: s)
        self.processed_data = False

        def predict_w_fn(params, states, actions):
            if actions.ndim == 1:
                actions = actions[:, None]
            xus = jnp.concatenate([states, actions], axis=-1)
            y = self.model.apply(params, xus)
            return y.reshape(-1, 1)

        @jit
        def fqe_loss_fn(params, q1_target_params, q2_target_params, states, actions, rewards, next_states, next_actions, dones, gamma, clip):
            if actions.ndim == 1:
                actions = actions[:, None]
            if next_actions.ndim == 1:
                next_actions = next_actions[:, None]
            rewards = rewards.reshape(-1, 1)
            dones = dones.reshape(-1, 1).astype(jnp.float32)
            current_q = predict_w_fn(params, states, actions)
            next_q1 = jax.lax.stop_gradient(predict_w_fn(q1_target_params, next_states, next_actions))
            next_q2 = jax.lax.stop_gradient(predict_w_fn(q2_target_params, next_states, next_actions))
            next_q = jnp.minimum(next_q1, next_q2)
            target = rewards + gamma * (1.0 - dones) * next_q
            target = jnp.clip(target, -clip, clip)
            loss = jnp.mean(jnp.square(current_q - target))
            return loss

        def train_fn(mlp_state, q1_target_params, q2_target_params, states, actions, rewards, next_states, next_actions, dones, gamma, clip):
            loss, grads = jax.value_and_grad(fqe_loss_fn, argnums=0)(mlp_state.params, q1_target_params, q2_target_params, states, actions, rewards, next_states, next_actions, dones, gamma, clip)
            mlp_state = mlp_state.apply_gradients(grads=grads)
            return mlp_state, loss

        def soft_update(x, y):
            return jax.tree_util.tree_map(lambda a, b: self.tau * b + (1 - self.tau) * a, x, y)

        self.soft_update = jit(soft_update)
        self.train_fn = jit(train_fn)
        self.predict_w_fn = jit(predict_w_fn)
        self.loss_fn = jit(fqe_loss_fn)

    def _to_numpy_writable(self, x):
        return np.array(np.asarray(x), dtype=np.float32, copy=True)

    def train_q_network(self, data, target: Policy, behavior: Policy, gamma: float):
        key = jrandom.PRNGKey(self.seed)
        states, states_un, actions, next_states, next_states_un, rewards, policy_ratio, terminals = data
        use_states = states_un
        use_next_states = next_states_un
        if actions.ndim == 1:
            actions = actions[:, None]
        xus = jnp.concatenate([use_states, actions], axis=-1)
        key, init_key1, init_key2 = jrandom.split(key, 3)
        q1_params = self.model.init(init_key1, xus[:20])
        q1_mlp_state = train_state.TrainState.create(apply_fn=self.model.apply, params=q1_params, tx=self.optimizer)
        q1_target_params = jax.tree_util.tree_map(lambda x: x, q1_params)
        q2_params = self.model.init(init_key2, xus[:20])
        q2_mlp_state = train_state.TrainState.create(apply_fn=self.model.apply, params=q2_params, tx=self.optimizer)
        q2_target_params = jax.tree_util.tree_map(lambda x: x, q2_params)
        max_rew = jnp.max(jnp.abs(rewards))
        clip = jnp.where(gamma < 0.999999, max_rew / (1 - gamma), jnp.inf)
        N = use_states.shape[0]
        with tqdm(range(self.epochs)) as tp:
            for _ in tp:
                key, b_key = jrandom.split(key)
                batch_ordering = jrandom.permutation(b_key, jnp.arange(N))
                q1_epoch_loss = []
                q2_epoch_loss = []
                for start in range(0, N, self.batch_size):
                    batch = batch_ordering[start:start + self.batch_size]
                    batch_states = use_states[batch]
                    batch_actions = actions[batch]
                    batch_rewards = rewards[batch].reshape(-1, 1)
                    batch_next_states = use_next_states[batch]
                    batch_dones = terminals[batch].reshape(-1, 1).astype(jnp.float32)
                    bns_np = self._to_numpy_writable(batch_next_states)
                    bna_np = target.sample(bns_np)
                    batch_next_actions = jnp.asarray(bna_np)
                    if batch_next_actions.ndim == 1:
                        batch_next_actions = batch_next_actions[:, None]
                    q1_mlp_state, q1_loss_val = self.train_fn(q1_mlp_state, q1_target_params, q2_target_params, batch_states, batch_actions, batch_rewards, batch_next_states, batch_next_actions, batch_dones, gamma, clip)
                    q2_mlp_state, q2_loss_val = self.train_fn(q2_mlp_state, q1_target_params, q2_target_params, batch_states, batch_actions, batch_rewards, batch_next_states, batch_next_actions, batch_dones, gamma, clip)
                    q1_target_params = self.soft_update(q1_target_params, q1_mlp_state.params)
                    q2_target_params = self.soft_update(q2_target_params, q2_mlp_state.params)
                    q1_epoch_loss.append(float(q1_loss_val))
                    q2_epoch_loss.append(float(q2_loss_val))
                tp.set_postfix(q1_loss=np.mean(q1_epoch_loss), q2_loss=np.mean(q2_epoch_loss))
        self.q1_params = q1_mlp_state.params
        self.q2_params = q2_mlp_state.params
        return self.q1_params, self.q2_params

    def evaluate(self, data: DataType, target: Policy, behavior: Policy, gamma: float = 0.99, reward_estimator=None) -> float:
        traj_data = data
        if not self.processed_data:
            self.data = to_numpy(data, target=target, behavior=behavior, return_terminals=True)
            data = self.data
            self.processed_data = True
        else:
            data = self.data
        states, states_un, actions, next_states, next_states_un, rewards, policy_ratio, terminals = data
        q1_params, q2_params = self.train_q_network(data, target, behavior, gamma)
        init_states = jnp.stack([jnp.asarray(tau['states'][0]) for tau in traj_data])
        init_states_np = self._to_numpy_writable(init_states)
        init_actions = jnp.asarray(target.sample(init_states_np))
        if init_actions.ndim == 1:
            init_actions = init_actions[:, None]
        xus = jnp.concatenate([init_states, init_actions], axis=-1)
        q1 = self.model.apply(q1_params, xus).reshape(-1, 1)
        q2 = self.model.apply(q2_params, xus).reshape(-1, 1)
        return jnp.minimum(q1, q2).mean().item()
