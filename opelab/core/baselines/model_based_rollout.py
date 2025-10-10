import os
import pickle
import random
from typing import Sequence, Tuple, Optional  # NOTE: no KeyArray here

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap

import numpy as np
from tqdm import tqdm

import flax.linen as nn
from flax.training import train_state
import optax

from opelab.core.baseline import Baseline
from opelab.core.data import DataType, to_numpy
from opelab.core.mlp import MLP
from opelab.core.policy import Policy


def _to_numpy_writable(x, dtype=np.float32):
    """Ensure a writable, contiguous NumPy array (copies if needed)."""
    return np.array(x, dtype=dtype, copy=True)

def _policy_sample_np(policy: Policy, states_np: np.ndarray) -> np.ndarray:
    """Call policy.sample with NumPy states; return NumPy actions (handles torch tensors)."""
    act = policy.sample(states_np)
    if hasattr(act, "detach"):  # torch.Tensor
        act = act.detach().cpu().numpy()
    else:
        act = np.array(act, copy=True)
    return act


class MBR(Baseline):
    def __init__(
        self,
        state_dim: int,
        savepath: str = "models/hopper/model_based2.pkl",
        N: int = 100,
        tau: float = 0.05,
        horizon: int = 100,
        lr: float = 3e-4,
        layers: Sequence[int] = (600, 600),
        epochs: int = 30,
        batch_size: int = 1024,
        verbose: int = 0,
        seed: int = 0,
        terminated_fn=None,
    ):
        self.terminate_fn = terminated_fn
        self.lr = lr
        self.epochs = epochs
        self.N = N
        self.batch_size = batch_size
        self.horizon = horizon
        self.verbose = verbose
        self.seed = seed
        self.tau = tau
        self.optimizer = optax.adam(lr)

        # Models
        self.model = MLP(list(layers) + [state_dim], nn.relu, output_activation=lambda s: s)  # dynamics: (s,a)-> s'
        self.reward_model = MLP(list(layers) + [1], nn.relu, output_activation=lambda s: s)   # reward: (s,a)-> r
        self.done_model = MLP(list(layers) + [1], nn.relu, output_activation=lambda s: s)     # done-logit: (s,a)-> logit
        self.Q = MLP(list(layers) + [1], nn.relu, output_activation=lambda s: s)              # Q(s,a)

        self.trained = False
        self.savepath = savepath

        # ---------- helper fn definitions (JAX friendly) ----------
        def predict_w_fn(params, states, actions):
            xus = jnp.concatenate([states, actions], axis=-1)
            return vmap(self.model.apply, in_axes=(None, 0))(params, xus)

        def predict_r_w_fn(params, states, actions):
            xus = jnp.concatenate([states, actions], axis=-1)
            out = vmap(self.reward_model.apply, in_axes=(None, 0))(params, xus)
            return out.squeeze(-1)  # (B,)

        def predict_d_w_fn(params, states, actions):
            xus = jnp.concatenate([states, actions], axis=-1)
            logit = vmap(self.done_model.apply, in_axes=(None, 0))(params, xus).squeeze(-1)
            return jax.nn.sigmoid(logit)  # (B,)

        def predict_q_w_fn(params, states, actions):
            xus = jnp.concatenate([states, actions], axis=-1)
            out = vmap(self.Q.apply, in_axes=(None, 0))(params, xus)
            return out.squeeze(-1)  # (B,)

        def soft_update(x, y):
            # new_target = tau * online + (1 - tau) * old_target
            return jax.tree_util.tree_map(lambda a, b: self.tau * b + (1.0 - self.tau) * a, x, y)

        self.predict_w_fn = jit(predict_w_fn)
        self.predict_r_w_fn = jit(predict_r_w_fn)
        self.predict_d_w_fn = jit(predict_d_w_fn)
        self.predict_q_w_fn = jit(predict_q_w_fn)
        self.soft_update = jit(soft_update)

        # ---------- losses & train-steps ----------
        def model_loss(params, states, actions, next_states):
            pred_next_state = predict_w_fn(params, states, actions)
            return 0.5 * jnp.mean((pred_next_state - next_states) ** 2)

        def reward_loss_fn(params, states, actions, rewards):
            pred_rewards = predict_r_w_fn(params, states, actions)
            return 0.5 * jnp.mean((pred_rewards - rewards) ** 2)

        def done_loss_fn(params, states, actions, dones, weights):
            eps = 1e-6
            p = predict_d_w_fn(params, states, actions)  # (B,)
            weights = weights.reshape(-1)
            dones = dones.reshape(-1)
            per_ex = dones * jnp.log(eps + p) + (1.0 - dones) * jnp.log(eps + 1.0 - p)
            return -jnp.mean(weights * per_ex)

        def fqe_loss_fn(params, q1_t_params, q2_t_params,
                        states, actions, rewards, next_states, next_actions, dones,
                        gamma, clip):
            current_q = predict_q_w_fn(params, states, actions)  # (B,)
            next_q1 = jax.lax.stop_gradient(predict_q_w_fn(q1_t_params, next_states, next_actions))
            next_q2 = jax.lax.stop_gradient(predict_q_w_fn(q2_t_params, next_states, next_actions))
            next_q = jnp.minimum(next_q1, next_q2)
            target = rewards + gamma * (1.0 - dones) * next_q
            target = jnp.clip(target, -clip, clip)
            return jnp.mean((current_q - target) ** 2)

        @jit
        def train_dyn_step(mlp_state, states, actions, next_states):
            loss, grads = jax.value_and_grad(model_loss)(mlp_state.params, states, actions, next_states)
            mlp_state = mlp_state.apply_gradients(grads=grads)
            return mlp_state, loss

        @jit
        def train_rew_step(mlp_state, states, actions, rewards):
            loss, grads = jax.value_and_grad(reward_loss_fn)(mlp_state.params, states, actions, rewards)
            mlp_state = mlp_state.apply_gradients(grads=grads)
            return mlp_state, loss

        @jit
        def train_done_step(mlp_state, states, actions, dones, weights):
            loss, grads = jax.value_and_grad(done_loss_fn)(mlp_state.params, states, actions, dones, weights)
            mlp_state = mlp_state.apply_gradients(grads=grads)
            return mlp_state, loss

        @jit
        def train_q_step(mlp_state, q1_t_params, q2_t_params,
                         states, actions, rewards, next_states, next_actions, dones, gamma, clip):
            loss, grads = jax.value_and_grad(fqe_loss_fn)(
                mlp_state.params, q1_t_params, q2_t_params,
                states, actions, rewards, next_states, next_actions, dones,
                gamma, clip
            )
            mlp_state = mlp_state.apply_gradients(grads=grads)
            return mlp_state, loss

        self.model_loss = model_loss
        self.train_fn = train_dyn_step
        self.train_reward_fn = train_rew_step
        self.train_d_fn = train_done_step
        self.train_q_fn = train_q_step

    # ------------------------------- training: dynamics / reward / done -------------------------------
    def train_dynamics_network(self, data: DataType, target: Policy, behavior: Policy):
        states, states_un, actions, next_states, next_states_un, rewards, policy_ratio, terminals = to_numpy(
            data, target, behavior, return_terminals=True
        )

        states = jnp.asarray(states)
        actions = jnp.asarray(actions)
        next_states = jnp.asarray(next_states)
        rewards = jnp.asarray(rewards).reshape(-1)   # (N,)
        terminals = jnp.asarray(terminals).reshape(-1)

        # clips for evaluation
        rew_min, rew_max = float(rewards.min()), float(rewards.max())
        state_min = jnp.min(states, axis=0)
        state_max = jnp.max(states, axis=0)

        # class-imbalance weights for done BCE (avoid div-by-zero)
        done_probability = float(terminals.mean())
        weights = 1.0 / (done_probability * terminals + (1.0 - done_probability) * (1.0 - terminals) + 1e-6)

        # init params
        xus0 = jnp.concatenate([states[0], actions[0]], axis=-1)
        dyn_params = self.model.init(jr.PRNGKey(self.seed), xus0)
        rew_params = self.reward_model.init(jr.PRNGKey(self.seed + 1), xus0)
        done_params = self.done_model.init(jr.PRNGKey(self.seed + 2), xus0)

        dyn_state = train_state.TrainState.create(apply_fn=self.model.apply, params=dyn_params, tx=self.optimizer)
        rew_state = train_state.TrainState.create(apply_fn=self.reward_model.apply, params=rew_params, tx=self.optimizer)
        done_state = train_state.TrainState.create(apply_fn=self.done_model.apply, params=done_params, tx=self.optimizer)

        num_batches = int(states.shape[0]) // self.batch_size
        key = jr.PRNGKey(self.seed)

        with tqdm(range(self.epochs)) as tp:
            for _ in tp:
                key, kperm = jr.split(key)
                order = jr.permutation(kperm, jnp.arange(states.shape[0]))

                dyn_losses, rew_losses, done_losses = [], [], []

                for j in range(num_batches):
                    idx = order[j * self.batch_size : (j + 1) * self.batch_size]
                    bs = states[idx]
                    ba = actions[idx]
                    bns = next_states[idx]
                    br = rewards[idx]
                    bd = terminals[idx]
                    bw = weights[idx]

                    dyn_state, dloss = self.train_fn(dyn_state, bs, ba, bns)
                    rew_state, rloss = self.train_reward_fn(rew_state, bs, ba, br)
                    done_state, tloss = self.train_d_fn(done_state, bs, ba, bd, bw)

                    dyn_losses.append(dloss)
                    rew_losses.append(rloss)
                    done_losses.append(tloss)

                tp.set_postfix(
                    dynamics_loss=float(jnp.mean(jnp.asarray(dyn_losses))),
                    rewards_loss=float(jnp.mean(jnp.asarray(rew_losses))),
                    done_loss=float(jnp.mean(jnp.asarray(done_losses))),
                )

        self.params = dyn_state.params
        self.rew_params = rew_state.params
        self.done_params = done_state.params
        return self.params, self.rew_params, self.done_params, (rew_min, rew_max), (state_min, state_max)

    # ------------------------------- training: Q (optional) -------------------------------
    def train_q_network(self, data, target, behavior, dynamics_params, reward_params, done_params, gamma):
        states, states_un, actions, next_states, next_states_un, rewards, policy_ratio, terminals = to_numpy(
            data, target, behavior, return_terminals=True
        )

        states = jnp.asarray(states)
        actions = jnp.asarray(actions)
        next_states = jnp.asarray(next_states)
        rewards = jnp.asarray(rewards).reshape(-1)
        terminals = jnp.asarray(terminals).reshape(-1)

        xus0 = jnp.concatenate([states[0], actions[0]], axis=-1)

        key = jr.PRNGKey(self.seed)
        key, k1, k2 = jr.split(key, 3)

        q1_params = self.Q.init(k1, xus0)
        q2_params = self.Q.init(k2, xus0)

        q1_state = train_state.TrainState.create(apply_fn=self.Q.apply, params=q1_params, tx=self.optimizer)
        q2_state = train_state.TrainState.create(apply_fn=self.Q.apply, params=q2_params, tx=self.optimizer)

        q1_t_params = jax.tree_util.tree_map(lambda x: x, q1_params)
        q2_t_params = jax.tree_util.tree_map(lambda x: x, q2_params)

        max_rew = float(jnp.abs(rewards).max())
        clip = max_rew / max(1e-6, (1.0 - gamma))

        num_batches = int(states.shape[0]) // self.batch_size

        with tqdm(range(self.epochs)) as tp:
            for _ in tp:
                key, kperm = jr.split(key)
                order = jr.permutation(kperm, jnp.arange(states.shape[0]))
                q1_losses, q2_losses = [], []

                for j in range(num_batches):
                    idx = order[j * self.batch_size : (j + 1) * self.batch_size]
                    bs = states[idx]

                    # policy actions via NumPy bridge (batched)
                    bs_np = _to_numpy_writable(bs)
                    ba_np = _policy_sample_np(target, bs_np)
                    if ba_np.ndim == 1:
                        ba_np = ba_np[:, None]
                    ba = jnp.asarray(ba_np)

                    bns = self.predict_w_fn(dynamics_params, bs, ba)
                    bns_np = _to_numpy_writable(bns)
                    bna_np = _policy_sample_np(target, bns_np)
                    if bna_np.ndim == 1:
                        bna_np = bna_np[:, None]
                    bna = jnp.asarray(bna_np)

                    key, kdone = jr.split(key)
                    p_done = self.predict_d_w_fn(done_params, bs, ba)  # (B,)
                    bd = jr.bernoulli(kdone, p_done).astype(jnp.float32)

                    br = self.predict_r_w_fn(reward_params, bs, ba)

                    q1_state, q1_loss = self.train_q_fn(
                        q1_state, q1_t_params, q2_t_params,
                        bs, ba, br, bns, bna, bd, gamma, clip
                    )
                    q2_state, q2_loss = self.train_q_fn(
                        q2_state, q1_t_params, q2_t_params,
                        bs, ba, br, bns, bna, bd, gamma, clip
                    )

                    if j % 5 == 0:
                        q1_t_params = self.soft_update(q1_t_params, q1_state.params)
                        q2_t_params = self.soft_update(q2_t_params, q2_state.params)

                    q1_losses.append(q1_loss)
                    q2_losses.append(q2_loss)

                tp.set_postfix(
                    q1_loss=float(jnp.mean(jnp.asarray(q1_losses))),
                    q2_loss=float(jnp.mean(jnp.asarray(q2_losses))),
                )

        self.q1_params = q1_state.params
        self.q2_params = q2_state.params
        return self.q1_params, self.q2_params

    # ------------------------------- parallel rollouts -------------------------------
    def simulate_rollouts_batch(
        self,
        key,  
        dynamics_params,
        rewards_params,
        done_params,
        data: DataType,
        target: Policy,
        gamma: float,
        reward_clip: Tuple[float, float],
        state_clip: Tuple[jnp.ndarray, jnp.ndarray],
        num_rollouts: int = 50,
        record_trajs: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run num_rollouts rollouts in parallel with masking.
        Returns (returns, trajs) where trajs has shape (B, T, Ds+Da) if record_trajs else None.
        """
        # sample initial states from dataset (first state of random trajs)
        taus = random.choices(data, k=num_rollouts)
        init_states = jnp.stack([jnp.asarray(t["states"][0]) for t in taus], axis=0)  # (B, Ds)

        B, Ds = init_states.shape
        discounts = (gamma ** jnp.arange(self.horizon)).astype(jnp.float32)  # (T,)

        returns = jnp.zeros((B,), dtype=jnp.float32)
        state = init_states  # (B, Ds)
        alive = jnp.ones((B,), dtype=bool)

        # allocate trajectory buffer if needed (we fill with zeros for ended envs)
        trajs = None

        for t in range(self.horizon):
            if not bool(alive.any()):
                break

            # policy actions for all alive states (bridge to NumPy once)
            state_np = _to_numpy_writable(np.asarray(state))
            act_np = _policy_sample_np(target, state_np)  # (B, Da) or (B,)
            if act_np.ndim == 1:
                act_np = act_np[:, None]
            action = jnp.asarray(act_np)  # (B, Da)

            # lazily allocate trajs with correct Da
            if record_trajs and trajs is None:
                Da = action.shape[-1]
                trajs = np.zeros((B, self.horizon, int(Ds + Da)), dtype=np.float32)

            # concatenate (s,a) for model heads
            xus = jnp.concatenate([state, action], axis=-1)  # (B, Ds+Da)

            # reward + next state + done probs (vectorized/JAX)
            r = self.reward_model.apply(rewards_params, xus).squeeze(-1)  # (B,)
            r = jnp.clip(r, reward_clip[0], reward_clip[1])

            next_state = self.model.apply(dynamics_params, xus)  # (B, Ds)

            # optional Python termination fn (evaluated per-env)
            if self.terminate_fn is not None:
                term_np = np.array([self.terminate_fn(np.array(ns)) for ns in np.array(next_state)], dtype=bool)
                term = jnp.asarray(term_np)
            else:
                term = jnp.zeros((B,), dtype=bool)

            key, k_t = jr.split(key)
            p_done = jax.nn.sigmoid(self.done_model.apply(done_params, xus).squeeze(-1))  # (B,)
            done = jr.bernoulli(k_t, p_done)  # (B,)

            # accumulate masked reward
            returns = returns + discounts[t] * r * alive.astype(r.dtype)

            # record traj row (only for alive envs; dead rows remain zero)
            if record_trajs:
                row_np = np.array(jnp.concatenate([state, action], axis=-1))
                trajs[:, t, :] = row_np

            # update mask and state
            still_alive = alive & (~done) & (~term)
            clipped_next = jnp.clip(next_state, state_clip[0], state_clip[1])
            state = jnp.where(still_alive[:, None], clipped_next, state)
            alive = still_alive

        return np.array(returns), trajs  # convert returns to NumPy for np.nanmean etc.

    def evaluate(self, data, target, behavior, gamma: float = 1.0, reward_estimator=None):
        print("evaluating")
        if self.terminate_fn is not None:
            print("using terminate function and not the trained model")
        print(self.savepath)

        # load / train dynamics + heads
        if os.path.exists(self.savepath):
            print("loading model weights")
            weights = pickle.load(open(self.savepath, "rb"))
            self.dynamic_params = weights["dynamics"]
            self.reward_params = weights["reward"]
            self.done_params = weights["done"]
            self.reward_clip = weights["reward_clip"]
            self.state_clip = weights["state_clip"]
        elif not self.trained:
            print("training model")
            (self.dynamic_params,
             self.reward_params,
             self.done_params,
             self.reward_clip,
             self.state_clip) = self.train_dynamics_network(data, target, behavior)
            self.trained = True
            os.makedirs(os.path.dirname(self.savepath), exist_ok=True)
            pickle.dump(
                {
                    "dynamics": self.dynamic_params,
                    "reward": self.reward_params,
                    "done": self.done_params,
                    "reward_clip": self.reward_clip,
                    "state_clip": self.state_clip,
                },
                open(self.savepath, "wb"),
            )
        # else: already trained in-session

        # ---- PARALLEL ROLLOUTS ----
        key = jr.PRNGKey(self.seed)
        returns, trajs = self.simulate_rollouts_batch(
            key,
            self.dynamic_params,
            self.reward_params,
            self.done_params,
            data,
            target,
            gamma,
            self.reward_clip,
            self.state_clip,
            num_rollouts=50,           # parallel instead of sequential
            record_trajs=False,         
        )

        mean_return = float(np.nanmean(returns))
        print(mean_return)

        # # Save trajectories (optional)
        # with open("model_based_rollouts.pkl", "wb") as f:
        #     pickle.dump(trajs, f)

        return mean_return
