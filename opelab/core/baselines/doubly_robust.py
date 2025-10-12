import numpy as np
from typing import Sequence
from tqdm import tqdm
import random 

import flax.linen as nn
from flax.training import train_state
import jax 
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, grad, jit 
import optax 

from opelab.core.baseline import Baseline
from opelab.core.data import DataType, to_numpy
from opelab.core.mlp import MLP
from opelab.core.policy import Policy


class DoublyRobustEstimator(Baseline):

    
    def __init__(self, lr:float = 3e-5, tau=0.0005, layers: Sequence[int] = [400, 400], epochs:int=100, batch_size: int=32, verbose: int = 0, seed: int = 0):
        
        self.lr = lr 
        self.epochs = epochs 
        self.batch_size = batch_size
        self.verbose = verbose
        self.seed = seed
        self.tau = tau
        self.optimizer = optax.adamw(lr)
        self.q_model = MLP(layers+[1,], nn.sigmoid, output_activation=lambda s: s)
        self.value_model = MLP(layers+[1,], nn.sigmoid, output_activation=lambda s: s)
        self.processed_data=False
        
        
        def predict_w_fn(params, states, actions):
            actions = actions[:, None] if actions.ndim == 1 else actions
            xus = jnp.concatenate([states, actions], axis=-1)
            return vmap(self.q_model.apply, in_axes=(None, 0))(params, xus)
        
        def predict_value_w_fn(params, states):
            
            
            return vmap(self.value_model.apply, in_axes=(None,0))(params, states)
        
                
        def fqe_loss(mlp_state, target_params, states, actions, rewards, next_states, next_actions, dones, gamma):
            params = mlp_state.params
            current_q = predict_w_fn(params, states, actions)
            next_q = jax.lax.stop_gradient(predict_w_fn(target_params, next_states, next_actions))
            
            loss = jnp.mean(jnp.square(current_q - rewards - gamma*(1-dones)*next_q))
            return loss 
        
        def soft_update(x, y):
            return jax.tree_util.tree_map(lambda a, b: self.tau*b + (1-self.tau)*a, x, y)
        
        self.soft_update =  jit(soft_update)
        
            
        def train_fn(mlp_state, q1_target_params, q2_target_params, states, actions, rewards, next_states, next_actions, dones, gamma, clip):
            
            def loss_fn(params):
                
                current_q = predict_w_fn(params, states, actions)
                next_q1 = jax.lax.stop_gradient(predict_w_fn(q1_target_params, next_states, next_actions))
                next_q2 = jax.lax.stop_gradient(predict_w_fn(q2_target_params, next_states, next_actions))
                next_q = jnp.minimum(next_q1, next_q2)
                target = rewards + gamma*(1-dones)*next_q
                target = jnp.clip(target, -1*clip, clip)
                loss = jnp.mean((1-dones)*jnp.square(current_q - target))
                return loss 
            
            loss, grads = jax.value_and_grad(loss_fn)(mlp_state.params)
            
            mlp_state = mlp_state.apply_gradients(grads=grads)
            return mlp_state, loss 

        def train_value_fn(value_mlp, q1_params, q2_params, states, actions):
            def loss_fn(params):
                qs = 0.5*jnp.add(self.predict_w_fn(q1_params, states, actions), self.predict_w_fn(q2_params, states, actions))
                
                predicted_values = self.predict_value_w_fn(params, states)
                
                return jnp.mean(jnp.square(predicted_values-qs))
            loss, grads = jax.value_and_grad(loss_fn)(value_mlp.params)
            
            value_mlp = value_mlp.apply_gradients(grads=grads)
            return value_mlp, loss
        
        
        self.train_fn = jit(train_fn)
        self.train_value_fn = jit(train_value_fn)
        self.predict_w_fn = jit(predict_w_fn)
        self.loss_fn = jit(fqe_loss)
        self.predict_value_w_fn = jit(predict_value_w_fn)
        
    def train_value_networks(self, data, target: Policy, behavior:Policy, gamma: float):
        ##initialize q_network
        key = jax.random.PRNGKey(self.seed)
        
        _, states, actions, _, next_states, rewards, policy_ratio, terminals = data #to_numpy(data, target=target, behavior=behavior, return_terminals=True)
        xus = jnp.concatenate([states, actions], axis=-1)
        
        key = jrandom.PRNGKey(self.seed)        
        key, init_key1, init_key2 = jrandom.split(key, 3)
        
        q1_params = self.q_model.init(init_key1, xus[:20])
        q1_mlp_state = train_state.TrainState.create(
            apply_fn = self.q_model.apply,
            params = q1_params,
            tx = self.optimizer
        )
        
        q1_target_params = jax.tree_util.tree_map(lambda x: x, q1_params)
        
        
        q2_params = self.q_model.init(init_key2, xus[:20])
        q2_mlp_state = train_state.TrainState.create(
            apply_fn = self.q_model.apply,
            params = q2_params,
            tx = self.optimizer
        )
        
        q2_target_params = jax.tree_util.tree_map(lambda x: x, q2_params)
        
        
        
        v_params = self.value_model.init(jax.random.PRNGKey(self.seed), states[0])
        v_mlp_state = train_state.TrainState.create(
            apply_fn = self.predict_value_w_fn,
            params = v_params,
            tx = self.optimizer
        )
        
        
        max_rew = np.absolute(rewards).max()
        clip = max_rew/(1-gamma)
        ## Training loop
        
        with tqdm(range(self.epochs)) as tp:
            for i in tp:

                
                key, b_key = jrandom.split(key)
                batch_ordering = jrandom.permutation(b_key, jnp.arange(len(states)))
                epoch_q1_loss = []
                epoch_q2_loss = []
                epoch_v_loss = []
                #batch_states, batch_actions, batch_rewards, batch_next_states, batch_next_actions, batch_dones= None, None, None, None, None, None
                
                    
                for j in range(len(states)//self.batch_size):
                        batch = batch_ordering[j*self.batch_size: (j+1)*self.batch_size]
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = states[batch], actions[batch], rewards[batch], next_states[batch], terminals[batch]
                        batch_next_actions = target.sample(batch_next_states)
                        
                        target_batch_actions = target.sample(batch_states)
                        
                
                        q1_mlp_state, q1_loss_val = self.train_fn(q1_mlp_state, q1_target_params, q2_target_params, batch_states, batch_actions, batch_rewards, batch_next_states, batch_next_actions, batch_dones, gamma, clip=clip)
                        q2_mlp_state, q2_loss_val = self.train_fn(q2_mlp_state, q1_target_params, q2_target_params, batch_states, batch_actions, batch_rewards, batch_next_states, batch_next_actions, batch_dones, gamma, clip=clip)
                        
                        if j % 20 == 0:
                            q1_target_params = self.soft_update(q1_target_params, q1_mlp_state.params)
                            q2_target_params = self.soft_update(q2_target_params, q2_mlp_state.params)
                        
                        v_mlp_state, v_loss_val = self.train_value_fn(v_mlp_state, q1_mlp_state.params, q2_mlp_state.params, batch_states, target_batch_actions)
                        
                        
                        epoch_q1_loss.append(q1_loss_val)
                        epoch_q2_loss.append(q2_loss_val)
                        epoch_v_loss.append(v_loss_val)
                        
                
                tp.set_postfix(q1_loss = np.mean(epoch_q1_loss), q2_loss = np.mean(epoch_q2_loss), value_loss = np.mean(epoch_v_loss))
        self.q1_params = q1_mlp_state.params 
        self.q2_params = q2_mlp_state.params 
        self.v_params = v_mlp_state.params
        return self.q1_params, self.q2_params, self.v_params

    def evaluate(self, data: DataType, target: Policy, behavior: Policy, gamma:float=1, reward_estimator=None )-> float:
        traj_data = data
        if not self.processed_data:
            self.data = to_numpy(data, target=target, behavior=behavior, return_terminals=True)
            data = self.data
            self.processed_data=True
        else:
            data = self.data 
            

        states, states_un, actions, next_states, next_states_un, rewards, policy_ratio, terminals = data
        
        
        traj_data = random.choices(traj_data, k=50)
        ##train "Model-based" networks for DR
        q1_params, q2_params, v_params = self.train_value_networks(data, target, behavior, gamma)
        epsilon=1e-20
        mean_est_reward = 0.0
        is_comp, mb_comp = [], []
        initial_qs = []
        for tau in tqdm(traj_data):
            log_prob_ratio = 0.0
            total_reward = 0.0
            discounted_t = 1.0
            normalizer = 0.0
            
            
            total_reward = 0
            
            xus = jnp.concatenate([tau['states'][0], tau['actions'][0]], axis=-1)
            initial_q =  np.minimum(np.array(self.q_model.apply(q1_params, xus)), np.array(self.q_model.apply(q2_params, xus)))
            initial_qs.append(initial_q)
            for i, (state, action, reward) in enumerate(zip(tau['states'], tau['actions'], tau['rewards'])):
                
                target_prob = target.prob(state, action)+epsilon
                behaviour_prob = behavior.prob(state, action)+epsilon
                new_log_prob_ratio = np.log(target_prob+epsilon) - np.log(max(behaviour_prob+epsilon, 1e-6))
                
                xus = jnp.concatenate([state, action])
                q_est = np.minimum(np.array(self.q_model.apply(q1_params, xus)), np.array(self.q_model.apply(q2_params, xus)))
                v_est = np.array(self.value_model.apply(v_params, state))
                
                is_rew_comp = np.exp(new_log_prob_ratio+log_prob_ratio) * reward * discounted_t
                mb_rew_comp = discounted_t * (jnp.exp(new_log_prob_ratio+log_prob_ratio)*q_est - jnp.exp(log_prob_ratio)*v_est)
                
                total_reward += is_rew_comp - mb_rew_comp
                is_comp.append(is_rew_comp)
                mb_comp.append(mb_rew_comp)
                log_prob_ratio += new_log_prob_ratio
                normalizer += discounted_t
                discounted_t *= gamma
            total_reward #/= normalizer
            mean_est_reward += total_reward
        mean_est_reward /= len(traj_data)
        
        print(np.mean(initial_qs))
        print(mean_est_reward, np.mean(is_comp), np.max(is_comp),np.max(mb_comp), np.mean(mb_comp))
        return np.array(mean_est_reward)[0]