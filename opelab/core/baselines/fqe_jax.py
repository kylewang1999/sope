import numpy as np
from typing import Sequence
from tqdm import tqdm

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

class FQE(Baseline):
    
    def __init__(self, lr:float = 3e-3, tau=0.0005, layers: Sequence[int] = [500, 500], epochs:int=100, batch_size: int=256, verbose: int = 0, seed: int = 0):
        
        self.lr = lr 
        self.epochs = epochs 
        self.batch_size = batch_size
        self.verbose = verbose
        self.seed = seed
        self.tau = tau
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr))
        self.model = MLP(layers+[1,], nn.relu, output_activation=lambda s: s)
        self.processed_data=False
        
        def predict_w_fn(params, states, actions):
            actions = actions[:, None] if actions.ndim == 1 else actions
            xus = jnp.concatenate([states, actions], axis=-1)
            return vmap(self.model.apply, in_axes=(None, 0))(params, xus)
        
        
        @jit
        def fqe_loss_fn(params, q1_target_params, q2_target_params, states, actions, rewards, next_states, next_actions, dones, gamma, clip):
                
                current_q = predict_w_fn(params, states, actions)
                next_q1 = jax.lax.stop_gradient(predict_w_fn(q1_target_params, next_states, next_actions))
                
                
                next_q2 = jax.lax.stop_gradient(predict_w_fn(q2_target_params, next_states, next_actions))
                
                
                next_q = jnp.minimum(next_q1, next_q2)
                target = rewards + gamma*(1-dones)*next_q
                target = jnp.clip(target, -1*clip, clip)
                loss = jnp.mean(jnp.square(current_q - target))
                print(target.shape, current_q.shape)
                return loss 
            
        def train_fn(mlp_state, q1_target_params, q2_target_params, states, actions, rewards, next_states, next_actions, dones, gamma, clip):
            
            
            
            loss, grads = jax.value_and_grad(fqe_loss_fn, argnums=(0))(mlp_state.params, q1_target_params, q2_target_params, states, actions, rewards, next_states, next_actions, dones, gamma, clip)
            
            mlp_state = mlp_state.apply_gradients(grads=grads)
            return mlp_state, loss 
        def soft_update(x, y):
            return jax.tree_util.tree_map(lambda a, b: self.tau*b + (1-self.tau)*a, x, y)
        
        self.soft_update =  jit(soft_update)
    
        self.train_fn = jit(train_fn)
        self.predict_w_fn = jit(predict_w_fn)
        self.loss_fn = jit(fqe_loss_fn)
        
    def train_q_network(self, data, target: Policy, behavior:Policy, gamma: float):
        ##initialize q_network
        key = jax.random.PRNGKey(self.seed)
        
        _, states, actions, _, next_states, rewards, policy_ratio, terminals = data #to_numpy(data, target=target, behavior=behavior, return_terminals=True)
        xus = jnp.concatenate([states, actions], axis=-1)
        
        key = jrandom.PRNGKey(self.seed)        
        key, init_key1, init_key2 = jrandom.split(key, 3)
        q1_params = self.model.init(init_key1, xus[:20])
        q1_mlp_state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            params = q1_params,
            tx = self.optimizer
        )
        
        q1_target_params = jax.tree_util.tree_map(lambda x: x, q1_params)
        
        
        q2_params = self.model.init(init_key2, xus[:20])
        q2_mlp_state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            params = q2_params,
            tx = self.optimizer
        )
        
        q2_target_params = jax.tree_util.tree_map(lambda x: x, q2_params)
        
        
        max_rew = np.absolute(rewards).max()
        clip = max_rew/(1-gamma)
        ## Training loop
        
        with tqdm(range(self.epochs)) as tp:
            for i in tp:

                
                key, b_key = jrandom.split(key)
                batch_ordering = jrandom.permutation(b_key, jnp.arange(len(states)))
                q1_epoch_loss = []
                q2_epoch_loss = []
                # print(len(states)//self.batch_size)
                #batch_states, batch_actions, batch_rewards, batch_next_states, batch_next_actions, batch_dones= None, None, None, None, None, None
                
                    
                for j in range(len(states)//self.batch_size):
                        batch = batch_ordering[j*self.batch_size: (j+1)*self.batch_size]
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = states[batch], actions[batch], rewards[batch], next_states[batch], terminals[batch]
                        
                        batch_next_actions = target.sample(batch_next_states)
                        batch_dones = batch_dones.astype(jnp.float32)

                        q1_mlp_state, q1_loss_val = self.train_fn(q1_mlp_state, q1_target_params, q2_target_params, batch_states, batch_actions, batch_rewards, batch_next_states, batch_next_actions, batch_dones, gamma, clip=clip)
                        q2_mlp_state, q2_loss_val = self.train_fn(q2_mlp_state, q1_target_params, q2_target_params, batch_states, batch_actions, batch_rewards, batch_next_states, batch_next_actions, batch_dones, gamma, clip=clip)
                        
                        if j % 5 == 0:
                            q1_target_params = self.soft_update(q1_target_params, q1_mlp_state.params)
                            q2_target_params = self.soft_update(q2_target_params, q2_mlp_state.params)


                        q1_epoch_loss.append(q1_loss_val)
                        q2_epoch_loss.append(q2_loss_val)
                        
                
                tp.set_postfix(q1_loss = np.mean(q1_epoch_loss), q2_loss = np.mean(q2_epoch_loss))
        self.q1_params = q1_mlp_state.params 
        self.q2_params = q2_mlp_state.params 
        return self.q1_params, self.q2_params

    def evaluate(self, data: DataType, target: Policy, behavior: Policy, gamma:float=1, reward_estimator=None )-> float:
        
        traj_data = data
        if not self.processed_data:
            self.data = to_numpy(data, target=target, behavior=behavior, return_terminals=True)
            data = self.data
            self.processed_data = True
        else:
            data = self.data 

        states, states_un, actions, next_states, next_states_un, rewards, policy_ratio, terminals = data
        
        q1_params, q2_params = self.train_q_network(data, target, behavior, gamma)
        
        
        initial_qs = []
        for tau in traj_data:
            first_state = tau['states'][0]
            target_action = target.sample(first_state)
            xus = jnp.concatenate([first_state, target_action], axis=-1)
            q_value = jnp.minimum(self.model.apply(q1_params, xus), self.model.apply(q2_params, xus))
            initial_qs.append(q_value)
            
        print(np.mean(initial_qs))
        return np.mean(initial_qs)