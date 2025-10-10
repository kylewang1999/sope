import numpy as np
from typing import Sequence
from tqdm import tqdm

import flax.linen as nn
from flax.training import train_state
import jax 
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, jit, value_and_grad
import optax 

from opelab.core.baseline import Baseline
from opelab.core.data import to_numpy
from opelab.core.mlp import MLP
from opelab.core.policy import Policy


@jax.jit
def cdist(x, y):
    return jnp.sqrt(jnp.sum((x[:, None] - y[None, :]) ** 2, -1))


class BlackBox(Baseline):
    
    def __init__(self, lr_w:float=1e-3, kernel_scale:float=1.0,
                 layers: Sequence[int] = [256, 256], 
                 epochs:int=20, 
                 batch_size: int=512, 
                 seed: int = 0):        
        self.lr_w = lr_w
        self.kernel_scale = kernel_scale
        self.epochs = epochs 
        self.batch_size = batch_size
        self.seed = seed
        self.w_opt = optax.adam(lr_w)
        self.processed_data=False
        print(self.kernel_scale)
        
        self.w = MLP(layers+[1,], nn.leaky_relu, output_activation=nn.softplus)
        
        def kernel(states, actions, states2, actions2, scale):
            states = states[:, None] if states.ndim == 1 else states
            actions = actions[:, None] if actions.ndim == 1 else actions
            states2 = states2[:, None] if states2.ndim == 1 else states2
            actions2 = actions2[:, None] if actions2.ndim == 1 else actions2            
            x1 = jnp.concatenate([states, actions], axis=-1)
            x2 = jnp.concatenate([states2, actions2], axis=-1)
            return jnp.exp(-scale * cdist(x1, x2))
        
        def predict_w_fn(params, states, actions):
            states = states[:, None] if states.ndim == 1 else states
            actions = actions[:, None] if actions.ndim == 1 else actions
            xs = jnp.concatenate([states, actions], axis=-1)
            return vmap(self.w.apply, in_axes=(None, 0))(params, xs)
        
        def train_fn(w_state, states, actions, next_states, next_actions, scale):            
            def loss_fn(w_params):
                K0 = kernel(states, actions, states, actions, scale)
                K1 = kernel(states, actions, next_states, next_actions, scale)
                K2 = kernel(next_states, next_actions, next_states, next_actions, scale)
                K = K0 - 2 * K1 + K2
                ws = predict_w_fn(w_params, states, actions).reshape((-1, 1))
                ws = ws / (jnp.sum(ws) + 1e-12)
                return jnp.sum(ws * ws.reshape((1, -1)) * K) 
            loss, grad = value_and_grad(loss_fn)(w_state.params)  
            w_state = w_state.apply_gradients(grads=grad)                      
            return w_state, loss
        
        self.predict_w_fn = jit(predict_w_fn)
        self.train_fn = jit(train_fn)
    
    def train_networks(self, data, traj_data, target: Policy , behavior: Policy, gamma:float):
        seed = np.random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)
        batch_size = self.batch_size

        states, _, actions, next_states, next_states_un, *_ = data
        
        if batch_size is None or batch_size == "":
            batch_size = len(states)
        
        print(len(states) // batch_size)
        xus = np.concatenate([states, actions], axis=-1)

        w_params = self.w.init(key, xus[0])
        w_state = train_state.TrainState.create(
            apply_fn = self.predict_w_fn, params = w_params, tx = self.w_opt)        
        
        with tqdm(range(self.epochs)) as tp:
            for epoch in tp:
                key, b_key = jrandom.split(key)                
                batch_ordering = jrandom.permutation(b_key, jnp.arange(len(states)))
                epoch_loss = [] 
                density_ratios_list = []
                density_ratios_list_max = []

                for j in  range(len(states)//batch_size):
                        batch = batch_ordering[j*batch_size:(j+1)*batch_size]

                        batch_states, batch_actions, batch_next_states = states[batch], actions[batch], next_states[batch]
                        batch_next_actions = target.sample(next_states_un[batch])

                        w_state, loss_val = self.train_fn(
                            w_state, batch_states, batch_actions, 
                            batch_next_states, batch_next_actions, self.kernel_scale)
                                                
                        density_ratios = self.predict_w_fn(w_state.params, batch_states, batch_actions)
                        epoch_loss.append(loss_val)
                        density_ratios_list_max.append(jnp.max(density_ratios))
                        density_ratios_list.append(jnp.mean(density_ratios))
                        
                tp.set_postfix(loss = np.mean(epoch_loss), dratios = np.max(density_ratios_list_max), 
                               dratios_mean = np.mean(density_ratios_list))

        
        self.w_params = w_state.params 
        return self.w_params
    
    def evaluate(self, data, target, behavior, gamma = 1, reward_estimator=None):
        
        traj_data = data
        if not self.processed_data:
            self.data = to_numpy(data, target=target, behavior=behavior, return_terminals=True)
            data = self.data 
            self.processed_data=True
        else:
            data = self.data 
        states, _, actions, *_, rewards, _, _ = data
        
        w_params = self.train_networks(data, traj_data, target, behavior, gamma)

        ws = self.predict_w_fn(w_params, states, actions).reshape((-1))
        rs = rewards.reshape((-1))
        estimate = np.sum(rs * ws).item() / (np.sum(ws).item() + 1e-12)
        estimate = estimate / (1 - gamma)
        
        print(estimate)
        return estimate