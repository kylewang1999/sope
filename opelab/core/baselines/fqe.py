import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from opelab.core.baseline import Baseline
from opelab.core.data import to_numpy, Normalization

# Optional: For mixed precision optimization (uncomment to enable)
# from torch.cuda.amp import autocast, GradScaler

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_dim + action_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(action.device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(state.device)

        if action.dim() == 1:
            action = action.unsqueeze(-1)
            
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class FQE(Baseline):
    def __init__(self, state_dim, action_dim, device='cuda', gamma=0.99, epochs=1000,
                 batch_size=2048, lr=3e-4, tau=0.0005, target_update_freq=50,
                 preprocess_once=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau
        self.device = device
        self.target_update_freq = target_update_freq
        self.preprocess_once = preprocess_once

        # Placeholders for networks, will be initialized in evaluate()
        self.q_net1 = None
        self.q_net2 = None
        self.target_q_net1 = None
        self.target_q_net2 = None
        self.optimizer = None
        self.criterion = nn.MSELoss(reduction='sum')

        # Normalization statistics
        self.state_mean = None
        self.state_std = None
        self.state_mean_tensor = None
        self.state_std_tensor = None
        self.reward_mean = None
        self.reward_std = None

        self.data = None
        self.processed_data = None
        self.initial_states_tensor = None
        self.behavior_policy = None
        self.target_policy = None

        # Optional: For mixed precision (uncomment to enable)
        # self.scaler = GradScaler() if 'cuda' in device else None

    def load_data(self, data):
        """Load data and reset processed data."""
        self.data = data
        self.processed_data = None
        self.initial_states_tensor = None
        # Reset normalization stats if new data is loaded
        self.state_mean = None
        self.state_std = None
        self.state_mean_tensor = None
        self.state_std_tensor = None
        self.reward_mean = None
        self.reward_std = None

    def soft_update(self, net, target_net):
        """Soft update target network."""
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _prepare_data(self, data, target_policy, behavior_policy):
        """Process data once if flag is set, or every time if not."""
        if self.preprocess_once and self.processed_data is not None:
            return self.processed_data

        print("Preprocessing data...")
        states, _, actions, next_states, _, rewards, _, terminals = to_numpy(
            data, target_policy, behavior_policy,
            normalization=Normalization.NONE, return_terminals=True # We handle normalization manually
        )

        if self.state_mean is None or self.state_std is None:
            print("Calculating state normalization statistics...")
            self.state_mean = np.mean(states, axis=0, keepdims=True)
            self.state_std = np.std(states, axis=0, keepdims=True) + 1e-8 # Add epsilon

        if self.reward_mean is None or self.reward_std is None:
            print("Calculating reward normalization statistics...")
            self.reward_mean = np.mean(rewards)
            self.reward_std = np.std(rewards) + 1e-8

        states = (states - self.state_mean) / self.state_std
        next_states = (next_states - self.state_mean) / self.state_std
        rewards = (rewards - self.reward_mean) / self.reward_std

        def to_tensor(x):
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        # Set tensor versions for unnormalization
        self.state_mean_tensor = to_tensor(self.state_mean)
        self.state_std_tensor = to_tensor(self.state_std)

        processed_data = (
            to_tensor(states),
            to_tensor(actions),
            to_tensor(next_states),
            to_tensor(rewards),
            to_tensor(terminals)
        )

        if self.preprocess_once:
            self.processed_data = processed_data
            self.target_policy = target_policy
            self.behavior_policy = behavior_policy

            print("Preprocessing initial states...")
            initial_states = np.array([ep['states'][0] for ep in data])
            normalized_initial_states = (initial_states - self.state_mean) / self.state_std
            self.initial_states_tensor = to_tensor(normalized_initial_states)

        return processed_data

    def evaluate(self, data, target_policy, behavior_policy, gamma=0.999, reward_estimator=None):
        # Per user request: Re-initialize networks and optimizer for a fresh run
        self.q_net1 = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.q_net2 = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_q_net1 = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_q_net2 = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        self.optimizer = optim.Adam(
            list(self.q_net1.parameters()) + list(self.q_net2.parameters()),
            lr=self.lr
        )

        # Prepare data (now with normalization)
        if self.preprocess_once:
            if self.processed_data is None:
                states, actions, next_states, rewards, terminals = self._prepare_data(
                    data, target_policy, behavior_policy
                )
            else:
                states, actions, next_states, rewards, terminals = self.processed_data
                self.target_policy = target_policy
                self.behavior_policy = behavior_policy
        else:
            states, actions, next_states, rewards, terminals = self._prepare_data(
                data, target_policy, behavior_policy
            )
        
        # --- Training Loop (no changes here) ---
        dataset_size = states.shape[0]
        indices = np.arange(dataset_size)
        self.q_net1.train()
        self.q_net2.train()
        global_idx = 0
        pbar = tqdm(range(self.epochs), desc="Training FQE")
        for epoch in pbar:
            np.random.shuffle(indices)
            epoch_loss = 0.0
            for i in range(0, dataset_size, self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                s, a, ns, r, d = (states[batch_indices], actions[batch_indices], 
                                  next_states[batch_indices], rewards[batch_indices], 
                                  terminals[batch_indices])
                
                with torch.no_grad():
                    ns_unnorm = ns * self.state_std_tensor + self.state_mean_tensor
                    na = target_policy.sample_tensor(ns_unnorm, deterministic=True)
                    q1_next = self.target_q_net1(ns, na)
                    q2_next = self.target_q_net2(ns, na)
                    q_target = torch.min(q1_next, q2_next)
                    y = r + gamma * (1 - d) * q_target
                
                # Optional mixed precision (uncomment to enable):
                # with autocast():
                q1_val, q2_val = self.q_net1(s, a), self.q_net2(s, a)
                loss = (self.criterion(q1_val, y) + self.criterion(q2_val, y)) / s.shape[0]
                
                self.optimizer.zero_grad()
                # Optional mixed precision (uncomment to enable):
                # self.scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(...)
                # self.scaler.step(self.optimizer)
                # self.scaler.update()
                # else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.q_net1.parameters()) + list(self.q_net2.parameters()), 
                    max_norm=10.0
                )
                self.optimizer.step()
                
                epoch_loss += loss.item() * s.shape[0]
                global_idx += 1
                
                if global_idx % self.target_update_freq == 0:
                    self.soft_update(self.q_net1, self.target_q_net1)
                    self.soft_update(self.q_net2, self.target_q_net2)
            
            pbar.set_postfix({"loss": f"{epoch_loss / dataset_size:.6f}"})
        
        # --- Final Value Estimation ---
        self.q_net1.eval()
        self.q_net2.eval()
        
        if self.preprocess_once and self.initial_states_tensor is not None:
            initial_states_tensor = self.initial_states_tensor
        else:
            initial_states = np.array([ep['states'][0] for ep in data])
            # Normalize with the stats computed during data preparation
            normalized_initial_states = (initial_states - self.state_mean) / self.state_std
            initial_states_tensor = torch.tensor(
                normalized_initial_states, dtype=torch.float32, device=self.device
            )
        
        print("Evaluating policy on normalized initial states...")
        with torch.no_grad():
            initial_unnorm = initial_states_tensor * self.state_std_tensor + self.state_mean_tensor
            actions = target_policy.sample_tensor(initial_unnorm, deterministic=True)
            q1_vals = self.q_net1(initial_states_tensor, actions)
            q2_vals = self.q_net2(initial_states_tensor, actions)
            q_vals = 0.5 * (q1_vals + q2_vals)
            mean_value = q_vals.mean().item()
            # Unnormalize the mean value
            unnorm_mean_value = mean_value * self.reward_std + self.reward_mean / (1 - gamma)
            print(f"Mean value (normalized): {mean_value:.6f}")
            print(f"Mean value (unnormalized): {unnorm_mean_value:.6f}")
        
        return unnorm_mean_value