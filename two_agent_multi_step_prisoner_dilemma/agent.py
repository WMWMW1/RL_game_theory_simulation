# agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Device use (GPU or apple mps
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print (x)

# Hyperparameters
PPO_EPOCHS = 10
PPO_EPSILON = 0.2
GAMMA = 0.99
TAU = 0.95
LR = 3e-4
ENTROPY_COEFF = 0.01
VALUE_LOSS_COEFF = 0.5
MAX_GRAD_NORM = 0.5
BATCH_SIZE = 64


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, n_heads=4, n_transformer_layers=2, transformer_hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, transformer_hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_hidden_dim, 
                nhead=n_heads
            ), 
            num_layers=n_transformer_layers
        )
        self.fc2 = nn.Linear(transformer_hidden_dim, 64)
        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(1)  # Transformer expects (seq_len, batch, feature)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        state_values = self.value_head(x)
        return action_logits, state_values


class PPOAgent:
    def __init__(self, policy_network, action_dim):
        self.policy_network = policy_network.to(device)
        self.action_dim = action_dim
        self.optimizer = torch.optim.Adam(policy_network.parameters(), lr=LR)
        
        # Buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        with torch.no_grad():
            action_logits, value = self.policy_network(state)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def compute_gae(self, next_value):
        returns = []
        gae = 0
        values = self.values + [next_value]
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + GAMMA * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + GAMMA * TAU * (1 - self.dones[step]) * gae
            returns.insert(0, gae + values[step])
        advantages = [ret - val for ret, val in zip(returns, self.values)]
        return returns, advantages
    
    def update(self):
        # Convert buffers to tensors
        states = torch.tensor(self.states, dtype=torch.float32).to(device)
        actions = torch.tensor(self.actions, dtype=torch.int64).to(device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        rewards = self.rewards
        dones = self.dones
        values = torch.tensor(self.values, dtype=torch.float32).to(device)
        
        # Compute next value
        if len(self.states) > 0:
            next_state = torch.tensor(self.states[-1], dtype=torch.float32).to(device).unsqueeze(0)
            with torch.no_grad():
                _, next_value = self.policy_network(next_state)
                next_value = next_value.item()
        else:
            next_value = 0
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae(next_value)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        dataset_size = states.size(0)
        for _ in range(PPO_EPOCHS):
            # Shuffle indices for mini-batch
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                action_logits, state_values = self.policy_network(batch_states)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                
                # New log probs
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Ratio for PPO
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                state_values = state_values.squeeze()
                value_loss = F.mse_loss(state_values, batch_returns)
                
                # Total loss
                loss = policy_loss + VALUE_LOSS_COEFF * value_loss - ENTROPY_COEFF * entropy
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
