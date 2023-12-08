"""
agent.py

Purpose:
    Contains the implementation of the PPOAgent and its Policy Network.

Description:
    - The PolicyNetwork class defines the neural network architecture for the agent's policy.
    - The PPOAgent class includes methods for action selection, training, and calculating advantages.

Classes:
    - PolicyNetwork: Neural network model for generating action probabilities and state values.
    - PPOAgent: Handles the decision-making and learning process of an agent.

Methods:
    - PolicyNetwork.forward(x): Forward pass through the network.
    - PPOAgent.compute_advantages(...): Computes the Generalized Advantage Estimation (GAE).
    - PPOAgent.get_values(states): Gets the value estimates for given states.
    - PPOAgent.select_action(state): Selects an action based on the current state.
    - PPOAgent.train(state, action, reward, ...): Trains the agent using the given state, action, and reward.

Variables:
    - PPO_EPOCHS: Number of training epochs per update.
    - PPO_EPSILON: Hyperparameter for PPO's loss clipping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values

class PPOAgent:
    def __init__(self, policy_network, action_dim):
        self.policy_network = policy_network.to(device)  # Move model to CUDA
        self.action_dim = action_dim
        self.optimizer = torch.optim.Adam(policy_network.parameters(), lr=1e-3)

    def compute_advantages(self, rewards, values, next_values, masks, gamma=0.99, tau=0.95):
        # Make sure all tensors are on the same device
        values = values + [next_values.to(device)]
        rewards = [r.to(device) for r in rewards]
        masks = [m.to(device) for m in masks]

        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def get_values(self, states):
        states = states.to(device)  # Move states to CUDA
        _, values = self.policy_network(states)
        return values.squeeze(-1)

    def select_action(self, state):
        state = state.to(device)  # Move state to CUDA
        with torch.no_grad():
            action_probs, _ = self.policy_network(state.unsqueeze(0))
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs.squeeze(0)[action])
        return action, log_prob

    def train(self, state, action, reward, PPO_EPOCHS=10, PPO_EPSILON=0.2):
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.int64).to(device)
        reward = torch.tensor([reward], dtype=torch.float32).to(device)

        for _ in range(PPO_EPOCHS):
            action_probs, state_value = self.policy_network(state)
            dist = torch.distributions.Categorical(action_probs)
            action_log_prob = dist.log_prob(action).float()
            state_value = state_value.squeeze()
            value_loss = F.mse_loss(state_value, reward)
            policy_loss = -action_log_prob * (reward - state_value.detach())
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.optimizer.step()