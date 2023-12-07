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
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, n_heads=4, n_transformer_layers=2, transformer_hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, transformer_hidden_dim)
        
        # Transformer编码层
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
        
        # 为了使用Transformer层，我们需要将x调整为[seq_len, batch, feature]
        x = x.unsqueeze(0)  # 假设seq_len=1
        x = self.transformer(x)
        x = x.squeeze(0)  # 恢复到[batch, feature]

        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)

        return action_probs, state_values

class PPOAgent:
    def __init__(self, policy_network, action_dim):
        self.policy_network = policy_network
        self.action_dim = action_dim  # Store action_dim as an instance variable
        self.optimizer = torch.optim.Adam(policy_network.parameters(), lr=1e-3)
    def compute_advantages(self, rewards, values, next_values, masks, gamma=0.99, tau=0.95):
        values = values + [next_values]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
    def get_values(self, states):
        _, values = self.policy_network(states)
        return values.squeeze(-1)

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            action = torch.tensor([random.randrange(self.action_dim)], dtype=torch.int64)
            with torch.no_grad():
                action_probs, _ = self.policy_network(state.unsqueeze(0))
            log_prob = torch.log(action_probs.squeeze(0)[action])
        else:
            with torch.no_grad():
                action_probs, _ = self.policy_network(state.unsqueeze(0))
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs.squeeze(0)[action])

        return action, log_prob
    # def select_action(self, state):
    #     with torch.no_grad():
    #         action_probs, _ = self.policy_network(state.unsqueeze(0))
    #     action = torch.multinomial(action_probs, 1).item()
    #     log_prob = torch.log(action_probs.squeeze(0)[action])

    #     return action, log_prob



    def train(self, state, action, reward, PPO_EPOCHS=10, PPO_EPSILON=0.2):
        # Use clone().detach() if state is already a tensor
        state = state.clone().detach().unsqueeze(0) if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.int64)
        reward = torch.tensor([reward], dtype=torch.float32)

        for _ in range(PPO_EPOCHS):
            action_probs, state_value = self.policy_network(state)
            dist = torch.distributions.Categorical(action_probs)
            action_log_prob = dist.log_prob(action).float()

            # Ensure state_value has the same shape as reward
            state_value = state_value.squeeze()

            # Compute the losses
            value_loss = F.mse_loss(state_value, reward)
            policy_loss = -action_log_prob * (reward - state_value.detach())

            # Optimization step
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.optimizer.step()