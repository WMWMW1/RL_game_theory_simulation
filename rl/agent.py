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

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values

class PPOAgent:
    def __init__(self, policy_network):
        self.policy_network = policy_network
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

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.policy_network(state)
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs.squeeze(0)[action])
        return action, log_prob


    def train(self, state, action, reward, PPO_EPOCHS=10, PPO_EPSILON=0.2):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.int64)
        reward = torch.tensor([reward], dtype=torch.float32)

        for _ in range(PPO_EPOCHS):
            action_probs, state_value = self.policy_network(state)
            dist = torch.distributions.Categorical(action_probs)
            action_log_prob = dist.log_prob(action).float()

            # 确保 state_value 的形状与 reward 相同
            state_value = state_value.squeeze()

            # 计算价值函数的损失
            value_loss = F.mse_loss(state_value, reward)

            # 计算策略的损失
            policy_loss = -action_log_prob * (reward - state_value.detach())

            # 优化更新
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.optimizer.step()
