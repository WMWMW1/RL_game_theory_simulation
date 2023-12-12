"""
environment.py

Purpose:
    Defines the environment for the multi-agent reinforcement learning scenario.

Description:
    - The MultiAgentEnvironment class simulates interactions between agents.
    - It provides methods to step through the environment and reset it.
    - The step function calculates rewards based on agents' actions.

Classes:
    - MultiAgentEnvironment: Handles the logic for the agent interactions and state transitions.

Methods:
    - step(actions): Receives actions from the agents and returns corresponding rewards.
    - reset(): Resets the environment to its initial state.

Variables:
    - n_agents: Number of agents in the environment.
    - n_actions: Number of possible actions each agent can take.
"""

import numpy as np
import torch
class MultiAgentEnvironment:
    def __init__(self):
        self.n_agents = 2
        self.n_actions = 2
        self.history_length = 2
        self.history = torch.full((self.n_agents, self.history_length, 2), 3, dtype=torch.float32)
        self.cumulative_rewards = np.zeros(self.n_agents)
        self.current_step = 0
        self.max_steps_per_episode = 10

    def step(self, actions):
        if actions[0] == 0 and actions[1] == 0:
            # Both agents are friendly
            print("Both agents are friendly")
            rewards = [1.9, 1.9]
        elif actions[0] == 1 and actions[1] == 1:
            # Both agents are unfriendly
            print("Both agents are unfriendly")
            rewards = [-1, -1]
        elif actions[0] == 0 and actions[1] == 1:
            # First agent is friendly, second is unfriendly
            print("First agent is friendly, second is unfriendly")
            rewards = [-1, 2]
        else:
            # First agent is unfriendly, second is friendly
            print("First agent is unfriendly, second is friendly")
            rewards = [2, -1]

        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1, :] = torch.tensor(actions, dtype=torch.float32).view(self.n_agents, -1)

        # 计算新的环境状态（包含历史信息）
        new_state = self.history.view(-1)

        for i in range(self.n_agents):
            self.cumulative_rewards[i] += rewards[i]

        self.current_step += 1
        done = self.is_terminal()

        if done:
            final_rewards = self.cumulative_rewards.tolist()
            self.cumulative_rewards = np.zeros(self.n_agents)  # 重置累积奖励
            return new_state, final_rewards, done
        rewards=[0,0]
        return new_state, rewards, done
    def is_terminal(self):
        return self.current_step >= self.max_steps_per_episode


    def calculate_cumulative_reward(self):
        return self.calculate_cumulative_reward
    def reset(self):
        self.history = torch.full((self.n_agents, self.history_length, 2), 3, dtype=torch.float32)
        self.cumulative_rewards = np.zeros(self.n_agents)
        self.current_step = 0
        return self.history.view(-1)