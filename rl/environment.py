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

class MultiAgentEnvironment:
    def __init__(self):
        self.n_agents = 2
        self.n_actions = 2  # 0 for 'friendly', 1 for 'unfriendly'
    
    def step(self, actions):
        if actions[0] == 0 and actions[1] == 0:
            # Both agents are friendly
            print("Both agents are friendly")
            rewards = [1, 1]
        elif actions[0] == 1 and actions[1] == 1:
            # Both agents are unfriendly
            print("Both agents are unfriendly")
            rewards = [-2, -2]
        elif actions[0] == 0 and actions[1] == 1:
            # First agent is friendly, second is unfriendly
            print("First agent is friendly, second is unfriendly")
            rewards = [-1, 2]
        else:
            # First agent is unfriendly, second is friendly
            print("First agent is unfriendly, second is friendly")
            rewards = [2, -1]

        return rewards

    def reset(self):
        # 实现环境状态的重置逻辑
        return np.zeros(4)  #