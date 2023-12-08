"""
train.py

Purpose:
    The main training script for a multi-agent reinforcement learning environment using
    the Proximal Policy Optimization (PPO) algorithm.

Description:
    - Initializes the learning environment and two agents with policy networks.
    - Conducts training over a specified number of episodes.
    - In each episode, executes a fixed number of decision steps (4 in this case).
    - Agents select actions based on their policy networks and the environment state.
    - Environment calculates rewards based on these actions.
    - Agents are trained using the computed rewards.

Variables:
    - state_dim: Dimension of the state space in the environment.
    - action_dim: Number of possible actions in the action space.
    - num_episodes: Total number of training episodes.
"""
import torch
import numpy as np
from agent import PPOAgent, PolicyNetwork
from environment import MultiAgentEnvironment

def main():
    # 初始化环境和智能体
    env = MultiAgentEnvironment()
    state_dim = 2 * 2* 2  # 现在的状态维度为 2 agents * 4 history_length * 2 actions
    action_dim = env.n_actions
    agent1_policy_net = PolicyNetwork(state_dim, action_dim)
    agent2_policy_net = PolicyNetwork(state_dim, action_dim)

    agent1 = PPOAgent(agent1_policy_net,action_dim)
    agent2 = PPOAgent(agent2_policy_net,action_dim)

    num_episodes = 1000  # 总的训练回合数

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action1, _ = agent1.select_action(state)
            action2, _ = agent2.select_action(state)

            new_state, rewards, done = env.step([action1, action2])

            if not done:
                # 训练智能体
                agent1.train(state, action1, rewards[0])
                agent2.train(state, action2, rewards[1])
            else:
                # 使用累积奖励进行训练
                agent1.train(state, action1, rewards[0])
                agent2.train(state, action2, rewards[1])

            state = new_state


if __name__ == "__main__":
    main()
