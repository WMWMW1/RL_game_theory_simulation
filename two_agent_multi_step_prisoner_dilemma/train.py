# train.py

import torch
import numpy as np
from agent import PPOAgent, PolicyNetwork
from environment import MultiAgentEnvironment

def main():
    # Initialize environment and agents
    env = MultiAgentEnvironment()
    state_dim = env.n_agents * env.history_length * 2  # 2 agents * history_length * 2 actions
    action_dim = env.n_actions
    
    # Initialize policy networks
    agent1_policy_net = PolicyNetwork(state_dim, action_dim)
    agent2_policy_net = PolicyNetwork(state_dim, action_dim)
    
    # Initialize PPO agents
    agent1 = PPOAgent(agent1_policy_net, action_dim)
    agent2 = PPOAgent(agent2_policy_net, action_dim)
    
    num_episodes = 1000  # Total number of training episodes
    update_every = 5      # Update agents every 'update_every' episodes
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        episode_reward1 = 0
        episode_reward2 = 0
        
        while not done:
            # Agents select actions
            action1, log_prob1, value1 = agent1.select_action(state)
            action2, log_prob2, value2 = agent2.select_action(state)
            
            # Environment responds
            new_state, rewards, done = env.step([action1, action2])
            reward1, reward2 = rewards
            episode_reward1 += reward1
            episode_reward2 += reward2
            
            # Store transitions
            agent1.store_transition(state, action1, log_prob1, reward1, done, value1)
            agent2.store_transition(state, action2, log_prob2, reward2, done, value2)
            
            state = new_state
        
        # After episode ends, perform PPO update
        agent1.update()
        agent2.update()
        
        if episode % 50 == 0:
            print(f"Episode {episode} | Agent1 Total Reward: {episode_reward1:.2f} | Agent2 Total Reward: {episode_reward2:.2f}")
    
    print("Training completed.")

if __name__ == "__main__":
    main()
