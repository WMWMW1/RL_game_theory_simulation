# train.py

import numpy as np
from agent import QAgent
from environment import MultiAgentEnvironment

def main():
    # Initialize environment and agents
    env = MultiAgentEnvironment()
    n_actions = env.n_actions

    # Initialize two agents with UCB exploration
    c_parameter = 2.0  # Exploration parameter for UCB
    agent1 = QAgent(n_actions=n_actions, c=c_parameter)
    agent2 = QAgent(n_actions=n_actions, c=c_parameter)

    num_episodes = 10 # Maximum number of training episodes
    max_steps = env.max_steps_per_episode
    delta_threshold = 1e-4  # Convergence threshold

    # Track rewards
    rewards_history = {'agent1': [], 'agent2': []}

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward1 = 0
        total_reward2 = 0

        # Track maximum Delta for convergence
        max_delta1 = 0
        max_delta2 = 0

        for step in range(max_steps):
            # Agents choose actions based on UCB
            action1 = agent1.choose_action(state)
            action2 = agent2.choose_action(state)

            # Environment responds
            new_state, rewards, done = env.step([action1, action2])
            reward1, reward2 = rewards
            total_reward1 += reward1
            total_reward2 += reward2

            # Update Q-tables and get Delta
            agent1.update_q_value(state, action1, reward1, new_state, done)
            agent2.update_q_value(state, action2, reward2, new_state, done)

            state = new_state

            if done:
                break

        # Record rewards
        rewards_history['agent1'].append(total_reward1)
        rewards_history['agent2'].append(total_reward2)

        # Check convergence
        # Since UCB doesn't use Delta for convergence in this setup,
        # you can implement a different convergence criterion or rely on fixed episodes.
        # Alternatively, track recent reward improvements.

        # Logging
        if episode % 50 == 0:
            avg_reward1 = np.mean(rewards_history['agent1'][-50:])
            avg_reward2 = np.mean(rewards_history['agent2'][-50:])
            print(f"Episode {episode} | Avg Reward Agent1: {avg_reward1:.2f} | Avg Reward Agent2: {avg_reward2:.2f}")

    print("Training completed.")

    # Optional: Save Q-tables for later use
    # import pickle
    # with open('agent1_q_table.pkl', 'wb') as f:
    #     pickle.dump(agent1.q_table, f)
    # with open('agent2_q_table.pkl', 'wb') as f:
    #     pickle.dump(agent2.q_table, f)

if __name__ == "__main__":
    main()
