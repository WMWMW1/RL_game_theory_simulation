# environment.py

import numpy as np
import torch

class MultiAgentEnvironment:
    def __init__(self):
        self.n_agents = 2
        self.n_actions = 2  # 0: Friendly, 1: Unfriendly
        self.history_length = 1
        self.reset()
        self.max_steps_per_episode = 10

    def step(self, actions):
        # Compute immediate reward based on actions
        if actions[0] == 0 and actions[1] == 0:
            # Both agents are friendly
            print("Both agents are friendly")
            rewards = [1, 1]
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

        # Update history
        actions_tensor = torch.tensor(actions, dtype=torch.float32).view(self.n_agents, -1)
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1, :] = actions_tensor

        # New state includes history
        new_state = self.history.view(-1).numpy()

        # Update cumulative rewards
        for i in range(self.n_agents):
            self.cumulative_rewards[i] += rewards[i]

        self.current_step += 1
        done = self.is_terminal()

        if done:
            final_rewards = self.cumulative_rewards.tolist()
            self.cumulative_rewards = np.zeros(self.n_agents)  # Reset cumulative rewards
            return new_state, final_rewards, done
        else:
            return new_state, rewards, done

    def is_terminal(self):
        return self.current_step >= self.max_steps_per_episode

    def reset(self):
        self.history = torch.full((self.n_agents, self.history_length, 2), 3, dtype=torch.float32)
        self.cumulative_rewards = np.zeros(self.n_agents)
        self.current_step = 0
        return self.history.view(-1).numpy()
