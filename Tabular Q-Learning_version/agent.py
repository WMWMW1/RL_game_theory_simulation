# agent.py

import numpy as np
import math
import random

class QAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, c=1.0):
        """
        Initializes the Q-learning agent with UCB exploration.

        Args:
            n_actions (int): Number of possible actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            c (float): Exploration parameter for UCB.
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.c = c  # Exploration parameter for UCB
        self.q_table = {}  # Q-table {state: {action: value}}
        self.state_counts = {}  # Count of state visits N(s)
        self.action_counts = {}  # Count of state-action pairs N(s,a)

    def get_state_key(self, state):
        """
        Converts the state array into a tuple to be used as a key in Q-table.

        Args:
            state (numpy.ndarray): Current state.

        Returns:
            tuple: Tuple representation of the state.
        """
        return tuple(state.astype(int))

    def choose_action(self, state):
        """
        Chooses an action based on the UCB exploration strategy.

        Args:
            state (numpy.ndarray): Current state.

        Returns:
            int: Chosen action.
        """
        state_key = self.get_state_key(state)
        
        # Initialize Q-values and counts if state is new
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in range(self.n_actions)}
            self.state_counts[state_key] = 0
            self.action_counts[state_key] = {action: 0 for action in range(self.n_actions)}
        
        # Increment state visit count
        self.state_counts[state_key] += 1
        total_state_visits = self.state_counts[state_key]
        
        # Compute UCB values for each action
        ucb_values = []
        for action in range(self.n_actions):
            q_value = self.q_table[state_key][action]
            action_count = self.action_counts[state_key][action]
            
            if action_count == 0:
                # Assign a large bonus to ensure each action is tried at least once
                ucb = float('inf')
            else:
                # UCB1 formula
                exploration_bonus = self.c * math.sqrt(math.log(total_state_visits) / action_count)
                ucb = q_value + exploration_bonus
            ucb_values.append(ucb)
        
        # Choose the action with the highest UCB value
        chosen_action = np.argmax(ucb_values)
        return chosen_action

    def update_q_value(self, state, action, reward, next_state, done):
        """
        Updates the Q-table based on the transition.

        Args:
            state (numpy.ndarray): Previous state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (numpy.ndarray): Current state.
            done (bool): Whether the episode has ended.
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values and counts for next_state if not present
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in range(self.n_actions)}
            self.state_counts[next_state_key] = 0
            self.action_counts[next_state_key] = {a: 0 for a in range(self.n_actions)}
        
        # Current Q value
        current_q = self.q_table[state_key][action]
        
        # Max future Q value
        if done:
            max_future_q = 0.0
        else:
            max_future_q = max(self.q_table[next_state_key].values())
        
        # Q-learning update rule
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Increment action count
        self.action_counts[state_key][action] += 1

    def reset_counts(self):
        """
        Resets the state and action counts. Useful if starting fresh.
        """
        self.state_counts = {}
        self.action_counts = {}
