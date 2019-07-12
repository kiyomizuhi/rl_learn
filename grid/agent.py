import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Args:
        state

    Return:
        action
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def policy(self):
        pass

class RandomAgent(Agent):
    """
    Args:
        state

    Return:
        action
    """
    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        return np.random.choice(self.actions)

class OffPolicyEpsilonGreedyAgent(Agent):
    """
    eps
    """
    def __init__(self, env, epsilon):
        self.actions = env.actions
        self._epsilon = epsilon
        self.state_to_index = dict([(state, i) for i, state in env.states])
        self.Q = [list(range(4))] * len(env.states)

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self._epsilon = epsilon

    def policy(self, state):
        if np.random.rand() < self._epsilon:
            return np.random.choice(self.actions)
        else:
            idx_state = self.state_to_index[state]
            idx_action = np.argmax(self.Q[idx_state])
            return self.actions[idx_action]

    def train(self):

class OnPolicyEpsilonGreedyAgent(Agent):
    pass

if __name__ == '__main__':
    from env_config import *
    env = Environment(grid)
    agent = RandomAgent(env)