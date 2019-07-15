import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

class Agent(ABC):
    """
    abstract base class of Agent
    """
    @abstractmethod
    def policy(self):
        pass

    @abstractmethod
    def log_experience(self):
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
        self.log_experiences = []

    def policy(self, state):
        return np.random.choice(self.actions)

    def log_experience(self, exp):
        self.log_exps.append(exp)

class OffPolicyEpsilonGreedyAgent(Agent):
    """
    eps
    """
    def __init__(self, env, epsilon):
        self.actions = env.actions
        self.__epsilon = epsilon
        self.state_to_index = dict([(state, i) for i, state in env.states])
        self.Q = [list(range(4))] * len(env.states)
        self.log_experiences = []

    @property
    def epsilon(self):
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self.__epsilon = epsilon

    def policy(self, s):
        if np.random.rand() < self.__epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.Q[s])

    def log_experience(self, exp):
        self.log_exps.append(exp)

    def train(self):
        pass

class GymFrozenLakeEpsilonGreedyAgent(Agent):
    """
    eps
    """
    def __init__(self, env, epsilon=0.5):
        self.env = env
        self.gamma = 0.95
        self.__epsilon = epsilon
        self.actions = list(range(env.action_space.n))
        self.log_experiences = []
        self.log_episodes = []
        print(env.render())

    @property
    def epsilon(self):
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self.__epsilon = epsilon

    def policy(self, s):
        if np.random.rand() < self.__epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.Q[s])

    def init_log_experiences(self):
        self.log_experiences = []

    def init_Q(self):
        self.Q = defaultdict(lambda: 0.001*np.random.rand(self.env.action_space.n))

    def init_sa_counts(self):
        self.sa_counts = defaultdict(lambda: [0] * self.env.action_space.n)

    def log_experience(self, exp):
        self.log_experiences.append(exp)

    def log_episode(self):
        self.log_episodes.append(self.log_experiences)

    def train(self, num_episodes=1000):
        self.init_Q()
        self.init_sa_counts()
        for _ in range(num_episodes):
            self.init_log_experiences()
            # self.init_sa_counts()
            self.train_episode()
            self.log_episode()
            self.epsilon = 0.999 * self.epsilon

    def train_episode(self):
        this_s = self.env.reset()
        done = False
        while not done:
            a = self.policy(this_s)
            next_s, reward, done, info = self.env.step(a)
            reward = self.env.dict_state_reward[next_s]
            experience = (this_s, next_s, a, reward, done)
            self.log_experience(experience)
            this_s = next_s

class MonteCarloAgent(GymFrozenLakeEpsilonGreedyAgent):
    def __init__(self, env):
        super(MonteCarloAgent, self).__init__(env)

    def train_episode(self):
        super().train_episode()
        self.update_Q()

    def update_Q(self):
        for i, exp0 in enumerate(self.log_experiences):
            s, _, a, _, _ = exp0
            gs = [np.power(self.gamma, j) * exp1[3] for j, exp1 in enumerate(self.log_experiences[i:])]
            G = sum(gs)
            self.sa_counts[s][a] += 1
            alpha = 1 / self.sa_counts[s][a]
            self.Q[s][a] += alpha * (G - self.Q[s][a])









if __name__ == '__main__':
    from env_config import *
    env = Environment(grid)
    agent = RandomAgent(env)