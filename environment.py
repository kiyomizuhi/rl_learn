import enum

class State():
    def __init__(self, row=-1, column=-1):
        pass

    def __repr__(self):
        pass

    def clone(self):
        pass

    def __hash__(self):
        pass

    def __eq__(self):


class Environment():
    """
    Defines environment.
        - available states
        - available actions
        - transition_function: state, action -> prob
        - reward_function: state, state' -> reward
    """
    def __init__(self, grid, move_prob=0.8):
        pass

    @property
    def row_length(self):
        pass

    @property
    def column_length(self):
        pass

    @property
    def actions(self):
        pass

    @property
    def states(self):
        pass

    def transit_func(self, this_state, action):
        pass

    def can_action_at(self, this_state):
        pass

    def _move(self):
        pass

    def reward_func(self, this_state, next_state):
        pass

    def reset(self):
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, this_state, action):
        next_state, reward, done = self.transit(this_state, action)
        return next_state, reward, done

    def transit(self, this_state, action):
        pass