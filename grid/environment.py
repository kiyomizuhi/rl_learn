import enum
import numpy as np

class Action(enum.Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

class State():
    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, state):
        return self.row == state.row and self.column == state.column

class Environment():
    """
    Defines environment.
        - available states
        - available actions
        - transition_function: state, action -> prob
        - reward_function: state, state' -> reward
    """
    def __init__(self, grid, forward_prob=0.8):
        self.grid = grid
        self.agent_state = State()
        self.default_reward = -0.04
        self.forward_prob = forward_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN,
                Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Block cells are not included to the state.
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    def reset(self):
        self.agent_state = State(0, 0)
        return self.agent_state

    def trans_func(self, state, action):
        trans_probs = {}
        if not self._check_normal_state(state):
            return trans_probs

        av = action.value
        # action_left = Action((av + 1) % 4)
        action_back = Action((av + 2) % 4)
        # action_right = Action((av + 3) % 4)
        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.forward_prob
            elif a != action_back:
                prob = (1 - self.forward_prob) / 2

            next_state = self._move(state, a)
            if next_state not in trans_probs.keys():
                trans_probs[next_state] = prob
            else:
                trans_probs[next_state] += prob

        return trans_probs

    def reward_func(self, state):
        # in general, reward func depends both on
        # current state and next state
        reward = self.default_reward
        done = False
        if self.grid[state.row][state.column] == 1:
            reward = 1
            done = True
        elif self.grid[state.row][state.column] == -1:
            reward = -1
            done = True
        return reward, done

    def step(self, action):
        trans_probs = self.trans_func(self.agent_state, action)
        next_states = []
        ps = []
        for s, p in trans_probs.items():
            next_states.append(s)
            ps.append(p)
        next_state = np.random.choice(next_states, p=ps)
        reward, done = self.reward_func(next_state)
        if next_state is not None:
            self.agent_state = next_state
        return next_state, reward, done

    def _check_normal_state(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _check_blocked_state(self, state):
        if self.grid[state.row][state.column] == 9:
            return False
        else:
            return True

    def _check_within_grid(self, state):
        if not (0 <= state.row < self.row_length):
            return False
        if not (0 <= state.column < self.column_length):
            return False
        return True

    def _move(self, this_state, action):
        next_state = this_state.clone()

        if action == Action.UP:
            next_state.row += 1
        elif action == Action.DOWN:
            next_state.row -= 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1
        if not self._check_within_grid(next_state):
            next_state = this_state
        if not self._check_blocked_state(next_state):
            next_state = this_state
        return next_state
