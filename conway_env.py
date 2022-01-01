import gym
import numpy as np
from lib import fft_convolve2d


class ConwayEnv(gym.Env):

    def __init__(self, action_shape=(3, 3), state_shape=(8, 8), goal_location=(5, 5), start_state=None, k=None):
        """
        state_shape dimensions must be even!!
        """
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.action_space = gym.spaces.MultiBinary(action_shape)
        self.observation_space = gym.spaces.MultiBinary(state_shape)

        if start_state is None:
            start_state = np.zeros(state_shape, dtype=np.int8)
        self.state = start_state
        self.goal_location = goal_location
        self.state_reset()
        self.goal_view.fill(1)

        if k is None:
            m, n = state_shape
            k = np.zeros((m, n))
            k[m // 2 - 1: m // 2 + 2, n // 2 - 1: n // 2 + 2] = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.k = k

    def step(self, action):
        # apply actions
        np.logical_xor(action, self.action_view, out=self.action_view, dtype=np.int8, casting='unsafe')

        # Shit there is environment wrap-around. TODO: Fix it
        b = fft_convolve2d(self.state, self.k).round()
        c = np.zeros(b.shape)

        c[np.where((b == 2) & (self.state == 1))] = 1
        c[np.where((b == 3) & (self.state == 1))] = 1

        c[np.where((b == 3) & (self.state == 0))] = 1
        self.state = c.astype(np.int8)
        self.state_reset()

        done = not not np.all(np.logical_not(self.goal_view).astype(np.int8))

        # This reward function encourages keeping at least one square 'on' which prevents termination
        # reward = float(np.sum(np.logical_not(self.goal_view).astype(np.int8)))

        reward = 10.0 if done else 0.0

        return self.state, reward, done, {}

    def state_reset(self):
        self.action_view = self.state[1:1 + self.action_shape[0], 1:1 + self.action_shape[1]]
        self.goal_view = self.state[self.goal_location[0]:self.goal_location[0] + 2,
                         self.goal_location[1]:self.goal_location[1] + 2]

    def reset(self):
        start_state = np.zeros(self.state_shape, dtype=np.int8)
        self.state = start_state
        self.state_reset()
        self.goal_view.fill(1)
        return self.state


class FlatObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_shape = self.observation_space.n
        self.observation_space = gym.spaces.MultiBinary(n=np.product(self.observation_space.n))

    def observation(self, obs):
        observation = obs.reshape(np.product(self.observation_space.n))
        return observation


class FlatActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_shape = self.action_space.n
        self.action_space = gym.spaces.MultiBinary(n=np.product(self.action_space.n))

    def action(self, act):
        action = act.reshape(self.original_shape)
        return action
