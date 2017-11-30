import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class GuessingGame(gym.Env):
	def __init__(self, level=2):
		self.action_space = spaces.Discrete(10 ** level)
		self.observation_space = spaces.Discrete(2)
		self._seed()
		self._state = self._reset()

	def _step(self, action):
		assert self.action_space.contains(action)
		return self._state, 1000 if action == self._state.sum() else -abs(action - self._state.sum()), action == self._state.sum(), {}

	def _reset(self):
		a = self.np_random.choice([i for i in range(int(self.action_space.n / 2))])
		b = self.np_random.choice([i for i in range(int(self.action_space.n / 2))])
		self._state = np.array([a, b], dtype=np.float32)
		return self._state

	def get_state(self):
		return self._state

	def _configure(self):
		pass

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
