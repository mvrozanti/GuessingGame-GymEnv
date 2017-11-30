import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import guessing_game
import matplotlib.pyplot as plt

class GuessingGameSolver():
	def __init__(self, n_episodes=1000, n_win_ticks=20, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01,
	             epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=256, monitor=False, quiet=False, plot=False):
		self.memory = deque(maxlen=100000)
		self.env = gym.make('guessing_game-v0')
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_log_decay
		self.alpha = alpha
		self.alpha_decay = alpha_decay
		self.n_episodes = n_episodes
		self.n_win_ticks = n_win_ticks
		self.batch_size = batch_size
		self.quiet = quiet
		self.plot = plot
		if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
		# Init models
		self.model = self._build_model()
		self.target_model = self._build_model()

	def _build_model(self):
		model = Sequential()
		model.add(Dense(3, input_dim=2, activation='tanh'))
		model.add(Dense(3, activation='tanh'))
		model.add(Dense(self.env.action_space.n, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
		return model


	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def choose_action(self, state, epsilon):
		return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(
			self.model.predict(state))

	def get_epsilon(self, t):
		return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

	def preprocess_state(self, state):
		return np.reshape(state, [1,2])

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
		for state, action, reward, next_state, done in minibatch:
			target = self.model.predict(state)
			if done:
				target[0][action] = reward
			else:
				a = self.model.predict(next_state)[0]
				t = self.target_model.predict(next_state)[0]
				target[0][action] = reward + self.gamma * t[np.argmax(a)]
			self.model.fit(state, target, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def update_target_model(self):
		# copy weights from model to target_model
		self.target_model.set_weights(self.model.get_weights())

	def run(self):
		try: self.model.load_weights('guessing.ai')
		except: print('COULD NOT LOAD WEIGHTS')
		scores = deque(maxlen=100)
		# if self.plot: plt.ion()
		for e in range(self.n_episodes):
			state = self.env.reset()
			state = self.preprocess_state(state)
			done = False
			i = 0
			total_reward = 0
			while not done:
				action = self.choose_action(state, self.get_epsilon(e))
				next_state, reward, done, _ = self.env.step(action)
				total_reward += reward
				next_state = self.preprocess_state(next_state)
				self.remember(state, action, reward, next_state, done)
				state = next_state
				i += 1
			self.update_target_model()
			if self.plot:
				plt.scatter(e, total_reward)
				plt.pause(0.05)
			print(total_reward/i)
			scores.append(i)
			mean_score = np.mean(scores)
			if mean_score <= self.n_win_ticks and e >= 100:
				if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
				return e - 100
			if e % 100 == 0 and not self.quiet:
				print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
			self.replay(self.batch_size)
			self.model.save_weights('guessing.ai')
		if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
		return e


if __name__ == '__main__':
	agent = GuessingGameSolver(alpha=0.1, epsilon=0, plot=True)
	agent.run()