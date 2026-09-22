import gymnasium as gym


class Timeout(gym.Wrapper):
	"""
	Wrapper for enforcing a time limit on the environment.
	"""

	def __init__(self, env, max_episode_steps):
		super().__init__(env)
		# Plain attributes (not a read-only property) so domain make_env
		# helpers can assign max_episode_steps under Gymnasium>=1.0.
		self._max_episode_steps = max_episode_steps
		self.max_episode_steps = max_episode_steps

	def reset(self, **kwargs):
		self._t = 0
		return self.env.reset(**kwargs)

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self._t += 1
		done = done or self._t >= self.max_episode_steps
		return obs, reward, done, info
