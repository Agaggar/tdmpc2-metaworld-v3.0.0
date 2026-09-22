import numpy as np
import gymnasium as gym
from envs.wrappers.timeout import Timeout

# Local Meta-World install is V3 + Gymnasium (no ALL_V2_*). Keep the
# historical alias so task ids still look like the TD-MPC2 paper naming.
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.camera_name = "corner2"
		# Match TD-MPC2 Meta-World camera framing from the original wrapper.
		self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
		self.env.camera_name = self.camera_name
		if getattr(self.env, "mujoco_renderer", None) is not None:
			self.env.mujoco_renderer.camera_name = self.camera_name
			self.env.mujoco_renderer.width = 384
			self.env.mujoco_renderer.height = 384
		self.env._freeze_rand_vec = False

	def reset(self, **kwargs):
		obs, _ = self.env.reset(**kwargs)
		obs = obs.astype(np.float32)
		# Side-effect zero action (same as upstream TD-MPC2); return reset obs.
		self.env.step(np.zeros(self.env.action_space.shape, dtype=np.float32))
		return obs

	def step(self, action):
		reward = 0
		info = {}
		for _ in range(2):
			obs, r, terminated, truncated, info = self.env.step(action.copy())
			reward += r
		obs = obs.astype(np.float32)
		# Timeout owns episode termination; keep done=False here (upstream behavior).
		info = dict(info)
		info["terminated"] = False
		return obs, reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return np.asarray(self.env.render()).copy()


def make_env(cfg):
	"""
	Make Meta-World environment.
	"""
	env_id = cfg.task.split("-", 1)[-1] + "-v3-goal-observable"
	if not cfg.task.startswith('mw-') or env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=cfg.seed, render_mode="rgb_array")
	env = MetaWorldWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=100)
	env.max_episode_steps = env._max_episode_steps
	return env
