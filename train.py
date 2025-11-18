"""Train script for StevalPendelEnv using Stable-Baselines3 TD3.

This script provides configurable hyperparameters at the top, wraps the
environment with a Monitor for logging, writes tensorboard logs, uses
verbose=2 and shows a tqdm progress bar during training.

Usage examples:
  python train.py
  python train.py --total-timesteps 500000

Make sure you have a Python environment with: stable-baselines3, gymnasium,
tqdm, tensorboard, and mujoco installed.
"""

from pathlib import Path
import argparse
import os
import sys
import time

import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

# Make sure local package can be imported for the custom env
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

from stevalPendelEnv import StevalPendelEnv


# =========================
# Hyperparameters (edit here)
# =========================
ENV_ID = "StevalPendel-v0"
TOTAL_TIMESTEPS = 50_000
LEARNING_RATE = 1e-3
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
GAMMA = 0.99
TAU = 0.005
VERBOSE = 2
TENSORBOARD_LOG_DIR = str(ROOT / "logs" / "tb")
MODEL_SAVE_PATH = str(ROOT / "models" / "td3_steval_pendel")
MONITOR_DIR = str(ROOT / "logs" / "monitor")


class TQDMProgressBarCallback(BaseCallback):
	"""Progress bar callback that updates a tqdm bar using timesteps.

	It expects `total_timesteps` to be provided via the callback's `__init__`.
	"""
	def __init__(self, total_timesteps: int, verbose=0):
		super().__init__(verbose)
		self.total_timesteps = int(total_timesteps)
		self.pbar = None

	def _on_training_start(self) -> None:
		# Create progress bar at start
		self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="steps")

	def _on_step(self) -> bool:
		# Update bar by number of environment steps in this call
		# `self.num_timesteps` is the total timesteps done so far
		if self.pbar is None:
			return True
		self.pbar.n = min(self.num_timesteps, self.total_timesteps)
		self.pbar.refresh()
		return True

	def _on_training_end(self) -> None:
		if self.pbar is not None:
			self.pbar.close()


def make_env(render_mode=None):
	def _init():
		env = StevalPendelEnv(render_mode=render_mode)
		return env
	return _init


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
	parser.add_argument("--tensorboard-log", type=str, default=TENSORBOARD_LOG_DIR)
	parser.add_argument("--save-path", type=str, default=MODEL_SAVE_PATH)
	parser.add_argument("--render", action="store_true", help="Run env in human render mode during training (slows down)")
	args = parser.parse_args()

	total_timesteps = args.total_timesteps

	# Create folders
	Path(args.tensorboard_log).mkdir(parents=True, exist_ok=True)
	Path(MONITOR_DIR).mkdir(parents=True, exist_ok=True)
	Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

	# Vectorized env with Monitor
	render_mode = "human" if args.render else None
	env = DummyVecEnv([make_env(render_mode=render_mode)])
	# Wrap with Monitor for logging episode rewards & lengths
	env.envs[0] = Monitor(env.envs[0], filename=os.path.join(MONITOR_DIR, "monitor.csv"))

	# Logger for stable-baselines3
	new_logger = configure(folder=args.tensorboard_log, format_strings=["stdout", "tensorboard"])

	# Model
	policy_kwargs = dict()
	model = TD3(
		"MlpPolicy",
		env,
		learning_rate=LEARNING_RATE,
		buffer_size=BUFFER_SIZE,
		batch_size=BATCH_SIZE,
		gamma=GAMMA,
		tau=TAU,
		policy_noise=POLICY_NOISE,
		noise_clip=NOISE_CLIP,
		policy_freq=POLICY_FREQ,
		verbose=VERBOSE,
		tensorboard_log=args.tensorboard_log,
		policy_kwargs=policy_kwargs,
	)
	model.set_logger(new_logger)

	# Callbacks
	tqdm_cb = TQDMProgressBarCallback(total_timesteps)

	# Train
	print(f"Starting training for {total_timesteps} timesteps")
	start = time.time()
	model.learn(total_timesteps=total_timesteps, callback=tqdm_cb)
	duration = time.time() - start

	# Save
	model.save(args.save_path)
	print(f"Model saved to {args.save_path}")
	print(f"Training took {duration:.1f} seconds")

	env.close()


if __name__ == "__main__":
	main()

