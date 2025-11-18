"""
TD3 Training für PendelEnv
Alle Hyperparameter und policy_kwargs sind einfach editierbar.
"""

import numpy as np
import gymnasium as gym
import torch.nn as nn
import argparse
from pendel_env import PendelEnv
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import time

# Environment erstellen
env = PendelEnv(render_mode=None)

# Optional: Environment-Check (hilft bei Fehlern)
check_env(env)

# TD3 Hyperparameter
TOTAL_TIMESTEPS = 200_000
LEARNING_RATE = 3e-4
BUFFER_SIZE = 500_000
BATCH_SIZE = 256
TRAIN_FREQ = 1
#TRAIN_FREQ = (1, "episode")
GRADIENT_STEPS = 1
GAMMA = 0.99
TAU = 0.005
POLICY_DELAY = 2
TARGET_POLICY_NOISE = 0.2
TARGET_NOISE_CLIP = 0.5
ACTION_NOISE = NormalActionNoise(
    mean=np.zeros(1),   
    sigma=0.1 * np.ones(1)  
)

POLICY_KWARGS = dict(
	net_arch=dict(
		pi=[128, 128],      # Actor-Netzwerk
		qf=[512, 512]       # Critic-Netzwerk
	),
	activation_fn=nn.ReLU,
)

class RewardTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.writer = None
        self.episode_rewards = []

    def _on_training_start(self) -> None:
        self.writer = SummaryWriter(log_dir=self.model.tensorboard_log)

    def _on_step(self) -> bool:
        # Episodenreward aus Infos extrahieren
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][-1]
            reward = self.locals["rewards"][-1]
            self.episode_rewards.append(reward)
            if info.get("episode"):
                mean_reward = np.mean(self.episode_rewards[-info["episode"]["l"]:])
                self.writer.add_scalar("reward/mean", mean_reward, self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self.writer.close()

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', help='Nur Modell evaluieren, nicht trainieren')
args = parser.parse_args()

if args.eval:
	# Modell laden und evaluieren
	model = TD3.load("td3_pendel")
	env = PendelEnv(render_mode="human")
	obs, _ = env.reset()
	print("Starte Evaluation...")
	for _ in range(1500):
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = env.step(action)
		# Realzeit-Sync für 60 FPS
		time.sleep(1/60)
		if terminated or truncated:
			obs, _ = env.reset()
	env.close()
else:
	# Parallele Envs für Training
	train_env = PendelEnv(render_mode="none")

	# TD3-Agent initialisieren
	model = TD3(
		"MlpPolicy",
		train_env,
		verbose=2,
		learning_rate=LEARNING_RATE,
		buffer_size=BUFFER_SIZE,
		batch_size=BATCH_SIZE,
		train_freq=TRAIN_FREQ,
		gradient_steps=GRADIENT_STEPS,
		gamma=GAMMA,
		tau=TAU,
		policy_delay=POLICY_DELAY,
		target_policy_noise=TARGET_POLICY_NOISE,
		target_noise_clip=TARGET_NOISE_CLIP,
		action_noise=ACTION_NOISE,
		policy_kwargs=POLICY_KWARGS,
		tensorboard_log="./runs/"
	)

	# Training starten
	callback = RewardTensorboardCallback()
	#model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="td3_run", progress_bar=True, callback=callback)
	model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="td3_run", progress_bar=True) 
	# Modell speichern
	model.save("td3_pendel")
	env.close()