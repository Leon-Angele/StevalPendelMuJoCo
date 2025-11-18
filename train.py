"""
TD3 Training für PendelEnv
Alle Hyperparameter und policy_kwargs sind einfach editierbar.
"""

import numpy as np
import gymnasium as gym
import torch.nn as nn
import argparse
import time
from stable_baselines3.common.monitor import Monitor 


from pendel_env import PendelEnv
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

# --- Hyperparameter ---
TOTAL_TIMESTEPS = 10_000_000
LEARNING_RATE = 3e-4
BUFFER_SIZE = 500_000
BATCH_SIZE = 128
TRAIN_FREQ = 1
# TRAIN_FREQ = (1, "episode")
GRADIENT_STEPS = 1
GAMMA = 0.99
TAU = 0.005
POLICY_DELAY = 2
TARGET_POLICY_NOISE = 0.2
TARGET_NOISE_CLIP = 0.5

# Noise Definition
ACTION_NOISE = NormalActionNoise(
    mean=np.zeros(1),
    sigma=0.1 * np.ones(1)
)

# Netzwerk Architektur
POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[512, 512],      # Actor-Netzwerk
        qf=[512, 512]       # Critic-Netzwerk
    ),
    activation_fn=nn.ReLU,
)

NUM_ENVS = 32  # Anzahl paralleler Envs beim Training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Nur Modell evaluieren, nicht trainieren')
    parser.add_argument('--human', action='store_true', help='Training mit Render-Modus "human"')
    args = parser.parse_args()

    # --- Optionaler Check (nur einmalig zum Debuggen) ---
    # temp_env = PendelEnv(render_mode=None)
    # check_env(temp_env)
    # temp_env.close()

    if args.eval:
        # Modell laden und evaluieren
        try:
            model = TD3.load("td3_pendel")
            print("Modell geladen.")
        except FileNotFoundError:
            print("Kein gespeichertes Modell gefunden. Bitte erst trainieren.")
            exit()

        env = PendelEnv(render_mode="human")
        obs, _ = env.reset()
        print("Starte Evaluation...")
        
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Realzeit-Sync für ~60 FPS
            time.sleep(1/60)
            
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()

    else:
        # --- TRAINING ---
        
        # Render-Modus Logik
        train_render_mode = "human" if args.human else None
        
        # Wrapper-Funktion für parallele Umgebungen
        def make_env():
            env = PendelEnv(render_mode=train_render_mode)
            # --- KORREKTUR: Monitor Wrapper hinzufügen ---
            # Dies ermöglicht das Logging von ep_rew_mean und ep_len_mean
            env = Monitor(env)
            return env

        # Vektorisierte Umgebung erstellen
        if NUM_ENVS == 1:
            train_env = DummyVecEnv([make_env])
        else:
            # Bei SubprocVecEnv werden separate Prozesse gestartet
            train_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

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

        print(f"Starte Training mit {NUM_ENVS} Environments für {TOTAL_TIMESTEPS} Steps...")
        
        # Training starten
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name="td3_run",
            progress_bar=True
            # log_interval=4 # Optional: Wie oft (in Episoden) geloggt wird
        )
        
        # Modell speichern
        model.save("td3_pendel")
        print("Training abgeschlossen und Modell gespeichert.")
        
        train_env.close()
        


