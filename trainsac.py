"""
SAC Training für dein reales Schrittmotor-Pendel
– KEIN Early Stopping
– KEIN best_model.zip
– Nur ein finales Modell am Ende
– Aber trotzdem top Performance!
"""

import numpy as np
import gymnasium as gym
import torch.nn as nn
import argparse
import time
from stable_baselines3.common.monitor import Monitor
from pendel_env_full import PendelEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# =========================== DEINE WUNSCH-HYPERPARAMETER ===========================
TOTAL_TIMESTEPS = 2_500_000          # Du bestimmst, wann Schluss ist
LEARNING_RATE   = 5e-4
BUFFER_SIZE     = 1_000_000
BATCH_SIZE      = 1024                # SAC liebt das bei deinem Problem
GAMMA           = 0.99
TAU             = 0.01
ENT_COEF        = "auto"             # Bleibt die beste Wahl

# Beste Architektur für dein Pendel (breiter = besser)
POLICY_KWARGS = dict(
    net_arch=[400, 300, 256],
    activation_fn=nn.Tanh,
)

NUM_ENVS = 64                         # Oder 32/128 je nach CPU/RAM

# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Nur evaluieren')
    parser.add_argument('--human', action='store_true', help='Training mit Render')
    parser.add_argument('--model', type=str, default="sac_pendel", help='Modellname')
    args = parser.parse_args()

    if args.eval:
        model = SAC.load(args.model)
        env = PendelEnv(render_mode="human")
        obs, _ = env.reset()
        print("Evaluation läuft – deterministische Policy")

        while True:  # Endlos laufen lassen mit ESC zum Beenden oder Ctrl+C
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(1/60)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()

    else:
        def make_env():
            env = PendelEnv(render_mode="human" if args.human else None)
            env = Monitor(env)
            return env

        # Vektorisierte Umgebung
        if NUM_ENVS == 1:
            train_env = DummyVecEnv([make_env])
        else:
            train_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            tau=TAU,
            gamma=GAMMA,
            ent_coef=ENT_COEF,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=POLICY_KWARGS,
            tensorboard_log="./runs/",
            device="auto",
        )

        print(f"Starte SAC-Training mit {NUM_ENVS} Envs für genau {TOTAL_TIMESTEPS:,} Steps")
        print("– Kein Early Stopping – Kein best_model – Nur finales Modell am Ende")

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name="sac_pendel_final",
            progress_bar=True,
            # KEIN callback!
        )

        model.save(args.model)
        print(f"Training fertig → Modell gespeichert als: {args.model}.zip")
        train_env.close()