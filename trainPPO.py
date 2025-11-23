"""
Recurrent PPO (RNN-PPO) Training für Furuta-Pendel (partial observability)
- Nur Pendel-Winkel und Pendel-Winkelgeschwindigkeit als Observation
- LSTM-Policy → kann fehlenden Arm-Zustand aus Historie rekonstruieren
- Unterstützt --load zum Fortsetzen und --eval für Demo
"""

import os
import time
import argparse
import numpy as np
import gymnasium as gym
import torch.nn as nn
from pendel_env import PendelEnv 

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sb3_contrib import RecurrentPPO  # <-- WICHTIG: aus sb3-contrib!

# -------------------------- Hyperparameter --------------------------
TOTAL_TIMESTEPS = 5_000_000

# PPO Hyperparameter (gut für rekurrente Policies)
LEARNING_RATE   = 3e-4
N_STEPS         = 1000       # Rollout-Länge pro Environment
BATCH_SIZE      = 64         # Mini-Batch-Größe (klein bei RNNs empfohlen)
N_EPOCHS        = 10
GAMMA           = 0.995
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
ENT_COEF        = 0.01       # etwas Entropie hilft bei Exploration
VF_COEF         = 0.5
MAX_GRAD_NORM   = 0.5

NUM_ENVS = 32  # parallele Environments

# LSTM & Netz-Architektur
POLICY_KWARGS = dict(
    activation_fn=nn.Tanh,
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    # --- LSTM spezifisch ---
    n_lstm_layers=1,
    lstm_hidden_size=256,
    shared_lstm=False,           # separate LSTMs für Actor & Critic (besser)
    enable_critic_lstm=True,     # Critic braucht auch Historie bei POMDPs!
)

# -------------------------- Environment --------------------------
def make_env(render_mode=None):
    env = PendelEnv(render_mode=render_mode, max_steps=1000)  # dein Furuta-Env
    env = Monitor(env)
    return env


# -------------------------- Main --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Nur evaluieren (mit Rendering)")
    parser.add_argument("--human", action="store_true", help="Training mit human rendering (langsam!)")
    parser.add_argument("--load", action="store_true", help="Vorhandenes Modell + Stats laden und weitertrainieren")
    args = parser.parse_args()

    MODEL_PATH  = "rppo_furuta_pendel"
    STATS_PATH  = "vec_normalize_rppo.pkl"

    # ---------------------- Evaluation Mode ----------------------
    if args.eval:
        print("=== EVALUATION MODE ===")
        env = DummyVecEnv([lambda: make_env(render_mode="human")])
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False
        env.norm_reward = False

        model = RecurrentPPO.load(MODEL_PATH, env=env)

        obs = env.reset()
        lstm_states = None                   # initial hidden states
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        print("Starte Rendering... (Strg+C zum Beenden)")
        try:
            while True:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )
                obs, reward, done, info = env.step(action)
                episode_starts = done
                time.sleep(0.02)
        except KeyboardInterrupt:
            env.close()

    # ---------------------- Training Mode ----------------------
    else:
        print("=== TRAINING MODE ===")

        # Basis-Envs (mit oder ohne Rendering)
        if args.human:
            base_env = DummyVecEnv([lambda: make_env(render_mode="human")])
        else:
            base_env = DummyVecEnv([lambda: make_env() for _ in range(NUM_ENVS)])

        # --- Laden oder neu erstellen ---
        if args.load and os.path.exists(MODEL_PATH + ".zip") and os.path.exists(STATS_PATH):
            print(f"Lade gespeichertes Modell und Normalisierung von:\n  {MODEL_PATH}\n  {STATS_PATH}")

            train_env = VecNormalize.load(STATS_PATH, base_env)
            train_env.training = True
            train_env.norm_reward = True

            model = RecurrentPPO.load(MODEL_PATH, env=train_env, tensorboard_log="./runs/")

            print("Modell erfolgreich geladen → Training wird fortgesetzt")

        else:
            if args.load:
                print("⚠️  --load angegeben, aber Modell/Stats nicht gefunden → starte NEUES Training")
            else:
                print("Starte NEUES Training")

            train_env = VecNormalize(
                base_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                gamma=GAMMA
            )

            model = RecurrentPPO(
                policy="MlpLstmPolicy",
                env=train_env,
                verbose=1,
                learning_rate=LEARNING_RATE,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_range=CLIP_RANGE,
                ent_coef=ENT_COEF,
                vf_coef=VF_COEF,
                max_grad_norm=MAX_GRAD_NORM,
                policy_kwargs=POLICY_KWARGS,
                tensorboard_log="./runs/",
                device="auto"
            )

        # ---------------------- Training starten ----------------------
        print(f"Starte Training für {TOTAL_TIMESTEPS:,} Timesteps")
        print(f"→ {NUM_ENVS} parallele Environments | LSTM hidden size = {POLICY_KWARGS['lstm_hidden_size']}")

        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                tb_log_name="rppo_furuta",
                reset_num_timesteps=not args.load,
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\nTraining durch Benutzer abgebrochen")

        # ---------------------- Speichern ----------------------
        print("Speichere Modell und Normalisierungs-Stats...")
        model.save(MODEL_PATH)
        train_env.save(STATS_PATH)
        print(f"fertig → gespeichert unter '{MODEL_PATH}.zip' und '{STATS_PATH}'")
        train_env.close()