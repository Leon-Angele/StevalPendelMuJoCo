"""
TD3 Training für PendelEnv - Optimiert mit OU-Noise
Option: --load zum Fortsetzen des Trainings
"""
import numpy as np
import gymnasium as gym
import torch.nn as nn
import argparse
import time
import os
from stable_baselines3.common.monitor import Monitor
from pendel_env import PendelEnv 
from stable_baselines3 import TD3

from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# --- Hyperparameter ---
TOTAL_TIMESTEPS = 5_000_000 
LEARNING_RATE = 1e-3
BUFFER_SIZE = 1_000_000  
BATCH_SIZE = 256  
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
GAMMA = 0.995  
TAU = 0.005
POLICY_DELAY = 2
TARGET_POLICY_NOISE = 0.2  
TARGET_NOISE_CLIP = 0.5

# Noise Stärke (Sigma)
ACTION_NOISE_SIGMA = 0.1

NUM_ENVS = 32 

POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 256],      
        qf=[256, 256]       
    ),
    activation_fn=nn.Tanh,
)

def make_env(render_mode=None):
    env = PendelEnv(render_mode=render_mode, max_steps=1000)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Nur Modell evaluieren')
    parser.add_argument('--human', action='store_true', help='Training mit Render-Modus "human" (nicht empfohlen für Speed)')
    # NEU: Argument zum Laden
    parser.add_argument('--load', action='store_true', help='Lade existierendes Modell und trainiere weiter')
    
    args = parser.parse_args()

    model_path = "td3_pendel"
    stats_path = "vec_normalize.pkl"
    log_path = "./logs/"
    os.makedirs(log_path, exist_ok=True)

    # --- EVALUATION MODUS ---
    if args.eval:
        print("Lade Modell und Normalisierungs-Statistiken...")
        env = DummyVecEnv([lambda: make_env(render_mode="human")])
        try:
            env = VecNormalize.load(stats_path, env)
            env.training = False 
            env.norm_reward = False 
        except FileNotFoundError:
            print(f"Fehler: {stats_path} nicht gefunden.")
            exit()

        try:
            model = TD3.load(model_path, env=env)
        except FileNotFoundError:
            print(f"Fehler: {model_path}.zip nicht gefunden.")
            exit()

        obs = env.reset()
        print("Starte Evaluation...")
        try:
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                time.sleep(0.005) 
        except KeyboardInterrupt:
            env.close()

    # --- TRAINING MODUS ---
    else:
        # Ornstein-Uhlenbeck Noise Definition
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(1),
            sigma=ACTION_NOISE_SIGMA * np.ones(1),
            theta=0.15,
            dt=1e-2,
            initial_noise=None
        )

        # 1. Basis-Environment erstellen
        if args.human:
            base_env = DummyVecEnv([lambda: make_env(render_mode=None)])
        else:
            base_env = DummyVecEnv([lambda: make_env(render_mode=None) for _ in range(NUM_ENVS)])
        
        # 2. Modell & VecNormalize laden ODER neu erstellen
        if args.load and os.path.exists(model_path + ".zip") and os.path.exists(stats_path):
            print(f"--- LADE MODELL: {model_path} ---")
            
            # A) Statistiken laden (Wichtig!)
            train_env = VecNormalize.load(stats_path, base_env)
            # Wichtig: Wir müssen dem Env sagen, dass es weiter trainieren soll (Updaten von Mean/Std)
            train_env.training = True 
            train_env.norm_reward = True
            
            # B) Modell laden
            # Wir übergeben tensorboard_log explizit, damit Logs weitergeschrieben werden
            model = TD3.load(model_path, env=train_env, tensorboard_log="./runs/")
            
            # C) Noise neu setzen (wird beim Loaden oft nicht perfekt übernommen)
            model.action_noise = action_noise
            
            print("Modell und Statistiken erfolgreich geladen. Training wird fortgesetzt.")
            
        else:
            if args.load:
                print(f"Warnung: --load gesetzt, aber {model_path}.zip oder {stats_path} nicht gefunden.")
                print("Starte stattdessen NEUES Training.")
            else:
                print("--- NEUES TRAINING ---")

            # A) Neuer Normalizer
            train_env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=GAMMA)

            # B) Neues Modell
            model = TD3(
                "MlpPolicy",
                train_env,
                verbose=1,
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
                action_noise=action_noise,
                policy_kwargs=POLICY_KWARGS,
                tensorboard_log="./runs/"
            )
        
        print(f"Starte Training für {TOTAL_TIMESTEPS} Steps...")
        print(f"Action Noise: Ornstein-Uhlenbeck (Sigma={ACTION_NOISE_SIGMA})")
        
        try:
            # reset_num_timesteps=False sorgt dafür, dass Tensorboard nicht bei 0 anfängt,
            # wenn man ein Modell lädt (optional, aber schön für Graphen)
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                tb_log_name="td3_pendel_ou",
                progress_bar=True,
                callback=None,
                reset_num_timesteps=not args.load 
            )
        except KeyboardInterrupt:
            print("Abbruch durch User.")

        # Speichern am Ende
        model.save(model_path)
        train_env.save(stats_path)
        print("Gespeichert.")
        train_env.close()