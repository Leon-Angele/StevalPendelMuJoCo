import numpy as np
import gymnasium as gym
import torch.nn as nn
import argparse
import os
import pickle # Wichtig für VecNormalize Speicherung
from stable_baselines3.common.monitor import Monitor
from pendel_env_full import PendelEnv 
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # <--- IMPORTIEREN!
from stable_baselines3.common.callbacks import EvalCallback

# --- Hyperparameter ---
TOTAL_TIMESTEPS = 1_000_00
LEARNING_RATE = 1e-3          
BUFFER_SIZE = 1_000_000  
BATCH_SIZE = 256  
TRAIN_FREQ = (1, "step")
GRADIENT_STEPS = 1          
GAMMA = 0.99
TAU = 0.005
POLICY_DELAY = 2
TARGET_POLICY_NOISE = 0.2  
TARGET_NOISE_CLIP = 0.5
LEARNING_STARTS = 10_000
ACTION_NOISE_SIGMA = 0.3      

POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 256], qf=[256, 256]),
    activation_fn=nn.ReLU,
)

def make_env(render_mode=None):
    # max_steps hier wichtig, damit Monitor korrekte Ep-Längen loggt
    env = PendelEnv(render_mode=render_mode, max_steps=1000)
    env = Monitor(env) 
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Nur Modell abspielen')
    parser.add_argument('--human', action='store_true', help='Training mit Render')
    parser.add_argument('--load', action='store_true', help='Training fortsetzen')
    args = parser.parse_args()

    model_path = "Modelle/td3_pendel"
    vec_norm_path = "Modelle/vec_normalize.pkl" # Pfad für die Normalisierungs-Statistik
    log_path = "./runs/"
    
    os.makedirs("Modelle", exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # 1. Action Noise
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(1),
        sigma=ACTION_NOISE_SIGMA * np.ones(1)
    )

    if args.eval:
        print("Lade Modell & Normalisierung...")
        
        # 1. Env erstellen
        env = DummyVecEnv([lambda: make_env(render_mode="human")])
        
        # 2. Normalisierung laden (SEHR WICHTIG BEI EVAL!)
        # Wir laden die Statistik, aber stellen training=False, damit sie sich beim Testen nicht ändert
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False 
            env.norm_reward = False # Beim Testen wollen wir den echten Reward sehen
        else:
            print("Warnung: Keine Normalisierungs-Daten gefunden. Modell wird vermutlich schlecht performen.")

        model = TD3.load(model_path, env=env)
        
        obs = env.reset()
        try:
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
        except KeyboardInterrupt:
            env.close()

    else:
        # --- TRAINING ---
        env = DummyVecEnv([lambda: make_env(render_mode=None)])
        
        # 3. VecNormalize anwenden
        # Das skaliert Observations und Rewards automatisch.
        if args.load and os.path.exists(vec_norm_path):
            print("Lade existierende Env-Normalisierung...")
            env = VecNormalize.load(vec_norm_path, env)
        else:
            # clip_obs=10, clip_reward=10 verhindert zu extreme Werte für das Netz
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

        if args.load and os.path.exists(model_path + ".zip"):
            print(f"--- Lade Modell ---")
            model = TD3.load(model_path, env=env, tensorboard_log=log_path)
            model.action_noise = action_noise
        else:
            print("--- Neues Training ---")
            model = TD3(
                "MlpPolicy",
                env,
                learning_rate=LEARNING_RATE,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                tau=TAU,
                gamma=GAMMA,
                train_freq=TRAIN_FREQ,
                gradient_steps=GRADIENT_STEPS,
                action_noise=action_noise,
                policy_delay=POLICY_DELAY,
                policy_kwargs=POLICY_KWARGS,
                tensorboard_log=log_path,
                learning_starts=LEARNING_STARTS,
                verbose=1
            )

        print(f"Start Training... Logs in {log_path}")

        try:
            # Wir speichern die VecNormalize Stats zwischendurch nicht automatisch im Callback hier (einfachheitshalber),
            # aber beim finalen Speichern.
            model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4,tb_log_name="td3_vecNorm" ,progress_bar=True, reset_num_timesteps=not args.load)
        except KeyboardInterrupt:
            print("Stop durch User.")

        # Speichern
        model.save(model_path)
        env.save(vec_norm_path) # <--- WICHTIG: Normalisierungs-Statistik speichern!
        print("Modell und Normalisierung gespeichert.")
        env.close()