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
from pendel_env_full import PendelEnv
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize  # NEU: Für Normalisierung
from stable_baselines3.common.callbacks import EvalCallback  # NEU: Für periodische Eval
from torch.utils.tensorboard import SummaryWriter

# --- Hyperparameter ---
TOTAL_TIMESTEPS = 5_000_000 
LEARNING_RATE = 3e-4
BUFFER_SIZE = 1_000_000  
BATCH_SIZE = 256  # REDUZIERT: Kleiner Batch für stabileres Update (TD3 ist sensitiv)
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
GAMMA = 0.98  # LEICHT REDUZIERT: Mildere Diskontierung für lange Horizons
TAU = 0.005
POLICY_DELAY = 2
TARGET_POLICY_NOISE = 0.3  
TARGET_NOISE_CLIP = 0.5
# Noise Definition
ACTION_NOISE = NormalActionNoise(
    mean=np.zeros(1),
    sigma=0.3 * np.ones(1)  
)
# Netzwerk Architektur
POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[400, 300],      
        qf=[400, 300]       
    ),
    activation_fn=nn.ReLU,
)
NUM_ENVS = 32  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Nur Modell evaluieren, nicht trainieren')
    parser.add_argument('--human', action='store_true', help='Training mit Render-Modus "human"')
    args = parser.parse_args()


    if args.eval:
        # Modell laden und evaluieren
        try:
            model = TD3.load("td3_pendel")
            print("Modell geladen.")
        except FileNotFoundError:
            print("Kein gespeichertes Modell gefunden. Bitte erst trainieren.")
            exit()
        env = PendelEnv(render_mode="human", max_steps=2000)  # ERHÖHT: Längere Episodes für Eval
        obs, _ = env.reset()
        print("Starte Evaluation...")
        
        for _ in range(150):  
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
            env = PendelEnv(render_mode=train_render_mode, max_steps=1000)  
            env = Monitor(env)
            return env
        
        # Vektorisierte Umgebung erstellen
        if NUM_ENVS == 1:
            train_env = DummyVecEnv([make_env])
        else:
            train_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])
        
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        
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
        
        # NEU: Eval-Callback für periodische Evaluation (alle 10k Steps, speichert Best-Model)
        eval_env = VecNormalize(DummyVecEnv([lambda: Monitor(PendelEnv(max_steps=2000))]), norm_obs=True, norm_reward=True)
        eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/", log_path="./logs/",
                                     eval_freq=10000, n_eval_episodes=10, deterministic=True, render=False)
        
        # Training starten
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name="td3_run",
            progress_bar=True,
            callback=eval_callback  # NEU: Hinzugefügt
        )
        
        # Modell speichern (inkl. VecNormalize-Stats)
        model.save("td3_pendel")
        train_env.save("vec_normalize.pkl")  # NEU: Speichere Normalizer für Eval
        print("Training abgeschlossen und Modell gespeichert.")
        
        train_env.close()