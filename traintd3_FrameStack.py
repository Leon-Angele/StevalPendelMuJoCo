"""
TD3 Training für PendelEnv - OPTIMIERTE VERSION FÜR "BLINDEN" AGENTEN
Mit VecFrameStack (Ersetzt fehlenden Arm-Sensor durch Kurzzeitgedächtnis)
"""
import numpy as np
import gymnasium as gym
import torch.nn as nn
import argparse
import os
from stable_baselines3.common.monitor import Monitor

from pendel_env_full import PendelEnv 
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack 
from stable_baselines3.common.callbacks import EvalCallback

# --- Hyperparameter ---
TOTAL_TIMESTEPS = 1_000_000  
LEARNING_RATE = 1e-3
BUFFER_SIZE = 1_000_00 
BATCH_SIZE = 512  
TRAIN_FREQ = 1
GRADIENT_STEPS = 1           #
GAMMA = 0.99 
TAU = 0.005
POLICY_DELAY = 2
TARGET_POLICY_NOISE = 0.2  
TARGET_NOISE_CLIP = 0.5
LEARNING_STARTS = 10_000

ACTION_NOISE_SIGMA = 0.1 

NUM_ENVS = 1 

N_STACK = 4

POLICY_KWARGS = dict(
    net_arch=dict(pi=[128, 128], qf=[128, 128]),
    activation_fn=nn.ReLU,
)

def make_env(render_mode=None):
    # max_steps erhöht, damit er Zeit für den Swing-Up hat
    env = PendelEnv(render_mode=render_mode, max_steps=1000)
    env = Monitor(env) 
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Nur Modell abspielen')
    parser.add_argument('--human', action='store_true', help='Training mit Render (langsam)')
    parser.add_argument('--load', action='store_true', help='Training fortsetzen')
    args = parser.parse_args()

    model_path = "Modelle/model"
    log_path = "./runs/"
    best_model_path = "./logs/best_model/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(best_model_path, exist_ok=True)

    # --- SETUP ---
    
    # Action Noise (Wichtig für TD3 Exploration)
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(1),
        sigma=ACTION_NOISE_SIGMA * np.ones(1),
        theta=0.15,
        dt=1e-2
    )

    # --- EVALUATION ---
    
    if args.eval:
        if not os.path.exists(model_path + ".zip"):
            print("Kein Modell gefunden!")
            exit()
            
        print("Lade Modell für Evaluation...")
        
        # 1. Env erstellen
        env = DummyVecEnv([lambda: make_env(render_mode="human")])
        
        # 2. WICHTIG: Auch bei Eval muss gestackt werden!
        # Sonst erwartet das Modell 12 Inputs, bekommt aber nur 3 -> Crash.
        env = VecFrameStack(env, n_stack=N_STACK)
        
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
        
        # 1. Training Environment
        if args.human:
            env = DummyVecEnv([lambda: make_env(render_mode="human")])
        else:
            env = DummyVecEnv([lambda: make_env(render_mode=None)])

        # 2. Frame Stacking anwenden
        env = VecFrameStack(env, n_stack=N_STACK)

        # Modell laden oder neu erstellen
        if args.load and os.path.exists(model_path + ".zip"):
            print(f"--- Lade existierendes Modell: {model_path} ---")
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
                target_policy_noise=TARGET_POLICY_NOISE,
                target_noise_clip=TARGET_NOISE_CLIP,
                policy_kwargs=POLICY_KWARGS,
                tensorboard_log=log_path,
                learning_starts=LEARNING_STARTS,
                verbose=1
            )

        # 3. Eval Environment (Muss exakt gleich wie Train Env aufgebaut sein)
        eval_env = DummyVecEnv([lambda: make_env(render_mode=None)])
        eval_env = VecFrameStack(eval_env, n_stack=N_STACK) # <--- Auch hier stacken!
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_path,
            log_path=log_path,
            eval_freq=5000,
            deterministic=True,
            render=False
        )

        print(f"Start... Logs in {log_path}")

        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                log_interval=50,
                tb_log_name="td3_framestack",
                callback=eval_callback,
                progress_bar=True,
                reset_num_timesteps=not args.load
            )
        except KeyboardInterrupt:
            print("Stop durch User.")

        model.save(model_path)
        print("Gespeichert.")
        env.close()