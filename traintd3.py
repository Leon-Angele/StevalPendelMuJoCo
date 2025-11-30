"""
TD3 Training für PendelEnv - CLEAN VERSION (ohne VecNormalize)
"""
import numpy as np
import gymnasium as gym
import torch.nn as nn
import argparse
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from pendel_env_full import PendelEnv 
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# first     run   = mit batch size 256, lr 1e-3, 75k steps, noise 0.1, [128, 128, 32]
# second    run  = [128, 128],[128, 128]
# third     run   = gamma 0.99 -> 0.995
# fourth    run   = gamma 0.995 -> 0.98



# --- Hyperparameter (Optimiert für Stabilität) ---
TOTAL_TIMESTEPS = 1_000_000  
LEARNING_RATE = 7e-4
BUFFER_SIZE = 1_000_000  
BATCH_SIZE = 256  
TRAIN_FREQ = 1
GRADIENT_STEPS = 1          
GAMMA = 0.99
TAU = 0.005
POLICY_DELAY = 2
TARGET_POLICY_NOISE = 0.2  
TARGET_NOISE_CLIP = 0.5
LEARNING_STARTS = 10_000

# Noise: 0.2 für guten Swing-Up
ACTION_NOISE_SIGMA = 0.2

NUM_ENVS = 32

POLICY_KWARGS = dict(

    net_arch=dict(pi=[128, 128], qf=[128, 128, 32]),
    activation_fn=nn.ReLU,
)


def make_env(render_mode=None):
    env = PendelEnv(render_mode=render_mode, max_steps=2000)
    env = Monitor(env) # Monitor loggt Reward für Tensorboard
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Nur Modell abspielen')
    parser.add_argument('--plot', action='store_true', help='Im Eval-Modus Rewards über Zeit plotten')
    parser.add_argument('--human', action='store_true', help='Training mit Render (langsam)')
    parser.add_argument('--load', action='store_true', help='Training fortsetzen')
    args = parser.parse_args()

    model_path = "Modelle/model"
    log_path = "./runs/"
    best_model_path = "./logs/best_model/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(best_model_path, exist_ok=True)

    # --- SETUP ---
    
    # 1. Action Noise (Wichtig für TD3 Exploration)
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(1),
        sigma=ACTION_NOISE_SIGMA * np.ones(1),
        theta=0.15,
        dt=1e-2
    )

    # --- EVALUATION ODER TRAINING ---
    
    if args.eval:
        # Laden ohne VecNormalize ist super einfach:
        if not os.path.exists(model_path + ".zip"):
            print("Kein Modell gefunden!")
            exit()
            
        print("Lade Modell (Raw)...")
        # Einfach Env erstellen, Modell laden, fertig.
        env = DummyVecEnv([lambda: make_env(render_mode="human")])
        model = TD3.load(model_path, env=env)
        
        obs = env.reset()
        # VecEnv returns (obs, infos) for reset; keep track for compatibility
        rewards = []
        try:
            while True:
                action, _ = model.predict(obs, deterministic=True)
                step_result = env.step(action)
                # VecEnv can return either 4-tuples (obs, reward, done, info)
                # or 5-tuples (obs, reward, terminated, truncated, info).
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                else:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False

                # reward may be an array (vec env); convert to scalar if single env
                if isinstance(reward, (list, tuple, np.ndarray)):
                    r = np.asarray(reward).ravel()[0]
                else:
                    r = float(reward)
                rewards.append(r)
        except KeyboardInterrupt:
            pass
        finally:
            env.close()

        if args.plot:
            if len(rewards) == 0:
                print("Keine Rewards gesammelt - nichts zu plotten.")
            else:
                plt.figure(figsize=(8,4))
                plt.plot(np.arange(len(rewards)), rewards, label='Reward')
                plt.xlabel('Steps')
                plt.ylabel('Reward')
                plt.title('Eval Rewards over Time')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                try:
                    plt.show()
                except Exception:
                    out_path = os.path.join(log_path, 'eval_rewards.png')
                    plt.savefig(out_path)
                    print(f"Plot saved to {out_path}")

    else:
        # --- TRAINING ---
        
        # Environment
        if args.human:
            env = DummyVecEnv([lambda: make_env(render_mode="human")])
        else:
            env = DummyVecEnv([lambda: make_env(render_mode=None)])

        # Modell laden oder neu erstellen
        if args.load and os.path.exists(model_path + ".zip"):
            print(f"--- Lade existierendes Modell: {model_path} ---")
            model = TD3.load(model_path, env=env, tensorboard_log=log_path)
            model.action_noise = action_noise # Noise neu setzen
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

        print(f"Start... Logs in {log_path}")

        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                log_interval=1,
                tb_log_name="td3",
                progress_bar=True,
                reset_num_timesteps=not args.load
            )
        except KeyboardInterrupt:
            print("Stop durch User.")

        model.save(model_path)
        print("Gespeichert.")
        env.close()