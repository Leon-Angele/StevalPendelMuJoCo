"""
SAC Training mit gSDE für PendelEnv (Schrittmotor Sim-to-Real)
Erweitert um: --load zum Fortsetzen des Trainings
"""

import numpy as np
import gymnasium as gym
import torch.nn as nn
import argparse
import time
import os

# --- Stable Baselines3 Imports ---
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# --- Import deiner Environment ---
try:
    from pendel_env import PendelEnv
except ImportError:
    print("Fehler: 'pendel_env.py' nicht gefunden oder Klasse heißt anders.")
    exit()

# ==========================================
# === HYPERPARAMETER (Sim-to-Real Tuned) ===
# ==========================================

TOTAL_TIMESTEPS = 2_000_000 
LEARNING_RATE = 1e-3
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256
GAMMA = 0.995 
TAU = 0.005
TRAIN_FREQ = 1
GRADIENT_STEPS = 1

# === gSDE Einstellungen ===
USE_SDE = True
SDE_SAMPLE_FREQ = 8 
SDE_CONST_LOG = -2.0 # Konstantes Log Std für gSDE (optional, wenn nicht in policy_kwargs)

# === Netzwerk Architektur ===
POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 256], qf=[256, 256]),
    activation_fn=nn.Tanh,
    log_std_init=-2, 
)

NUM_ENVS = 32

# Pfade
LOG_DIR = "./runs/"
MODEL_PATH = "Modelle/sac_pendel_gsde"
STATS_PATH = "Modelle/vec_normalize_sac.pkl"

# ==========================================
# === CODE ===
# ==========================================

def make_env(render_mode=None):
    """Erstellt eine Instanz der Umgebung"""
    env = PendelEnv(render_mode=render_mode, max_steps=1000)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Modell nur evaluieren (kein Training)')
    parser.add_argument('--human', action='store_true', help='Training live ansehen (render_mode="human")')
    # NEU: Argument zum Laden des Modells
    parser.add_argument('--load', action='store_true', help='Lade existierendes Modell und trainiere weiter')
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    if args.eval:
        # ==================
        # === EVALUATION ===
        # (Unverändert)
        # ==================
        print(f"Lade Modell '{MODEL_PATH}' und Stats...")
        
        env = DummyVecEnv([lambda: make_env(render_mode="human")])
        
        try:
            env = VecNormalize.load(STATS_PATH, env)
            env.training = False 
            env.norm_reward = False
        except FileNotFoundError:
            print(f"FEHLER: {STATS_PATH} nicht gefunden. Du musst erst trainieren!")
            exit()

        try:
            model = SAC.load(MODEL_PATH, env=env)
        except FileNotFoundError:
            print(f"FEHLER: {MODEL_PATH}.zip nicht gefunden.")
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
    else:
        # ================
        # === TRAINING ===
        # ================

        # 1. Basale Environment(s) erstellen
        if args.human:
            base_env = DummyVecEnv([lambda: make_env(render_mode="human")])
        else:
            base_env = DummyVecEnv([lambda: make_env(render_mode=None) for _ in range(NUM_ENVS)])
        
        # 2. Lade-Logik (NEU)
        if args.load and os.path.exists(MODEL_PATH + ".zip") and os.path.exists(STATS_PATH):
            print(f"--- LADE MODELL: {MODEL_PATH} und Statistiken ---")
            
            # A) Statistiken laden (KRITISCH!)
            train_env = VecNormalize.load(STATS_PATH, base_env)
            train_env.training = True 
            train_env.norm_reward = True
            
            # B) Modell laden
            model = SAC.load(MODEL_PATH, env=train_env, tensorboard_log=os.path.join(LOG_DIR, "tensorboard"))
            
            print("Modell und Statistiken erfolgreich geladen. Training wird fortgesetzt.")
            
        else:
            if args.load:
                print(f"Warnung: --load gesetzt, aber {MODEL_PATH}.zip oder {STATS_PATH} nicht gefunden. Starte NEU.")
            else:
                print("--- NEUES TRAINING ---")

            # A) Neuer Normalizer
            train_env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=GAMMA)

            # B) Neues Modell initialisieren
            model = SAC(
                "MlpPolicy",
                train_env,
                verbose=1,
                learning_rate=LEARNING_RATE,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                tau=TAU,
                train_freq=TRAIN_FREQ,
                gradient_steps=GRADIENT_STEPS,
                
                # --- gSDE Konfiguration ---
                use_sde=USE_SDE,
                sde_sample_freq=SDE_SAMPLE_FREQ,
                
                policy_kwargs=POLICY_KWARGS,
                tensorboard_log=os.path.join(LOG_DIR, "tensorboard")
            )

        
        # 4. Lernen starten
        print(f"Starte SAC Training für {TOTAL_TIMESTEPS} Steps...")
        
        try:
            # reset_num_timesteps=False stellt sicher, dass die Steps in Tensorboard weiterlaufen
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                tb_log_name="sac_pendel_gsde",
                progress_bar=True,
                callback=None,
                reset_num_timesteps=not args.load 
            )
        except KeyboardInterrupt:
            print("\nTraining manuell abgebrochen. Speichere aktuellen Stand...")

        # 5. Speichern
        model.save(MODEL_PATH)
        train_env.save(STATS_PATH) # WICHTIG: Normalisierung speichern!

        print(f"Modell gespeichert: {MODEL_PATH}.zip")
        print(f"Stats gespeichert:  {STATS_PATH}")
        
        train_env.close()