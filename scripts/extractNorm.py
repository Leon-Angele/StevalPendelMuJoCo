import numpy as np
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pendel_env import PendelEnv  # Importiere deine Env-Klasse

# Pfad zur Datei
STATS_PATH = "vec_normalize_sac.pkl"

def main():
    if not os.path.exists(STATS_PATH):
        print(f"Fehler: {STATS_PATH} nicht gefunden.")
        return

    # 1. Wir brauchen eine Dummy-Env, um VecNormalize zu laden
    # (Die Parameter hier sind egal, solange die Observation-Shape stimmt)
    env = DummyVecEnv([lambda: PendelEnv(max_steps=100)])

    # 2. Laden der Statistiken
    try:
        norm_env = VecNormalize.load(STATS_PATH, env)
    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        return

    # 3. Werte extrahieren
    # SB3 speichert 'mean' und 'var' (Varianz).
    # Wir brauchen Std (Standardabweichung) = sqrt(var + epsilon)
    
    means = norm_env.obs_rms.mean
    variances = norm_env.obs_rms.var
    epsilon = norm_env.epsilon
    
    stds = np.sqrt(variances + epsilon)

    # 4. Ausgabe für Copy-Paste
    print("\n=== ERGEBNISSE FÜR DEINEN CONTROLLER ===\n")
    print(f"Observation Shape: {means.shape}\n")

    print("# Mean (Mittelwert) zum Kopieren:")
    print(f"OBS_MEAN = {np.array2string(means, separator=', ', precision=6)}")
    
    print("\n# Std (Standardabweichung) zum Kopieren:")
    print(f"OBS_STD  = {np.array2string(stds, separator=', ', precision=6)}")

    print("\n========================================\n")
    
    # Beispielrechnung
    print("Beispiel für die Implementierung:")
    print("obs_raw = [sin_p, cos_p, vel_p, sin_r, cos_r, vel_r]")
    print("obs_norm = (obs_raw - OBS_MEAN) / OBS_STD")

if __name__ == "__main__":
    main()