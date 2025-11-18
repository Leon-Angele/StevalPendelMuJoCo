from pendel_env_full import PendelEnv
import time
import numpy as np

# Env erstellen mit Render-Modus
env = PendelEnv(render_mode="human")
obs, _ = env.reset()

print("Starte Test-Lauf...")

for i in range(500):
    # Simuliere eine Policy: Schwinge hin und her
    # Wir ändern die Richtung alle 100 Schritte
    target_speed = 1.0 if (i % 100) < 50 else -1.0
    
    # Action ist im Bereich [-1, 1]
    action = np.array([target_speed], dtype=np.float32)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Print komplette Trajektorie
    print(f"Step {i}: Obs={obs} | Reward={reward:.2f} | Trunc={truncated} | Info={info}")

    time.sleep(0.02) # Damit wir Zeit zum Zuschauen haben

env.close()
