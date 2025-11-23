from pendel_env_full import PendelEnv
import time
import numpy as np

# Env erstellen mit Render-Modus
env = PendelEnv(render_mode="human")
obs, _ = env.reset()

print("Starte Test-Lauf...")

for i in range(500):
    # Simuliere eine Policy: Schwinge hin und her
    # Wir ändern die Richtung alle 200 Schritte
    target_speed = 0.5 if (i % 200) < 100 else -1.0
    
    # Action ist im Bereich [-1, 1]
    action = np.array([target_speed], dtype=np.float32)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Print komplette Trajektorie
    #print(f"Step {i}: Obs={obs} | Reward={reward:.2f} | Trunc={truncated} | Info={info}")

    time.sleep(0.005) # Damit wir Zeit zum Zuschauen haben

# Nach dem regulären Lauf: ca. 5 Sekunden keine Aktion (action = 0)
print("Starte 5s Pause mit action=0...")
pause_duration = 10.0  # Sekunden
start_time = time.time()
while time.time() - start_time < pause_duration:
    action = np.array([0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
    time.sleep(0.005)

env.close()
