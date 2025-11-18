from pendel_env import PendelEnv
import time
import numpy as np

# Env erstellen mit Render-Modus
env = PendelEnv(render_mode="human")
obs, _ = env.reset()

print("Starte Test-Lauf...")

for i in range(500):
    # Simuliere eine Policy: Schwinge hin und her
    # Wir ändern die Richtung alle 100 Schritte
    target_speed = 1.0 if (i % 200) < 100 else -1.0
    
    # Action ist im Bereich [-1, 1]
    action = np.array([target_speed], dtype=np.float32)
    
    obs, reward, done, trunc, info = env.step(action)
    
    # Debug Print alle 20 Schritte
    if i % 20 == 0:
        print(f"Step {i}: Obs={obs[:2]} (Cos/Sin) | Reward={reward:.2f} | MotorSpeed={info['motor_speed']:.2f}")

    time.sleep(0.02) # Damit wir Zeit zum Zuschauen haben

env.close()