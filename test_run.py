from pendel_env_full import PendelEnv
import time
import numpy as np

# Env erstellen mit Render-Modus
env = PendelEnv(render_mode="human")
obs, _ = env.reset()

print("Starte Test-Lauf...")

obs_list = []
reward_list = []
for i in range(500):
    # Simuliere eine Policy: Schwinge hin und her
    # Wir ändern die Richtung alle 200 Schritte
    target_speed = 1 if (i % 50) < 25 else -1.0
    # Action ist im Bereich [-1, 1]
    action = np.array([target_speed], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    obs_list.append(obs)
    reward_list.append(reward)
    #print(f"Step {i}: Obs={obs} | Reward={reward:.2f} ")
    print("Action:", action)
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

obs_array = np.array(obs_list)
min_obs = obs_array.min(axis=0)
max_obs = obs_array.max(axis=0)
print(f"Min obs: {min_obs}")
print(f"Max obs: {max_obs}")

min_reward = np.min(reward_list)
max_reward = np.max(reward_list)
print(f"Min reward: {min_reward}")
print(f"Max reward: {max_reward}")

env.close()
