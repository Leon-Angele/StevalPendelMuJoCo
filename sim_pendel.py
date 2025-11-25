import time
import math
import mujoco
import mujoco.viewer

# 1. Modell und Daten laden
xml_path = 'Pendel_description/pendel_roboter.xml'
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

# Parameter für den Sinus
AMPLITUDE = 0.5 # Wie weit dreht es sich (in Radiant)
SPEED = 5      # Wie schnell (Frequenz)

print("Starte Simulation... Drücke ESC im Viewer zum Beenden.")

# 2. Viewer starten (Passiv-Modus erlaubt uns, die Kontrolle zu behalten)
with mujoco.viewer.launch_passive(m, d) as viewer:
    start_time = time.time()
    
    while viewer.is_running():
        step_start = time.time()
        
        # Aktuelle Zeit berechnen
        current_time = time.time() - start_time
        
        # --- SINUS BERECHNUNG ---
        # Wir steuern den Aktuator "motor_rotary" an.
        # Da es der erste (und einzige) Aktuator im XML ist, nutzen wir Index 0.
        target_pos = AMPLITUDE * math.sin(SPEED * current_time)
        
        d.ctrl[0] = target_pos
        # ------------------------

        # Physik-Schritt berechnen
        mujoco.mj_step(m, d)

        # Viewer aktualisieren
        viewer.sync()

        # Zeit-Synchronisation (damit es nicht zu schnell läuft)
        # MuJoCo default timestep ist 0.002, wir versuchen das einzuhalten
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)