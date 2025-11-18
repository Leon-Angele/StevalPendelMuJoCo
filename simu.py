import mujoco
import mujoco.viewer
import os
import math
import time

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "Pendel_description", "pendel_roboter.xml")

if not os.path.exists(xml_path):
    print(f"FEHLER: Datei nicht gefunden unter: {xml_path}")
    exit()
    
# 1. Modell und Daten laden
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2. Joint- und Actuator-IDs finden (macht den Code robust)
rotary_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'rotaryGelenk')
pendel_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'pendelGelenk')
motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor_rotary')

if rotary_joint_id < 0 or pendel_joint_id < 0 or motor_id < 0:
    print("ERROR: Konnte Joint- oder Actuator-Namen nicht im Modell finden.")
    exit()

# --- SIMULATION ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    print("Simulation läuft. Beobachte die Werte im Terminal. (Druckt alle 100 Schritte)")
    
    start_time = time.time()
    print_counter = 0 # Zähler für die Druckfrequenz
    
    while viewer.is_running():
        
        # --- A. STEUERUNG ---
        sim_time = data.time
        
        # Sinus-Input: 1.0 rad/s Amplitude, 5.0 rad/s Frequenz
        control_signal = 5.0 * math.sin(5 * sim_time)  # Geschwindigkeit in rad/s
        data.ctrl[motor_id] = control_signal
        
        # --- B. PHYSIK BERECHNEN ---
        mujoco.mj_step(model, data)

        # --- C. DEBUG PRINTS (Neu!) ---
        print_counter += 1
        
        if print_counter % 100 == 0: # Druckt nur jeden 100. Schritt
            
            # Position und Geschwindigkeit des Rotary-Gelenks abrufen
            rotary_pos = data.qpos[rotary_joint_id]
            rotary_vel = data.qvel[rotary_joint_id]
            
            # Position und Geschwindigkeit des Pendel-Gelenks abrufen
            pendel_pos = data.qpos[pendel_joint_id]
            pendel_vel = data.qvel[pendel_joint_id]
            
            print(f"⏱️ Time: {sim_time:.3f} s")
            print(f"  ⚡️ Input (rad/s): {control_signal:.3f}")
            print(f"  🔄 Rotary Pos: {rotary_pos:.3f} rad | Vel: {rotary_vel:.2f} rad/s")
            print(f"  ⚖️ Pendel Pos: {pendel_pos:.3f} rad | Vel: {pendel_vel:.2f} rad/s")
            print("-" * 40)
        
        # --- D. GRAFIK UPDATE ---
        viewer.sync()

        # Realzeit-Synchronisation (optional)
        time_until_next_step = model.opt.timestep - (time.time() - start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            start_time += model.opt.timestep
        else:
            start_time = time.time()