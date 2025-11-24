import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
import time

class PendelEnv(gym.Env):
    """
    MuJoCo-Umgebung für ein Inverses Pendel (Rotary Inverted Pendulum).
    Simuliert spezifische Hardware-Eigenschaften wie Schrittmotor-Logik und Latenz.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()

        # --- 1. Parameter ---
        self.MAX_SPEED = 5.0  # Max. Drehgeschwindigkeit des Motors in rad/s
        self.dt = 0.005       # Zeitintervall pro Steuerungsschritt (typisch für Agenten-Frequenz)
        self.latency = 0.001  # Simuliere Steuerungs-Latenz (z.B. Bus-Verzögerung)
        self.max_steps = max_steps
        self.current_step = 0
        
        # Stepper Motor Parameter
        self.accel_ramp = 100.0  # Max. Beschleunigung/Verzögerung in rad/s^2
        self.effective_step = np.deg2rad(1.8) / 16  # Quantisierung des Schrittmotors (z.B. 1/16 Mikroschritt)
        self.limit_threshold = 0.05  # Nicht verwendet in den bereitgestellten Methoden
        
        # --- 2. MuJoCo Setup & Caching ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "Pendel_description", "pendel_roboter.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Setze MuJoCo Timestep auf einen kleinen Wert für präzisere Integration
        self.model.opt.timestep = 0.0005 # Beispiel: Kleiner als self.dt für Sub-Steps

        # IDs & Adressen cachen
        self.pendel_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendelGelenk")
        self.rotary_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rotaryGelenk")
        self.actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_rotary")
        self.pendel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Pendel_1")
        
        self.pendel_qpos_adr = self.model.jnt_qposadr[self.pendel_joint_id]
        self.pendel_qvel_adr = self.model.jnt_dofadr[self.pendel_joint_id]
        self.rotary_qpos_adr = self.model.jnt_qposadr[self.rotary_joint_id]
        
        self.rotary_min = self.model.jnt_range[self.rotary_joint_id][0]
        self.rotary_max = self.model.jnt_range[self.rotary_joint_id][1]
        
        # Referenzwerte für Randomization (Gespeichert, um sie in reset zu modifizieren)
        self.default_pendel_mass = self.model.body_mass[self.pendel_body_id]
        self.default_pendel_damping = self.model.dof_damping[self.pendel_qvel_adr]

        # --- 3. Spaces & Rendering ---
        # Action Space: Ein Wert ([-1.0, 1.0]) repräsentiert die gewünschte Motorgeschwindigkeit
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space: [sin(theta), cos(theta), Pendelgeschwindigkeit]
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -np.inf], dtype=np.float32),
            high=np.array([1, 1, np.inf], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        
        # --- 4. Controller States (Zur Simulation der Steuerung) ---
        self.target_position = 0.0      # Kontinuierliche Zielposition (vor Quantisierung)
        self.current_velocity = 0.0     # Aktuell erlaubte Motorgeschwindigkeit (nach Rampe)
        self.last_ctrl_target = 0.0     # Letzter an MuJoCo gesendeter quantisierter Befehl (für Latenz)
        self.prev_height = 0.0          # Für Potential Shaping im Reward

    # ------------------------------------------------------------------
    # --- Hauptmethoden der Environment ---
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Setzt die Umgebung zurück und wendet Domain Randomization an."""
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Domain Randomization (Veränderung der Physik-Parameter)
        # Masse: +/- 5%
        mass_factor = self.np_random.uniform(0.95, 1.05)
        self.model.body_mass[self.pendel_body_id] = self.default_pendel_mass * mass_factor
        
        # Dämpfung: 0.8x bis 1.2x
        damping_factor = self.np_random.uniform(0.8, 1.2)
        self.model.dof_damping[self.pendel_qvel_adr] = self.default_pendel_damping * damping_factor

        # Start State (Pendel leicht ausgelenkt, Motorarm bei 0)
        self.data.qpos[self.pendel_qpos_adr] = 0 #self.np_random.uniform(low=-0.2, high=0.2)
        self.data.qpos[self.rotary_qpos_adr] = 0.0
        
        # Erster MuJoCo-Schritt, um alle Daten zu initialisieren
        mujoco.mj_step(self.model, self.data)

        # Controller States zurücksetzen
        self.target_position = 0.0
        self.current_velocity = 0.0
        self.last_ctrl_target = 0.0
        self.prev_height = 0.0 # Für Potential Shaping

        return self._get_obs(), {}

    def step(self, action):
        """Führt einen Simulationsschritt mit der gegebenen Aktion durch."""
        desired_velocity = float(action[0]) * self.MAX_SPEED

        # --- 1. Stepper-Logik (Rampe & Quantisierung) ---
        # Beschleunigungsrampe anwenden
        velocity_diff = desired_velocity - self.current_velocity
        max_accel_step = self.accel_ramp * self.dt
        self.current_velocity += np.clip(velocity_diff, -max_accel_step, max_accel_step)
        
        # Kontinuierliche und dann quantisierte Zielposition berechnen
        self.target_position += self.current_velocity * self.dt
        quantized_target = round(self.target_position / self.effective_step) * self.effective_step

        # --- 2. Latenz- und MuJoCo-Simulation ---
        physics_timestep = self.model.opt.timestep
        n_latency_steps = int(self.latency / physics_timestep)
        n_total_steps = int(self.dt / physics_timestep)
        n_remaining_steps = max(0, n_total_steps - n_latency_steps)

        # Phase 1: Latenz-Phase (alter Befehl aktiv)
        self.data.ctrl[self.actuator_id] = self.last_ctrl_target
        for _ in range(n_latency_steps):
            mujoco.mj_step(self.model, self.data)

        # Phase 2: Aktions-Phase (neuer Befehl aktiv)
        self.data.ctrl[self.actuator_id] = quantized_target
        for _ in range(n_remaining_steps):
            mujoco.mj_step(self.model, self.data)

        self.last_ctrl_target = quantized_target  # Speichern für nächste Latenz-Phase

        # --- 3. Checks & Rewards ---
        rotary_pos = self.data.qpos[self.rotary_qpos_adr]
        terminated = False  # Derzeit keine Terminierung definiert

        obs = self._get_obs()
        tracking_error = np.abs(rotary_pos - quantized_target)
        reward = self.compute_rewards(obs, action, tracking_error)
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps  # Abbruch bei maximaler Schrittzahl

        info = {"rotary_angle": rotary_pos}

        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Gibt den aktuellen Beobachtungsvektor zurück."""
        theta_p = self.data.qpos[self.pendel_qpos_adr] # Pendelwinkel (0 = unten, +/-pi = oben)
        vel_p = self.data.qvel[self.pendel_qvel_adr]   # Pendelwinkelgeschwindigkeit

        # Optional: Hier Sensorrauschen hinzufügen (auskommentiert in Original)
        # theta_p += self.np_random.normal(0, 0.001) 
        # ...

        # Observation: [sin(theta), cos(theta), vel_p]
        return np.array([
            np.sin(theta_p),
            np.cos(theta_p),
            vel_p
        ], dtype=np.float32)

    def compute_rewards(self, obs, action, tracking_error):
        """Berechnet die Belohnung für den aktuellen Schritt (Reward Engineering)."""
        sin_theta, cos_theta, theta_dot = obs[0], obs[1], obs[2]
        
        # Skalierungen für Rewards
        theta_dot_scaled = theta_dot / 50.0  # Normalisiert auf eine Max-Geschw. von 50
        action_scaled = action[0]

        # A. Swing-Up Reward (Potential Energy)
        height = (1.0 - cos_theta) / 2.0  # Höhe: 0 (unten) bis 1 (oben)
        r_swingup = 2.0 * height

        # B. Balance Bonus (Gaussian/Exponential)
        dist_to_top = np.abs(np.pi - np.arccos(cos_theta))  # Distanz zum oberen Gleichgewichtspunkt (pi)
        r_balance = 3.0 * np.exp(-(dist_to_top ** 2) / (2 * 0.1 ** 2)) # Scharfe Gauß-Funktion bei pi

        # C. Stability Penalty (Dämpft Geschwindigkeit nahe der Spitze)
        balance_factor = np.exp(-dist_to_top / 0.5)  # Aktiv nur nahe der Spitze
        r_stability = -0.5 * (theta_dot_scaled ** 2) * balance_factor

        # D. Effort Penalty (Belohnung für sparsamen Motor-Einsatz)
        effort_weight = 0.01 if height < 0.5 else 0.1  # Geringere Strafe beim Aufschwingen
        r_effort = -effort_weight * (action_scaled ** 2)

        # E. Slip Penalty (Bestrafung für großen Tracking Error zwischen Soll- und Ist-Position)
        full_step_rad = np.deg2rad(1.8)
        if tracking_error > 0.5 * full_step_rad:
            r_slip = -5.0 * (tracking_error ** 2)
        else:
            r_slip = 0.0

        # F. Potential Shaping (Belohnt positive Höhenzunahme)
        # **Achtung**: `self.prev_height` muss im `reset` initialisiert werden.
        if not hasattr(self, 'prev_height'):
             self.prev_height = 0.0
             
        height_delta = height - self.prev_height
        r_shaping = 1.0 * height_delta
        self.prev_height = height  # Update für nächsten Step

        # Gesamte Belohnung
        reward = r_swingup + r_balance + r_stability + r_effort + r_slip + r_shaping

        return float(reward)

    def render(self):
        """Rendert die Umgebung, falls der Render-Modus 'human' ist."""
        if self.viewer is None:
            # Initialisiert den passiven MuJoCo-Viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Synchronisiert MuJoCo-Daten mit dem Viewer
        self.viewer.sync()
        
        # Fügt eine kleine Pause hinzu, um die Visualisierung zu verlangsamen
        if self.render_mode == "human":
            time.sleep(self.dt)

    def close(self):
        """Schließt den Viewer und gibt Ressourcen frei."""
        if self.viewer is not None:
            self.viewer.close()

# Beispiel für die Nutzung (wird nicht ausgeführt, nur zur Demonstration)
if __name__ == '__main__':
    env = PendelEnv(render_mode="human")
    obs, info = env.reset()
    print("Umgebung initialisiert. Beobachtung:", obs)
    
    # Führe 100 Schritte mit zufälligen Aktionen aus
    for _ in range(100):
        action = env.action_space.sample()  # Zufällige Aktion
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode beendet nach {env.current_step} Schritten.")
            obs, info = env.reset()
            
    env.close()