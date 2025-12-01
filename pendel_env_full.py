import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
import time

MAX_SPEED = 10.0   # Max. Drehgeschwindigkeit des Motors in rad/s
MIN_SPEED = 2.0    # Min. Drehgeschwindigkeit des Motors in rad/s
ACCEL_RAMP = 20.0  # Beschleunigungsrampe in rad/s²
EFFECTIVE_STEP = np.deg2rad(1.8) / 16

DT = 0.005      # Steuerungsintervall in Sekunden
LATENCY = 0.001 # Latenz in Sekunden

class PendelEnv(gym.Env):
    """
    MuJoCo-Umgebung für ein Inverses Pendel (Rotary Inverted Pendulum).
    Simuliert spezifische Hardware-Eigenschaften wie Schrittmotor-Logik und Latenz.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=3000):
        super().__init__()

        # --- Filter-Parameter ---
        self.FILTER_ALPHA = 0.4
        self.last_filtered_vel_p = 0.0
        self.last_filtered_vel_r = 0.0

        # --- 1. Parameter ---
        self.MAX_SPEED = MAX_SPEED
        self.MIN_SPEED = MIN_SPEED 
        self.DEADZONE_MAGNITUDE = 0.05 # Deadzone für die Aktion
        self.dt = DT
        self.latency = LATENCY
        self.max_steps = max_steps
        self.current_step = 0

        # Stepper Motor Parameter
        self.accel_ramp = ACCEL_RAMP
        self.effective_step = EFFECTIVE_STEP 
        
        # --- 2. MuJoCo Setup ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Pfad zum XML-Modell anpassen
        xml_path = os.path.join(current_dir, "Pendel_description", "pendel_roboter_mutter.xml") 

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.model.opt.timestep = 0.0005 

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
        
        # Referenzwerte für Randomization
        self.default_pendel_mass = self.model.body_mass[self.pendel_body_id]
        self.default_pendel_damping = self.model.dof_damping[self.pendel_qvel_adr]

        # --- 3. Spaces ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -np.inf, -1, -np.inf], dtype=np.float32),
            high=np.array([1, 1, np.inf, 1, np.inf], dtype=np.float32),
            shape=(5,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        
        # Controller States
        self.target_position = 0.0
        self.current_velocity = 0.0
        self.last_ctrl_target = 0.0
        # Zustand für Action Smoothness Penalty
        self.last_action = np.array([0.0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Domain Randomization
        mass_factor = self.np_random.uniform(0.95, 1.05)
        self.model.body_mass[self.pendel_body_id] = self.default_pendel_mass * mass_factor
        
        damping_factor = self.np_random.uniform(0.8, 1.2)
        self.model.dof_damping[self.pendel_qvel_adr] = self.default_pendel_damping * damping_factor

        self.MAX_SPEED = self.np_random.uniform(0.9, 1.1) * MAX_SPEED
        self.MIN_SPEED = self.np_random.uniform(0.9, 1.1) * MIN_SPEED 
        self.accel_ramp = self.np_random.uniform(0.9, 1.1) * ACCEL_RAMP
        self.latency = self.np_random.uniform(0.9, 1.1) * LATENCY

        # Start State
        self.data.qpos[self.pendel_qpos_adr] = self.np_random.uniform(low=-0.2, high=0.2)
        self.data.qpos[self.rotary_qpos_adr] = 0.0
        
        mujoco.mj_step(self.model, self.data)

        self.last_filtered_vel_p = 0.0
        self.last_filtered_vel_r = 0.0

        self.target_position = 0
        self.current_velocity = 0.0
        self.last_ctrl_target = 0.0
        # last_action initialisieren
        self.last_action = np.array([0.0]) 

        return self._get_obs(), {}

    def step(self, action):
        """Führt einen Simulationsschritt mit der gegebenen Aktion durch."""
        raw_action = float(action[0]) # Wert zwischen -1.0 und 1.0

        # --- NEUE SKALIERUNGS-LOGIK für MIN/MAX Speed und Deadzone ---
        deadzone = self.DEADZONE_MAGNITUDE 
        
        if abs(raw_action) < deadzone:
            # Aktion ist zu klein -> Motor stoppt
            desired_velocity = 0.0
        else:
            # 1. Betrag normalisieren: Skaliere den Eingabebereich [deadzone, 1.0] auf [0.0, 1.0]
            magnitude_normalized = (np.clip(abs(raw_action), deadzone, 1.0) - deadzone) / (1.0 - deadzone)
            
            # 2. Skalieren auf den Geschwindigkeitsbereich [MIN_SPEED, MAX_SPEED]
            speed_range = self.MAX_SPEED - self.MIN_SPEED
            speed = self.MIN_SPEED + magnitude_normalized * speed_range
            
            # 3. Vorzeichen wiederherstellen
            desired_velocity = np.sign(raw_action) * speed
        # --- ENDE SKALIERUNGS-LOGIK ---

        # --- 1. Stepper-Logik ---
        velocity_diff = desired_velocity - self.current_velocity
        max_accel_step = self.accel_ramp * self.dt
        self.current_velocity += np.clip(velocity_diff, -max_accel_step, max_accel_step)
        
        self.target_position += self.current_velocity * self.dt
        quantized_target = round(self.target_position / self.effective_step) * self.effective_step

        # --- 2. Physik-Simulation ---
        physics_timestep = self.model.opt.timestep
        n_latency_steps = int(self.latency / physics_timestep)
        n_total_steps = int(self.dt / physics_timestep)
        n_remaining_steps = max(0, n_total_steps - n_latency_steps)

        self.data.ctrl[self.actuator_id] = self.last_ctrl_target
        for _ in range(n_latency_steps):
            mujoco.mj_step(self.model, self.data)

        self.data.ctrl[self.actuator_id] = quantized_target
        for _ in range(n_remaining_steps):
            mujoco.mj_step(self.model, self.data)

        self.last_ctrl_target = quantized_target 

        # --- 3. Rewards & Terminierung ---
        rotary_pos = self.data.qpos[self.rotary_qpos_adr]
        
        terminated = False 

        obs = self._get_obs()
        reward = self.compute_rewards(obs, action)
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        info = {"rotary_angle": rotary_pos}

        if self.render_mode == "human":
            self.render()
            
        # Speichere die aktuelle Aktion für den nächsten Smoothness Penalty
        self.last_action = action.copy()
            
        return obs, reward, terminated, truncated, info

    
    def _get_obs(self):
        """Gibt den aktuellen Beobachtungsvektor zurück, gefiltert mit EMA."""
        
        # 1. Rohe, unnormalisierte Werte aus MuJoCo lesen
        theta_p_raw = self.data.qpos[self.pendel_qpos_adr]
        vel_p_raw = self.data.qvel[self.pendel_qvel_adr]

        theta_r_raw = self.data.qpos[self.rotary_qpos_adr]
        vel_r_raw = self.data.qvel[self.rotary_qpos_adr]

        # 2. Rauschen hinzufügen (Domain Randomization)
        theta_p_noise = theta_p_raw + self.np_random.normal(0, 0.001)
        vel_p_noise = vel_p_raw + self.np_random.normal(0, 0.05)
        theta_r_noise = theta_r_raw + self.np_random.normal(0, 0.001)
        vel_r_noise = vel_r_raw + self.np_random.normal(0, 0.05)
        
        # 3. Filterung der Geschwindigkeiten
        alpha = self.FILTER_ALPHA
        one_minus_alpha = 1.0 - alpha
        
        # Pendel-Geschwindigkeit (vel_p)
        filtered_vel_p = (alpha * vel_p_noise) + \
                             (one_minus_alpha * self.last_filtered_vel_p)
        
        # Rotary-Geschwindigkeit (vel_r)
        filtered_vel_r = (alpha * vel_r_noise) + \
                             (one_minus_alpha * self.last_filtered_vel_r)
        
        # 4. Filter-Zustand speichern 
        self.last_filtered_vel_p = filtered_vel_p
        self.last_filtered_vel_r = filtered_vel_r

        # 5. Normalisieren und Rückgabe
        vel_p_norm = filtered_vel_p 
        vel_r_norm = filtered_vel_r 
        theta_r_norm = theta_r_noise / 6.3 
        
        return np.array([
            np.sin(theta_p_noise),
            np.cos(theta_p_noise),
            vel_p_norm,
            theta_r_norm,
            vel_r_norm
        ], dtype=np.float32)
    

    def compute_rewards2(
        self,
        state,
        action,
        # Gewichte für die Nebenziele (Armposition, Energie, etc.)
        q_theta=1.0,           # Bestrafung für falsche Arm-Position
        q_theta_dot=0.1,       # Bestrafung für wilde Arm-Bewegungen
        q_phi_dot=0.5,         # Bestrafung für wilde Pendel-Bewegungen
        r_control=0.01,        # Bestrafung für hohen Energieverbrauch (u)
        q_smooth=0.0,          # NEU: Bestrafung für Aktionsänderungen (Zittern)
        
        # Parameter für die Gauß-Glocke
        sigma_phi=np.pi/2,         # SCHÄRFERER WERT für Stabilität
        
        phi_target=np.pi,
        theta_target=0.0
    ):

        sin_phi, cos_phi, phi_dot, theta, theta_dot = state
        
        phi_dot = phi_dot / 20
        theta_dot = theta_dot / 20

        phi = np.arctan2(sin_phi, cos_phi)
        u = action[0]

        # --- 1. Fehler berechnen ---
        phi_err_raw = phi - phi_target
        theta_err_raw = theta - theta_target

        # Normalisieren auf [-pi, pi]
        phi_err = ((phi_err_raw + np.pi) % (2 * np.pi)) - np.pi
        theta_err = ((theta_err_raw + np.pi) % (2 * np.pi)) - np.pi

        # --- 2. Das Hauptziel: Die Gauß-Glocke ---
        reward_phi = 1 * (np.exp(- (phi_err**2) / (2 * sigma_phi**2)))
        
        # --- 3. Der "Sweet Spot" Bonus ---
        dist_angle = np.abs(phi_err)
        dist_vel   = np.abs(phi_dot)

        # Ein Bonus, der explodiert, wenn beides gegen 0 geht
        stabilization_bonus = 2.0 * np.exp(-(dist_angle**2 + dist_vel**2) * 10.0)

        # --- 4. Die Kosten (Penalties) für Nebenziele ---
        
        # Berechne die Differenz zur letzten Aktion (für Smoothness Penalty)
        u_diff = u - self.last_action[0] 
        
        costs = (q_theta       * (theta_err**2) +
                 q_theta_dot   * (theta_dot**2) +
                 q_phi_dot     * (phi_dot**2) +
                 r_control     * (u**2) +
                 q_smooth      * (u_diff**2)) # <--- SMOOTHNESS COST

        # --- 5. Gesamt-Reward ---
        reward = reward_phi - costs + stabilization_bonus # <-- Bonus hier addiert!
        
        return float(reward)


    def compute_rewards(
        self,
        state,
        action,
        # Gewichte für die Nebenziele (Armposition, Energie, etc.)
        q_theta=1.0,           # Bestrafung für falsche Arm-Position
        q_theta_dot=0.1,       # Bestrafung für wilde Arm-Bewegungen
        q_phi_dot=0.5,         # Bestrafung für wilde Pendel-Bewegungen
        r_control=0.01,        # Bestrafung für hohen Energieverbrauch (u)
        q_smooth=0.0,          # NEU: Bestrafung für Aktionsänderungen (Zittern)
        
        # Parameter für die Gauß-Glocke
        sigma_phi=np.pi/2,         # SCHÄRFERER WERT für Stabilität
        
        phi_target=np.pi,
        theta_target=0.0
    ):
        sin_phi, cos_phi, phi_dot, theta, theta_dot = state
        
        phi_dot = phi_dot / 20
        theta_dot = theta_dot / 40

        phi = np.arctan2(sin_phi, cos_phi)
        u = action[0]

        # --- 1. Fehler berechnen ---
        phi_err_raw = phi - phi_target
        theta_err_raw = theta - theta_target
        # Normalisieren auf [-pi, pi]
        phi_err = ((phi_err_raw + np.pi) % (2 * np.pi)) - np.pi
        theta_err = ((theta_err_raw + np.pi) % (2 * np.pi)) - np.pi

        # --- 1. & 2. Hauptziel (Grobe Orientierung) ---
        # Die breite Glocke hilft beim Aufschwingen (Dense Reward)
        reward_phi = 1.0 * np.exp(-(phi_err**2) / (2 * sigma_phi**2))
        
        # --- 3. VERBESSERT: Der Stabilisierungs-Bonus ---
        # Statt harter If-Abfrage: Ein scharfer Peak, der Winkel UND Geschwindigkeit prüft.
        target_accuracy = 0.1  # Wie "scharf" soll der Bonus sein?
        combined_error = (phi_err**2) + 0.5 * (phi_dot**2) 
        bonus = 2.0 * np.exp(-combined_error / (target_accuracy**2))

        # --- 4. Kosten ---
        
        u_diff = u - self.last_action[0] 
        
        costs = (q_theta       * (theta_err**2) +
                 q_theta_dot   * (theta_dot**2) +
                 q_phi_dot     * (phi_dot**2) +
                 r_control     * (u**2) +
                 q_smooth      * (u_diff**2))

        # --- 5. Gesamt ---
        reward = reward_phi + bonus - costs
        
        return float(reward)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        if self.render_mode == "human":
            time.sleep(self.dt)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

