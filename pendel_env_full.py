import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
import time

SLIPPING_ENABLED = False

class PendelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()

        # --- Timing Einstellungen ---
        self.MAX_SPEED = 5.0  # Max Geschwindigkeit (rad/s)
        
        # Zykluszeit: 5ms (200 Hz) wie dein echtes System
        self.dt = 0.005  
        
        # Latenz: 1ms (Verzögerung bis Motor reagiert)
        self.latency = 0.001 
        
        self.max_steps = max_steps
        self.current_step = 0

        # --- Schrittmotor-Parameter (STEVAL-EDUKIT01 / NEMA17) ---
        self.step_angle = np.deg2rad(1.8)
        self.holding_torque_max = 0.6 
        self.accel_ramp = 100.0 
        self.microsteps = 16 
        self.effective_step = self.step_angle / self.microsteps

        # Pfad zur XML laden
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "Pendel_description", "pendel_roboter.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # --- IDs und Adressen holen ---
        self.pendel_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendelGelenk")
        self.rotary_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rotaryGelenk")
        self.actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_rotary")
        
        # Body ID für Domain Randomization (Masse ändern)
        self.pendel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Pendel_1")

        self.pendel_qpos_adr = self.model.jnt_qposadr[self.pendel_joint_id]
        self.pendel_qvel_adr = self.model.jnt_dofadr[self.pendel_joint_id]
        self.rotary_qpos_adr = self.model.jnt_qposadr[self.rotary_joint_id]
        self.rotary_qvel_adr = self.model.jnt_dofadr[self.rotary_joint_id]

        # --- Standardwerte für Randomization speichern ---
        # Wir speichern die Werte aus der XML, um sie als Basis für die Variation zu nutzen
        self.default_pendel_mass = self.model.body_mass[self.pendel_body_id]
        self.default_pendel_damping = self.model.dof_damping[self.pendel_qvel_adr]

        # Action Space: -1 bis 1 (wird auf MAX_SPEED skaliert)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -60, -1, -1, -25], dtype=np.float32),
            high=np.array([1, 1, 60, 1, 1, 25], dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None

        # --- Zustandsvariablen ---
        self.target_position = 0.0
        self.current_velocity = 0.0
        self.last_ctrl_target = 0.0 # Für Latenz-Logik

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # MuJoCo Reset
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # =========================================================
        # === DOMAIN RANDOMIZATION (Sim-to-Real Robustness) ===
        # =========================================================
        
        # 1. Masse Randomization (+/- 20%)
        # Simuliert unterschiedliche Dichten im Material oder Schrauben/Kabel
        mass_factor = self.np_random.uniform(0.8, 1.2)
        self.model.body_mass[self.pendel_body_id] = self.default_pendel_mass * mass_factor

        # 2. Dämpfung/Reibung Randomization (0.5x bis 2.0x)
        # Reibung ist in der Realität schwer zu modellieren und variiert stark
        damping_factor = self.np_random.uniform(0.5, 2.0)
        self.model.dof_damping[self.pendel_qvel_adr] = self.default_pendel_damping * damping_factor

        # 3. Zufälliger Startzustand
        # Startet nicht immer perfekt bei 0, sondern leicht wackelnd (+/- 0.2 rad)
        # Hilft dem Agenten, sich aus ungünstigen Lagen zu erholen
        self.data.qpos[self.pendel_qpos_adr] = self.np_random.uniform(low=-0.2, high=0.2)
        self.data.qpos[self.rotary_qpos_adr] = 0.0
        
        # Initialen Schritt ausführen, um Physik zu aktualisieren
        mujoco.mj_step(self.model, self.data)

        # Interne Variablen zurücksetzen
        self.target_position = 0.0
        self.current_velocity = 0.0
        self.last_ctrl_target = 0.0

        return self._get_obs(), {}

    def step(self, action):
        # 1. Action Verarbeitung (OHNE Smoothing -> direktes Mapping)
        desired_velocity = float(action[0]) * self.MAX_SPEED

        # 2. Beschleunigungsrampe (Schrittmotor Physik)
        velocity_diff = desired_velocity - self.current_velocity
        max_accel_step = self.accel_ramp * self.dt
        self.current_velocity += np.clip(velocity_diff, -max_accel_step, max_accel_step)

        # 3. Integriere zu Position
        position_increment = self.current_velocity * self.dt
        self.target_position += position_increment

        # 4. Quantisierung (Stepper Microsteps)
        quantized_target = round(self.target_position / self.effective_step) * self.effective_step

        # 5. --- LATENZ SIMULATION (1ms Latenz bei 5ms Zyklus) ---
        physics_timestep = self.model.opt.timestep # XML Timestep (z.B. 0.001)
        
        # Berechnung der Sub-Steps
        n_latency_steps = int(self.latency / physics_timestep) # 1ms / 1ms = 1 Step
        n_total_steps = int(self.dt / physics_timestep)        # 5ms / 1ms = 5 Steps
        n_remaining_steps = n_total_steps - n_latency_steps
        
        if n_remaining_steps < 0: n_remaining_steps = 0

        # Phase A: Latenzzeit -> Motor hält noch den ALTEN Wert
        self.data.ctrl[self.actuator_id] = self.last_ctrl_target
        for _ in range(n_latency_steps):
            mujoco.mj_step(self.model, self.data)

        # Phase B: Aktionszeit -> Motor bekommt den NEUEN Wert
        self.data.ctrl[self.actuator_id] = quantized_target
        for _ in range(n_remaining_steps):
            mujoco.mj_step(self.model, self.data)

        # Speichern für nächsten Step
        self.last_ctrl_target = quantized_target

        # 6. Beobachtung & Reward
        obs = self._get_obs()
        reward = self.compute_rewards(obs, action)
        
        self.current_step += 1
        terminated = False 
        truncated = self.current_step >= self.max_steps

        info = {
            "target_speed": desired_velocity, 
            "current_speed": self.current_velocity
        }

        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Echte Daten aus MuJoCo
        theta_p = self.data.qpos[self.pendel_qpos_adr]
        vel_p = self.data.qvel[self.pendel_qvel_adr]
        theta_r = self.data.qpos[self.rotary_qpos_adr]
        vel_r = self.data.qvel[self.rotary_qvel_adr]

        # === OBSERVATION NOISE (Sim-to-Real) ===
        # Simuliert Rauschen von Encodern und diskreter Differenzierung
        # Winkel-Rauschen: Standardabweichung ca. 0.05 Grad (~0.001 rad)
        theta_p += self.np_random.normal(0, 0.001)
        theta_r += self.np_random.normal(0, 0.001)
        
        # Geschwindigkeits-Rauschen: Ist in echt viel höher durch Differenzierung
        vel_p += self.np_random.normal(0, 0.05)
        vel_r += self.np_random.normal(0, 0.05)

        # Clipping für Sicherheit im Netz
        vel_p = np.clip(vel_p, -50.0, 50.0)   
        vel_r = np.clip(vel_r, -20.0, 20.0)   

        return np.array([
            np.sin(theta_p), np.cos(theta_p), vel_p,
            np.sin(theta_r), np.cos(theta_r), vel_r
        ], dtype=np.float32)

    def compute_rewards(self, obs, action):
        sin_phi, cos_phi, phi_dot = obs[0], obs[1], obs[2]
        
        # Skalierte Geschwindigkeit
        phi_dot_norm = phi_dot / 36.0 
        
        # Ziel: Pendel oben (cos_phi = -1, da MuJoCo Gravity -z ist und 0 = unten)
        # Distanz zum Ziel "Oben" (0 bis 2)
        dist_to_top = (1 + cos_phi) 
        
        # Grundstrafe für Distanz
        distance_reward = -dist_to_top 
        
        # Starker Bonus nur wenn wirklich fast oben
        bonus_width = 0.1 
        bonus_amplitude = 5.0 
        upright_bonus = bonus_amplitude * np.exp(- (dist_to_top**2) / (2 * bonus_width**2))
        
        # Velocity Penalty (Vibration vermeiden)
        # Wenn fast oben: Strikter sein
        if dist_to_top < 0.2:
            velocity_penalty = -0.5 * phi_dot_norm**2
        else:
            velocity_penalty = -0.05 * phi_dot_norm**2
        
        # Action Penalty (Energieeffizienz und Motorshonung)
        action_penalty = -0.002 * float(action[0])**2
        
        reward = distance_reward + upright_bonus + velocity_penalty + action_penalty
        
        return float(reward)

    def render(self):
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            
            # --- Bremse für Echtzeit-Wahrnehmung ---
            if self.render_mode == "human":
                time.sleep(self.dt)  # Schläft ca. 5ms pro Frame

    def close(self):
        if self.viewer is not None:
            self.viewer.close()