import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
import time
from collections import deque

# --- Hardware Parameter (Basiswerte des Steppers) ---
ROTOR_POS_LIMIT_DEG = 240.0 
ROTOR_POS_LIMIT_RAD = np.deg2rad(ROTOR_POS_LIMIT_DEG)

MAX_SPEED_PPS = 2000.0       
MAX_ACCEL_PPS = 6000.0       
STEPS_PER_REV = 3200.0       # 1/16 Microsteps (200 * 16)
ENCODER_COUNTS_PER_REV = 2400.0 # 600 PPR * 4 (typischer Quadraturzähler)

DT = 0.005       # 5ms Steuerungsintervall
LATENCY_STEPS = 1 # 1 Step Delay simuliert die Totzeit

class PendelEnv(gym.Env):
    """
    MuJoCo-Umgebung für ein Inverses Pendel (Rotary Inverted Pendulum).
    Simuliert Positionssteuerung (GoTo) mit Beschleunigungslimit, 
    Quantisierung und Encoder-Auflösung, um den Sim-to-Real Gap zu minimieren.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=3000):
        super().__init__()

        # --- 1. Systemparameter ---
        self.FILTER_ALPHA = 0.4
        self.DEADZONE_MAGNITUDE = 0.05 
        self.dt = DT
        self.max_steps = max_steps
        self.current_step = 0

        # Motor Umrechnung in physikalische Einheiten
        self.MAX_SPEED_RAD = (MAX_SPEED_PPS / STEPS_PER_REV) * 2 * np.pi
        self.MAX_ACCEL_RAD = (MAX_ACCEL_PPS / STEPS_PER_REV) * 2 * np.pi
        self.max_pos_rad = ROTOR_POS_LIMIT_RAD
        
        # Quantisierungs-Werte
        self.step_angle_rad = np.deg2rad(1.8) / 16.0 
        self.encoder_res_rad = 2 * np.pi / ENCODER_COUNTS_PER_REV
        
        # Controller Interne Zustände
        self.current_motor_pos = 0.0
        self.current_motor_vel = 0.0 # Trackt die simulierte Motorgeschw. (für Accel Limit)
        self.last_ctrl_target = 0.0
        
        # Verzögerungs-Puffer für Actions (Totzeit Simulation)
        self.action_delay_buffer = deque([0.0] * LATENCY_STEPS, maxlen=LATENCY_STEPS)
        self.last_action = np.array([0.0])
        self.last_filtered_vel_p = 0.0
        self.last_filtered_vel_r = 0.0

        # --- 2. MuJoCo Setup ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "Pendel_description", "pendel_roboter_mutter.xml") 

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.0005 

        # IDs & Adressen cachen
        self.pendel_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendelGelenk")
        self.rotary_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rotaryGelenk")
        self.actuator_id = mujuco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_rotary")
        self.pendel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Pendel_1")
        
        self.pendel_qpos_adr = self.model.jnt_qposadr[self.pendel_joint_id]
        self.pendel_qvel_adr = self.model.jnt_dofadr[self.pendel_joint_id]
        self.rotary_qpos_adr = self.model.jnt_qposadr[self.rotary_joint_id]
        
        # Referenzwerte für Randomization (Reibung und Masse)
        self.default_pendel_mass = self.model.body_mass[self.pendel_body_id]
        self.default_pendel_damping = self.model.dof_damping[self.pendel_qvel_adr]
        self.default_pendel_friction = self.model.dof_frictionloss[self.pendel_qvel_adr]


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
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Domain Randomization
        mass_factor = self.np_random.uniform(0.95, 1.05)
        self.model.body_mass[self.pendel_body_id] = self.default_pendel_mass * mass_factor
        
        self.model.dof_damping[self.pendel_qvel_adr] = self.default_pendel_damping * self.np_random.uniform(0.8, 1.2)
        self.model.dof_frictionloss[self.pendel_qvel_adr] = self.default_pendel_friction * self.np_random.uniform(0.5, 1.5)

        # Motor Zustände resetten
        self.current_motor_pos = 0.0
        self.current_motor_vel = 0.0
        self.last_ctrl_target = 0.0
        self.last_action = np.array([0.0])
        self.last_filtered_vel_p = 0.0
        self.last_filtered_vel_r = 0.0
        self.action_delay_buffer = deque([0.0] * LATENCY_STEPS, maxlen=LATENCY_STEPS)

        # Start State
        self.data.qpos[self.pendel_qpos_adr] = self.np_random.uniform(low=-0.2, high=0.2)
        self.data.qpos[self.rotary_qpos_adr] = 0.0
        
        mujoco.mj_step(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        """Führt einen Simulationsschritt mit der gegebenen Aktion durch (Positionssteuerung)."""
        
        # 1. Action Delay (Systemlatenz)
        self.action_delay_buffer.append(action[0])
        delayed_action = self.action_delay_buffer[0] 
        
        raw_action = float(np.clip(delayed_action, -1.0, 1.0))

        # 2. Action Mapping: [-1, 1] -> Zielwinkel [-240°, 240°]
        target_pos_rad = raw_action * self.max_pos_rad

        # 3. Stepper Simulation (Trapezprofil-Annäherung)
        
        # A) Gewünschte Geschwindigkeit (begrenzt durch P-Controller/Target Position)
        dist_to_target = target_pos_rad - self.current_motor_pos
        desired_vel = dist_to_target / self.dt
        
        # B) Geschwindigkeitslimit (MAX_SPEED)
        desired_vel = np.clip(desired_vel, -self.MAX_SPEED_RAD, self.MAX_SPEED_RAD)
        
        # C) Beschleunigungslimit (MAX_ACCEL)
        max_vel_change = self.MAX_ACCEL_RAD * self.dt
        vel_diff = desired_vel - self.current_motor_vel
        actual_vel_change = np.clip(vel_diff, -max_vel_change, max_vel_change)
        
        # D) Neue Geschwindigkeit und Position
        self.current_motor_vel += actual_vel_change
        self.current_motor_pos += self.current_motor_vel * self.dt

        # 4. Stepper Quantisierung (Der "Stepper-Effekt")
        quantized_motor_pos = round(self.current_motor_pos / self.step_angle_rad) * self.step_angle_rad

        # 5. Physik-Simulation in MuJoCo durchführen
        physics_timestep = self.model.opt.timestep
        n_total_steps = int(self.dt / physics_timestep)

        # MuJoCo Actor Steuerung
        self.data.ctrl[self.actuator_id] = quantized_motor_pos
        
        for _ in range(n_total_steps):
            mujoco.mj_step(self.model, self.data)

        self.last_ctrl_target = quantized_motor_pos 

        # --- 6. Rewards & Obs ---
        
        terminated = False 

        obs = self._get_obs()
        reward = self.compute_rewards(obs, action)
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        info = {"target_pos": target_pos_rad, "motor_vel": self.current_motor_vel}
        
        if self.render_mode == "human":
            self.render()
            
        self.last_action = action.copy()
            
        return obs, reward, terminated, truncated, info

    
    def _get_obs(self):
        """Gibt den aktuellen Beobachtungsvektor zurück, inkl. Quantisierung und Rauschen."""
        
        # 1. Rohe Werte aus MuJoCo lesen
        theta_p_raw = self.data.qpos[self.pendel_qpos_adr]
        vel_p_raw = self.data.qvel[self.pendel_qvel_adr]
        theta_r_raw = self.data.qpos[self.rotary_qpos_adr]
        vel_r_raw = self.data.qvel[self.rotary_qpos_adr]

        # 2. Rauschen
        theta_p_noisy = theta_p_raw + self.np_random.normal(0, 0.001)
        vel_p_noisy = vel_p_raw + self.np_random.normal(0, 0.05)
        theta_r_noisy = theta_r_raw + self.np_random.normal(0, 0.001)
        vel_r_noisy = vel_r_raw + self.np_random.normal(0, 0.05)
        
        # 3. Quantisierung der Beobachtung (Encoder-Effekt)
        theta_p_discrete = round(theta_p_noisy / self.encoder_res_rad) * self.encoder_res_rad
        theta_r_discrete = round(theta_r_noisy / self.encoder_res_rad) * self.encoder_res_rad

        # 4. Filterung der Geschwindigkeiten (EMA)
        alpha = self.FILTER_ALPHA
        filtered_vel_p = (alpha * vel_p_noisy) + ((1.0 - alpha) * self.last_filtered_vel_p)
        filtered_vel_r = (alpha * vel_r_noisy) + ((1.0 - alpha) * self.last_filtered_vel_r)
        
        self.last_filtered_vel_p = filtered_vel_p
        self.last_filtered_vel_r = filtered_vel_r

        # 5. Normalisieren und Rückgabe
        theta_r_norm = theta_r_discrete / (2 * np.pi) 
        
        return np.array([
            np.sin(theta_p_discrete),
            np.cos(theta_p_discrete),
            filtered_vel_p,
            theta_r_norm,
            filtered_vel_r
        ], dtype=np.float32)
    

    def compute_rewards(
        self,
        state,
        action,
        q_theta=1.0, q_theta_dot=0.1, q_phi_dot=0.5, r_control=0.01, q_smooth=0.0, 
        sigma_phi=np.pi/2, phi_target=np.pi, theta_target=0.0
    ):
        sin_phi, cos_phi, phi_dot, theta, theta_dot = state
        
        phi_dot = phi_dot / 20
        theta_dot = theta_dot / 40

        phi = np.arctan2(sin_phi, cos_phi)
        u = action[0]

        # Fehler berechnen
        phi_err_raw = phi - phi_target
        theta_err_raw = theta - theta_target
        phi_err = ((phi_err_raw + np.pi) % (2 * np.pi)) - np.pi
        theta_err = ((theta_err_raw + np.pi) % (2 * np.pi)) - np.pi

        # 1. Hauptziel (Grobe Orientierung)
        reward_phi = 1.0 * np.exp(-(phi_err**2) / (2 * sigma_phi**2))
        
        # 2. Stabilisierungs-Bonus (Scharfer Peak)
        target_accuracy = 0.1
        combined_error = (phi_err**2) + 0.5 * (phi_dot**2) 
        bonus = 2.0 * np.exp(-combined_error / (target_accuracy**2))

        # 3. Kosten
        u_diff = u - self.last_action[0] 
        
        costs = (q_theta       * (theta_err**2) +
                 q_theta_dot   * (theta_dot**2) +
                 q_phi_dot     * (phi_dot**2) +
                 r_control     * (u**2) +
                 q_smooth      * (u_diff**2))

        # 4. Gesamt
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