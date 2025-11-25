import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
import time

class PendelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()

        self.MAX_SPEED = 5.0
        self.dt = 0.005
        self.latency = 0.001
        self.max_steps = max_steps
        self.current_step = 0
        
        # Stepper Motor Parameter
        self.accel_ramp = 100.0
        self.effective_step = np.deg2rad(1.8) / 16 
        self.limit_threshold = 0.05 

        # MuJoCo Setup
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "Pendel_description", "pendel_roboter.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

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

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Obs: [sin(theta), cos(theta), vel]
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -np.inf], dtype=np.float32),
            high=np.array([1, 1, np.inf], dtype=np.float32),
            shape=(3,), 
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        
        # Controller States
        self.target_position = 0.0
        self.current_velocity = 0.0
        self.last_ctrl_target = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Domain Randomization
        # Masse: +/- 5% (da Hardware bekannt)
        mass_factor = self.np_random.uniform(0.95, 1.05)
        self.model.body_mass[self.pendel_body_id] = self.default_pendel_mass * mass_factor
        
        # Dämpfung: 0.8x bis 1.2x (da XML geschätzt)
        damping_factor = self.np_random.uniform(0.8, 1.2)
        self.model.dof_damping[self.pendel_qvel_adr] = self.default_pendel_damping * damping_factor

        # Start State
        self.data.qpos[self.pendel_qpos_adr] = self.np_random.uniform(low=-0.2, high=0.2)
        self.data.qpos[self.rotary_qpos_adr] = 0.0 # Sicherstellen, dass Arm bei 0 startet
        
        mujoco.mj_step(self.model, self.data)

        self.target_position = 0.0
        self.current_velocity = 0.0
        self.last_ctrl_target = 0.0

        return self._get_obs(), {}

    def step(self, action):
        desired_velocity = float(action[0]) * self.MAX_SPEED

        # Stepper-Logik (Rampe & Quantisierung)
        velocity_diff = desired_velocity - self.current_velocity
        max_accel_step = self.accel_ramp * self.dt
        self.current_velocity += np.clip(velocity_diff, -max_accel_step, max_accel_step)
        
        self.target_position += self.current_velocity * self.dt
        quantized_target = round(self.target_position / self.effective_step) * self.effective_step

        # Latenz-Simulation
        physics_timestep = self.model.opt.timestep 
        n_latency_steps = int(self.latency / physics_timestep)
        n_total_steps = int(self.dt / physics_timestep)
        n_remaining_steps = max(0, n_total_steps - n_latency_steps)

        # 1. Latenz-Phase (alter Befehl)
        self.data.ctrl[self.actuator_id] = self.last_ctrl_target
        for _ in range(n_latency_steps):
            mujoco.mj_step(self.model, self.data)

        # 2. Aktions-Phase (neuer Befehl)
        self.data.ctrl[self.actuator_id] = quantized_target
        for _ in range(n_remaining_steps):
            mujoco.mj_step(self.model, self.data)

        self.last_ctrl_target = quantized_target

        # Checks & Rewards
        rotary_pos = self.data.qpos[self.rotary_qpos_adr]
        terminated = False

        obs = self._get_obs()
        reward = self.compute_rewards(obs, action) 
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        info = {"rotary_angle": rotary_pos}

        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        theta_p = self.data.qpos[self.pendel_qpos_adr]
        vel_p = self.data.qvel[self.pendel_qvel_adr]

        """         # Sensor Noise
        theta_p += self.np_random.normal(0, 0.001)
        vel_p += self.np_random.normal(0, 0.05)
        vel_p = np.clip(vel_p, -50.0, 50.0)   
        """
        return np.array([
            np.sin(theta_p), 
            np.cos(theta_p), 
            vel_p
        ], dtype=np.float32)

    def compute_rewards(self, obs, action):

        obs_history = obs.reshape(-1, 3)
        
        # Extrahiere die Spalten über alle Frames hinweg
        # col 0: sin_phi, col 1: cos_alpha, col 2: alpha_dot
        cos_alpha_hist = obs_history[:, 1]
        alpha_dot_hist = obs_history[:, 2]
        
        # Die aktuellsten Werte (für Basis-Berechnungen)
        current_cos_alpha = cos_alpha_hist[-1]
        current_alpha_dot = alpha_dot_hist[-1]
        
        # --- 2. Berechnung der Historien-Metriken ---
        
        # A. Durschnittliche Position (Konsistenz)
        mean_cos_alpha = np.mean(cos_alpha_hist[-5:]) 
        
        # B. Trend (für besseres Aufschwingen)

        trend_improvement = np.mean(cos_alpha_hist[-3:]) - np.mean(cos_alpha_hist[:3])
        
        # C. Laufruhe (Smoothness / Anti-Jitter)

        alpha_dot_std = np.std(alpha_dot_hist)

        # --- 3. Reward Komponenten ---
        
        # Skalierungen
        alpha_dot_scaled = current_alpha_dot / 50.0 
        action_scaled = action[0]

        # A. Swing-Up Reward (BASIERT JETZT AUF MEAN)
        # Nutzung des Mittelwerts macht den Gradienten stabiler.
        r_swingup = (1.0 - mean_cos_alpha) / 2.0 
        
        # B. Balance Bonus (Gauß)
        dist_to_top = (1.0 + current_cos_alpha) 
        r_balance = np.exp(-(dist_to_top**2) / (2 * 0.2**2)) 
        
        # C. Stability Penalty (Kombiniert absolute Speed + Varianz)
        # Wir bestrafen jetzt auch, wenn er zittert (std), nicht nur wenn er schnell ist.
        r_stability = -1.0 * (alpha_dot_scaled**2) - 0.5 * (alpha_dot_std / 50.0)
        
        # D. Swing-Up Boost (NEU)

        r_trend = 0
        if r_balance < 0.5 and trend_improvement < 0: 
            # Hinweis: Ziel ist -1. Wenn wir von 1 auf 0 gehen, wird der Wert KLEINER.
            # Ein negativer Trend (Richtung -1) ist hier also gut!
            r_trend = abs(trend_improvement) * 5.0

        # E. Effort Penalty
        r_effort = -0.05 * (action_scaled**2)
        
        # --- 4. Gesamtsumme ---
        # Gewichtung des Trends nicht zu hoch, sonst "wackelt" er, um Rewards zu farmen.
        reward = (1.0 * r_swingup) + (2.0 * r_balance) + r_stability + r_trend + r_effort
        
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