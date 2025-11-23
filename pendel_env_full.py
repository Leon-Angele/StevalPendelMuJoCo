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
        self.dt = 0.005          # 200 Hz Control Loop
        self.latency = 0.001     # 1ms Verzögerung (State -> Action)
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
        self.rotary_qvel_adr = self.model.jnt_dofadr[self.rotary_joint_id]
        
        self.rotary_min = self.model.jnt_range[self.rotary_joint_id][0]
        self.rotary_max = self.model.jnt_range[self.rotary_joint_id][1]
        
        # Referenzwerte für Randomization
        self.default_pendel_mass = self.model.body_mass[self.pendel_body_id]
        self.default_pendel_damping = self.model.dof_damping[self.pendel_qvel_adr]
        self.default_pendel_frictionloss = self.model.dof_frictionloss[self.pendel_qvel_adr]

        # Action Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space: [sin(p), cos(p), vel_p, rotary_norm, vel_r]
        # rotary_norm ist -1 bis 1
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -60, -1.0, -self.MAX_SPEED*2], dtype=np.float32),
            high=np.array([1, 1, 60, 1.0, self.MAX_SPEED*2], dtype=np.float32),
            shape=(5,), 
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        
        self.target_position = 0.0
        self.current_velocity = 0.0
        self.last_ctrl_target = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # --- Domain Randomization ---
        # Masse (+/- 5%)
        mass_factor = self.np_random.uniform(0.95, 1.05)
        self.model.body_mass[self.pendel_body_id] = self.default_pendel_mass * mass_factor
        
        # Dämpfung (Viskos) (0.8x bis 1.2x, da unsicher)
        damping_factor = self.np_random.uniform(0.8, 1.2)
        self.model.dof_damping[self.pendel_qvel_adr] = self.default_pendel_damping * damping_factor

        # Reibung (Trocken) (0.8x bis 1.2x, da unsicher)
        friction_factor = self.np_random.uniform(0.8, 1.2)
        self.model.dof_frictionloss[self.pendel_qvel_adr] = self.default_pendel_frictionloss * friction_factor

        # Start State
        self.data.qpos[self.pendel_qpos_adr] = self.np_random.uniform(low=-0.2, high=0.2)
        self.data.qpos[self.rotary_qpos_adr] = 0.0 
        
        mujoco.mj_step(self.model, self.data)

        # Controller Reset
        self.target_position = 0.0
        self.current_velocity = 0.0
        self.last_ctrl_target = 0.0

        return self._get_obs(), {}

    def step(self, action):
        desired_velocity = float(action[0]) * self.MAX_SPEED

        # --- 1. Stepper-Berechnung (Planung) ---
        velocity_diff = desired_velocity - self.current_velocity
        max_accel_step = self.accel_ramp * self.dt
        self.current_velocity += np.clip(velocity_diff, -max_accel_step, max_accel_step)
        
        self.target_position += self.current_velocity * self.dt
        quantized_target = round(self.target_position / self.effective_step) * self.effective_step

        # --- 2. Physik mit Latenz (Ausführung) ---
        physics_timestep = self.model.opt.timestep 
        n_latency_steps = int(self.latency / physics_timestep)
        n_total_steps = int(self.dt / physics_timestep)
        n_remaining_steps = max(0, n_total_steps - n_latency_steps)

        # Phase A: Latenzzeit (Motor hält alten Wert)
        self.data.ctrl[self.actuator_id] = self.last_ctrl_target
        for _ in range(n_latency_steps):
            mujoco.mj_step(self.model, self.data)

        # Phase B: Neuer Befehl aktiv
        self.data.ctrl[self.actuator_id] = quantized_target
        for _ in range(n_remaining_steps):
            mujoco.mj_step(self.model, self.data)

        self.last_ctrl_target = quantized_target

        # --- 3. Checks ---
        rotary_pos = self.data.qpos[self.rotary_qpos_adr]
        terminated = False
        limit_penalty = 0.0

        # Harte Limits
        if rotary_pos <= (self.rotary_min + self.limit_threshold) or \
           rotary_pos >= (self.rotary_max - self.limit_threshold):
            terminated = False # kein Abbruch, nur Penalty
            limit_penalty = -10.0

        obs = self._get_obs()
        reward = self.compute_rewards(obs, action) + limit_penalty 
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        info = {"rotary_angle": rotary_pos}

        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        theta_p = self.data.qpos[self.pendel_qpos_adr]
        vel_p = self.data.qvel[self.pendel_qvel_adr]
        theta_r = self.data.qpos[self.rotary_qpos_adr]
        vel_r = self.data.qvel[self.rotary_qvel_adr]

        # Sensor Noise hinzufügen
        theta_p += self.np_random.normal(0, 0.001)
        vel_p += self.np_random.normal(0, 0.05)
        vel_p = np.clip(vel_p, -50.0, 50.0)
        
        theta_r += self.np_random.normal(0, 0.0005) 
        vel_r += self.np_random.normal(0, 0.01)

        # Normalisierung Rotary (-1 bis 1 für das Netz)

        theta_r_norm = theta_r / self.rotary_max

        return np.array([
            np.sin(theta_p), 
            np.cos(theta_p), 
            vel_p,
            theta_r_norm, 
            vel_r
        ], dtype=np.float32)

    def compute_rewards(self, obs, action):
        sin_phi, cos_phi, phi_dot, theta_r_norm, vel_r = obs 
        
        # Distance to top
        dist_to_top = (1 + cos_phi)
        distance_reward = -dist_to_top 
        
        # Balance Bonus (Gaussian peak)
        bonus_width = 0.1 
        bonus_amplitude = 5.0 
        upright_bonus = bonus_amplitude * np.exp(- (dist_to_top**2) / (2 * bonus_width**2))
        
        # Velocity Penalty
        phi_dot_norm = phi_dot / 36.0 
        if dist_to_top < 0.2:
            velocity_penalty = -0.5 * phi_dot_norm**2
        else:
            velocity_penalty = -0.05 * phi_dot_norm**2
        
        action_penalty = -0.002 * float(action[0])**2
        
        # Center Penalty (basierend auf normiertem Winkel)
        # Bestraft, wenn der Arm nah an die Limits (-1 oder 1) kommt
        center_penalty = -0.5 * (theta_r_norm**2)

        return float(distance_reward + upright_bonus + velocity_penalty + action_penalty + center_penalty)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        if self.render_mode == "human":
            time.sleep(self.dt)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()