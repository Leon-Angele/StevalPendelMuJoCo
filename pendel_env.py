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
        self.accel_ramp = 100.0
        self.effective_step = np.deg2rad(1.8) / 16 
        self.limit_threshold = 0.05 # Puffer für Endanschlag (rad)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "Pendel_description", "pendel_roboter.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

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
        
        self.default_pendel_mass = self.model.body_mass[self.pendel_body_id]
        self.default_pendel_damping = self.model.dof_damping[self.pendel_qvel_adr]

        # Action Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space: [sin(theta_p), cos(theta_p), vel_p, last_action]
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -60], dtype=np.float32),
            high=np.array([1, 1, 60], dtype=np.float32),
            shape=(3,), 
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        self.target_position = 0.0
        self.current_velocity = 0.0
        self.last_ctrl_target = 0.0
        self.last_action = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Domain Randomization
        mass_factor = self.np_random.uniform(0.8, 1.2)
        self.model.body_mass[self.pendel_body_id] = self.default_pendel_mass * mass_factor
        damping_factor = self.np_random.uniform(0.5, 2.0)
        self.model.dof_damping[self.pendel_qvel_adr] = self.default_pendel_damping * damping_factor

        # Start State (unten hängend)
        self.data.qpos[self.pendel_qpos_adr] = self.np_random.uniform(low=-0.2, high=0.2)
        self.data.qpos[self.rotary_qpos_adr] = 0.0
        mujoco.mj_step(self.model, self.data)

        self.target_position = 0.0
        self.current_velocity = 0.0
        self.last_ctrl_target = 0.0
        self.last_action = 0.0 

        return self._get_obs(), {}

    def step(self, action):
        desired_velocity = float(action[0]) * self.MAX_SPEED

        # Stepper-Motor Logik (Beschleunigungsrampe & Quantisierung)
        velocity_diff = desired_velocity - self.current_velocity
        max_accel_step = self.accel_ramp * self.dt
        self.current_velocity += np.clip(velocity_diff, -max_accel_step, max_accel_step)
        position_increment = self.current_velocity * self.dt
        self.target_position += position_increment
        quantized_target = round(self.target_position / self.effective_step) * self.effective_step

        # Latenz & Physics Stepping
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

        # Interne Checks & Rewards
        rotary_pos = self.data.qpos[self.rotary_qpos_adr]
        terminated = False
        limit_penalty = 0.0

        # Check: Endanschlag erreicht?
        hit_min = rotary_pos <= (self.rotary_min + self.limit_threshold)
        hit_max = rotary_pos >= (self.rotary_max - self.limit_threshold)
        
        if hit_min or hit_max:
            terminated = True
            limit_penalty = -10.0

        obs = self._get_obs()
        reward = self.compute_rewards(obs, action) + limit_penalty 
        
        self.last_action = float(action[0]) 
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        info = {"rotary_angle": rotary_pos}

        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        theta_p = self.data.qpos[self.pendel_qpos_adr]
        vel_p = self.data.qvel[self.pendel_qvel_adr]

        # Noise
        theta_p += self.np_random.normal(0, 0.001)
        vel_p += self.np_random.normal(0, 0.05)
        vel_p = np.clip(vel_p, -50.0, 50.0)   

        # Observation Space: [sin(theta_p), cos(theta_p), vel_p, last_action]
        return np.array([
            np.sin(theta_p), 
            np.cos(theta_p), 
            vel_p
        ], dtype=np.float32)

    def compute_rewards(self, obs, action):
        sin_phi, cos_phi, phi_dot = obs 
        
        # 1. Distance (Ziel: cos=-1, also oben)
        dist_to_top = (1 + cos_phi)
        distance_reward = -dist_to_top 
        
        # 2. Bonus für Upright
        bonus_width = 0.1 
        bonus_amplitude = 5.0 
        upright_bonus = bonus_amplitude * np.exp(- (dist_to_top**2) / (2 * bonus_width**2))
        
        # 3. Velocity Penalty
        phi_dot_norm = phi_dot / 36.0 
        if dist_to_top < 0.2:
            velocity_penalty = -0.5 * phi_dot_norm**2
        else:
            velocity_penalty = -0.05 * phi_dot_norm**2
        
        # 4. Action Penalty
        action_penalty = -0.002 * float(action[0])**2
        
        reward = distance_reward + upright_bonus + velocity_penalty + action_penalty
        
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