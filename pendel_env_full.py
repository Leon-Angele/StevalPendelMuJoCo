import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os

class PendelEnvFull(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=500):
        super().__init__()
        
        self.MAX_SPEED = 5.0  
        self.dt = 0.01        
        self.max_steps = max_steps
        self.current_step = 0

        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "Pendel_description", "pendel_roboter.xml")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # IDs holen
        self.pendel_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendelGelenk")
        self.rotary_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rotaryGelenk")
        self.actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_rotary")

        # --- NEU: Adressen für schnellen Zugriff cachen ---
        self.pendel_qpos_adr = self.model.jnt_qposadr[self.pendel_joint_id]
        self.pendel_qvel_adr = self.model.jnt_dofadr[self.pendel_joint_id]
        self.rotary_qpos_adr = self.model.jnt_qposadr[self.rotary_joint_id]
        self.rotary_qvel_adr = self.model.jnt_dofadr[self.rotary_joint_id]

        # Action Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # --- NEU: Observation Space auf 6 erweitert ---
        # [sin_p, cos_p, vel_p, sin_r, cos_r, vel_r]
        low = np.array([-1, -1, -np.inf, -1, -1, -np.inf], dtype=np.float32)
        high = np.array([1, 1, np.inf, 1, 1, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # Pendel leicht zufällig starten
        self.data.qpos[self.pendel_qpos_adr] = np.random.uniform(-0.1, 0.1)
        
        # Optional: Auch Rotary-Arm zufällig drehen?
        # self.data.qpos[self.rotary_qpos_adr] = np.random.uniform(-3.14, 3.14)
        
        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        target_speed = float(action[0]) * self.MAX_SPEED
        self.data.ctrl[self.actuator_id] = target_speed

        n_steps = int(self.dt / self.model.opt.timestep)
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self.calc_reward(obs, action)

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = {"target_speed": target_speed}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # --- NEU: Werte für BEIDE Gelenke holen ---
        
        # 1. Pendel Werte
        theta_p = self.data.qpos[self.pendel_qpos_adr]
        vel_p = self.data.qvel[self.pendel_qvel_adr]

        # 2. Rotary Werte
        theta_r = self.data.qpos[self.rotary_qpos_adr]
        vel_r = self.data.qvel[self.rotary_qvel_adr]

        # Array zusammenbauen (6 Werte)
        return np.array([
            np.sin(theta_p), # Pendel sin
            np.cos(theta_p), # Pendel cos
            vel_p,           # Pendel speed
            np.sin(theta_r), # Rotary sin
            np.cos(theta_r), # Rotary cos
            vel_r            # Rotary speed
        ], dtype=np.float32)

    def calc_reward(self, obs, action):
        # Obs entpacken
        # [sin_p, cos_p, vel_p, sin_r, cos_r, vel_r]
        cos_pendel = obs[1]
        vel_pendel = obs[2]
        vel_rotary = obs[5] # Rotary Speed nutzen wir für Kosten
        
        # --- Kosten Pendel (Ziel: Oben / cos=-1) ---
        dist_to_top = (cos_pendel + 1.0) 
        theta_cost = dist_to_top**2
        
        # --- Kosten Geschwindigkeit ---
        # Wir bestrafen jetzt auch Rotary-Speed leicht, damit er nicht unnötig rast
        vel_cost = 0.1 * (vel_pendel**2) + 0.01 * (vel_rotary**2)
        
        action_cost = 0.001 * (action[0]**2)
        
        return float(-(theta_cost + vel_cost + action_cost))

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()