import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os

class PendelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=500):
        super().__init__()
        
        self.MAX_SPEED = 7.5
        self.dt = 0.02        
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
        self.data.qpos[self.pendel_qpos_adr] = 0
        self.data.qpos[self.rotary_qpos_adr] = 0

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
        vel_pendel = obs[2] /50
        vel_rotary = obs[5] / self.MAX_SPEED

        # --- Kosten Pendel (Ziel: Oben / cos=-1) ---
        theta_cost = (cos_pendel + 1.0)
    
        return -(theta_cost**2 + 0.1*vel_pendel**2 + 0.01*vel_rotary**2)


    def compute_rewards(self, obs, action):
    
        sin_phi, cos_phi, phi_dot = obs[0], obs[1], obs[2]
        phi_dot_norm = phi_dot / 50.0 
        
        # 1. Die Basis-Kosten (Dein alter Ansatz)
        # Ziel ist 0, Maximum penalty ist -2
        distance_penalty = -(1 + cos_phi)
        
        # 2. Der "Extra-Bonus" (Glockenkurve)
        # Wir berechnen, wie weit wir von -1 entfernt sind.
        # Da cos_phi zwischen -1 und 1 liegt, ist (1 + cos_phi) unser "Abstand" zum Ziel.
        # Bei Zielerreichung ist dist_to_target = 0.
        dist_to_target = (1 + cos_phi)
        
        # Bonus-Einstellungen:
        # width: Wie "breit" ist der Bereich, in dem es Bonus gibt?
        # Ein kleinerer Wert (z.B. 0.1) bedeutet, man muss sehr präzise sein.
        bonus_width = 0.1 
        # amplitude: Wie viel extra Punkte gibt es im perfekten Zentrum?
        bonus_amplitude = 2.0 
        
        # Gauß-Funktion: Wird 2.0 bei perfektem Ziel, fällt sanft auf 0 ab, wenn man wegdriftet.
        extra_reward = bonus_amplitude * np.exp(- (dist_to_target**2) / (2 * bonus_width**2))
        
        reward = distance_penalty + extra_reward - 0.1 * phi_dot_norm**2 - 0.001 * action**2
        return float(reward)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()