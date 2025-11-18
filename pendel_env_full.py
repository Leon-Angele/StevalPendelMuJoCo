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

        self.MAX_SPEED = 10.0  # Max Geschwindigkeit (rad/s)
        self.dt = 0.01  # Simulationsschritt
        self.max_steps = max_steps
        self.current_step = 0

        # --- Schrittmotor-Parameter basierend auf STEVAL-EDUKIT01 (NEMA17) ---
        self.step_angle = np.deg2rad(1.8)  # Standard Step-Winkel für NEMA17
        self.holding_torque_max = 0.25  # Geschätzter max Holding Torque (Nm) für 0.8A NEMA17
        self.accel_ramp = 20.0  # Beschleunigungsrampe (rad/s²), angepasst für Realismus
        self.microsteps = 16  # 1/16 Microstepping
        self.rated_current = 0.8  # Nennstrom (A) aus Kit-Specs (nicht direkt verwendet, aber für Skalierung)

        # Effektive Schrittgröße
        self.effective_step = self.step_angle / self.microsteps

        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "Pendel_description", "pendel_roboter.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # IDs holen
        self.pendel_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendelGelenk")
        self.rotary_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rotaryGelenk")
        self.actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_rotary")

        self.pendel_qpos_adr = self.model.jnt_qposadr[self.pendel_joint_id]
        self.pendel_qvel_adr = self.model.jnt_dofadr[self.pendel_joint_id]
        self.rotary_qpos_adr = self.model.jnt_qposadr[self.rotary_joint_id]
        self.rotary_qvel_adr = self.model.jnt_dofadr[self.rotary_joint_id]

        # Action Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space
        low = np.array([-1, -1, -np.inf, -1, -1, -np.inf], dtype=np.float32)
        high = np.array([1, 1, np.inf, 1, 1, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None

        # --- Zustandsvariablen für Schrittmotor ---
        self.target_position = 0.0  # Gewünschte Position (rad)
        self.current_velocity = 0.0  # Aktuelle gerampte Geschwindigkeit

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Pendel leicht zufällig starten
        self.data.qpos[self.pendel_qpos_adr] = self.np_random.uniform(low=-0.4, high=0.4)
        self.data.qpos[self.rotary_qpos_adr] = 0
        mujoco.mj_step(self.model, self.data)

        # Schrittmotor resetten
        self.target_position = 0.0
        self.current_velocity = 0.0

        return self._get_obs(), {}

    def step(self, action):
        # Action ist normalisierte Geschwindigkeit (-1 bis 1)
        desired_velocity = float(action[0]) * self.MAX_SPEED

        # Beschleunigungsrampe anwenden
        velocity_diff = desired_velocity - self.current_velocity
        max_accel_step = self.accel_ramp * self.dt
        self.current_velocity += np.clip(velocity_diff, -max_accel_step, max_accel_step)

        # Integriere zu Position
        position_increment = self.current_velocity * self.dt
        self.target_position += position_increment

        # Diskrete Schritte quantisieren
        quantized_target = round(self.target_position / self.effective_step) * self.effective_step

        # Setze Aktuator (Position-Actuator)
        self.data.ctrl[self.actuator_id] = quantized_target

        # MuJoCo-Schritte simulieren
        n_steps = int(self.dt / self.model.opt.timestep)
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        # Schrittverlust (Slipping) simulieren
        current_torque = self.data.qfrc_actuator[self.rotary_qvel_adr]
        # Sinusförmiges Holding Torque (typisch für Stepper, skaliert mit Current)
        effective_holding_torque = self.holding_torque_max * np.sin((self.data.qpos[self.rotary_qpos_adr] / self.step_angle) * np.pi)

        if abs(current_torque) > effective_holding_torque:
            slip_amount = np.sign(current_torque) * self.effective_step * self.np_random.uniform(0.5, 2.0)
            self.data.qpos[self.rotary_qpos_adr] += slip_amount
            self.target_position += slip_amount

        obs = self._get_obs()
        reward = self.compute_rewards(obs, action)
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {"target_speed": desired_velocity, "slipped": abs(current_torque) > effective_holding_torque}

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        theta_p = self.data.qpos[self.pendel_qpos_adr]
        vel_p = self.data.qvel[self.pendel_qvel_adr]
        theta_r = self.data.qpos[self.rotary_qpos_adr]
        vel_r = self.data.qvel[self.rotary_qvel_adr]
        return np.array([
            np.sin(theta_p), np.cos(theta_p), vel_p,
            np.sin(theta_r), np.cos(theta_r), vel_r
        ], dtype=np.float32)

    def compute_rewards_old(self, obs, action, slipped=False):
        sin_phi, cos_phi, phi_dot = obs[0], obs[1], obs[2]
        phi_dot_norm = phi_dot / 36
        distance_penalty = -(1 + cos_phi)
        dist_to_target = (1 + cos_phi)
        bonus_width = 0.1
        bonus_amplitude = 2.0
        extra_reward = bonus_amplitude * np.exp(- (dist_to_target**2) / (2 * bonus_width**2))
        reward = distance_penalty + extra_reward - 0.1 * phi_dot_norm**2 - 0.001 * action**2
        #reward -= 0.5 if slipped else 0.0  # Oder -1.0 für stärkeren Anreiz
        return float(reward)
    
    def compute_rewards(self, obs, action):
        sin_phi, cos_phi, phi_dot = obs[0], obs[1], obs[2]
        phi_dot_norm = phi_dot / 36  # Bleibt
        
        # Basis: -2 unten (cos=1), 0 oben (cos=-1) – korrekt
        distance_penalty = -(1 + cos_phi)
        
        # Gauss-Bonus: Stark für perfektes Oben (cos≈-1)
        dist_to_target = (1 + cos_phi)  # 2 unten, 0 oben
        bonus_width = 0.05
        bonus_amplitude = 5.0
        extra_reward = bonus_amplitude * np.exp(- (dist_to_target**2) / (2 * bonus_width**2))
        
        # Korrigierte Velocity-Strafe: Mild unten (für Swing-Up-Schwingen), hart oben (gegen Oszillationen)
        if cos_phi > -0.5:  # Unten/Übergang (cos > -0.5, also weit von oben) – toleriere Velocity
            velocity_penalty = -0.01 * phi_dot_norm**2  # Sehr mild, erlaubt Schwingen
        else:  # Nah oben (cos < -0.5) – straf Oszillationen
            velocity_penalty = -0.3 * phi_dot_norm**2  # Stärker für Stabilität
        
        # Action-Strafe: Mild
        action_penalty = -0.005 * action**2
        
        # Slipping-Strafe: Mild, um Realismus zu halten
        current_torque = self.data.qfrc_actuator[self.rotary_qvel_adr]  # Oder Torque-Sensor
        slipped = abs(current_torque) > self.holding_torque_max * 0.9
        slip_penalty = -0.5 if slipped else 0.0
        
        reward = distance_penalty + extra_reward + velocity_penalty + action_penalty + slip_penalty
        return float(reward)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()