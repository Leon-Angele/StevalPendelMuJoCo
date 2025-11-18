import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os

class PendelEnv(gym.Env):
    """
    Custom Environment für den Pendel-Roboter mit Schrittmotor-Physik.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=1500):
        super().__init__()

        # --- KONFIGURATION SCHRITTMOTOR ---
        self.MAX_SPEED = 5.0  # rad/s (Maximalgeschwindigkeit)
        self.MAX_ACC = 50   # rad/s² (Maximale Beschleunigung)
        self.dt = 0.01        # Simulationsschritt pro Action (10ms)
        
        # Interner Speicher für die aktuelle Motorgeschwindigkeit (für die Rampe)
        self.current_motor_speed = 0.0

        # --- MODEL LADEN ---
        # Pfad anpassen!
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "Pendel_description", "pendel_roboter.xml")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # IDs für schnellen Zugriff finden
        self.pendel_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendelGelenk")
        self.rotary_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rotaryGelenk")
        self.actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_rotary")

        # --- ACTION SPACE ---
        # Wir steuern die ZIEL-GESCHWINDIGKEIT (-1 bis +1, wird skaliert)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # --- OBSERVATION SPACE ---
        # [sin(theta), cos(theta), theta_dot]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Renderer Setup
        self.render_mode = render_mode
        self.viewer = None

        # Maximalanzahl an Schritten
        self.max_steps = max_steps
        self.current_step = 0

    def calc_reward(self,obs_next, action):

        sin_phi, cos_phi, phi_dot = obs_next[0], obs_next[1], obs_next[2]
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
        
        # 3. Gesamtreward
        # Wir behalten die Strafen für Geschwindigkeit und Energie bei.
        reward = distance_penalty + extra_reward - 0.1 * phi_dot_norm**2 - 0.001 * action**2
        
        return float(reward)

    def _get_obs(self):
        # 1. Winkel des Pendels holen (qpos)
        # Hinweis: qpos Adressen sind nicht immer gleich der Joint ID, 
        # aber bei einfachen Hinge-Joints meistens qpos[joint_id + offset]
        # Sicherer Weg über Address-Lookup:
        pendel_qpos_addr = self.model.jnt_qposadr[self.pendel_joint_id]
        pendel_qvel_addr = self.model.jnt_dofadr[self.pendel_joint_id]

        theta = self.data.qpos[pendel_qpos_addr]
        theta_dot = self.data.qvel[pendel_qvel_addr]

        return np.array([
            np.sin(theta),
            np.cos(theta),
            theta_dot
        ], dtype=np.float32)

    def step(self, action):
        # 1. Observation vor Action berechnen
        obs_before = self._get_obs()

        # 2. 1ms Physikschritt ohne Action
        #n_physics_steps = int(0.001 / self.model.opt.timestep)
        #for _ in range(n_physics_steps):
        #    mujoco.mj_step(self.model, self.data)

        # 3. Action wie gewohnt anwenden
        target_speed = float(action[0]) * self.MAX_SPEED
        speed_diff = target_speed - self.current_motor_speed
        max_change = self.MAX_ACC * self.dt
        actual_change = np.clip(speed_diff, -max_change, max_change)
        self.current_motor_speed += actual_change
        self.data.ctrl[self.actuator_id] = target_speed

        # 4. Simulation vorantreiben (wie gehabt)
        n_steps = int(self.dt / self.model.opt.timestep)
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        # 5. Observation holen
        obs = self._get_obs()
        reward = self.calc_reward(obs, action)

        terminated = False
        truncated = False
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        info = {"motor_speed": self.current_motor_speed}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Simulation zurücksetzen
        mujoco.mj_resetData(self.model, self.data)
        
        # Motor Rampe zurücksetzen
        self.current_motor_speed = 0.0

        # Schrittzähler zurücksetzen
        self.current_step = 0

        # Zufällige Startposition? (Optional, hilft beim Lernen)
        # self.data.qpos[self.model.jnt_qposadr[self.pendel_joint_id]] = np.random.uniform(-0.1, 0.1)

        # Ein erster Schritt, um alles zu initialisieren
        mujoco.mj_step(self.model, self.data)

        return self._get_obs(), {}

    def render(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()