import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
import time

class PendelEnv(gym.Env):
    """
    MuJoCo-Umgebung für ein Inverses Pendel (Rotary Inverted Pendulum).
    Simuliert spezifische Hardware-Eigenschaften wie Schrittmotor-Logik und Latenz.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()

        # --- 1. Parameter ---
        self.MAX_SPEED = 5.0  # Max. Drehgeschwindigkeit des Motors in rad/s
        self.dt = 0.005       # Zeitintervall pro Steuerungsschritt
        self.latency = 0.00  # Latenz
        self.max_steps = max_steps
        self.current_step = 0
        self.current_total_step = 0


        self.total_timesteps = 500_000

        # Stepper Motor Parameter
        self.accel_ramp = 100.0 
        self.effective_step = np.deg2rad(1.8) / 16 
        
        # --- 2. MuJoCo Setup ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "Pendel_description", "pendel_roboter.xml")

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
            low=np.array([-1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            shape=(5,),
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

        #"""
        # Domain Randomization
        mass_factor = self.np_random.uniform(0.95, 1.05)
        self.model.body_mass[self.pendel_body_id] = self.default_pendel_mass * mass_factor
        
        damping_factor = self.np_random.uniform(0.9, 1.1)
        self.model.dof_damping[self.pendel_qvel_adr] = self.default_pendel_damping * damping_factor

        self.MAX_SPEED = self.np_random.uniform(0.9, 1.1) * 5.0
        self.accel_ramp = self.np_random.uniform(0.9, 1.1) * 100.0
        
        #"""
        # Start State
        self.data.qpos[self.pendel_qpos_adr] = self.np_random.uniform(low=-0.2, high=0.2)
        self.data.qpos[self.rotary_qpos_adr] = 0.0
        
        mujoco.mj_step(self.model, self.data)

        self.target_position = 0
        self.current_velocity = 0.0
        self.last_ctrl_target = 0.0

        return self._get_obs(), {}

    def step(self, action):
        """Führt einen Simulationsschritt mit der gegebenen Aktion durch."""
        desired_velocity = float(action[0]) * self.MAX_SPEED

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
        
        # WICHTIG: Keine vorzeitige Terminierung (Endlos-Loop bis max_steps)
        terminated = False 

        obs = self._get_obs()
        reward = self.compute_rewards(obs, action)
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        self.current_total_step +=1

        info = {"rotary_angle": rotary_pos}

        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Gibt den aktuellen Beobachtungsvektor zurück."""
        theta_p = self.data.qpos[self.pendel_qpos_adr]
        vel_p = self.data.qvel[self.pendel_qvel_adr]/20 

        theta_r =self.data.qpos[self.rotary_qpos_adr]/6.3
        vel_r = self.data.qvel[self.rotary_qpos_adr]/20   

        self.noise_scale = min(1.0, self.current_total_step / self.total_timesteps)

        theta_p += self.np_random.normal(0, 0.001 * self.noise_scale)
        vel_p += self.np_random.normal(0, 0.05 * self.noise_scale)
        theta_r += self.np_random.normal(0, 0.001 * self.noise_scale)
        vel_r += self.np_random.normal(0, 0.05 * self.noise_scale)

        return np.array([
            np.sin(theta_p),
            np.cos(theta_p),
            vel_p,
            theta_r,
            vel_r
        ], dtype=np.float32)
    

    def compute_rewards(
        self,
        state,
        action,
        # Gewichte für die Nebenziele (Armposition, Energie, etc.)
        q_theta=1.0,       # Bestrafung für falsche Arm-Position
        q_theta_dot=0.1,   # Bestrafung für wilde Arm-Bewegungen
        q_phi_dot=0.5,     # Bestrafung für wilde Pendel-Bewegungen
        r_control=0.01,    # Bestrafung für hohen Energieverbrauch (u)
        
        # Parameter für die Gauß-Glocke
        sigma_phi=1.57079633,     
        
        phi_target=np.pi,
        theta_target=0.0
    ):

        sin_phi, cos_phi, phi_dot, theta, theta_dot = state
        phi = np.arctan2(sin_phi, cos_phi)
        u = action[0]

        # --- 1. Fehler berechnen ---
        phi_err_raw = phi - phi_target
        theta_err_raw = theta - theta_target

        # Normalisieren auf [-pi, pi]
        phi_err = ((phi_err_raw + np.pi) % (2 * np.pi)) - np.pi
        theta_err = ((theta_err_raw + np.pi) % (2 * np.pi)) - np.pi

        # --- 2. Das Hauptziel: Die Gauß-Glocke für das Pendel ---
        # Das Ergebnis ist ein Wert zwischen 0.0 (ganz unten) und 1.0 (perfekt oben).
        # Die Division durch (2 * sigma^2) steuert die Steilheit.
        reward_phi =1 * (np.exp(- (phi_err**2) / (2 * sigma_phi**2)))

        # --- 3. Die Kosten (Penalties) für Nebenziele ---
        # Wir ziehen diese von der Belohnung ab, um Energieeffizienz und
        # Stabilität zu erzwingen.
        costs = (q_theta      * (theta_err**2) +
                 q_theta_dot  * (theta_dot**2) +
                 q_phi_dot    * (phi_dot**2) +
                 r_control    * (u**2))

        # --- 4. Gesamt-Reward ---
        # Formel: "Tu das Richtige (Glocke)" minus "Mach nichts Dummes (Kosten)"
        reward = reward_phi - costs
        """         print(  f"Reward Components => "
                f"Reward_phi: {reward_phi:.3f}, "
                f"Costs: {costs:.3f}, "
                f"Total Reward: {reward:.3f}"
        ) """

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

if __name__ == '__main__':
    env = PendelEnv(render_mode="human")
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()