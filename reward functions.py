import numpy as np

def compute_rotary_pendulum_reward(
    self,
    state,                    # [theta, phi, theta_dot, phi_dot]
    action,                   # scalar oder 1D-array (u_t)
    q_theta=1.0,
    q_phi=10.0,               # Pendelwinkel ist am wichtigsten!
    q_theta_dot=0.1,
    q_phi_dot=0.1,
    r_control=0.001,
    phi_target=np.pi,         # oben = π, unten = 0
    theta_target=0.0,
    velocity_bonus_thresh=3.0 # rad/s, Bonus wenn langsam
):
    """
    Fortgeschrittene Reward-Funktion für Rotary Inverted Pendulum
    (Swing-Up + Balancing)  perfekt für TD3 / DDPG / SAC
    """
    sin_phi, cos_phi,phi_dot,theta,theta_dot= state
    phi = np.arctan2(sin_phi, cos_phi)

    u = action[0]

    # Optional: kleinsten Winkelabstand (sehr empfohlen!)
    phi   = ((phi   - phi_target   + np.pi) % (2*np.pi)) - np.pi + phi_target
    theta = ((theta - theta_target + np.pi) % (2*np.pi)) - np.pi + theta_target

    # 1) Binary Bonus: +1 wenn das System fast still steht
    if abs(theta_dot) < velocity_bonus_thresh and abs(phi_dot) < velocity_bonus_thresh:
        bonus = 1.0
    else:
        bonus = 0.0

    # 2) Quadratische Kosten
    cost = (q_theta      * (theta - theta_target)**2 +
            q_phi        * (phi   - phi_target)**2 +
            q_theta_dot  * theta_dot**2 +
            q_phi_dot    * phi_dot**2 +
            r_control    * u**2)

    # Reward = Bonus − Kosten  (maximieren!)
    reward = bonus - cost

    return reward