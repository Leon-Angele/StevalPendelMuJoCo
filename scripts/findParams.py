import numpy as np
from scipy.optimize import minimize
import mujoco

# 1. Deine echten Daten (Beispiel)
# real_times = [0.0, 0.01, ...]
# real_angles = [1.57, 1.55, ...] 

def sim_trajectory(params):
    damp, frict = params
    
    # Werte im Model setzen
    model.dof_damping[pendel_adr] = damp
    model.dof_frictionloss[pendel_adr] = frict
    
    # Reset auf exakt den Startwinkel deiner echten Daten
    mujoco.mj_resetData(model, data)
    data.qpos[pendel_adr] = real_angles[0] 
    
    sim_angles = []
    for _ in real_times:
        mujoco.mj_step(model, data)
        sim_angles.append(data.qpos[pendel_adr])
        
    return np.array(sim_angles)

def cost_function(params):
    sim_angles = sim_trajectory(params)
    # Fehlerquadratsumme (MSE)
    return np.sum((sim_angles - real_angles)**2)

# Optimierer finden lassen
res = minimize(cost_function, x0=[0.0002, 0.0005], bounds=((0, 0.01), (0, 0.01)))
print(f"Gefundene Dämpfung: {res.x[0]}, Gefundene Reibung: {res.x[1]}")