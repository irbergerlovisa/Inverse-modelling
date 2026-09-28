import numpy as np
import icemodel

# constants that describe the setup — these are known/given, not unknowns
angle = 0.2
L = 1.0e5
A = 300.0
rho = 9.138e-19
grav = 9.7692e15
n = 3.0

def predict_velocity(amplitudes, x, h_obs, dx):
    '''
    this function takes a guess of amplitudes and returns what the velocity would be if that guess were correct, using
    real observed surface of h_obs. It uses a 'candidate bedrock', never a real one.
    '''
    b_candidate = icemodel.CreateBedRock(angle, amplitudes, x, L)
    dhdx, ux_pred, uz_pred = icemodel.VelocityAndSurfaceGradient(
        b_candidate, h_obs, dx, rho, grav, A, n)
    return ux_pred, uz_pred

def cost(amplitudes, x, h_obs, dx, ux_obs):
    ux_pred, uz_pred = predict_velocity(amplitudes, x, h_obs, dx)
    return np.sum((ux_pred - ux_obs)**2)