import numpy as np
import icemodel
import matplotlib.pyplot as plt

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

# load all of the npy-files
x = np.load('coordinates.npy')
h_obs = np.load('surface.npy')
ux_obs = np.load('ux.npy')
uz_obs = np.load('uz.npy')

dx = x[1] - x[0]
#ux0, uz0 = predict_velocity([0, 0], x, h_obs, dx)
ux1, uz1 = predict_velocity([0.8, -0.4], x, h_obs, dx)
print(ux1, uz1)

# small cost-function 
ux_obs = np.load('ux.npy')
diff = ux1 - ux_obs
print(np.max(np.abs(diff)))
print(np.sum(diff**2))


# an inverse model which discovers the bedrock amplitudes ffrom the surface data
a1_vals = np.linspace(-2, 2, 41)
a2_vals = np.linspace(-2, 2, 41)
C = np.zeros((len(a1_vals), len(a2_vals)))

for i, a1 in enumerate(a1_vals):
    for j, a2 in enumerate(a2_vals):
        C[i, j] = cost([a1, a2], x, h_obs, dx, ux_obs)

best_i, best_j = np.unravel_index(np.argmin(C), C.shape)
print(a1_vals[best_i], a2_vals[best_j])

