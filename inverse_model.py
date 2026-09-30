import numpy as np
import icemodel
import matplotlib.pyplot as plt

# loading all of the npy-files
ux_obs = np.load('ux.npy')
uz_obs = np.load('uz.npy')
x = np.load('coordinates.npy')
h_obs = np.load('surface.npy')

# constants of the setup
angle = 0.2
L = 1.0e5
A = 300.0
rho = 9.138e-19
grav = 9.7692e15
n = 3.0

def predict_velocity(amplitudes, x, h_obs, dx):
    '''
    this function takes a guess of amplitudes and returns what the velocity would be if that guess were correct, using
    real observed surface of h_obs. It uses a 'candidate bedrock', not a real one.
    '''
    b_candidate = icemodel.CreateBedRock(angle, amplitudes, x, L)
    dhdx, ux_pred, uz_pred = icemodel.VelocityAndSurfaceGradient(
        b_candidate, h_obs, dx, rho, grav, A, n)
    return ux_pred, uz_pred

dx = x[1] - x[0]


def cost(amplitudes, x, h_obs, dx, ux_obs):
    '''compares the predicted velocity of a certain amplitude (from the predict_velocity func) with the real observed
    veloocity (loaded from the ux.npy file). This is compared for the horizontal compenent of the ice surface velocity. '''
    ux_pred, uz_pred = predict_velocity(amplitudes, x, h_obs, dx)
    return np.sum((ux_pred - ux_obs)**2)

# an inverse model which searches for the bedrock amplitudes ffrom the surface data
# linspace between [-2,2] with stepsize 0.1: 41 evenly spaced steps for each amplitude
# 41 x 41 = 1681 combinations in total to search in
# intervall of [-2,2] seems reasonable given estimated amplitudes in the assignment, and the solution was confirmed
# by using a normal optimiser 

a1_vals = np.linspace(-2, 2, 41)
a2_vals = np.linspace(-2, 2, 41)
C = np.zeros((len(a1_vals), len(a2_vals))) 

for i, a1 in enumerate(a1_vals):
    for j, a2 in enumerate(a2_vals):
        '''
        searches each amplitude combination [a1, a2] to determine where the cost of the corresponding
        ux velocity is the smallest. C[i,j] overwrites to the corresponding value of the error function.
        '''
        C[i, j] = cost([a1, a2], x, h_obs, dx, ux_obs)

best_i, best_j = np.unravel_index(np.argmin(C), C.shape)
best_a1, best_a2 = round(a1_vals[best_i], 1), round(a2_vals[best_j], 1)
# Retrievs the amplitudes from where the error was the smallest  


print('The first amplitude is estimated to:', best_a1)
print('The second amplitude is estimated to:', best_a2)

# Cost function plotted in logarithmic scale:
plt.figure(figsize=(8, 6))
C_log = np.log10(C + 1e-12) 
plt.contourf(a2_vals, a1_vals, C_log, levels=40)
plt.colorbar(label="log10(cost)")

plt.plot(
    best_a2,
    best_a1,
    "r*",
    markersize=15,
    label="Minimum cost"
)

plt.xlabel("Amplitude 2")
plt.ylabel("Amplitude 1")
plt.title("Cost for different bedrock amplitudes")
plt.legend()
plt.show()