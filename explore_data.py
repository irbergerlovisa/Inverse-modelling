import numpy as np
import matplotlib.pyplot as plt

# load all of the npy-files
x = np.load('coordinates.npy')
h_obs = np.load('surface.npy')
ux_obs = np.load('ux.npy')
uz_obs = np.load('uz.npy')

# check to see if all four arrays have the same length
# (they should all have the same length)
print(x.shape, h_obs.shape, ux_obs.shape, uz_obs.shape)
# prints (51,) (51,) (51,) (51,)
print(x.min(), x.max())
# prints 0.0 100000.0 (0 to L=1e5)

# check of the background slope (understand this part better)
# compare to angle = 0.2 in RunIcesheetSimulation.py
# -- is param angle known?, look 
tan_from_data = (h_obs[0] - h_obs[-1]) / (x[-1] - x[0])
angle_from_data = np.degrees(np.arctan(tan_from_data))
print(angle_from_data)
# prints 0.19966383824589268


# plot the surface and the velocities
# the surface should look like a straight slope with two smooths bumps
# the velocities should not be constant
# fig, axes = plt.subplots(3, 1, figsize=(7,8), sharex=True)
# axes[0].plot(x, h_obs); axes[0].set_ylabel('surface h (m)')
# axes[1].plot(x, ux_obs); axes[1].set_ylabel('ux (m/a)')
# axes[2].plot(x, uz_obs); axes[2].set_ylabel('uz (m/a)')
# axes[2].set_xlabel('x (m)')
# plt.tight_layout()
# plt.show()

# Try amplitude = [0,0] as a reference
# Before writing any fitting code, just run the forward model 
# once with no bumps at all and compare:
# import icemodel
# L = 1e5
# dx = x[1]-x[0]
# b_flat = icemodel.CreateBedRock(0.2, [0,0], x, L)
# dhdx, ux_flat, uz_flat = icemodel.VelocityAndSurfaceGradient(
#     b_flat, h_obs, dx, rho=9.138e-19, grav=9.7692e15, A=300, n=3.)

# plt.plot(x, ux_obs, label='observed')
# plt.plot(x, ux_flat, label='predicted with amp=[0,0]')
# plt.legend()
# plt.show()


from inverse_model import predict_velocity
x = np.load('coordinates.npy')
h_obs = np.load('surface.npy')
dx = x[1] - x[0]
ux0, uz0 = predict_velocity([0, 0], x, h_obs, dx)
#print(ux0, uz0)
# this should reproduce exactly what you already plotted in Step 1
ux1, uz1 = predict_velocity([0.8, -0.4], x, h_obs, dx)
print(ux1, uz1)

# small cost-function 
ux_obs = np.load('ux.npy')
diff = ux1 - ux_obs
print(np.max(np.abs(diff)))
print(np.sum(diff**2))

# cost-function
def cost(amplitudes, x, h_obs, dx, ux_obs):
    ux_pred, uz_pred = predict_velocity(amplitudes, x, h_obs, dx)
    return np.sum((ux_pred - ux_obs)**2)

# 
print(cost([0, 0], x, h_obs, dx, ux_obs))       # should be huge, from the spike
print(cost([0.8, -0.4], x, h_obs, dx, ux_obs))  # should match your 3.28e-10

# an inverse model which discovers the bedrock amplitudes ffrom the surface data
a1_vals = np.linspace(-2, 2, 41)
a2_vals = np.linspace(-2, 2, 41)
C = np.zeros((len(a1_vals), len(a2_vals)))

for i, a1 in enumerate(a1_vals):
    for j, a2 in enumerate(a2_vals):
        C[i, j] = cost([a1, a2], x, h_obs, dx, ux_obs)

best_i, best_j = np.unravel_index(np.argmin(C), C.shape)
print(a1_vals[best_i], a2_vals[best_j])

# real optimiser to compare:
from scipy.optimize import least_squares

def residuals(amplitudes, x, h_obs, dx, ux_obs):
    ux_pred, uz_pred = predict_velocity(amplitudes, x, h_obs, dx)
    return ux_pred - ux_obs

result = least_squares(residuals, x0=[0, 0], args=(x, h_obs, dx, ux_obs))
print(result.x)
# 0.8000000000000003 -0.3999999999999999 (grid search). vs.
# [ 0.80000078 -0.40000147] (true optimiser)


# plot which explains why you chose those amplitudes 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# plt.contourf(a2_vals, a1_vals, np.clip(C, 1e-10, None), levels=40, norm=LogNorm())
# plt.colorbar(label='cost (log scale)')
# plt.plot(a1_vals[best_j], a1_vals[best_i], 'r*')  # careful with i/j order, see below
# plt.xlabel('amp2')
# plt.ylabel('amp1')
# plt.show()

#import numpy as np

# Load the .npy file
#data = np.load('x.npy')

# View the contents, shape, and data type
print("Data:\n", x)
print("Shape:", x.shape)
print("Data Type:", x.dtype)
print("h_obs:", h_obs)
print("ux_obs:", ux_obs)
print("uz_obs:", uz_obs)