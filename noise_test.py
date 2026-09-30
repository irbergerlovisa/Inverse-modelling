import numpy as np
import matplotlib.pyplot as plt

from inverse_model import cost


# Load observed data
x = np.load("coordinates.npy")
h_obs = np.load("surface.npy")
ux_obs = np.load("ux.npy")

dx = x[1] - x[0]


# Values of amplitudes tested by the inverse model
a1_vals = np.linspace(-2, 2, 41)
a2_vals = np.linspace(-2, 2, 41)


# Noise levels that well test
noise_levels = [0.00, 0.01, 0.03, 0.05, 0.10]


# Use one random pattern and only change its size
np.random.seed(1)
random_pattern = np.random.uniform(-1, 1, size=ux_obs.shape)


# Store results
best_a1_values = []
best_a2_values = []
minimum_costs = []


for noise_level in noise_levels:
#h_noisy = h_obs * (
 #   1 + np.random.uniform(-noise_level, noise_level, size=h_obs.shape)
#) adding noise to the height surface changed the amplitudes drastically so i added noise only to the horizontal velocity

    # Add random noise to observed horizontal velocity
    ux_noisy = ux_obs * (1 + noise_level * random_pattern)

    # Matrix for costs
    C = np.zeros((len(a1_vals), len(a2_vals)))

    # Run the same inverse model
    for i, a1 in enumerate(a1_vals):
        for j, a2 in enumerate(a2_vals):

            C[i, j] = cost(
                [a1, a2],
                x,
                h_obs,
                dx,
                ux_noisy
            )

    # Find amplitudes with the smallest cost
    best_i, best_j = np.unravel_index(np.argmin(C), C.shape)

    best_a1 = a1_vals[best_i]
    best_a2 = a2_vals[best_j]
    minimum_cost = C[best_i, best_j]

    # Save results
    best_a1_values.append(best_a1)
    best_a2_values.append(best_a2)
    minimum_costs.append(minimum_cost)


# Printing results in a table
print("Noise (%)    Amplitude 1    Amplitude 2    Minimum cost")

for noise, a1, a2, min_cost in zip(
    noise_levels,
    best_a1_values,
    best_a2_values,
    minimum_costs
):
    print(
        f"{noise*100:6.1f}       "
        f"{a1:8.2f}       "
        f"{a2:8.2f}       "
        f"{min_cost:.6f}"
    )


# Convert noise levels from decimals to percentages
noise_percent = np.array(noise_levels) * 100

# Plot minimum cost
plt.figure(figsize=(8, 5))

plt.plot(
    noise_percent,
    minimum_costs,
    marker="o"
)

plt.xlabel("Noise level (%)")
plt.ylabel("Minimum cost")
plt.title("Effect of measurement noise on model fit")

plt.xticks(noise_percent)
plt.grid()

plt.show()