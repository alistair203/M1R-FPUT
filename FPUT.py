import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.fft import fft

# Parameters
N = 32                                                # Default 32
alpha = 1.00                                          # Default 0.25
TMAX = 100000                                          # Default 10 000
DT = 20                                               # Default 20 
t_eval = np.arange(0, TMAX, DT)

# Initial conditions: excite first mode
A = 1.0                                               # Default 1.0
q0 = A * np.sin(np.pi * np.arange(1, N+1) / (N + 1))  # initial positions
p0 = np.zeros(N)                                      # initial momenta
y0 = np.concatenate([q0, p0])                         # state vector: [q, p]

# FPUT-α differential equation
def fpu_alpha(t, y):
    q = y[:N]
    p = y[N:]
    dq_dt = p

    dp_dt = np.zeros(N)
    for i in range(1, N - 1):
        dp_dt[i] = (
            q[i + 1] + q[i - 1] - 2 * q[i] +
            alpha * ((q[i + 1] - q[i])**2 - (q[i] - q[i - 1])**2)
        )
    dp_dt[0] = q[1] - 2 * q[0] + alpha * ((q[1] - q[0])**2 - q[0]**2)
    dp_dt[-1] = q[-2] - 2 * q[-1] + alpha * (q[-1]**2 - (q[-1] - q[-2])**2)

    return np.concatenate([dq_dt, dp_dt])

# Solve the system
sol = solve_ivp(fpu_alpha, [0, TMAX], y0, t_eval=t_eval, method='RK45', rtol=1e-4)
Q = sol.y[:N, :]    # positions over time
P = sol.y[N:, :]    # momenta over time

def compute_mode_energies(Q, P, N):
    energies = []
    for i in range(Q.shape[1]):
        q_ext = np.concatenate(([0], Q[:, i], [0], -Q[:, i][::-1]))
        p_ext = np.concatenate(([0], P[:, i], [0], -P[:, i][::-1]))

        qf = np.imag(fft(q_ext)) / np.sqrt(2 * (N + 1))
        pf = np.imag(fft(p_ext)) / np.sqrt(2 * (N + 1))

        omega_k2 = 4 * (np.sin(np.pi * np.arange(1, N + 1) / (2 * (N + 1))))**2
        e_k = 0.5 * (omega_k2 * qf[1:N + 1]**2 + pf[1:N + 1]**2)
        energies.append(e_k)

    return np.array(energies)

# Compute mode energies for plotting
E = compute_mode_energies(Q, P, N)
omega1 = 4 * (np.sin(np.pi / (2 * (N + 1))))**2
TIME = t_eval * np.sqrt(omega1) / (2 * np.pi)

# Plot energies of first few nodes over time
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(TIME, E[:, i]*100, label=f'Mode {i+1}')
plt.xlabel("Time (cycles of mode 1)", fontsize = 20)
plt.ylabel("Energy (arbritary units)", fontsize = 20)
plt.tick_params(axis='both', labelsize=16)
plt.legend(fontsize = 14)
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/experiment2big.png", dpi=600)
plt.show()

# Plot mode energies at specific times
plot_times = [0, 50000, 100000]  # simulation times to sample
time_indices = [np.argmin(np.abs(t_eval - t)) for t in plot_times]  # corresponding indices

plt.figure(figsize=(10, 6))
for idx, t_idx in enumerate(time_indices):
    plt.scatter(
        range(1, N + 1),
        E[t_idx] * 100,
        label=f"t (cycles of mode 1) = {round(t_eval[t_idx] * np.sqrt(omega1) / (2 * np.pi))}", 
    )
plt.xlabel("Mode number", fontsize=20)
plt.ylabel("Energy (arbritary units)", fontsize=20)
plt.tick_params(axis='y', labelsize=16)
plt.xticks(ticks=range(1, 33))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.grid(False)
plt.savefig("graphs/experiment2thermalisationenergies.png", dpi=600)
plt.show()
