import numpy as np
from src import plummer
from src import leapint

# G = 1
M_unit = 10e7 # M_solar = 1.99e41 kg
r_unit = 1.5 # Kpc
t_unit = 2.7327 # 10e8year

N = int(1e4)
X = plummer.rng_x(N) * r_unit
V = plummer.rng_v(X) * (r_unit / t_unit)
masses = np.ones(N) / N
steps = 150
dt = 1
epsilon = 0.05

# x_history, v_history = leapint.integrate(X, V, masses, steps, dt, epsilon)

# with open("plummer.tab", "w") as f:
#     for i in range(steps):
#         f.write(f"{N} {i*dt}\n")
#         for j in range(N):
#             f.write(
#                 (
#                     f"{masses[j]} {x_history[i, j, 0]:.6f} {x_history[i, j, 1]:.6f} {x_history[i, j, 2]:.6f} {v_history[i, j, 0]:.6f} {v_history[i, j, 1]:.6f} {v_history[i, j, 2]:.6f}\n"
#                 )
#             )


V_collapse = np.zeros_like(V)


x_collapse_history, v_collapse_history = leapint.integrate(X, V_collapse, masses, steps, dt, epsilon)

with open("plummer_collapse.tab", "w") as f:
    for i in range(steps):
        f.write(f"{N} {i*dt}\n")
        for j in range(N):
            f.write(
                (
                    f"{masses[j]} {x_collapse_history[i, j, 0]:.6f} {x_collapse_history[i, j, 1]:.6f} {x_collapse_history[i, j, 2]:.6f} {v_collapse_history[i, j, 0]:.6f} {v_collapse_history[i, j, 1]:.6f} {v_collapse_history[i, j, 2]:.6f}\n"
                )
            )