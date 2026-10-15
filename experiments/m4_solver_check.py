"""M4.1: the course's PQP against the exact bounded least squares (BVLS) on the C1 problem."""

import time

import numpy as np

from navigation_control_flight_simulation.mpc_incremental import IncrementalMPC
from navigation_control_flight_simulation.plant import course_plant
from navigation_control_flight_simulation.qp import solve_pqp

plant = course_plant()
c1 = IncrementalMPC(plant)
x, u_prev = np.zeros(plant.n), np.zeros(plant.m)
h, v = np.full(35, 100.0), np.full(35, -6.43)                # +100 ft climb with the energy speed target

t = time.perf_counter()
u_exact = c1.control(x, u_prev, h, v)
t_exact = time.perf_counter() - t

# the same problem in the course's form:  min dU'H dU + 2 f'dU,  M dU <= g
E = np.zeros((35, 5)); E[:, 0], E[:, 2] = h, 7 * 6.43
f = -c1.Gb_Q @ E.ravel()
L = np.kron(np.tril(np.ones((6, 6))), np.eye(2))
M = np.vstack([L, -L])
g = np.concatenate([np.tile(c1.u_max - u_prev, 6), -np.tile(c1.u_min - u_prev, 6)])
t = time.perf_counter()
r = solve_pqp(np.linalg.inv(c1.H), f, M, g)
t_pqp = time.perf_counter() - t

print(f"unknowns {c1.nb * 2}, limits {len(g)}, condition number of H {np.linalg.cond(c1.H):.1e}")
print(f"BVLS : {1000 * t_exact:7.1f} ms, first command {u_exact}")
print(f"PQP  : {1000 * t_pqp:7.1f} ms, {r.iterations} passes, converged {r.converged}, first command {r.U[:2] + u_prev}")
print(f"largest difference between the two first commands: {np.abs(r.U[:2] + u_prev - u_exact).max():.2e}")