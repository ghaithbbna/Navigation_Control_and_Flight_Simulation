"""Quadratic-programming solvers for the MPC problem.
----------------------------------------
    minimise    J(U) = U' H U + 2 f' U
    subject to  M U <= g

H must be symmetric positive definite. Box limits lo <= U <= hi are written as
M = [I; -I], g = [hi; -lo].

Two solvers
-----------
solve_pqp   : the course's dual multiplicative-update solver (fast, approximate)
solve_exact : enumerates active sets (exact, only for a few constraints; test oracle)
"""

from dataclasses import dataclass
from itertools import combinations
import numpy as np

@dataclass(frozen=True)
class QPResult:
    U: np.ndarray            # primal solution
    lam: np.ndarray          # Lagrange multipliers (one per constraint)
    iterations: int          # passes of the update (0 for the exact solver)
    converged: bool          # did the stopping rule trigger before max_iter?
    max_violation: float     # largest value of (M U - g), floored at 0

def cost(H, f, U):
    """The cost J(U) = U'HU + 2 f'U."""
    return float(U @ H @ U + 2.0 * f @ U)

def _prepare(M, g, f):
    M = np.atleast_2d(np.asarray(M, dtype=float))
    g = np.atleast_1d(np.asarray(g, dtype=float)).ravel()
    f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
    if M.shape[0] != g.shape[0]:
        raise ValueError(f"M has {M.shape[0]} rows but g has {g.shape[0]} entries")
    if M.shape[1] != f.shape[0]:
        raise ValueError(f"M has {M.shape[1]} columns but f has {f.shape[0]} entries")
    return M, g, f

def solve_pqp(H_inv, f, M, g, tol=1e-9, max_iter=10000, callback=None):
    """Course PQP: multiplicative updates on the dual variables lambda >= 0."""
    H_inv = np.asarray(H_inv, dtype=float)
    M, g, f = _prepare(M, g, f)
    m = M.shape[0]
    
    # Dual problem: minimise 0.5 lam' W lam + Z' lam  over lam >= 0
    W = 0.5 * M @ H_inv @ M.T
    Z = M @ H_inv @ f + g
    
    # Split into non-negative parts: W = Wp - Wm, Z = Zp - Zm
    Wp, Wm = np.maximum(W, 0.0), np.maximum(-W, 0.0)
    Zp, Zm = np.maximum(Z, 0.0), np.maximum(-Z, 0.0)
    
    lam = np.ones(m)
    iterations = 0
    converged = False
    if callback is not None:
        callback(0, lam.copy())
    while iterations < max_iter:
        lam_old = lam.copy()
        for i in range(m):
            numerator = Zm[i] + Wm[i] @ lam
            denominator = Zp[i] + Wp[i] @ lam
            if lam[i] > 0.0 and denominator > 0.0:
                lam[i] *= numerator / denominator
        iterations += 1
        if callback is not None:
            callback(iterations, lam.copy())
        if np.sum((lam - lam_old) ** 2) <= tol:
            converged = True
            break
        
    U = -H_inv @ (f + 0.5 * M.T @ lam)
    violation = max(0.0, float(np.max(M @ U - g)))
    return QPResult(U=U, lam=lam, iterations=iterations,
                    converged=converged, max_violation=violation)        
    
def solve_exact(H, f, M, g, feas_tol=1e-9):
    """Exact solution by enumerating which constraints are active.

    Cost grows as 2**m, so use it only for small m (test oracle).
    """
    H = np.asarray(H, dtype=float)
    M, g, f = _prepare(M, g, f)
    m, n = M.shape
    best = None
    for size in range(m + 1):
        for S in combinations(range(m), size): #generates every subset of constraint indices of that size
            S = list(S)
            if S:
                K = np.block([[2.0 * H, M[S].T], #builds the matrix from four block
                              [M[S], np.zeros((size, size))]])
                rhs = np.concatenate([-2.0 * f, g[S]])
                try:
                    sol = np.linalg.solve(K, rhs) 
                except np.linalg.LinAlgError:
                    continue                      # dependent constraints: skip
                U, lam_S = sol[:n], sol[n:] #split the answer into U and the multipliers of the active limits
            else:
                U, lam_S = -np.linalg.solve(H, f), np.zeros(0)
            if np.all(M @ U <= g + feas_tol) and np.all(lam_S >= -feas_tol):
                J = cost(H, f, U)
                if best is None or J < best[0]:
                    lam = np.zeros(m)
                    lam[S] = lam_S
                    best = (J, U, lam)
    if best is None:
        raise ValueError("no feasible point found (constraints inconsistent?)")
    _, U, lam = best
    violation = max(0.0, float(np.max(M @ U - g)))
    return QPResult(U=U, lam=lam, iterations=0, converged=True, max_violation=violation)