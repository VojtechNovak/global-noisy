"""Compact iSOMA-AR implementation used in the study.

The algorithmic choices are intentionally fixed to the manuscript settings:
PopSize=50, N_jump=10, Step=0.3, m=10, n=5, k=15, adaptive PRT,
reverse-path first-improvement acceptance, the original iSOMA stagnation
restart, and the adaptive-rotation extension.

`x0` is accepted for SciPy/Qiskit interface compatibility, but (as in the
study iSOMA implementation) the initial population is sampled uniformly and
x0 is not injected into it.

Example
-------
# import numpy as np
# from scipy.optimize import minimize
#
# def sphere(x):
#     return np.sum(x*x)
#
# res = minimize(
#     sphere,
#     np.zeros(5),                      # used for dimension
#     method=isoma_ar,
#     bounds=[(-5.0, 5.0)] * 5,
#     options={"max_evals": 5000, "seed": 1234},
# )
# print(res.x, res.fun, res.nfev)
"""

import numpy as np
from scipy.optimize import OptimizeResult


def isoma_ar(
    fun,
    x0,
    args=(),
    jac=None,
    bounds=None,
    callback=None,
    *,
    max_evals=10_000,
    max_migrations=100_000,
    seed=None,
    **_,
):
    """SciPy-compatible iSOMA-AR minimizer."""
    del jac
    x0 = np.asarray(x0, dtype=float)
    dim = len(x0)

    if bounds is None or len(bounds) != dim:
        raise ValueError("Finite bounds are required for every variable.")

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        raise ValueError("iSOMA-AR requires finite box bounds.")

    # Study settings.
    PopSize, N_jump, Step = 50, 10, 0.3
    m, n, k = 10, 5, 15

    # Adaptive-rotation settings.
    eta_base, eta_cap = 0.12, 0.35
    eig_every, ridge = 5, 1e-4
    rho_gate, p_rot_max = 0.18, 0.45
    min_successes = max(6, int(np.ceil(1.5 * dim)))
    eps = 1e-12

    # RandomState reproduces the np.random-style generator used by the
    # original study implementation when an integer seed is supplied.
    rng = np.random.RandomState(seed) if seed is not None else np.random

    def f(x):
        return float(fun(np.asarray(x), *args))

    # Original iSOMA initialization: fully uniform random population.
    pop = lb[:, None] + rng.rand(dim, PopSize) * (ub - lb)[:, None]
    fitness = np.array([f(pop[:, i]) for i in range(PopSize)])
    FEs = PopSize

    best_idx = np.argmin(fitness)
    best_cost = float(fitness[best_idx])
    best_x = pop[:, best_idx].copy()
    best_cost_old = best_cost

    # Adaptive-rotation state.
    C = np.eye(dim)
    Q = np.eye(dim)
    n_success = 0

    Migration = 0
    Count = 0

    def rho_value():
        offdiag = C - np.diag(np.diag(C))
        return np.linalg.norm(offdiag, "fro") / (np.linalg.norm(C, "fro") + eps)

    while FEs < max_evals and Migration < max_migrations:
        Migration += 1

        # Same migrant selection as the supplied iSOMA.
        M = rng.choice(range(PopSize), m, replace=False)
        M_sort = np.argsort(fitness[M])

        for j in range(n):
            migrant_idx = M[M_sort[j]]
            Migrant = pop[:, migrant_idx].copy()

            # Same leader selection as the supplied iSOMA:
            # sample k from the whole population, then use second-best if self.
            K = rng.choice(range(PopSize), k, replace=False)
            K_sort = np.argsort(fitness[K])
            leader_idx = K[K_sort[0]]
            if migrant_idx == leader_idx:
                leader_idx = K[K_sort[1]]
            Leader = pop[:, leader_idx].copy()

            flag = 0
            move = 1

            while flag == 0 and move <= N_jump:
                nstep = (N_jump - move + 1) * Step
                PRT = 0.1 + 0.9 * (FEs / max_evals)
                mask = (rng.rand(dim) < PRT).astype(float)
                delta = Leader - Migrant

                # Rotation gate from the manuscript.
                use_rotation = False
                if n_success >= min_successes:
                    rho = rho_value()
                    if rho > rho_gate:
                        p_rot = min(
                            p_rot_max,
                            p_rot_max * (rho - rho_gate) / (1.0 - rho_gate),
                        )
                        use_rotation = rng.rand() < p_rot

                if use_rotation:
                    direction = Q @ (mask * (Q.T @ delta))
                else:
                    direction = mask * delta

                offspring = np.clip(Migrant + nstep * direction, lb, ub)
                new_cost = f(offspring)
                FEs += 1

                # First non-worsening proposal is accepted.
                old_cost = float(fitness[migrant_idx])
                if new_cost <= old_cost:
                    flag = 1
                    fitness[migrant_idx] = new_cost
                    pop[:, migrant_idx] = offspring

                    # AR learns only from strictly improving accepted steps.
                    if new_cost < old_cost:
                        s = offspring - Migrant
                        s_norm = np.linalg.norm(s)
                        if s_norm > eps:
                            u = s / s_norm
                            gain = (old_cost - new_cost) / (abs(old_cost) + eps)
                            eta = min(eta_cap, eta_base * (1.0 + min(2.0, gain)))
                            C = (1.0 - eta) * C + eta * np.outer(u, u)
                            n_success += 1

                            if n_success % eig_every == 0:
                                _, Q = np.linalg.eigh(C + ridge * np.eye(dim))

                    # Same global-best rule as the supplied iSOMA (<=).
                    if new_cost <= best_cost:
                        best_cost = float(new_cost)
                        best_x = offspring.copy()
                        Count = 0
                        if callback is not None:
                            callback(best_x)

                move += 1

            if flag == 0:
                Count += 1

        # Same stagnation logic as the supplied study iSOMA.
        if Count > PopSize * 50:
            if best_cost >= best_cost_old:
                rat = round(0.1 * PopSize)
                pop_temp = lb[:, None] + rng.rand(dim, rat) * (ub - lb)[:, None]
                fit_temp = np.array([f(pop_temp[:, i]) for i in range(rat)])
                FEs += rat

                # Original implementation replaces random population members.
                D = rng.choice(range(PopSize), rat, replace=False)
                pop[:, D] = pop_temp
                fitness[D] = fit_temp

                # AR state resets with the same stagnation restart.
                C = np.eye(dim)
                Q = np.eye(dim)
                n_success = 0

            best_cost_old = best_cost
            Count = 0

    return OptimizeResult(
        x=best_x,
        fun=best_cost,
        nfev=FEs,
        njev=0,
        nit=Migration,
        success=True,
        message="Optimization finished",
    )


# Convenient direct-use alias.
minimize = isoma_ar
