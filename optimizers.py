"""Observed-value-only optimizer adapters and transparent reference ports.

The jSO port implements the defining published features, with FE-based stage
schedules. It is not claimed bitwise-equivalent to the original C++ release.
The positive-weight CMA-ES implementation follows Hansen's tutorial; active
negative weights and noise-specific resampling are not silently enabled.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .noise import BudgetReached
from .vendor.isoma_ar import isoma_ar
from .vendor.lsrtde import LSRTDE_Optimizer
from .vendor.jso_lite import JSO as UploadedJSO

NAMES = ["jSO", "iSOMA-AR", "Powell", "CMA-ES", "L-SRTDE", "SPSA", "COBYLA"]


def lehmer(values, weights):
    v, w = np.asarray(values), np.asarray(weights)
    denominator = np.sum(w * v)
    return float(np.sum(w * v * v) / denominator) if denominator > 0 else 0.0


def run_jso(f, dim, budget, seed, options, events):
    rng = np.random.default_rng(seed)
    proposed = max(4, int(25 * np.sqrt(dim) * np.log(max(dim, 2))))
    npop = min(int(options.get("population", proposed)), budget)
    pop = rng.uniform(0, 1, (npop, dim))
    fit = np.array([f(x) for x in pop])
    evals, initial = len(pop), len(pop)
    memory_f, memory_cr = np.full(5, 0.3), np.full(5, 0.8)
    memory_f[-1] = memory_cr[-1] = 0.9
    archive, k = [], 0
    while evals < budget and len(pop) >= 4:
        progress = evals / budget
        size = max(4, int(round(initial + (4 - initial) * progress)))
        ids = np.argsort(fit)[:size]
        pop, fit = pop[ids], fit[ids]
        m = len(pop)
        count_best = min(m, max(2, int(round((0.25 - 0.125 * progress) * m))))
        best_ids = np.argsort(fit)[:count_best]
        joined = np.vstack((pop, archive)) if len(archive) else pop.copy()
        sf, scr, delta = [], [], []
        newpop, newfit = pop.copy(), fit.copy()
        for i in range(m):
            if evals >= budget:
                break
            r = int(rng.integers(5))
            F = memory_f[r] + 0.1 * rng.standard_cauchy()
            while F <= 0:
                F = memory_f[r] + 0.1 * rng.standard_cauchy()
            F = min(F, 1.0)
            CR = 0.0 if memory_cr[r] < 0 else float(np.clip(rng.normal(memory_cr[r], 0.1), 0, 1))
            if progress < 0.25:
                CR = max(CR, 0.7)
            elif progress < 0.5:
                CR = max(CR, 0.6)
            if progress < 0.6:
                F = min(F, 0.7)
            fw = (0.7 if progress < 0.2 else 0.8 if progress < 0.4 else 1.2) * F
            pbest = pop[int(rng.choice(best_ids))]
            r1 = int(rng.choice(np.delete(np.arange(m), i)))
            available = np.array([j for j in range(len(joined)) if j not in (i, r1)])
            r2 = int(rng.choice(available))
            mutant = pop[i] + fw * (pbest - pop[i]) + F * (pop[r1] - joined[r2])
            # L-SHADE parent-midpoint repair, explicitly documented.
            mutant = np.where(mutant < 0, pop[i] / 2, mutant)
            mutant = np.where(mutant > 1, (pop[i] + 1) / 2, mutant)
            mask = rng.random(dim) < CR
            mask[int(rng.integers(dim))] = True
            trial = np.where(mask, mutant, pop[i])
            value = f(trial)
            evals += 1
            accepted = bool(value <= fit[i])
            events.append(dict(kind="comparison", fe=evals, parent=pop[i].tolist(),
                               trial=trial.tolist(), parent_observed=float(fit[i]),
                               trial_observed=float(value), accepted=accepted))
            if accepted:
                newpop[i], newfit[i] = trial, value
                if value < fit[i]:
                    archive.append(pop[i].copy())
                    sf.append(F); scr.append(CR); delta.append(fit[i] - value)
        if sf:
            memory_f[k] = 0.5 * (memory_f[k] + lehmer(sf, delta))
            if max(scr) == 0:
                memory_cr[k] = -1.0
            elif memory_cr[k] < 0:
                memory_cr[k] = -1.0
            else:
                memory_cr[k] = 0.5 * (memory_cr[k] + lehmer(scr, delta))
            k = (k + 1) % 4
        cap = max(1, int(round(2.6 * m)))
        if len(archive) > cap:
            archive = [archive[j] for j in rng.choice(len(archive), cap, replace=False)]
        pop, fit = newpop, newfit
    i = int(np.argmin(fit))
    return dict(x=pop[i], value=float(fit[i]), initial_population=initial,
                port="published-feature port; FE schedules, archive factor 2.6, midpoint repair")


def run_cma(f, dim, budget, seed, options, events):
    del events
    rng = np.random.default_rng(seed)
    lam = max(2, int(options.get("population", 4 + int(3 * np.log(dim)))))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1 / np.sum(weights ** 2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damp = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    mean = rng.uniform(0, 1, dim)
    sigma = float(options.get("sigma0", 0.2))
    C, pc, ps = np.eye(dim), np.zeros(dim), np.zeros(dim)
    evals, generation, best = 0, 0, (float("inf"), mean.copy())
    while evals < budget:
        eig, Q = np.linalg.eigh((C + C.T) / 2)
        eig = np.maximum(eig, 1e-16)
        root = Q @ np.diag(np.sqrt(eig))
        inverse = Q @ np.diag(1 / np.sqrt(eig)) @ Q.T
        count = min(lam, budget - evals)
        # Reflection repair has no FE charge; its displacement is used in the
        # update. Box repair reduces exact affine invariance and is reported.
        candidates = mean + sigma * (rng.normal(size=(count, dim)) @ root.T)
        candidates = 1 - np.abs(1 - (candidates % 2))
        values = np.array([f(x) for x in candidates])
        evals += count
        i = int(np.argmin(values))
        if values[i] < best[0]:
            best = float(values[i]), candidates[i].copy()
        if count < lam:
            break
        selected = candidates[np.argsort(values)[:mu]]
        old = mean.copy()
        mean = weights @ selected
        y = (mean - old) / sigma
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inverse @ y)
        generation += 1
        hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * generation)) /
                     chi < 1.4 + 2 / (dim + 1))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y
        steps = (selected - old) / sigma
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) +
             (1 - hsig) * cc * (2 - cc) * C) + cmu * (steps.T * weights) @ steps)
        sigma *= np.exp((cs / damp) * (np.linalg.norm(ps) / chi - 1))
        sigma = float(np.clip(sigma, 1e-14, 2.0))
        if sigma < float(options.get("tolx", 1e-10)):
            break
    return dict(x=best[1], value=best[0], mean=mean.tolist(),
                initial_population=lam, port="positive-weight CMA-ES; reflection repair")


def run_spsa(f, dim, budget, seed, options, events):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, dim)
    alpha, gamma = 0.602, 0.101
    a, c = float(options.get("a", 0.08)), float(options.get("c", 0.08))
    iterations = max(0, (budget - 1) // 2)
    A = float(options.get("A", 0.1 * max(1, iterations)))
    for iteration in range(iterations):
        ak, ck = a / (iteration + 1 + A) ** alpha, c / (iteration + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], dim)
        plus, minus = (x + ck * delta) % 1, (x - ck * delta) % 1
        fp, fm = f(plus), f(minus)
        grad = (fp - fm) / (2 * ck) * delta
        events.append(dict(kind="spsa_pair", fe=2 * iteration + 2,
                           plus=plus.tolist(), minus=minus.tolist(), ck=ck,
                           difference=float(fp - fm)))
        x = (x - ak * grad) % 1
    value = f(x)  # Final recommendation is measured once and charged.
    return dict(x=x, value=value, iterations=iterations,
                port="unblocked SPSA; periodic updates; 2 FE/step plus final FE")


def run_local(name, f, dim, budget, seed, options, events):
    del events
    rng = np.random.default_rng(seed)
    if name == "Powell":
        opts = dict(maxfev=budget, xtol=options.get("xtol", 1e-5),
                    ftol=options.get("ftol", 1e-7))
    else:
        opts = dict(maxiter=budget, rhobeg=options.get("rhobeg", 0.2),
                    tol=options.get("tol", 1e-5), catol=1e-8)
    # COBYLA can evaluate an infeasible trust-region trial despite finite bounds.
    # The VQA is periodic, so evaluate its smooth periodic extension and archive
    # the corresponding feasible point. This is not a penalty or extra query.
    periodic = name == "COBYLA" and options.get("periodic_extension", True)
    objective = (lambda x: f(np.asarray(x) % 1.0)) if periodic else f
    result = minimize(objective, rng.uniform(0, 1, dim), method=name, bounds=[(0, 1)] * dim,
                      options=opts)
    return dict(x=result.x % 1 if periodic else result.x,
                value=float(result.fun), message=str(result.message))


def optimize(name, f, dim, budget, seed, options=None):
    """No truth/variance/landscape capability is accessible to these adapters."""
    options, events = options or {}, []
    calls, selected_x, selected_y = 0, None, float("inf")
    def observed_only(x):
        nonlocal calls, selected_x, selected_y
        if calls >= budget:
            raise BudgetReached("Adapter FE budget exhausted")
        y = f(x)
        calls += 1
        if y < selected_y:
            selected_y, selected_x = float(y), np.asarray(x).copy()
        return y
    status, result = "completed", {}
    try:
        if name.startswith("MS-"):
            base = name[3:]
            if base not in {"Powell", "COBYLA", "CMA-ES"}:
                raise ValueError("Unsupported multistart baseline.")
            starts = 0
            while calls < budget:
                remaining = budget - calls
                # Fixed per-start ceiling; restarts are charged, not free.
                chunk = min(remaining, int(options.get("restart_fes", 2000)))
                inner = optimize(base, observed_only, dim, chunk,
                                 int(rng_seed(seed, starts)), options)
                events.extend(inner.pop("events", []))
                if inner["nfev"] == 0:
                    break
                starts += 1
            result = dict(x=selected_x, value=selected_y, restarts=starts)
        elif name == "jSO":
            result = run_jso(observed_only, dim, budget, seed, options, events)
        elif name == "jSO-lite":
            raw = UploadedJSO(dim, (0, 1), budget, seed).run(observed_only)
            result = dict(x=raw["best_solution"], value=raw["best_fitness"], port="uploaded jSO-lite")
        elif name == "CMA-ES":
            result = run_cma(observed_only, dim, budget, seed, options, events)
        elif name == "SPSA":
            result = run_spsa(observed_only, dim, budget, seed, options, events)
        elif name in {"Powell", "COBYLA"}:
            result = run_local(name, observed_only, dim, budget, seed, options, events)
        elif name == "L-SRTDE":
            optimizer = LSRTDE_Optimizer(observed_only, dim, (0, 1), budget, seed)
            optimizer.pop_size_max = int(options.get("population", 20 * dim))
            x, value = optimizer.optimize()
            result = dict(x=x, value=value, initial_population=min(optimizer.pop_size_max, budget),
                          port="uploaded two-population port; initial population corrected to 20D")
        elif name == "iSOMA-AR":
            raw = isoma_ar(observed_only, np.zeros(dim), bounds=[(0, 1)] * dim,
                           max_evals=budget, seed=seed)
            result = dict(x=raw.x, value=float(raw.fun), initial_population=50,
                          port="uploaded iSOMA-AR; hard budget enforced by wrapper")
        elif name == "Crowding-DE":
            from .niching import run_crowding_de
            result = run_crowding_de(observed_only, dim, budget, seed, options)
        elif name == "Uniform":
            from .niching import run_uniform
            result = run_uniform(observed_only, dim, budget, seed)
        else:
            raise ValueError(f"Unknown optimizer {name}")
    except BudgetReached:
        status = "budget_cutoff"
        result = dict(x=selected_x, value=selected_y,
                      recommendation_source="common observed incumbent after hard cutoff")
    result.update(nfev=calls, status=status, events=events)
    if "x" in result and result["x"] is not None:
        result["x"] = np.asarray(result["x"], dtype=float).tolist()
    return result


def rng_seed(seed, start):
    from .models import stable_seed
    return stable_seed(seed, "restart", start)
