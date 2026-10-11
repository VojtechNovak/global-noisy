"""Prospective supplementary niche archives and assessment-only catalogues.

Only coordinates and observed fitness maintain the archive. Independent exact
catalogues, fidelities, Hessians and diagnostic quenches are used after search.
Locating a minimum and accessing its basin are deliberately separate outcomes.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize
from .characterize import hessian_summary


def torus_distances(x, points):
    delta = (np.asarray(points) - np.asarray(x) + 0.5) % 1.0 - 0.5
    return np.sqrt(np.mean(delta * delta, axis=-1))


class ObservedArchive:
    def __init__(self, capacity=64, radius=0.08):
        if capacity < 1 or not 0 < radius <= 0.5:
            raise ValueError("Invalid archive capacity/radius")
        self.capacity, self.radius = int(capacity), float(radius)
        self.x, self.y, self.indices = [], [], []

    def add(self, x, observed, index):
        x = np.asarray(x, dtype=float) % 1
        if self.x:
            distances = torus_distances(x, self.x)
            near = np.flatnonzero(distances <= self.radius)
            if len(near):
                # Deterministic nearest member; no exact energy lookup.
                j = int(near[np.argmin(distances[near])])
                if observed < self.y[j]:
                    self.x[j], self.y[j], self.indices[j] = x.copy(), float(observed), int(index)
                return
        if len(self.x) < self.capacity:
            self.x.append(x.copy()); self.y.append(float(observed)); self.indices.append(int(index))
        else:
            j = int(np.argmax(self.y))
            if observed < self.y[j]:
                self.x[j], self.y[j], self.indices[j] = x.copy(), float(observed), int(index)

    def snapshot(self):
        order = np.argsort(self.y, kind="stable")
        return [(int(self.indices[j]), np.array(self.x[j]), float(self.y[j])) for j in order]


def run_crowding_de(f, dim, budget, seed, options):
    """Canonical crowding principle; transparent torus DE/rand/1/bin variant.

    Trial competes with the closest population member, rather than its source
    target. This is an inspired implementation, not an exact Thomsen code port.
    """
    rng = np.random.default_rng(seed)
    size = min(budget, max(4, int(options.get("population", 10 * dim))))
    X = rng.uniform(0, 1, (size, dim)); Y = np.array([f(x) for x in X])
    used = size
    F, CR = float(options.get("F", 0.7)), float(options.get("CR", 0.9))
    while used < budget:
        if size < 4:
            x = rng.uniform(0, 1, dim)
        else:
            target = (used - size) % size
            others = np.delete(np.arange(size), target)
            a, b, c = rng.choice(others, 3, replace=False)
            displacement = (X[b] - X[c] + 0.5) % 1 - 0.5
            mutant = (X[a] + F * displacement) % 1
            mask = rng.random(dim) < CR; mask[rng.integers(dim)] = True
            x = np.where(mask, mutant, X[target])
        y = f(x); used += 1
        nearest = int(np.argmin(torus_distances(x, X)))
        if y <= Y[nearest]:
            X[nearest], Y[nearest] = x, y
    j = int(np.argmin(Y))
    return dict(x=X[j], value=float(Y[j]), initial_population=size,
                port="DE/rand/1/bin with nearest-member crowding on the unit torus; F=.7, CR=.9")


def run_uniform(f, dim, budget, seed):
    rng = np.random.default_rng(seed); best_x, best_y = None, float("inf")
    for _ in range(budget):
        x = rng.uniform(0, 1, dim); y = f(x)
        if y < best_y: best_x, best_y = x, float(y)
    return dict(x=best_x, value=best_y, port="independent uniform-query coverage control")


class Catalogue:
    def __init__(self, problem, landscape, settings):
        self.problem, self.settings = problem, settings
        self.entries, self.states = [], []
        frozen = landscape.get("niching_catalogue")
        if frozen is not None:
            import json
            if frozen["problem"] != json.loads(json.dumps(problem.metadata())) or frozen["settings"] != settings:
                raise ValueError("Frozen niche catalogue metadata/settings mismatch")
            self.entries = frozen["entries"]
            self.certified, self.description = frozen["certified"], frozen["description"]
            if problem.ansatz == "qaoa": self.states = [problem.state(e["x"]) for e in self.entries]
            return
        grad_tol = settings.get("catalogue_gradient_tolerance", 1e-5)
        if problem.ansatz == "product":
            ids = np.arange(len(problem.h.normalized))
            gaps = np.array([problem.h.normalized[ids ^ (1 << q)] - problem.h.normalized
                             for q in range(problem.n)])
            for i in np.flatnonzero(np.all(gaps > 1e-12, axis=0)):
                v = problem.h.bits[i] / 2 + 0.5
                x = np.linalg.solve(problem.U.astype(float), v) % 1
                self.entries.append(dict(index=int(i), x=x.tolist(), energy=float(problem.h.normalized[i])))
            self.certified = True
            self.description = "all strict computational-basis vertex minima; physical-state equivalence"
        else:
            # Freeze before benchmark: only independently characterized starts.
            for endpoint in landscape["quench_endpoints"]:
                if not endpoint.get("stationary") or endpoint["grad_norm"] > grad_tol: continue
                x = np.asarray(endpoint["x"])
                H = hessian_summary(problem.hessian(x))
                if H["negative"] or not H["active_positive"]: continue
                energy = float(problem.objective(x)); state = problem.state(x)
                if any(abs(energy - e["energy"]) <= 1e-7 and
                       abs(np.vdot(s, state)) ** 2 >= 1 - 1e-6
                       for e, s in zip(self.entries, self.states)): continue
                self.entries.append(dict(index=len(self.entries), x=x.tolist(), energy=energy,
                                         grad_norm=float(np.linalg.norm(problem.gradient(x))), hessian=H))
                self.states.append(state)
            self.certified = False
            self.description = "independent stationarity/Hessian-screened empirical QAOA catalogue; incomplete"

    def match(self, x, energy=None, fidelity_tol=None, energy_tol=None):
        if not self.entries: return None
        f_tol = self.settings.get("fidelity_tolerance", 0.01) if fidelity_tol is None else fidelity_tol
        e_tol = self.settings.get("energy_tolerance", 0.005) if energy_tol is None else energy_tol
        energy = self.problem.objective(x) if energy is None else energy
        if self.problem.ansatz == "product":
            angles = self.problem.angles(x)
            p1 = np.sin(angles / 2) ** 2
            bits = np.array([self.problem.h.bits[e["index"]] for e in self.entries])
            fidelities = np.prod(np.where(bits, p1, 1 - p1), axis=1)
        else:
            psi = self.problem.state(x)
            fidelities = np.abs(np.asarray(self.states).conj() @ psi) ** 2
        gaps = np.abs(energy - np.array([e["energy"] for e in self.entries]))
        possible = np.flatnonzero((fidelities >= 1 - f_tol) & (gaps <= e_tol))
        return int(possible[np.argmax(fidelities[possible])]) if len(possible) else None

    def metadata(self):
        return dict(entries=self.entries, size=len(self.entries), certified=self.certified,
                    description=self.description, problem=self.problem.metadata(), settings=self.settings)


_catalogues = {}


def assess_trace(problem, arrays, landscape, settings, seed, search_fe_ceiling=None):
    start = time.perf_counter()
    key = (problem.problem_id, repr(settings), repr(landscape["reference"]))
    if key not in _catalogues:
        # Keep cache bounded in workers traversing several disorder sizes.
        if len(_catalogues) >= 8: _catalogues.pop(next(iter(_catalogues)))
        _catalogues[key] = Catalogue(problem, landscape, settings)
    catalogue = _catalogues[key]
    archive = ObservedArchive(settings.get("archive_capacity", 64), settings.get("archive_radius", 0.08))
    # Scheduled FE checkpoints are the same for early-stopping methods; the
    # final archive is carried forward, without free search restarts.
    N = len(arrays["truth"])
    horizon = search_fe_ceiling or N
    scheduled = sorted(set(max(1, int(horizon * f)) for f in settings.get("checkpoint_fractions", [1.])))
    checkpoints = sorted(set(min(N, fe) for fe in scheduled))
    curves, snapshot = [], []
    for i, (x, y) in enumerate(zip(arrays["x"], arrays["observed"])):
        archive.add(x, y, i)
        if i + 1 in checkpoints:
            snapshot = archive.snapshot()
            labels = [catalogue.match(x, float(arrays["truth"][idx])) for idx, x, _ in snapshot]
            found = sorted({j for j in labels if j is not None})
            strict_labels = [catalogue.match(x, float(arrays["truth"][idx]), 0.001, 0.001)
                             for idx, x, _ in snapshot]
            strict_found = sorted({j for j in strict_labels if j is not None})
            curves.append(dict(fe=i + 1, attained_minima=len(found), attained_indices=found,
                catalogue_coverage=len(found) / len(catalogue.entries) if catalogue.entries else None,
                archive_precision=sum(j is not None for j in labels) / len(labels),
                strict_attained_minima=len(strict_found), archive_size=len(snapshot)))
    snapshot = archive.snapshot()
    labels = [catalogue.match(x, float(arrays["truth"][idx])) for idx, x, _ in snapshot]
    # Diagnostic quenches measure basin access, not minimum attainment. Half
    # of the cap is observed-best and half uniform among the remaining archive.
    cap = min(int(settings.get("offline_quenches", 16)), len(snapshot))
    best = min((cap + 1) // 2, len(snapshot))
    rng = np.random.default_rng(seed)
    chosen = list(range(best))
    if cap > best:
        chosen += rng.choice(np.arange(best, len(snapshot)), cap - best, replace=False).tolist()
    quenches = []
    for j in chosen:
        idx, x0, observed = snapshot[j]
        res = minimize(problem.objective, x0, jac=problem.gradient, method="BFGS",
                       options={"maxiter": int(settings.get("quench_maxiter", 300)), "gtol": 2e-6})
        x = np.asarray(res.x) % 1
        energy = float(problem.objective(x)); grad = float(np.linalg.norm(problem.gradient(x)))
        H = hessian_summary(problem.hessian(x))
        valid = grad <= settings.get("catalogue_gradient_tolerance", 1e-5) and H["negative"] == 0
        label = catalogue.match(x, energy, 1e-5, 1e-6) if valid else None
        quenches.append(dict(query_index=idx, archive_position=j, observed=observed,
                            x=x.tolist(), energy=energy, grad_norm=grad, minimum_screen_passed=valid,
                            catalogue_index=label, offline_objective_nfev=int(res.nfev)))
    basin_labels = [q["catalogue_index"] for q in quenches if q["catalogue_index"] is not None]
    counts = np.array([basin_labels.count(j) for j in sorted(set(basin_labels))], float)
    probabilities = counts / counts.sum() if len(counts) else counts
    attained = sorted({j for j in labels if j is not None})
    if N < horizon:
        actual_curves = {c["fe"]:c for c in curves}
        curves = [{**actual_curves[min(fe,N)], "fe":fe, "actual_search_fes_at_checkpoint":min(fe,N)} for fe in scheduled]
    final = curves[-1] if curves else {}
    return dict(catalogue_size=len(catalogue.entries), catalogue_certified=catalogue.certified,
                catalogue_description=catalogue.description, curve=curves,
                attained_minima=len(attained), catalogue_coverage=final.get("catalogue_coverage"),
                archive_precision=final.get("archive_precision"),
                duplicate_catalogue_hits=sum(j is not None for j in labels)-len(attained),
                strict_attained_minima=final.get("strict_attained_minima"),
                all_catalogued_attained=bool(catalogue.entries and len(attained) == len(catalogue.entries)),
                ground_catalogue_attained=sum(catalogue.entries[j]["energy"] <= 1e-12 for j in attained),
                diagnostic_basin_access=len(set(basin_labels)),
                diagnostic_basin_entropy=float(-np.sum(probabilities * np.log(probabilities))) if len(counts) else 0.,
                diagnostic_unmatched_minima=sum(q["minimum_screen_passed"] and q["catalogue_index"] is None for q in quenches),
                archive=[dict(query_index=i, x=x.tolist(), observed=y, exact=float(arrays["truth"][i]),
                              catalogue_index=label) for (i, x, y), label in zip(snapshot, labels)],
                diagnostic_quenches=quenches, offline_seconds=time.perf_counter() - start,
                note="quench-discovered endpoints are basin-access diagnostics; not credited as search-located minima")
