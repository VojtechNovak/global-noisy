"""Independent exact landscape probes and scale-dependent noise diagnostics."""
from __future__ import annotations
import heapq
import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.special import ndtr
from .models import stable_seed
from .noise import Noise, FrozenField


def hessian_summary(H):
    eig = np.linalg.eigvalsh((H + H.T) / 2)
    tolerance = max(1e-8, 1e-6 * float(np.max(np.abs(eig))))
    pos = eig[eig > tolerance]
    norm = np.linalg.norm(H)
    return dict(eigenvalues=eig.tolist(), spectral_tolerance=tolerance,
                active_condition=float(pos.max() / pos.min()) if len(pos) else None,
                active_positive=int(len(pos)), negative=int(np.sum(eig < -tolerance)),
                null=int(np.sum(np.abs(eig) <= tolerance)),
                offdiagonal_ratio=float(np.linalg.norm(H - np.diag(np.diag(H))) / norm)
                if norm else 0.0)


def discrete_geometry(h):
    E = h.normalized
    ids = np.arange(len(E))
    neighbours = np.array([ids ^ (1 << q) for q in range(h.n)])
    diffs = E[neighbours] - E
    strict = np.all(diffs > 1e-12, axis=0)
    weak = np.all(diffs >= -1e-12, axis=0)
    # Operational steepest one-spin descent, not continuous optimizer basins.
    bestq = np.argmin(E[neighbours], axis=0)
    nxt = neighbours[bestq, ids]
    nxt = np.where(E[nxt] < E - 1e-12, nxt, ids)
    roots = nxt.copy()
    while True:
        updated = roots[roots]
        if np.array_equal(updated, roots):
            break
        roots = updated
    counts = np.bincount(roots, minlength=len(E))
    minima = np.flatnonzero(weak)
    ground = E <= 1e-12
    return dict(strict_one_flip_minima=int(strict.sum()), weak_one_flip_minima=int(weak.sum()),
                minimum_indices=minima.tolist(), minimum_values=E[minima].tolist(),
                discrete_basin_counts=counts[minima].tolist(),
                discrete_ground_basin_fraction=float(np.sum(counts[ground]) / len(E)),
                discrete_minimum_quality_iqr=float(np.subtract(*np.percentile(E[minima], [75, 25]))),
                minimum_flip_gap=float(np.min(diffs[:, strict])) if strict.any() else None)


def discrete_escape_barrier(h, source):
    """Exact minimax ONE-FLIP graph barrier to any ground state.

    A continuous product-state path can interpolate every one-spin edge
    without exceeding its endpoint energies. This is an upper bound on the
    smallest continuous escape barrier, not an equality assertion.
    """
    E = h.normalized
    bottleneck = np.full(len(E), np.inf)
    bottleneck[source] = E[source]
    queue = [(float(E[source]), int(source))]
    while queue:
        value, v = heapq.heappop(queue)
        if value > bottleneck[v]:
            continue
        if E[v] <= 1e-12:
            return float(max(0, value - E[source]))
        for q in range(h.n):
            w = v ^ (1 << q)
            candidate = max(value, float(E[w]))
            if candidate < bottleneck[w]:
                bottleneck[w] = candidate
                heapq.heappush(queue, (candidate, w))
    return None


def local_quenches(problem, starts, seed, maxiter=300, grad_tol=1e-5):
    rng = np.random.default_rng(seed)
    endpoints, representatives, counts = [], [], []
    for start in range(starts):
        x0 = rng.uniform(0, 1, problem.dim)
        res = minimize(problem.objective, x0, method="BFGS", jac=problem.gradient,
                       options={"maxiter": maxiter, "gtol": grad_tol / 5})
        x = res.x % 1
        gnorm = float(np.linalg.norm(problem.gradient(x)))
        energy = float(problem.objective(x))
        valid = gnorm <= grad_tol
        # Only stationarity-screened endpoints enter the empirical clustering.
        # Hessian confirmation is performed separately at cluster representatives.
        cluster = None
        psi = problem.state(x) if valid else None
        if valid:
            for j, rep in enumerate(representatives):
                if abs(energy - rep["energy"]) <= 1e-7:
                    delta = (x - rep["x"] + 0.5) % 1 - 0.5
                    same_state = abs(np.vdot(rep["state"], psi)) ** 2 >= 1 - 1e-6
                    if np.linalg.norm(delta) / np.sqrt(problem.dim) <= 0.02 or same_state:
                        cluster = j
                        break
            if cluster is None:
                cluster = len(representatives)
                representatives.append(dict(x=x.copy(), energy=energy, state=psi))
                counts.append(0)
            counts[cluster] += 1
        endpoints.append(dict(start=start, energy=energy, grad_norm=gnorm,
                              stationary=valid, cluster=cluster, x=x.tolist(),
                              scipy_success=bool(res.success)))
    return endpoints, representatives, counts


def information_content(y, epsilon):
    dy = np.diff(y)
    symbols = np.where(dy > epsilon, 1, np.where(dy < -epsilon, -1, 0))
    if len(symbols) < 2:
        return 0.0
    probabilities = [np.mean((symbols[:-1] == a) & (symbols[1:] == b))
                     for a in (-1, 0, 1) for b in (-1, 0, 1) if a != b]
    return float(-sum(p * np.log(p) / np.log(6) for p in probabilities if p > 0))


def structure_function(y, lag):
    return float(np.mean((y[lag:] - y[:-lag]) ** 2))


def line_diagnostics(problem, seed, n_lines=8, grid=257, shots=(128, 10000),
                     replicates=16, field_specs=None):
    rng = np.random.default_rng(seed)
    rows, plots = [], []
    noise_models = [("exact", None)]
    noise_models += [(f"iid_gaussian_S{s}", Noise("gaussian", int(s))) for s in shots]
    fields = [(f"frozen_amp{spec['amplitude']}_ell{spec['length_scale']}",
               FrozenField(problem.dim, **spec, seed=stable_seed(seed, "field", j)))
              for j, spec in enumerate(field_specs or [])]
    for line in range(n_lines):
        anchor = rng.uniform(0, 1, problem.dim)
        direction = rng.normal(size=problem.dim)
        direction /= np.linalg.norm(direction)
        t = np.linspace(-0.25, 0.25, grid)
        points = (anchor[None, :] + t[:, None] * direction) % 1
        true = np.array([problem.objective(x) for x in points])
        for label, noise in noise_models + fields:
            if label.startswith("frozen"):
                latent = true + np.array([noise(x) for x in points])
                observations, sd = latent[None, :], 0.0
            elif noise is None:
                latent, observations, sd = true, true[None, :], 0.0
            else:
                latent = true
                sd = noise.scale / np.sqrt(noise.shots)
                observations = true[None, :] + rng.normal(0, sd, (replicates, grid))
            masks = np.array([(y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
                              for y in observations])
            row = dict(line=line, model=label, grid=grid, spacing=float(t[1] - t[0]),
                       latent_slice_minima=int(np.sum((latent[1:-1] < latent[:-2]) &
                                                     (latent[1:-1] < latent[2:]))),
                       observed_slice_minima_mean=float(np.sum(masks, axis=1).mean()),
                       repeated_point_minimum_frequency_ge075=int(np.sum(masks.mean(axis=0) >= 0.75)),
                       sd=float(sd))
            for lag in (1, 2, 4, 8, 16):
                if lag >= grid:
                    continue
                raw = float(np.mean([structure_function(y, lag) for y in observations]))
                row[f"structure_raw_lag{lag}"] = raw
                row[f"structure_debiased_lag{lag}"] = raw - 2 * sd * sd
            for epsilon in (0.0, 1e-4, 1e-3, 1e-2, 0.05):
                row[f"information_epsilon{epsilon}"] = float(np.mean(
                    [information_content(y, epsilon) for y in observations]))
                row[f"prominent_minima_epsilon{epsilon}"] = float(np.mean(
                    [len(find_peaks(-y, prominence=epsilon)[0]) for y in observations]))
            rows.append(row)
            if line == 0:
                plots.append(dict(label=label, t=t.tolist(), latent=latent.tolist(),
                                  observed=observations[0].tolist()))
    return rows, plots


def misranking_probes(problem, seed, pairs=128, shots=(128, 10000)):
    rng = np.random.default_rng(seed)
    rows = []
    for distance in (0.005, 0.02, 0.05, 0.1):
        for pair in range(pairs):
            a = rng.uniform(0, 1, problem.dim)
            v = rng.normal(size=problem.dim); v /= np.linalg.norm(v)
            b = (a + distance * v) % 1
            fa, va = problem.moments(a, True)
            fb, vb = problem.moments(b, True)
            gap = abs(fa - fb)
            for S in shots:
                homo_sd = 0.5 * np.sqrt(2 / S)
                hetero_sd = np.sqrt((va + vb) / S)
                rows.append(dict(pair=pair, distance=distance, shots=S, exact_gap=gap,
                                 gaussian_misrank_probability=float(ndtr(-gap / homo_sd)),
                                 hetero_gaussian_misrank_probability=float(ndtr(-gap / hetero_sd))
                                 if hetero_sd > 0 else 0.0))
    return rows


def characterize(problem, settings, seed):
    starts = int(settings.get("quench_starts", 64))
    endpoints, reps, counts = local_quenches(problem, starts, seed,
                                             int(settings.get("quench_maxiter", 300)))
    checked = []
    for rep, count in sorted(zip(reps, counts), key=lambda t: t[0]["energy"])[:
            int(settings.get("hessian_minima", 8))]:
        checked.append(dict(energy=rep["energy"], count=count,
                            hessian=hessian_summary(problem.hessian(rep["x"])),
                            x=rep["x"].tolist()))
    # Independent reference starts are never selected using benchmark outputs.
    reference_starts = int(settings.get("reference_starts", 128))
    if problem.ansatz == "product":
        reference = 0.0  # Every computational basis state is representable.
        reference_certified = True
    else:
        ref_ep, _, _ = local_quenches(problem, reference_starts,
                                     stable_seed(seed, "independent_reference"),
                                     int(settings.get("reference_maxiter", 600)))
        reference = min(e["energy"] for e in ref_ep)
        reference_certified = False
    valid_values = np.array([e["energy"] for e in endpoints if e["stationary"]])
    rng = np.random.default_rng(stable_seed(seed, "gradient_probes"))
    norms = [float(np.linalg.norm(problem.gradient(rng.uniform(0, 1, problem.dim))))
             for _ in range(int(settings.get("gradient_samples", 32)))]
    record = dict(problem=problem.metadata(), reference=reference,
                  reference_certified=reference_certified,
                  reference_description="exact variational minimum" if reference_certified else
                  "independent best-known variational value; not certified",
                  normalized_variational_gap=reference, endpoint_clusters=len(reps),
                  stationary_endpoints=int(len(valid_values)), quench_starts=starts,
                  quench_endpoints=endpoints, cluster_counts=counts,
                  basin_good_fraction=float(np.mean(valid_values <= reference + 0.01))
                  if len(valid_values) else None,
                  endpoint_quality_iqr=float(np.subtract(*np.percentile(valid_values, [75, 25])))
                  if len(valid_values) else None,
                  grad_norm_median=float(np.median(norms)), gradient_norms=norms,
                  hessian_probes=checked,
                  clustering_note="stationarity-screened empirical endpoint clusters; not an exact basin count")
    if problem.ansatz == "product":
        discrete = discrete_geometry(problem.h)
        selected = sorted(discrete["minimum_indices"], key=lambda i: problem.h.normalized[i])[:
            int(settings.get("barrier_minima", 4))]
        discrete["escape_barrier_upper_bounds"] = [dict(index=int(i),
            barrier=discrete_escape_barrier(problem.h, i)) for i in selected]
        # The exact vertex Hessian relation is a strong conditioning diagnostic.
        conditions = []
        for i in discrete["minimum_indices"]:
            theta = np.pi * problem.h.bits[i]
            v = theta / (2 * np.pi) + 0.5
            u = np.linalg.solve(problem.U.astype(float), v) % 1
            conditions.append(hessian_summary(problem.hessian(u)))
        discrete["all_vertex_hessian_summaries"] = conditions
        record["discrete_geometry"] = discrete
    lines, traces = line_diagnostics(problem, stable_seed(seed, "lines"),
                      settings.get("lines", 8), settings.get("line_grid", 257),
                      settings.get("probe_shots", [128, 10000]),
                      settings.get("line_replicates", 16), settings.get("field_specs", []))
    record.update(line_diagnostics=lines, example_lines=traces,
                  misranking_probes=misranking_probes(problem, stable_seed(seed, "pairs"),
                    settings.get("misranking_pairs", 32), settings.get("probe_shots", [128, 10000])))
    return record
