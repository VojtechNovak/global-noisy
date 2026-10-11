"""Noise models and a budgeted observation oracle with private assessment logs."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


class BudgetReached(RuntimeError):
    pass


class FrozenField:
    """Smooth periodic random trigonometric field, fixed per condition.

    Integer frequencies keep torus periodicity. length_scale is in unit
    coordinates; it is a frequency-envelope parameter, not an exact measured
    correlation length. Distinct +/- modes and Gaussian coefficients give
    exact spatial RMS amplitude after normalization.
    """
    def __init__(self, dim, amplitude=0.04, length_scale=0.1, modes=128, seed=0):
        if amplitude < 0 or length_scale <= 0 or modes < 1:
            raise ValueError("Invalid field parameters.")
        rng = np.random.default_rng(seed)
        freq = set()
        width = 1 / (2 * np.pi * length_scale)
        attempts = 0
        while len(freq) < modes:
            attempts += 1
            if attempts > max(10000, 1000 * modes):
                raise ValueError("Too many distinct modes for this dimension/length_scale; reduce modes.")
            k = np.rint(rng.normal(0, width, dim)).astype(int)
            if not np.any(k):
                continue
            first = k[np.flatnonzero(k)[0]]
            if first < 0:
                k = -k
            freq.add(tuple(k))
        self.k = np.array(sorted(freq), dtype=float)
        self.a, self.b = rng.normal(size=(2, modes))
        rms = np.sqrt(0.5 * np.sum(self.a ** 2 + self.b ** 2))
        self.a *= amplitude / rms
        self.b *= amplitude / rms
        self.amplitude, self.length_scale = float(amplitude), float(length_scale)

    def __call__(self, x):
        t = 2 * np.pi * (self.k @ np.asarray(x))
        return float(self.a @ np.cos(t) + self.b @ np.sin(t))

    def metadata(self):
        return dict(amplitude=self.amplitude, length_scale=self.length_scale,
                    k=self.k.tolist(), a=self.a.tolist(), b=self.b.tolist())


@dataclass
class Noise:
    mode: str = "gaussian"
    shots: int = 128
    scale: float = 0.5
    repeats: int = 1

    def __post_init__(self):
        if self.mode not in {"exact", "gaussian", "heteroscedastic", "shots"}:
            raise ValueError("Noise mode must be exact, gaussian, heteroscedastic, or shots.")
        if self.shots < 1 or self.repeats < 1 or self.scale < 0:
            raise ValueError("Invalid noise specification.")

    @property
    def cost(self):
        return 0 if self.mode == "exact" else int(self.shots * self.repeats)

    def observe(self, problem, x, rng):
        need_var = self.mode == "heteroscedastic"
        truth, var = problem.moments(x, need_var)
        if self.mode == "exact":
            return truth, truth, 0.0
        if self.mode == "shots":
            obs = problem.sample_mean(x, self.cost, rng)
            # Store the true pointwise standard deviation for offline audit.
            var = problem.moments(x, True)[1]
            return obs, truth, float(np.sqrt(var / self.cost))
        sd = np.sqrt(var / self.cost) if need_var else self.scale / np.sqrt(self.cost)
        return float(truth + rng.normal(0, sd)), truth, float(sd)


class Recorder:
    """Only objective_only() is passed to the optimizer, never this object.

    One function call costs one FE even when repeats averages several estimates;
    each repeat costs shots. Oracle truth is assessment-only and cannot guide
    proposals, shortlist creation, acceptance, or stopping.
    """
    def __init__(self, problem, noise, max_fes, seed, field=None):
        self.problem, self.noise = problem, noise
        self.max_fes = int(max_fes)
        if self.max_fes < 1:
            raise ValueError("Search budget must allow at least one FE.")
        self.rng = np.random.default_rng(seed)
        self.field = field
        self.x, self.observed, self.truth, self.latent, self.sd = [], [], [], [], []

    def objective_only(self):
        def observe(x):
            if len(self.x) >= self.max_fes:
                raise BudgetReached("Function evaluation ceiling reached.")
            x = np.asarray(x, dtype=float)
            if x.shape != (self.problem.dim,) or not np.all(np.isfinite(x)):
                raise ValueError("Optimizer proposed invalid coordinates.")
            # Optimizers use finite box constraints. Periodic extension is
            # reserved for local diagnostics and exact coordinate maps.
            if np.any(x < -1e-10) or np.any(x > 1 + 1e-10):
                raise ValueError("Optimizer proposal outside [0,1].")
            x = np.clip(x, 0, 1)
            obs, truth, sd = self.noise.observe(self.problem, x, self.rng)
            bias = self.field(x) if self.field is not None else 0.0
            self.x.append(x.copy())
            self.observed.append(obs + bias)
            self.truth.append(truth)
            self.latent.append(truth + bias)
            self.sd.append(sd)
            return obs + bias
        return observe

    @property
    def nfev(self):
        return len(self.x)

    def arrays(self):
        return dict(x=np.array(self.x), observed=np.array(self.observed),
                    truth=np.array(self.truth), latent=np.array(self.latent), sd=np.array(self.sd))


def shortlist(x, observed, k):
    """Noise-only shortlist; deduplicate exact torus coordinates, not truth."""
    chosen, seen = [], set()
    for i in np.argsort(observed, kind="stable"):
        key = np.round(np.asarray(x[i]) % 1, 12).tobytes()
        if key not in seen:
            seen.add(key)
            chosen.append(int(i))
        if len(chosen) == k:
            break
    return chosen


def verify(recorder, reserved_shots, k, seed):
    """Prospective validation, charged to the common TOTAL shot budget.

    All shortlisted points get equal independent measurements. Selection uses
    their validation observations; truth is attached afterward for assessment.
    """
    ids = shortlist(recorder.x, recorder.observed, k)
    if not ids or reserved_shots < len(ids):
        return None
    each = int(reserved_shots // len(ids))
    noise = Noise(recorder.noise.mode, each, recorder.noise.scale, 1)
    rng = np.random.default_rng(seed)
    values = []
    for i in ids:
        y = noise.observe(recorder.problem, recorder.x[i], rng)[0]
        if recorder.field is not None:
            y += recorder.field(recorder.x[i])
        values.append(float(y))
    j = int(np.argmin(values))
    i = ids[j]
    return dict(index=i, shortlist_indices=ids, validation_values=values,
                validation_shots=each * len(ids), shots_per_candidate=each,
                exact_selected=float(recorder.truth[i]), observed_selected=values[j])
