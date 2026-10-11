"""L-SRTDE Python port aligned to the authors' released C++ implementation.

Reference algorithm: Stanovov & Semenkin, CEC 2024.
Reference source: VladimirStanovov/L-SRTDE_CEC-2024, L-SRTDE.cpp.

API kept compatible with the user's previous implementation::

    opt = LSRTDE_Optimizer(func, dim, bounds, max_evals, seed)
    best_x, best_f = opt.optimize()

Important reproducibility note
------------------------------
The paper's x_new/x_top pseudocode and the released C++ are not literally
identical in indexing/operand notation.  This implementation follows the
released C++ operational logic because it is the strongest reference for
reproducing the authors' reported implementation:

* Popul and PopulFront are maintained separately.
* p-best is selected from ranked Popul.
* the rank-biased vector is selected from ranked PopulFront.
* mutation is
    PopulFront[target]
    + F * (Popul[pbest] - PopulFront[target])
    + F * (PopulFront[r_rank] - Popul[r_random]).
* successful trials are appended to Popul and written into PopulFront through
  a persistent ring index.
* PopulFront is reduced by deleting its worst individuals; Popul is then
  rebuilt from the best old-Popul + successful-offspring candidates.

The C++ completes an entire generation even when the evaluation budget is
crossed.  For fair generic benchmarking this port enforces a strict budget by
default. Set strict_budget=False to emulate that source-level termination
behavior more closely (which can exceed max_evals).
"""

from __future__ import annotations

from typing import Callable, Tuple
import numpy as np

Array = np.ndarray


def _as_bounds(bounds, dim: int) -> Tuple[Array, Array]:
    b = np.asarray(bounds, dtype=float)
    if b.ndim == 1 and b.size == 2:
        lo = np.full(dim, b[0], dtype=float)
        hi = np.full(dim, b[1], dtype=float)
    elif b.shape == (dim, 2):
        lo = b[:, 0].copy()
        hi = b[:, 1].copy()
    else:
        raise ValueError("bounds must be (lower, upper) or shape (dim, 2)")
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(lo >= hi):
        raise ValueError("invalid finite bounds")
    return lo, hi


def _weighted_lehmer_cpp(values: Array, deltas: Array) -> float:
    """MeanWL behavior of the released C++ (including the zero-weight fallback)."""
    sw = float(np.sum(deltas))
    if not np.isfinite(sw) or sw <= 0.0:
        # In the C++ a zero SumWeight makes intermediate weights NaN and the
        # final fabs(Sum)>1e-8 test falls through to return 1.0.
        return 1.0
    w = deltas / sw
    s2 = float(np.sum(w * values * values))
    s1 = float(np.sum(w * values))
    return s2 / s1 if abs(s1) > 1e-8 else 1.0


class LSRTDE_Optimizer:
    def __init__(
        self,
        func: Callable[[Array], float],
        dim: int,
        bounds,
        max_evals: int,
        seed=None,
        *,
        strict_budget: bool = True,
    ):
        if dim < 1:
            raise ValueError("dim must be positive")
        if max_evals < 1:
            raise ValueError("max_evals must be positive")
        self.func = func
        self.dim = int(dim)
        self.lower, self.upper = _as_bounds(bounds, self.dim)
        self.bounds = np.column_stack((self.lower, self.upper))
        self.max_evals = int(max_evals)
        self.strict_budget = bool(strict_budget)
        self.rng = np.random.default_rng(seed)

        self.pop_size_max = 4 * self.dim
        self.pop_size_min = 4
        self.memory_size = 5
        self.kp = 3.0

        self.eval_count = 0
        self.best_fit = float("inf")
        self.best_ind = None
        self.history = []

        self.memory_cr = np.ones(self.memory_size, dtype=float)
        self.memory_pos = 0
        self.success_rate = 0.5
        self.front_write_pos = 0

    def _eval(self, x: Array) -> float:
        y = float(self.func(x))
        self.eval_count += 1
        if y < self.best_fit:
            self.best_fit = y
            self.best_ind = x.copy()
        self.history.append(self.best_fit)
        return y

    def _can_eval(self) -> bool:
        return (not self.strict_budget) or self.eval_count < self.max_evals

    def optimize(self):
        # C++ Nmax = 20D. With a tiny user budget we cannot initialize more
        # points than can be evaluated under strict-budget mode.
        nmax = self.pop_size_max
        if self.strict_budget:
            nmax = min(nmax, self.max_evals)
        if nmax <= 0:
            return None, float("inf")

        # C++ allocates 2*Nmax and randomizes the whole buffer. Only the first
        # Nmax are evaluated initially; unused random rows do not affect search.
        pop = self.rng.uniform(self.lower, self.upper, size=(2 * nmax, self.dim))
        fit = np.full(2 * nmax, np.inf, dtype=float)

        n = nmax
        for i in range(n):
            if not self._can_eval():
                break
            fit[i] = self._eval(pop[i])

        # If budget was smaller than initialization, keep only evaluated rows.
        n = min(n, self.eval_count)
        if n == 0:
            return None, float("inf")
        nmax = n

        # Initial PopulFront is the sorted Popul, exactly as MainCycle does.
        initial_order = np.argsort(fit[:n])
        front = pop[initial_order].copy()
        front_fit = fit[initial_order].copy()
        self.front_write_pos = 0

        while self.eval_count < self.max_evals and n >= self.pop_size_min:
            mean_f = 0.4 + np.tanh(self.success_rate * 5.0) * 0.25
            sigma_f = 0.02

            # Frozen rank maps at generation start, matching Indices/Indices2.
            pop_order = np.argsort(fit[:n])
            front_order = np.argsort(front_fit[:n])

            rank_weights = np.exp(-self.kp * np.arange(n, dtype=float) / n)
            rank_weights /= rank_weights.sum()

            psize = max(2, int(n * 0.7 * np.exp(-self.success_rate * 7.0)))
            psize = min(n, psize)

            success_cr = []
            success_delta = []
            offspring_x = []
            offspring_f = []

            # C++ performs NIndsFront trials per generation, with a randomly
            # chosen target on each iteration (targets may repeat).
            for _ in range(n):
                if self.strict_budget and self.eval_count >= self.max_evals:
                    break

                target = int(self.rng.integers(n))
                mem_idx = int(self.rng.integers(self.memory_size))

                # p-best from ranked Popul, not PopulFront.
                pbest = int(pop_order[int(self.rng.integers(psize))])
                while pbest == target and n > 1:
                    pbest = int(pop_order[int(self.rng.integers(psize))])

                # Rank-biased vector from ranked PopulFront. The C++ checks
                # numeric index inequality against pbest (despite different arrays).
                r_rank = int(front_order[int(self.rng.choice(n, p=rank_weights))])
                while r_rank == pbest and n > 1:
                    r_rank = int(front_order[int(self.rng.choice(n, p=rank_weights))])

                # Random vector from ranked Popul (uniform over ranks == uniform
                # over individuals), excluding pbest and r_rank numerically.
                r_random = int(pop_order[int(self.rng.integers(n))])
                while r_random in (pbest, r_rank) and n > 2:
                    r_random = int(pop_order[int(self.rng.integers(n))])

                # F ~ N(meanF, .02), repeated until F in [0,1] in C++.
                while True:
                    F = float(self.rng.normal(mean_f, sigma_f))
                    if 0.0 <= F <= 1.0:
                        break

                Cr = float(np.clip(self.rng.normal(self.memory_cr[mem_idx], 0.05), 0.0, 1.0))

                donor = (
                    front[target]
                    + F * (pop[pbest] - front[target])
                    + F * (front[r_rank] - pop[r_random])
                )

                jrand = int(self.rng.integers(self.dim))
                mask = self.rng.random(self.dim) < Cr
                mask[jrand] = True
                trial = np.where(mask, donor, front[target]).astype(float, copy=False)
                actual_cr = float(np.count_nonzero(mask)) / self.dim

                # Released C++ resamples each violated coordinate uniformly.
                bad = (trial < self.lower) | (trial > self.upper)
                if np.any(bad):
                    trial = trial.copy()
                    trial[bad] = self.rng.uniform(self.lower[bad], self.upper[bad])

                trial_fit = self._eval(trial)

                # C++ uses <= against current PopulFront[target].
                if trial_fit <= front_fit[target]:
                    # Append successful trial to the Popul candidate pool.
                    offspring_x.append(trial.copy())
                    offspring_f.append(trial_fit)

                    # Persistent ring-buffer update of PopulFront.
                    wp = self.front_write_pos % n
                    front[wp] = trial
                    front_fit[wp] = trial_fit

                    # Preserve the source ordering of operations: fitness delta is
                    # computed *after* the ring write, which can make delta zero if
                    # wp == target. This oddity is present in the released C++.
                    success_cr.append(actual_cr)
                    success_delta.append(abs(float(front_fit[target]) - trial_fit))

                    self.front_write_pos = (wp + 1) % n

            successes = len(offspring_x)
            self.success_rate = successes / float(n)

            # LPSR from the initial Nmax, applied to PopulFront regardless of
            # whether any success occurred. C++ uses int truncation.
            n_next = int(((self.pop_size_min - nmax) / float(self.max_evals)) * self.eval_count + nmax)
            n_next = max(self.pop_size_min, min(n, n_next))

            # Remove worst entries from PopulFront. Sequential worst deletion is
            # equivalent to keeping its n_next best entries, but preserve order of
            # survivors because the ring index addresses positions.
            if n_next < n:
                remove_count = n - n_next
                keep_mask = np.ones(n, dtype=bool)
                # delete worst one at a time from the current compact array
                ftmp = front_fit[:n].copy()
                xtmp = front[:n].copy()
                for _ in range(remove_count):
                    worst = int(np.argmax(ftmp))
                    ftmp = np.delete(ftmp, worst)
                    xtmp = np.delete(xtmp, worst, axis=0)
                front = xtmp
                front_fit = ftmp
            else:
                front = front[:n_next].copy()
                front_fit = front_fit[:n_next].copy()

            # Update MCr before clearing successes, as in released source.
            if successes > 0:
                crs = np.asarray(success_cr, dtype=float)
                deltas = np.asarray(success_delta, dtype=float)
                mean_cr = _weighted_lehmer_cpp(crs, deltas)
                self.memory_cr[self.memory_pos] = 0.5 * (
                    mean_cr + self.memory_cr[self.memory_pos]
                )
                self.memory_pos = (self.memory_pos + 1) % self.memory_size

            # Popul := best n_next from previous Popul[0:n] U successful trials.
            if successes:
                cand_x = np.vstack((pop[:n], np.asarray(offspring_x)))
                cand_f = np.concatenate((fit[:n], np.asarray(offspring_f)))
            else:
                cand_x = pop[:n].copy()
                cand_f = fit[:n].copy()

            best_idx = np.argsort(cand_f)[:n_next]
            new_pop_active = cand_x[best_idx].copy()
            new_fit_active = cand_f[best_idx].copy()

            # Keep a 2*nmax workspace for source-compatible candidate appends.
            if pop.shape[0] < 2 * nmax:
                pop = np.empty((2 * nmax, self.dim), dtype=float)
                fit = np.full(2 * nmax, np.inf, dtype=float)
            pop[:n_next] = new_pop_active
            fit[:n_next] = new_fit_active

            n = n_next
            self.front_write_pos %= max(1, n)

            if not self.strict_budget and self.eval_count >= self.max_evals:
                break

        return None if self.best_ind is None else self.best_ind.copy(), float(self.best_fit)


def run_lsrtde(f, dim, bounds, max_evals, seed=None, **kwargs):
    return LSRTDE_Optimizer(f, dim, bounds, max_evals, seed, **kwargs).optimize()
