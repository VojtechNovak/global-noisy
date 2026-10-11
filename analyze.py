"""Equal-instance summaries, paired bootstrap contrasts, and shot-response fits.

This produces estimation-first reports. Runs are nested in disorder instances;
they are not treated as independent instances. No universal shot threshold or
optimizer ranking is inferred from an engineering smoke test.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from .storage import records, load_trace, curve_at


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    idx = np.argsort(p); adjusted = np.empty(len(p)); previous = 0.0
    for rank, j in enumerate(idx):
        previous = max(previous, min(1.0, (len(p) - rank) * p[j]))
        adjusted[j] = previous
    return adjusted.tolist()


def isotonic(y, weights):
    """Weighted PAVA for a nondecreasing success curve."""
    blocks = []
    for i, (value, weight) in enumerate(zip(y, weights)):
        blocks.append([i, i + 1, float(value * weight), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    result = np.empty(len(y))
    for start, stop, total, weight in blocks:
        result[start:stop] = total / weight
    return result


def threshold_label(shots, probabilities, target):
    hits = np.flatnonzero(np.asarray(probabilities) >= target)
    if not len(hits):
        return f">{max(shots)}", None, "right_censored"
    s = int(shots[hits[0]])
    return (f"<={s}" if hits[0] == 0 else str(s)), s, ("left_censored" if hits[0] == 0 else "grid")


def nested_pair_bootstrap(paired, repetitions, seed):
    rng = np.random.default_rng(seed)
    keys = sorted(paired)
    estimate = float(np.mean([np.median(paired[k][:, 0]) - np.median(paired[k][:, 1]) for k in keys]))
    # Common matched run counts permit a much faster exact hierarchical draw.
    if len({len(paired[k]) for k in keys}) == 1:
        data = np.stack([paired[k] for k in keys])
        chosen = rng.integers(len(keys), size=(repetitions, len(keys)))
        run_indices = rng.integers(data.shape[1], size=(repetitions, len(keys), data.shape[1]))
        samples = data[chosen[..., None], run_indices]
        medians = np.median(samples, axis=2)
        values = np.mean(medians[..., 0] - medians[..., 1], axis=1)
        low, high = np.percentile(values, [2.5, 97.5])
        return estimate, float(low), float(high)
    values = []
    for _ in range(repetitions):
        indices = rng.integers(len(keys), size=len(keys))
        differences = []
        for i in indices:
            a = paired[keys[i]]
            sampled = a[rng.integers(len(a), size=len(a))]
            differences.append(np.median(sampled[:, 0]) - np.median(sampled[:, 1]))
        values.append(float(np.mean(differences)))
    low, high = np.percentile(values, [2.5, 97.5])
    return estimate, float(low), float(high)


def collect(out):
    return records(out)


def analyze(out, bootstrap=2000, target_regret=0.01, target_probability=0.8):
    rows = collect(out)
    if not rows:
        raise ValueError("No completed run records.")
    dest = out / "analysis"; dest.mkdir(exist_ok=True)
    fields = ["condition", "n", "instance_id", "optimizer", "run", "noise", "noise_mode",
              "shots", "budget_mode", "budget", "reserve_fraction", "search_fes", "search_shots",
              "validation_shots", "oracle_best_exact", "raw_selected_exact", "selected_exact",
              "oracle_reference_gap", "selected_reference_gap", "search_selection_gap",
              "winner_optimism", "incumbent_age_fes", "search_wall_seconds", "reference_improved"]
    write_csv(dest / "run_table.csv", [{k: r.get(k) for k in fields} for r in rows])
    stratum_keys = ["condition", "n", "noise", "budget_mode", "budget", "reserve_fraction"]
    grouped = {}
    for r in rows:
        key = tuple(r[k] for k in stratum_keys)
        grouped.setdefault(key, []).append(r)
    summaries, contrasts = [], []
    for key, group in grouped.items():
        labels = dict(zip(stratum_keys, key))
        names = sorted({r["optimizer"] for r in group})
        for name in names:
            subset = [r for r in group if r["optimizer"] == name]
            instances = sorted({r["instance_id"] for r in subset})
            row = {**labels, "optimizer": name, "instances": len(instances), "runs": len(subset)}
            for metric in ["oracle_best_exact", "selected_exact", "search_selection_gap",
                           "winner_optimism", "search_wall_seconds"]:
                meds = [np.median([r[metric] for r in subset if r["instance_id"] == i]) for i in instances]
                row[f"equal_instance_mean_of_medians_{metric}"] = float(np.mean(meds))
            summaries.append(row)
        # Preregistered primary contrasts against the canonical VQA baseline.
        # The broader seven-way ranking remains descriptive.
        if "COBYLA" not in names:
            continue
        for name in names:
            if name == "COBYLA":
                continue
            maps = {a: {(r["instance_id"], r["run"]): r["selected_exact"]
                        for r in group if r["optimizer"] == a} for a in [name, "COBYLA"]}
            common = sorted(set(maps[name]) & set(maps["COBYLA"]))
            paired = {}
            for i, run in common:
                paired.setdefault(i, []).append([maps[name][i, run], maps["COBYLA"][i, run]])
            if not paired:
                continue
            paired = {i: np.array(values) for i, values in paired.items()}
            estimate, low, high = nested_pair_bootstrap(paired, bootstrap, 4117)
            differences = np.array([np.median(a[:, 0]) - np.median(a[:, 1]) for a in paired.values()])
            differences[np.abs(differences) < 1e-12] = 0
            pvalue = float(wilcoxon(differences).pvalue) if np.any(differences) else 1.0
            contrasts.append({**labels, "optimizer": name, "baseline": "COBYLA",
                              "instances": len(paired), "mean_median_difference": estimate,
                              "bootstrap_95_low": low, "bootstrap_95_high": high,
                              "instance_wilcoxon_p": pvalue,
                              "note": "negative favors optimizer; instance resampling with paired run resampling"})
    if contrasts:
        adjusted = holm([r["instance_wilcoxon_p"] for r in contrasts])
        for r, p in zip(contrasts, adjusted):
            r["holm_all_reported_primary_contrasts_p"] = p
    write_csv(dest / "summary.csv", summaries)
    write_csv(dest / "paired_contrasts.csv", contrasts)
    # Dose-response, grouped so budget, measurement model and verification
    # policy never get mixed into one threshold.
    dose_keys = ["condition", "n", "optimizer", "budget_mode", "budget",
                 "reserve_fraction", "noise_mode", "repeats", "scale"]
    doses = {}
    for r in rows:
        if r["noise_mode"] != "exact" and r["selected_reference_gap"] is not None:
            doses.setdefault(tuple(r[k] for k in dose_keys), []).append(r)
    curves, thresholds = [], []
    for key, group in doses.items():
        levels = sorted({r["shots"] for r in group})
        if len(levels) < 3:
            continue
        raw, weights = [], []
        for S in levels:
            subset = [r for r in group if r["shots"] == S]
            ids = sorted({r["instance_id"] for r in subset})
            proportions = [np.mean([r["selected_reference_gap"] <= target_regret
                                    for r in subset if r["instance_id"] == i]) for i in ids]
            raw.append(float(np.mean(proportions))); weights.append(len(ids))
        # Monotonicity is an explicit descriptive model. In fixed-shot-budget
        # studies fewer FEs at larger S can produce genuine nonmonotonicity.
        labels = dict(zip(dose_keys, key))
        fitted = isotonic(raw, weights) if labels["budget_mode"] == "fe" else np.asarray(raw)
        for S, p, iso in zip(levels, raw, fitted):
            curves.append({**labels, "shots": S, "success_raw": p,
                           "success_isotonic": float(iso) if labels["budget_mode"] == "fe" else None,
                           "target_regret": target_regret, "target_probability": target_probability})
        label, S, censor = threshold_label(levels, fitted, target_probability)
        # Do not force a monotone shot threshold on fixed-total-shot curves.
        thresholds.append({**labels, "threshold_shots": S if labels["budget_mode"] == "fe" else None,
                           "threshold_label": label if labels["budget_mode"] == "fe" else "not_monotone_assumed",
                           "censoring": censor if labels["budget_mode"] == "fe" else "not_applicable",
                           "raw_nonmonotone": bool(np.any(np.diff(raw) < -1e-12)),
                           "warning": "grid point estimate only; confirm with instance-cluster confidence bands before scientific claims"})
    write_csv(dest / "shot_response.csv", curves)
    write_csv(dest / "threshold_point_estimates.csv", thresholds)
    improved = sum(r["reference_improved"] for r in rows)
    note = (f"Completed records: {len(rows)}. Reference-improving runs: {improved}.\n"
            "QAOA references are empirical. Retain signed gaps; do not silently clip improvements.\n"
            "Bootstrap intervals require enough independent instances; a one-instance smoke test is engineering validation.\n"
            "Threshold files contain point estimates, not a demonstrated convergence/failure phase transition.\n")
    (dest / "analysis_notes.txt").write_text(note)
    return dict(completed_runs=len(rows), output=str(dest.resolve()), reference_improvements=improved)


def plot(out, rows, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # First condition is illustrative; all run data remain in the CSV files.
    condition = rows[0]["condition"]
    subset = [r for r in rows if r["condition"] == condition]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in sorted({r["optimizer"] for r in subset}):
        group = [r for r in subset if r["optimizer"] == name and r["reserve_fraction"] == 0]
        if not group:
            continue
        # Show each noise level separately; do not blend exact and noisy curves.
        for noise in sorted({r["noise"] for r in group}):
            ng = [r for r in group if r["noise"] == noise]
            length = min(r["search_fes"] for r in ng)
            checkpoint = np.unique(np.linspace(0, length - 1, min(length, 80)).astype(int))
            ys = [curve_at(load_trace(out, r), "oracle_curve", checkpoint + 1) for r in ng]
            ax.plot(checkpoint + 1, np.median(ys, axis=0), label=f"{name}, {noise}", lw=1)
    ax.set(xlabel="Charged search function evaluations", ylabel="Best exact normalized energy queried",
           title=f"Engineering example: {condition}")
    ax.legend(fontsize=6, ncol=3); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out / "analysis" / "example_convergence.png", dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/smoke")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--target-regret", type=float, default=0.01)
    p.add_argument("--target-probability", type=float, default=0.8)
    a = p.parse_args()
    print(json.dumps(analyze(Path(a.out), a.bootstrap, a.target_regret, a.target_probability), indent=2))


if __name__ == "__main__":
    main()
