"""All manuscript figures/tables, clustered uncertainty and data-only LaTeX.

Reports remain explicitly partial until the expected complete run inventory and
the FE/shot audit pass. Statistical units are disorder instances, not runs.
"""
from __future__ import annotations
import csv
import gzip
import io
import itertools
import json
from pathlib import Path
import sqlite3
import numpy as np
from .analyze import analyze, write_csv, isotonic, threshold_label, nested_pair_bootstrap
from .storage import records, curve_at


def grouped(rows, keys):
    groups = {}
    for r in rows: groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    return groups


def cluster_interval(subset, metric, repetitions=2000, seed=4171):
    ids = sorted({r["instance_id"] for r in subset})
    values = {i: np.array([r[metric] for r in subset if r["instance_id"] == i], float) for i in ids}
    if not ids: return None, None, None
    rng = np.random.default_rng(seed)
    estimate = float(np.mean([np.median(values[i]) for i in ids]))
    if len({len(v) for v in values.values()}) == 1:
        a = np.stack([values[i] for i in ids])
        choices = rng.integers(len(ids), size=(repetitions, len(ids)))
        runids = rng.integers(a.shape[1], size=(repetitions, len(ids), a.shape[1]))
        boot = np.median(a[choices[..., None], runids], axis=2).mean(axis=1)
    else:
        boot = [np.mean([np.median(values[ids[j]][rng.integers(len(values[ids[j]]), size=len(values[ids[j]]))])
                         for j in rng.integers(len(ids), size=len(ids))]) for _ in range(repetitions)]
    low, high = np.percentile(boot, [2.5, 97.5])
    return estimate, float(low), float(high)


def dose_uncertainty(rows, dest, repetitions):
    keys = ["condition", "n", "optimizer", "budget_mode", "budget", "reserve_fraction", "noise_mode", "scale", "repeats"]
    curves, thresholds = [], []
    for key, group in grouped([r for r in rows if r["noise_mode"] != "exact" and r["reference"] is not None], keys).items():
        levels = sorted({r["shots"] for r in group})
        if len(levels) < 3: continue
        mapping = {(r["shots"], r["instance_id"], r["run"]): float(r["selected_reference_gap"] <= 0.01) for r in group}
        pairs = sorted(set.intersection(*[{(r["instance_id"], r["run"]) for r in group if r["shots"] == S} for S in levels]))
        ids = sorted({i for i, run in pairs})
        if not ids: continue
        nruns = min(sum(i == j for j, run in pairs) for i in ids)
        # Only complete matched panels enter threshold inference; partial runs
        # are still retained in the descriptive run tables.
        runids = {i: sorted(run for j, run in pairs if j == i)[:nruns] for i in ids}
        a = np.array([[[mapping[S, i, run] for run in runids[i]] for i in ids] for S in levels])
        raw = a.mean(axis=(1, 2))
        labels = dict(zip(keys, key)); fixed_fe = labels["budget_mode"] == "fe"
        fitted = isotonic(raw, np.ones(len(raw))) if fixed_fe else raw
        rng = np.random.default_rng(5164)
        selected_ids = rng.integers(len(ids), size=(repetitions, len(ids)))
        selected_runs = rng.integers(nruns, size=(repetitions, len(ids), nruns))
        boot = np.stack([level[selected_ids[..., None], selected_runs].mean(axis=(1, 2)) for level in a], axis=1)
        if fixed_fe: boot = np.array([isotonic(p, np.ones(len(p))) for p in boot])
        low, high = np.percentile(boot, [2.5, 97.5], axis=0)
        for S, p, fit, lo, hi in zip(levels, raw, fitted, low, high):
            curves.append({**labels, "shots": S, "success_raw": float(p), "success_fit": float(fit),
                           "cluster_bootstrap_95_low": float(lo), "cluster_bootstrap_95_high": float(hi),
                           "instances": len(ids), "matched_runs_per_instance": nruns,
                           "interval_type": "pointwise hierarchical bootstrap; no simultaneous-band claim"})
        if fixed_fe:
            label, threshold, censor = threshold_label(levels, fitted, .8)
            boot_thresholds = np.array([next((S for S, p in zip(levels, b) if p >= .8), np.inf) for b in boot])
            # Order statistics preserve right censoring instead of inventing a
            # finite extrapolated shot count above the tested grid.
            ordered = np.sort(boot_thresholds)
            lo = ordered[int(.025 * (len(ordered) - 1))]; hi = ordered[int(.975 * (len(ordered) - 1))]
            thresholds.append({**labels, "threshold_label": label, "threshold_shots": threshold,
                               "censoring": censor, "bootstrap_95_low_label": str(int(lo)) if np.isfinite(lo) else f">{max(levels)}",
                               "bootstrap_95_high_label": str(int(hi)) if np.isfinite(hi) else f">{max(levels)}",
                               "right_censored_bootstrap_fraction": float(np.mean(~np.isfinite(boot_thresholds))),
                               "target_gap": .01, "target_probability": .8,
                               "note": "operational success threshold on the tested shot grid; QAOA reference empirical"})
    write_csv(dest / "shot_response_uncertainty.csv", curves)
    write_csv(dest / "shot_threshold_uncertainty.csv", thresholds)
    return curves


def landscape_tables(out, dest):
    landscape_rows, line_rows, pair_rows, hessian_rows = [], [], [], []
    for path in sorted((out / "landscapes").glob("*.json")):
        d = json.loads(path.read_text()); p = d["problem"]
        label = dict(problem_id=p["problem_id"], n=p["hamiltonian"]["n"], dim=p["dim"], shear=p["shear"])
        dg = d.get("discrete_geometry", {})
        landscape_rows.append({**label, "reference": d["reference"], "reference_certified": d["reference_certified"],
            "empirical_endpoint_clusters": d["endpoint_clusters"], "stationary_endpoints": d["stationary_endpoints"],
            "quench_starts": d["quench_starts"], "basin_good_fraction": d["basin_good_fraction"],
            "endpoint_quality_iqr": d["endpoint_quality_iqr"], "gradient_norm_median": d["grad_norm_median"],
            **{k: dg.get(k) for k in ("strict_one_flip_minima", "weak_one_flip_minima", "discrete_ground_basin_fraction",
                                     "discrete_minimum_quality_iqr", "minimum_flip_gap")}})
        line_rows.extend({**label, **r} for r in d["line_diagnostics"])
        pair_rows.extend({**label, **r} for r in d["misranking_probes"])
        for probe in d["hessian_probes"]:
            hessian_rows.append({**label, "energy": probe["energy"], **{k:v for k,v in probe["hessian"].items() if k != "eigenvalues"}})
    write_csv(dest / "landscape_features.csv", landscape_rows)
    write_csv(dest / "ruggedness_probes.csv", line_rows)
    write_csv(dest / "misranking_probes.csv", pair_rows)
    write_csv(dest / "hessian_features.csv", hessian_rows)


def plots(out, rows, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["condition", "n", "noise", "budget_mode", "budget", "reserve_fraction"]
    curve_rows, diagnostic_rows = [], []
    db = sqlite3.connect(f"file:{(out / 'records.sqlite').resolve()}?mode=ro", uri=True) if (out / "records.sqlite").exists() else None
    colors = {name: plt.get_cmap("tab20")(i % 20) for i, name in enumerate(sorted({r["optimizer"] for r in rows}))}
    try:
        for counter, (key, group) in enumerate(grouped(rows, keys).items()):
            labels = dict(zip(keys, key))
            horizon = max(r["search_fe_ceiling"] for r in group)
            grid = np.unique(np.geomspace(1, horizon, min(horizon, 100)).astype(int))
            curves = {}
            for r in group:
                if db:
                    blob = db.execute("SELECT trace FROM records WHERE id=?", (r["record_id"],)).fetchone()[0]
                    with np.load(io.BytesIO(blob), allow_pickle=False) as z: trace = {k:z[k] for k in z.files}
                else:
                    with np.load(out / r["trace_file"], allow_pickle=False) as z: trace = {k:z[k] for k in z.files}
                checkpoints = trace.get("checkpoint_fes", np.arange(1,len(trace["oracle_curve"])+1))
                window = max(1,int(.9*r["search_fes"]))
                previous = max(0,int(np.searchsorted(checkpoints,window,side="right")-1))
                diagnostic_rows.append({k:r[k] for k in keys+ ["instance_id","run","optimizer"]} |
                    {"actual_search_fes":r["search_fes"],"stored_window_start_fe":int(checkpoints[previous]),
                     "oracle_improvement_final_stored_window":float(trace["oracle_curve"][previous]-trace["oracle_curve"][-1]),
                     "incumbent_age_fes":r["incumbent_age_fes"],
                     "note":"assessment-only stagnation proxy; no proof of optimizer trapping"} |
                     r["comparison_diagnostics"])
                for metric in ("oracle_curve", "raw_selected_curve"):
                    curves.setdefault((r["optimizer"], metric, r["instance_id"]), []).append(curve_at(trace, metric, grid))
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharex=True, sharey=True)
            for name in sorted({r["optimizer"] for r in group}):
                for ax, metric in zip(axes, ("oracle_curve", "raw_selected_curve")):
                    instances = sorted(i for opt, m, i in curves if opt == name and m == metric)
                    # A within-instance run median, then an equal-instance mean.
                    instance_curves = np.array([np.median(curves[name, metric, i], axis=0) for i in instances])
                    mean = instance_curves.mean(axis=0)
                    ax.plot(grid, mean, color=colors[name], label=name, lw=1.3)
                    for fe, value in zip(grid, mean):
                        curve_rows.append({**labels, "optimizer": name, "metric": metric, "available_search_fe": int(fe),
                                           "equal_instance_mean_of_run_medians": float(value), "instances": len(instances)})
            for ax, title in zip(axes, ("Exact oracle tracking", "Noisy-observation incumbent")):
                ax.set(xscale="log", xlabel="Available search FE budget", title=title); ax.grid(alpha=.2)
            axes[0].set_ylabel("Normalized exact energy")
            axes[1].legend(fontsize=7, ncol=2)
            fig.suptitle(f"{labels['condition']}, n={labels['n']}, {labels['noise']}, B={labels['budget']}, reserve={labels['reserve_fraction']}", fontsize=9)
            fig.tight_layout()
            filename = f"convergence_{counter:03d}"
            fig.savefig(dest / (filename + ".pdf")); fig.savefig(dest / (filename + ".png"), dpi=140); plt.close(fig)
            labels["figure"] = filename
            with (dest / "figure_index.jsonl").open("a") as f: f.write(json.dumps(labels) + "\n")
    finally:
        if db: db.close()
    write_csv(dest / "convergence_aggregates.csv", curve_rows)
    write_csv(dest / "stagnation_and_comparison_diagnostics.csv", diagnostic_rows)


def endpoint_and_dose_plots(uncertainty, dose_curves, dest):
    import matplotlib.pyplot as plt
    keys=["condition","n","budget_mode","budget","reserve_fraction"]
    sections=[]
    for key,group in grouped(uncertainty,keys).items():
        if len({r["noise"] for r in group})>5:
            sections.extend((key+mode,subset) for mode,subset in grouped(group,["noise_mode"]).items())
        else:sections.append((key,group))
    for counter,(key,group) in enumerate(sections):
        fig,axes=plt.subplots(1,2,figsize=(11,4.2))
        names=sorted({r["optimizer"] for r in group}); noises=sorted({r["noise"] for r in group})
        color=plt.get_cmap("tab10")
        for ax,metric in zip(axes,["selected_exact","search_selection_gap"]):
            for k,noise in enumerate(noises):
                shift=(k-(len(noises)-1)/2)*.65/max(1,len(noises))
                for j,name in enumerate(names):
                    matches=[r for r in group if r["optimizer"]==name and r["noise"]==noise and r["metric"]==metric]
                    if not matches:continue
                    r=matches[0];x=j+shift
                    ax.scatter(x,r["estimate"],s=13,color=color(k%10),label=noise if j==0 else None)
                    ax.vlines(x,r["cluster_bootstrap_95_low"],r["cluster_bootstrap_95_high"],color=color(k%10),lw=.8)
            ax.set(xticks=range(len(names)),xticklabels=names,ylabel=metric.replace('_',' '))
            ax.tick_params(axis="x",labelrotation=65);ax.grid(axis="y",alpha=.2)
        axes[1].legend(fontsize=6,ncol=2)
        mode="" if len(key)==5 else ", "+key[5]
        fig.suptitle(f"{key[0]}, n={key[1]}, {key[2]} B={key[3]}, reserve={key[4]}{mode}\nEqual-instance endpoint estimates and 95% hierarchical bootstrap intervals",fontsize=9)
        fig.tight_layout();fig.savefig(dest/f"endpoints_{counter:03d}.pdf");plt.close(fig)
    keys=["condition","n","budget_mode","budget","reserve_fraction","noise_mode"]
    for counter,(key,group) in enumerate(grouped(dose_curves,keys).items()):
        fig,ax=plt.subplots(figsize=(7.5,4))
        for name in sorted({r["optimizer"] for r in group}):
            selected=sorted([r for r in group if r["optimizer"]==name],key=lambda r:r["shots"])
            S=[r["shots"] for r in selected];p=[r["success_fit"] for r in selected]
            ax.plot(S,p,marker='o',ms=3,lw=1,label=name)
            ax.fill_between(S,[r["cluster_bootstrap_95_low"] for r in selected],[r["cluster_bootstrap_95_high"] for r in selected],alpha=.08)
        ax.axhline(.8,color='black',ls=':',lw=.8)
        ax.set(xscale='log',xlabel='Shots per observation',ylabel='Probability of selected reference gap <= 0.01',ylim=(-.03,1.03),
               title=f"{key[0]}, n={key[1]}, {key[2]} B={key[3]}, {key[5]}")
        ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(dest/f"shot_response_{counter:03d}.pdf");plt.close(fig)


def niche_tables(rows, dest, repetitions):
    subset = [r for r in rows if r.get("niching")]
    if not subset: return
    metrics = ["attained_minima", "catalogue_coverage", "archive_precision", "strict_attained_minima",
               "diagnostic_basin_access", "diagnostic_basin_entropy", "diagnostic_unmatched_minima"]
    keys = ["condition", "n", "noise", "optimizer"]
    summaries, curve_rows, run_rows = [], [], []
    for r in subset:
        run_rows.append({k:r[k] for k in ["problem_id", "condition", "n", "instance_id", "run", "noise", "optimizer"]} |
                        {k:r["niching"].get(k) for k in metrics + ["catalogue_size", "catalogue_certified", "all_catalogued_attained", "offline_seconds"]})
        curve_rows.extend({k:r[k] for k in keys + ["instance_id", "run"]} | c for c in r["niching"]["curve"])
    for key, group in grouped(run_rows, keys).items():
        for metric in metrics:
            valid = [r for r in group if r[metric] is not None]
            if not valid: continue
            estimate, low, high = cluster_interval(valid, metric, repetitions)
            summaries.append({**dict(zip(keys, key)), "metric": metric, "equal_instance_mean_of_medians": estimate,
                              "cluster_bootstrap_95_low": low, "cluster_bootstrap_95_high": high,
                              "instances": len({r["instance_id"] for r in valid}), "runs": len(valid)})
    write_csv(dest / "niching_runs.csv", run_rows); write_csv(dest / "niching_summary.csv", summaries)
    write_csv(dest / "niching_curves.csv", curve_rows)
    import matplotlib.pyplot as plt
    for counter, (key, group) in enumerate(grouped(summaries, ["condition", "n", "noise"]).items()):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ["catalogue_coverage", "archive_precision"]):
            points = [r for r in group if r["metric"] == metric]
            y = [r["equal_instance_mean_of_medians"] for r in points]
            error = np.array([[v-r["cluster_bootstrap_95_low"], r["cluster_bootstrap_95_high"]-v] for r, v in zip(points, y)]).T
            # Percentile bootstrap intervals need not straddle the point estimate.
            ax.scatter(range(len(points)), y, s=20)
            for j, r in enumerate(points): ax.vlines(j, r["cluster_bootstrap_95_low"], r["cluster_bootstrap_95_high"])
            ax.set(xticks=range(len(points)), xticklabels=[r["optimizer"] for r in points], ylabel=metric.replace("_", " "), ylim=(-.02,1.02))
            ax.tick_params(axis="x", labelrotation=60); ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"{key[0]}, n={key[1]}, {key[2]}: minimum attainment")
        fig.tight_layout(); fig.savefig(dest / f"niching_{counter:03d}.pdf"); plt.close(fig)


def stage_report(out, repetitions=2000, expected=None):
    out = Path(out); rows = records(out)
    if not rows: return {"completed_runs":0, "complete":False, "expected_runs":expected}
    result = analyze(out, repetitions)
    dest = out / "analysis"
    index = dest / "figure_index.jsonl"
    if index.exists(): index.unlink()
    audit = []
    for r in rows:
        if r["search_fes"] > r["search_fe_ceiling"] or r["search_shots"] + r["validation_shots"] > r["total_shot_ceiling"]:
            audit.append(r.get("record_id", r["trace_file"]))
    result.update(complete=len(rows) == expected if expected is not None else False,
                  expected_runs=expected, budget_violations=audit)
    write_csv(dest / "budget_audit.csv", [{"completed_runs":len(rows), "expected_runs":expected,
               "budget_violations":len(audit), "stage_complete":result["complete"]}])
    uncertainty = []
    keys = ["condition", "n", "noise", "budget_mode", "budget", "reserve_fraction", "optimizer"]
    for key, group in grouped(rows, keys).items():
        for metric in ("selected_exact", "oracle_best_exact", "search_selection_gap", "winner_optimism"):
            estimate, low, high = cluster_interval(group, metric, repetitions)
            uncertainty.append({**dict(zip(keys,key)), "metric":metric, "estimate":estimate,
                               "noise_mode":group[0]["noise_mode"],
                               "cluster_bootstrap_95_low":low, "cluster_bootstrap_95_high":high,
                               "instances":len({r["instance_id"] for r in group}), "runs":len(group)})
    write_csv(dest / "endpoint_uncertainty.csv", uncertainty)
    dose_curves=dose_uncertainty(rows, dest, repetitions)
    landscape_tables(out, dest)
    plots(out, rows, dest)
    endpoint_and_dose_plots(uncertainty,dose_curves,dest)
    niche_tables(rows, dest, repetitions)
    with gzip.open(dest / "complete_records.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, separators=(",", ":"), allow_nan=False)+"\n")
    (dest / "report_status.json").write_text(json.dumps(result, indent=2))
    return result


def tex_escape(s):
    table = {"&":r"\&", "%":r"\%", "$":r"\$", "#":r"\#", "_":r"\_", "{":r"\{", "}":r"\}"}
    return "".join(table.get(c,c) for c in str(s))


def latex_report(state, status):
    dest = Path(state) / "manuscript_outputs"; dest.mkdir(exist_ok=True)
    lines = [r"\documentclass[11pt]{article}", r"\usepackage[margin=25mm]{geometry}",
             r"\usepackage{booktabs,longtable,graphicx,hyperref,pdflscape}", r"\title{Global and Noisy Optimization of Variational Quantum Algorithms: Execution Report}",
             r"\date{\today}\begin{document}\maketitle",
             "Execution status: " + ("COMPLETE." if status["complete"] else r"\textbf{PARTIAL --- not a completed benchmark.}"),
             "This report contains computed summaries. QAOA references and minimum catalogues are empirical; signed reference improvements remain in the data.",
             r"\begin{longtable}{lrrr}\toprule Stage & Complete runs & Planned runs & Budget violations\\\midrule"]
    for stage, item in status["stages"].items():
        lines.append(f"{tex_escape(stage)} & {item['completed_runs']} & {item['expected_runs']} & {len(item.get('budget_violations',[]))} " + r"\\")
    lines += [r"\bottomrule\end{longtable}",
              "Independent optimizer runs share deterministic disorder instances and seed labels. Primary comparisons use paired hierarchical bootstrap resampling at instance and run levels.",
              "Noise-created sampled extrema are distinguished from minima of a persistent latent mean. Niche attainment requires both state fidelity and energy proximity; offline quenches measure basin access only."]
    for stage in status["stages"]:
        path = Path(state) / "results" / stage / "analysis" / "summary.csv"
        if not path.exists(): continue
        with path.open() as f: rows = list(csv.DictReader(f))
        lines += [r"\clearpage\begin{landscape}\section*{" + tex_escape(stage) + "}",
                  r"\scriptsize\begin{longtable}{p{45mm}p{35mm}p{28mm}rrrr}\toprule Model & Noise & Optimizer & $n$ & Reserve & Oracle & Selected\\\midrule"]
        for r in rows:
            a = float(r["equal_instance_mean_of_medians_oracle_best_exact"])
            b = float(r["equal_instance_mean_of_medians_selected_exact"])
            lines.append(" & ".join([tex_escape(r[k]) for k in ["condition","noise","optimizer","n","reserve_fraction"]] + [f"{a:.4f}",f"{b:.4f}"]) + r"\\")
        lines += [r"\bottomrule\end{longtable}\normalsize\end{landscape}"]
    lines += [r"\end{document}"]
    (dest / "results_report.tex").write_text("\n".join(lines)+"\n")
    inventory = [r"\section{Computed experiment inventory}",
                 "Execution status: " + ("COMPLETE." if status["complete"] else r"\textbf{PARTIAL.}"),
                 "Completed runs: " + str(sum(r["completed_runs"] for r in status["stages"].values())) + ".",
                 "The separate execution report and CSV tables contain the computed summaries. Optimizer conclusions require interpretation of paired contrasts, geometry controls, and implementation limits."]
    (dest / "results_summary.tex").write_text("\n".join(inventory)+"\n")
    return dest
