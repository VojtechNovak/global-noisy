"""Plan, independently characterize, and run resumable experiment cells.

Usage: python -m gnvqa.run --config configs/smoke.json --out results/smoke --phase all
Run a full experiment explicitly; the default action only prints its size.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import io
import tempfile
import numpy as np
import scipy
from .models import build_problem, stable_seed
from .noise import Noise, Recorder, FrozenField, verify
from .optimizers import optimize
from .characterize import characterize


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name,
                                     suffix=".tmp", delete=False, encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        temp = Path(f.name)
    temp.replace(path)


def source_hash():
    return hashlib.sha256(b"".join(p.read_bytes() for p in
                          sorted(Path(__file__).parent.rglob("*.py")))).hexdigest()


def cell_tag(problem_id, noise_spec, mode, budget, reserve, run, name):
    signature = [problem_id, noise_spec, mode, budget, reserve, run, name]
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def problems(cfg):
    for spec in cfg["problems"]:
        for n in spec.get("sizes", cfg["sizes"]):
            for instance in cfg["instance_ids"]:
                disorder_seed = stable_seed(cfg["seed"], "disorder", n, instance)
                yield spec, n, instance, build_problem(spec, n, disorder_seed)


def cell_specs(cfg):
    for spec, n, instance, p in problems(cfg):
        for noise_spec in cfg["noise"]:
            for mode in cfg.get("budget_modes", ["fe"]):
                budgets = cfg["fe_budgets"] if mode == "fe" else cfg["shot_budgets"]
                if mode == "shots" and noise_spec["mode"] == "exact":
                    continue
                for budget in budgets:
                    for reserve in cfg.get("reserve_fractions", [0]):
                        if noise_spec["mode"] == "exact" and reserve:
                            continue
                        for run in range(cfg["runs"]):
                            for optimizer in cfg["optimizers"]:
                                yield spec, n, instance, p, noise_spec, mode, budget, reserve, run, optimizer


def plan(cfg):
    tasks = 0; fes = 0; shots = 0
    for _, _, _, _, ns, mode, budget, reserve, _, _ in cell_specs(cfg):
        noise = Noise(**{k: v for k, v in ns.items() if k != "name"})
        total_fes = int(budget) if mode == "fe" else int(budget // noise.cost)
        total_shots = total_fes * noise.cost if mode == "fe" else int(budget)
        tasks += 1; fes += int(total_fes * (1 - reserve)); shots += total_shots
    return dict(problem_conditions=sum(1 for _ in problems(cfg)), optimizer_runs=tasks,
                search_fe_ceiling_total=fes, shot_ceiling_total=shots,
                note="Ceilings exclude offline characterization/reference/assessment simulation time.")


def comparison_diagnostics(recorder, events):
    # Coordinates index deterministic exact values; no reevaluations or oracle
    # lookups happen inside optimizer execution.
    lookup = {np.round(x, 12).tobytes(): (t, l)
              for x, t, l in zip(recorder.x, recorder.truth, recorder.latent)}
    comparisons, wrong_accept, wrong_reject = 0, 0, 0
    for e in events:
        if e["kind"] != "comparison":
            continue
        a = lookup.get(np.round(e["parent"], 12).tobytes())
        b = lookup.get(np.round(e["trial"], 12).tobytes())
        if a is None or b is None:
            continue
        comparisons += 1
        # In the persistent-field control, compare to its latent mean.
        wrong_accept += bool(e["accepted"] and b[1] > a[1] + 1e-12)
        wrong_reject += bool(not e["accepted"] and b[1] < a[1] - 1e-12)
    return dict(instrumented_comparisons=comparisons,
                accepted_latent_worsening=wrong_accept,
                rejected_latent_improvement=wrong_reject,
                comparison_note="jSO comparisons only; other methods are not given fabricated counts")


def run_cell(cfg, cell, out, storage="full", keep_events=True, landscape_dir=None):
    spec, n, instance, p, noise_spec, mode, budget, reserve, run, name = cell
    cid = config_hash(cfg)
    tag = cell_tag(p.problem_id, noise_spec, mode, budget, reserve, run, name)
    dest = out / "runs" / f"{cid}_{tag}.json"
    if dest.exists():
        previous = json.loads(dest.read_text())
        trace = out / previous.get("trace_file", "missing")
        if trace.is_file():
            try:
                with np.load(trace, allow_pickle=False) as z:
                    _ = z["oracle_curve"]
                return previous
            except (OSError, ValueError, KeyError):
                pass  # Interrupted trace writes are replayed with the original seeds.
    noise = Noise(**{k: v for k, v in noise_spec.items() if k != "name"})
    total_shots = int(budget * noise.cost if mode == "fe" else budget)
    search_fes = int(budget * (1 - reserve)) if mode == "fe" else int(
        total_shots * (1 - reserve) // noise.cost)
    if search_fes < 1:
        raise ValueError("Budget/reserve/shot count leaves zero search FEs.")
    reserved = total_shots - search_fes * noise.cost if reserve else 0
    field = None
    if spec.get("field"):
        field = FrozenField(p.dim, **spec["field"],
                            seed=stable_seed(cfg["seed"], "fixed_field", p.problem_id))
    optimizer_seed = stable_seed(cfg["seed"], "optimizer", p.problem_id, run)
    # Distinct streams per method and shot condition. No claim of pointwise
    # common-random-number pairing across adaptive trajectories is made.
    noise_seed = stable_seed(cfg["seed"], "noise", p.problem_id, run, name, noise_spec)
    recorder = Recorder(p, noise, search_fes, noise_seed, field)
    options = cfg.get("optimizer_options", {}).get(name, {})
    start = time.perf_counter()
    result = optimize(name, recorder.objective_only(), p.dim, search_fes, optimizer_seed, options)
    elapsed_search = time.perf_counter() - start
    verification = verify(recorder, reserved, cfg.get("shortlist_k", 5),
                           stable_seed(noise_seed, "independent_verify")) if reserved else None
    arrays = recorder.arrays()
    observed_id, oracle_id = int(np.argmin(arrays["observed"])), int(np.argmin(arrays["truth"]))
    ref_file = (Path(landscape_dir) if landscape_dir else out / "landscapes") / f"{p.problem_id}.json"
    landscape = json.loads(ref_file.read_text()) if ref_file.exists() else {}
    if cfg.get("niching") and landscape_dir:
        catalogue_file = Path(landscape_dir).parent / "catalogues" / (p.problem_id + ".json")
        if not catalogue_file.is_file(): raise ValueError("Independent frozen niche catalogue is missing")
        landscape["niching_catalogue"] = json.loads(catalogue_file.read_text())
    reference = 0.0 if p.ansatz == "product" else landscape.get("reference")
    if landscape and landscape["problem"] != json.loads(json.dumps(p.metadata())):
        raise ValueError("Reference Hamiltonian/ansatz/coordinate metadata mismatch.")
    events = result.pop("events", [])
    native = p.objective(np.array(result["x"])) if result.get("x") is not None else None
    chosen = verification["index"] if verification else observed_id
    oracle_truth, raw_truth = float(arrays["truth"][oracle_id]), float(arrays["truth"][observed_id])
    chosen_truth = float(arrays["truth"][chosen])
    record = dict(config_hash=cid, problem_id=p.problem_id, family=spec["family"],
                  condition=spec["name"], n=n, dim=p.dim, depth=p.depth, blocks=p.blocks,
                  instance_id=instance, fingerprint=p.h.fingerprint(), optimizer=name,
                  run=run, optimizer_seed=optimizer_seed, noise_seed=noise_seed,
                  noise=noise_spec["name"], noise_mode=noise.mode, shots=noise.shots,
                  repeats=noise.repeats, scale=noise.scale, budget_mode=mode, budget=budget,
                  reserve_fraction=reserve, search_fe_ceiling=search_fes,
                  search_fes=recorder.nfev, search_shots=recorder.nfev * noise.cost,
                  validation_shots=verification["validation_shots"] if verification else 0,
                  total_shot_ceiling=total_shots,
                  oracle_best_exact=oracle_truth, raw_selected_exact=raw_truth,
                  selected_exact=chosen_truth, native_recommendation_exact=native,
                  search_selection_gap=chosen_truth - oracle_truth,
                  raw_search_selection_gap=raw_truth - oracle_truth,
                  winner_optimism=raw_truth - float(arrays["observed"][observed_id]),
                  raw_observed_min=float(arrays["observed"][observed_id]),
                  incumbent_age_fes=int(recorder.nfev - observed_id - 1),
                  oracle_index=oracle_id, raw_selected_index=observed_id, selected_index=chosen,
                  best_latent_queried=float(np.min(arrays["latent"])),
                  reference=reference, reference_certified=p.ansatz == "product",
                  oracle_reference_gap=oracle_truth - reference if reference is not None else None,
                  selected_reference_gap=chosen_truth - reference if reference is not None else None,
                  reference_improved=bool(reference is not None and oracle_truth < reference - 1e-7),
                  verification=verification, optimizer_result=result,
                  comparison_diagnostics=comparison_diagnostics(recorder, events),
                  search_wall_seconds=elapsed_search, total_wall_seconds=time.perf_counter() - start,
                  frozen_field=field.metadata() if field is not None else None,
                  assessment_note="oracle truth never used by optimizer, shortlist, or verification selection")
    if cfg.get("niching"):
        from .niching import assess_trace
        if not landscape:
            raise ValueError("Niching assessment requires an independent catalogue.")
        record["niching"] = assess_trace(p, arrays, landscape, cfg["niching"],
                                        stable_seed(cfg["seed"], "niche_assessment", p.problem_id, run), search_fes)
    if record["search_fes"] > search_fes or record["search_shots"] + record["validation_shots"] > total_shots:
        raise AssertionError("Budget accounting failed.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    trace_file = dest.with_suffix(".npz")
    running_min = np.minimum.accumulate(arrays["observed"])
    improved = np.r_[True, arrays["observed"][1:] < running_min[:-1]]
    selected_ids = np.maximum.accumulate(np.where(improved, np.arange(recorder.nfev), 0))
    oracle_curve = np.minimum.accumulate(arrays["truth"])
    selected_curve = arrays["truth"][selected_ids]
    if storage == "compact":
        # Logarithmic checkpoints include FE 1 and the final charged FE. Native
        # early stops hold their final recommendation when comparing larger B.
        checkpoint = np.unique(np.r_[0, recorder.nfev - 1,
                      np.geomspace(1, recorder.nfev, min(100, recorder.nfev)).astype(int) - 1])
        trace_arrays = dict(checkpoint_fes=checkpoint + 1, oracle_curve=oracle_curve[checkpoint],
                            raw_selected_curve=selected_curve[checkpoint],
                            observed_curve=running_min[checkpoint],
                            oracle_x=arrays["x"][oracle_id], selected_x=arrays["x"][chosen],
                            raw_selected_x=arrays["x"][observed_id])
    elif storage == "full":
        trace_arrays = dict(**arrays, oracle_curve=oracle_curve, raw_selected_curve=selected_curve)
    else:
        raise ValueError("Unknown storage policy.")
    buf = io.BytesIO(); np.savez_compressed(buf, **trace_arrays)
    temp = trace_file.with_suffix(".npz.tmp")
    temp.write_bytes(buf.getvalue()); temp.replace(trace_file)
    if events and keep_events:
        write_json(dest.with_suffix(".events.json"), events)
    record["trace_file"] = f"runs/{trace_file.name}"
    record["trace_storage"] = storage
    record["comparison_events_retained"] = bool(events and keep_events)
    record["total_wall_seconds"] = time.perf_counter() - start
    write_json(dest, record)
    return record


def manifest(cfg, out):
    write_json(out / "manifest.json", dict(config=cfg, config_hash=config_hash(cfg),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                code_sha256=source_hash(), backend="NumPy; product analytical; diagonal QAOA statevector",
                thread_settings={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                               "MKL_NUM_THREADS")}, plan=plan(cfg)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--out", default="results/smoke")
    parser.add_argument("--phase", choices=["plan", "characterize", "benchmark", "all"], default="plan")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--problem", help="Optional substring filter for problem_id")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    print(json.dumps(plan(cfg), indent=2), flush=True)
    if args.phase == "plan":
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    existing = out / "manifest.json"
    if existing.exists() and json.loads(existing.read_text())["config_hash"] != config_hash(cfg):
        raise ValueError("Use a separate output directory for each immutable configuration.")
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.rglob("*.py")))).hexdigest()
    if existing.exists() and json.loads(existing.read_text())["code_sha256"] != code_hash:
        raise ValueError("Source code changed; use a new output directory to keep runs reproducible.")
    manifest(cfg, out)
    if args.phase in {"characterize", "all"}:
        for _, _, _, p in problems(cfg):
            if args.problem and args.problem not in p.problem_id:
                continue
            dest = out / "landscapes" / f"{p.problem_id}.json"
            if not dest.exists():
                start = time.perf_counter()
                data = characterize(p, cfg.get("characterization", {}),
                                    stable_seed(cfg["seed"], "characterize", p.problem_id))
                data["offline_wall_seconds"] = time.perf_counter() - start
                write_json(dest, data)
            print("characterized", p.problem_id, flush=True)
    if args.phase in {"benchmark", "all"}:
        completed = 0
        for cell in cell_specs(cfg):
            if args.problem and args.problem not in cell[3].problem_id:
                continue
            if args.max_tasks is not None and completed >= args.max_tasks:
                break
            r = run_cell(cfg, cell, out)
            completed += 1
            print(f'{completed}: {r["problem_id"]} {r["noise"]} {r["optimizer"]} '
                  f'FE={r["search_fes"]} oracle={r["oracle_best_exact"]:.5g} '
                  f'selected={r["selected_exact"]:.5g}', flush=True)
    print("Output:", out.resolve())


if __name__ == "__main__":
    main()
