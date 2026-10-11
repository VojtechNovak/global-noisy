"""Bounded one-node worker pool, resumable independent cells, reference reuse.

No MPI and no optimizer parallelism: only independent scientific runs execute
concurrently. Shared outputs have exactly one writer protected by a suite lock.
"""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import contextlib
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import tempfile
import time
import traceback
import platform
import numpy as np
import scipy
from .run import problems, cell_specs, cell_tag, config_hash, source_hash, manifest, plan, run_cell, write_json
from .models import build_problem, stable_seed
from .characterize import characterize
from .storage import open_store, insert_record


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def reference_key(problem, settings):
    return hashlib.sha256(canonical([problem.metadata(), settings]).encode()).hexdigest()[:24]


def load_suite(path):
    path = Path(path).resolve()
    root = path.parent.parent
    suite = json.loads(path.read_text())
    stages = [(s["name"], json.loads((root / s["config"]).read_text())) for s in suite["stages"]]
    if len({s[0] for s in stages}) != len(stages): raise ValueError("Stage names must be unique")
    return root, suite, stages


def input_hash(root):
    h=hashlib.sha256()
    for p in sorted((root / "inputs").rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root / "inputs")).encode());h.update(p.read_bytes())
    return h.hexdigest()


def suite_signature(root,suite,stages):
    return hashlib.sha256(canonical([suite,stages,source_hash(),input_hash(root)]).encode()).hexdigest()


def describe(suite_path, workers=30):
    root, suite, stages = load_suite(suite_path)
    pilot = json.loads((root / "inputs/pilot_summary.json").read_text())["pilot"]["timings"]
    stages_out, total = [], 0.
    for name, cfg in stages:
        hours = 0.
        for spec, n, instance, p, ns, mode, B, reserve, run, opt in cell_specs(cfg):
            model = "Q1" if p.ansatz == "product" else ("QAOA_glass_p3" if p.depth == 3 else "QAOA_glass_p6_tied")
            seconds_per_fe = pilot.get(f"{model}/{n}/{opt}", pilot.get(f"{model}/{n}/COBYLA", 0.001))
            cost = 0 if ns["mode"] == "exact" else ns["shots"] * ns.get("repeats", 1)
            fes = int(B * (1 - reserve)) if mode == "fe" else int(B * (1 - reserve) // cost)
            hours += seconds_per_fe * fes / 3600
        total += hours
        stages_out.append(dict(stage=name, config_hash=config_hash(cfg), **plan(cfg),
                        pilot_based_cpu_hour_ceiling=hours))
    return dict(stages=stages_out, total_runs=sum(s["optimizer_runs"] for s in stages_out),
                pilot_based_cpu_hour_ceiling=total, pilot_based_ideal_node_hours=total / workers,
                workers=workers, note="FE ceilings extrapolated from local pilots; excludes physical-shot sampling overhead, new characterization, niching quenches, database I/O and analysis. Not a guaranteed runtime.")


@functools.lru_cache(maxsize=8)
def cached_problem(spec_json, n, seed):
    return build_problem(json.loads(spec_json), n, seed)


def worker_characterize(payload):
    spec, n, instance, cfg = payload
    seed = stable_seed(cfg["seed"], "disorder", n, instance)
    p = cached_problem(canonical(spec), n, seed)
    start = time.perf_counter()
    data = characterize(p, cfg.get("characterization", {}), stable_seed(cfg["seed"], "characterize", p.problem_id))
    data["offline_wall_seconds"] = time.perf_counter() - start
    data["cluster_provenance"] = dict(code_sha256=source_hash(), characterization=cfg.get("characterization", {}))
    return data


def worker_cell(payload):
    cfg, descriptor, landscape_dir, temp_root, storage = payload
    spec, n, instance, ns, mode, B, reserve, run, optimizer = descriptor
    p = cached_problem(canonical(spec), n, stable_seed(cfg["seed"], "disorder", n, instance))
    with tempfile.TemporaryDirectory(dir=temp_root, prefix="cell-") as temp:
        record = run_cell(cfg, (spec, n, instance, p, ns, mode, B, reserve, run, optimizer),
                          Path(temp), storage=storage, keep_events=False, landscape_dir=landscape_dir)
        trace = (Path(temp) / record["trace_file"]).read_bytes()
    return record, trace


def worker_catalogue(payload):
    spec, n, instance, cfg, reference_file = payload
    p = cached_problem(canonical(spec), n, stable_seed(cfg["seed"], "disorder", n, instance))
    from .niching import Catalogue
    start = time.perf_counter()
    data = Catalogue(p, json.loads(Path(reference_file).read_text()), cfg["niching"]).metadata()
    data["offline_seconds"] = time.perf_counter() - start
    data["reference_file_sha256"] = hashlib.sha256(Path(reference_file).read_bytes()).hexdigest()
    return data


class Deadline:
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds
        self.stopped = False
        signal.signal(signal.SIGTERM, self.signal)
        signal.signal(signal.SIGINT, self.signal)
        if hasattr(signal, "SIGUSR1"): signal.signal(signal.SIGUSR1, self.signal)

    def signal(self, signum, frame):
        del signum, frame
        self.stopped = True

    def available(self):
        return not self.stopped and time.monotonic() < self.end


def bounded_pool(pool, items, worker, deadline, max_inflight):
    """Yield successful results; submit only a bounded number of small payloads."""
    items = iter(items); pending = {}; exhausted = False
    while pending or (not exhausted and deadline.available()):
        while len(pending) < max_inflight and not exhausted and deadline.available():
            try: key, payload = next(items)
            except StopIteration: exhausted = True; break
            pending[pool.submit(worker, payload)] = key
        if not pending: break
        ready, _ = futures.wait(pending, timeout=1, return_when=futures.FIRST_COMPLETED)
        for future in ready:
            key = pending.pop(future)
            yield key, future.result()


def initialize(root, suite, stages, state):
    state.mkdir(parents=True, exist_ok=True)
    signature = suite_signature(root,suite,stages)
    previous = state / "suite_manifest.json"
    if previous.exists():
        if json.loads(previous.read_text())["signature"] != signature:
            raise ValueError("Suite, configuration or code changed. Use a new --state directory.")
    else:
        write_json(previous, dict(signature=signature, suite=suite, configs=dict(stages), code_sha256=source_hash(), reference_inputs_sha256=input_hash(root),
                                  imported_input_root=str(root / "inputs"), created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    cache = state / "reference_cache"; cache.mkdir(exist_ok=True)
    imported = 0
    for relative in suite.get("import_roots", []):
        source = root / relative
        m = json.loads((source / "manifest.json").read_text())
        cfg = m["config"]
        for spec, n, instance, p in problems(cfg):
            src = source / "landscapes" / (p.problem_id + ".json")
            if not src.is_file(): continue
            data = json.loads(src.read_text())
            if data["problem"] != json.loads(json.dumps(p.metadata())):
                raise ValueError("Uploaded characterization metadata mismatch: " + str(src))
            data["import_provenance"] = dict(source=str(src.relative_to(root)), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                         original_code_sha256=m["code_sha256"], original_config_hash=m["config_hash"],
                         original_python=m["python"], original_numpy=m["numpy"], original_scipy=m["scipy"],
                         characterization=cfg.get("characterization", {}))
            target = cache / (reference_key(p, cfg.get("characterization", {})) + ".json")
            if not target.exists():
                # Imported metadata alone would miss a changed simulator kernel.
                for endpoint in data.get("quench_endpoints", []):
                    if abs(p.objective(endpoint["x"])-endpoint["energy"])>1e-9:
                        raise ValueError("Imported endpoint energy disagrees with the current simulator: "+str(src))
                write_json(target, data); imported += 1
    for name, cfg in stages:
        out = state / "results" / name; out.mkdir(parents=True, exist_ok=True)
        if not (out / "manifest.json").exists(): manifest(cfg, out)
        else:
            m=json.loads((out / "manifest.json").read_text())
            current=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
            if any(m[k]!=v for k,v in current.items()): raise ValueError("Numerical environment changed; use a fresh state directory")
    return imported


def stage_references(cfg, state, out, pool, deadline, workers):
    cache = state / "reference_cache"
    settings = cfg.get("characterization", {})
    all_problems = list(problems(cfg))
    def missing():
        for spec, n, instance, p in all_problems:
            key = reference_key(p, settings)
            if not (cache / (key + ".json")).is_file(): yield key, (spec, n, instance, cfg)
    for key, data in bounded_pool(pool, missing(), worker_characterize, deadline, 2 * workers):
        write_json(cache / (key + ".json"), data)
        print("reference", data["problem"]["problem_id"], flush=True)
    target_dir = out / "landscapes"; target_dir.mkdir(exist_ok=True)
    complete = True
    for _, _, _, p in all_problems:
        source = cache / (reference_key(p, settings) + ".json")
        if not source.exists(): complete = False; continue
        data = json.loads(source.read_text())
        if data["problem"] != json.loads(json.dumps(p.metadata())): raise ValueError("Cached reference mismatch")
        dest = target_dir / (p.problem_id + ".json")
        if not dest.exists(): shutil.copy2(source, dest)
    return complete


def run_stage(name, cfg, suite, state, pool, deadline, workers, temp_root, max_cells=None):
    out = state / "results" / name
    if not stage_references(cfg, state, out, pool, deadline, workers): return False
    if cfg.get("niching"):
        catalogue_dir = out / "catalogues"; catalogue_dir.mkdir(exist_ok=True)
        def missing_catalogues():
            for spec,n,instance,p in problems(cfg):
                dest = catalogue_dir / (p.problem_id + ".json")
                if not dest.exists():
                    yield p.problem_id, (spec,n,instance,cfg,str(out / "landscapes" / (p.problem_id + ".json")))
        for problem_id,data in bounded_pool(pool, missing_catalogues(), worker_catalogue, deadline, 2*workers):
            write_json(catalogue_dir / (problem_id + ".json"), data)
        if any(not (catalogue_dir / (p.problem_id + ".json")).exists() for *_,p in problems(cfg)): return False
    db = open_store(out / "records.sqlite")
    completed = {r[0] for r in db.execute("SELECT id FROM records")}
    before = len(completed)
    expected = plan(cfg)["optimizer_runs"]
    print(f"stage={name} completed={before}/{expected}", flush=True)
    def missing():
        submitted = 0
        for spec, n, instance, p, ns, mode, B, reserve, run, optimizer in cell_specs(cfg):
            tag = cell_tag(p.problem_id, ns, mode, B, reserve, run, optimizer)
            record_id = config_hash(cfg) + "_" + tag
            if record_id in completed: continue
            if max_cells is not None and submitted >= max_cells: break
            submitted += 1
            yield record_id, (cfg, (spec, n, instance, ns, mode, B, reserve, run, optimizer),
                           str(out / "landscapes"), temp_root, suite.get("storage", "compact"))
    try:
        for record_id, (record, trace) in bounded_pool(pool, missing(), worker_cell, deadline, 2 * workers):
            if record["config_hash"] != config_hash(cfg): raise AssertionError("Worker configuration mismatch")
            insert_record(db, record, trace); completed.add(record_id)
            if len(completed) % 32 == 0:
                db.commit()
                write_json(out / "progress.json", dict(stage=name, completed=len(completed), expected=expected))
            if len(completed) % 250 == 0 or len(completed) == expected:
                print(f"{name}: {len(completed)}/{expected}", flush=True)
    finally:
        db.commit(); db.close()
        write_json(out / "progress.json", dict(stage=name, completed=len(completed), expected=expected))
    return len(completed) == expected


def run(suite_path, state, workers=30, seconds=47 * 3600, max_cells=None):
    root, suite, stages = load_suite(suite_path)
    state = Path(state).resolve(); state.mkdir(parents=True, exist_ok=True)
    if (state / "COMPLETE.json").exists():
        print("All search stages are complete; finalization remains available."); return True
    if (state / "FAILED.json").exists():
        print("A recorded failure blocks automatic retries. Read FAILED.json; fix the cause, then remove that flag."); return False
    with (state / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        imported = initialize(root, suite, stages, state)
        print("Imported compatible references:", imported, flush=True)
        deadline = Deadline(seconds)
        started = time.monotonic()
        # Node-local scratch is preferable; no scientific result is left here.
        temp_base = os.environ.get("TMPDIR") or tempfile.gettempdir()
        if not Path(temp_base).is_dir() or not os.access(temp_base, os.W_OK):
            temp_base = tempfile.gettempdir()
        job = os.environ.get("SLURM_JOB_ID", os.environ.get("PBS_JOBID", "local"))
        status = "incomplete"
        try:
            with tempfile.TemporaryDirectory(dir=temp_base, prefix="gnvqa-") as temp_root:
                with futures.ProcessPoolExecutor(max_workers=workers) as pool:
                    all_complete = True
                    for name, cfg in stages:
                        if not deadline.available(): all_complete = False; break
                        ok = run_stage(name, cfg, suite, state, pool, deadline, workers, temp_root, max_cells)
                        if not ok: all_complete = False; break
            if all_complete:
                write_json(state / "COMPLETE.json", dict(search_complete=True, completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
                status = "complete"
            return all_complete
        except Exception as exc:
            status = "failed"
            write_json(state / "FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc(), job_id=job))
            raise
        finally:
            entry = dict(job_id=job, hostname=os.uname().nodename, workers=workers, status=status,
                         elapsed_seconds=time.monotonic() - started, nodes=1)
            with (state / "segments.jsonl").open("a") as f: f.write(json.dumps(entry) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="cluster/suite.json")
    p.add_argument("--state", default="work/full_study")
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--seconds", type=int, default=47 * 3600)
    p.add_argument("--max-cells", type=int, help="Integration check only; do not use for the full study")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--initialize-only", action="store_true", help="Create immutable manifests and import references, without any search")
    a = p.parse_args()
    if a.workers < 1 or a.seconds < 1: p.error("Positive workers/seconds required")
    if a.plan: print(json.dumps(describe(a.suite, a.workers), indent=2)); return
    if a.initialize_only:
        root, suite, stages = load_suite(a.suite)
        state = Path(a.state).resolve(); state.mkdir(parents=True, exist_ok=True)
        with (state / "suite.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print("Imported references:", initialize(root, suite, stages, state))
        return
    run(a.suite, a.state, a.workers, a.seconds, a.max_cells)


if __name__ == "__main__": main()
