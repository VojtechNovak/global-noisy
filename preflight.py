"""Compute-node validation and a local timing calibration, never a login benchmark."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import scipy
from .cluster import load_suite, canonical, describe
from .models import stable_seed, build_problem
from .noise import Noise
from .run import write_json, source_hash, cell_specs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="cluster/suite.json")
    p.add_argument("--state", default="work/full_study")
    p.add_argument("--workers", type=int, default=30)
    a = p.parse_args(); state = Path(a.state); state.mkdir(parents=True, exist_ok=True)
    marker = state / "validation.json"
    import qiskit
    versions = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, qiskit=qiskit.__version__)
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["code_sha256"] != source_hash() or any(previous[k] != v for k,v in versions.items()):
            raise SystemExit("Changed code or numerical environment requires a fresh state directory")
        print("Existing validation applies to this source tree."); return
    root, suite, stages = load_suite(a.suite)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-v"], check=True)
    subprocess.run([sys.executable, "-m", "gnvqa.validate_qiskit"], check=True)
    timings = {}
    rng = np.random.default_rng(7823)
    total = 0.
    per_stage = {}
    for name, cfg in stages:
        seconds = 0.
        for spec, n, instance, problem, ns, mode, B, reserve, run, optimizer in cell_specs(cfg):
            timing_key = canonical([spec, n, ns])
            if timing_key not in timings:
                noise = Noise(**{k:v for k,v in ns.items() if k != "name"})
                samples = []
                for _ in range(4):
                    x = rng.uniform(0, 1, problem.dim)
                    start = time.perf_counter(); noise.observe(problem, x, rng)
                    samples.append(time.perf_counter() - start)
                timings[timing_key] = float(np.median(samples))
            cost = 0 if ns["mode"] == "exact" else ns["shots"] * ns.get("repeats",1)
            fes = int(B * (1-reserve)) if mode == "fe" else int(B * (1-reserve) // cost)
            # A modest measured-pilot optimizer overhead, added separately.
            seconds += (timings[timing_key] + 4e-5) * fes
        per_stage[name] = seconds / 3600; total += seconds
    write_json(state / "calibration.json", dict(single_core_fe_ceiling_hours=total/3600,
               ideal_worker_pool_hours=total/3600/a.workers, illustrative_threefold_overhead_hours=3*total/3600/a.workers,
               per_stage_cpu_hours=per_stage, timings=timings, workers=a.workers,
               note="A timing extrapolation, not a guaranteed duration or confidence interval. Excludes offline characterization, niche assessment, analysis and archive export."))
    write_json(marker, dict(code_sha256=source_hash(), python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, qiskit=qiskit.__version__, hostname=os.uname().nodename,
                tests="passed", qiskit_crosscheck="passed", thread_settings={k:os.getenv(k) for k in
                ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]}))
    print(json.dumps(json.loads((state / "calibration.json").read_text()), indent=2), flush=True)


if __name__ == "__main__": main()
