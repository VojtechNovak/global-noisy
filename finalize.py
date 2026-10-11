"""Audit a complete or partial suite and export one self-contained results ZIP."""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import zipfile
from .cluster import load_suite, canonical, suite_signature
from .run import plan, source_hash, write_json, config_hash, cell_specs, cell_tag
from .report import stage_report, latex_report


def audit_stage(out, cfg):
    expected = {config_hash(cfg) + "_" + cell_tag(c[3].problem_id, *c[4:]) for c in cell_specs(cfg)}
    dbfile = out / "records.sqlite"
    if not dbfile.exists(): return dict(expected=len(expected), completed=0, missing=len(expected), unexpected=0, integrity="no_database")
    with sqlite3.connect(f"file:{dbfile.resolve()}?mode=ro", uri=True) as db:
        integrity = db.execute("PRAGMA quick_check").fetchone()[0]
        completed = {r[0] for r in db.execute("SELECT id FROM records")}
        bad_budget, bad_configuration = 0, 0
        for payload, trace_bytes in db.execute("SELECT payload, length(trace) FROM records"):
            r = json.loads(payload)
            bad_budget += r["search_fes"] > r["search_fe_ceiling"] or r["search_shots"] + r["validation_shots"] > r["total_shot_ceiling"]
            bad_configuration += r["config_hash"] != config_hash(cfg) or trace_bytes < 1
    return dict(expected=len(expected), completed=len(completed), missing=len(expected-completed),
                unexpected=len(completed-expected), integrity=integrity, bad_budget=int(bad_budget), bad_configuration=int(bad_configuration))


def archive_results(root, state, status, target):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    candidates = []
    for folder in ("gnvqa", "configs", "cluster", "tests"):
        candidates.extend((p, "reproduce/" + str(p.relative_to(root))) for p in (root / folder).rglob("*")
                          if p.is_file() and "__pycache__" not in p.parts)
    for filename in ("README_CLUSTER.md", "NICHING_STUDY.md", "STUDY_EXECUTION.md", "manuscript.tex", "references.bib", "CODE_AUDIT.md", "study_design.md", "literature_review.md", "VALIDATION_CLUSTER.md"):
        if (root / filename).is_file(): candidates.append((root / filename, "reproduce/" + filename))
    for folder in ("results", "manuscript_outputs"):
        candidates.extend((p, str(p.relative_to(state))) for p in (state / folder).rglob("*")
                          if p.is_file() and not p.name.endswith(("-journal", ".tmp", ".aux", ".log", ".out")))
    for name in ("suite_manifest.json", "COMPLETE.json", "FAILED.json", "validation.json", "calibration.json", "segments.jsonl", "final_status.json"):
        if (state / name).exists(): candidates.append((state / name, name))
    # Original exact references and local pilot manifests retain independent
    # provenance; uploaded pilot traces are not mixed into the full benchmark.
    candidates.extend((p, "reproduce/inputs/" + str(p.relative_to(root / "inputs"))) for p in (root / "inputs").rglob("*") if p.is_file())
    if (root / ".cluster").exists():
        candidates.extend((p, "environment/" + p.name) for p in (root / ".cluster").iterdir()
                          if p.is_file() and p.suffix in (".txt", ".json"))
    logs = root / "logs"
    if logs.exists(): candidates.extend((p, "logs/" + p.name) for p in logs.iterdir() if p.is_file())
    checksums = []
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        archive.writestr("READ_FIRST.txt", "Global and Noisy VQA study\nStatus: " + ("COMPLETE" if status["complete"] else "PARTIAL") +
            "\nRead final_status.json before interpreting results. results/<stage>/analysis contains manuscript tables/figures.\n"
            "records.sqlite contains all run records and retained traces; reproduce/gnvqa/storage.py reads it.\n"
            "Compact traces contain exact endpoint values and approximately 100 log-spaced FE checkpoints, not every queried coordinate.\n"
            "QAOA references and niche catalogues are empirical. Offline quenches are not search successes.\n")
        for path, relative in sorted(candidates, key=lambda item:item[1]):
            h = hashlib.sha256()
            with path.open("rb") as f:
                while chunk := f.read(1024 * 1024): h.update(chunk)
            checksums.append(h.hexdigest() + "  " + relative)
            archive.write(path, relative)
        archive.writestr("SHA256SUMS.txt", "\n".join(checksums)+"\n")
    # Validate the central directory; full byte checksums accompany every file.
    with zipfile.ZipFile(temporary) as archive:
        if "final_status.json" not in archive.namelist(): raise RuntimeError("Incomplete results archive")
    temporary.replace(target)
    checksum = hashlib.sha256()
    with target.open("rb") as f:
        while chunk := f.read(1024 * 1024): checksum.update(chunk)
    target.with_suffix(".zip.sha256").write_text(checksum.hexdigest() + "  " + target.name + "\n")
    return target


def finalize(suite_path, state, workers=6, bootstrap=None):
    root, suite, stages = load_suite(suite_path)
    state = Path(state).resolve(); state.mkdir(parents=True, exist_ok=True)
    signature = suite_signature(root,suite,stages)
    old = json.loads((state / "suite_manifest.json").read_text())
    if signature != old["signature"]: raise ValueError("Cannot finalize a changed suite/source tree")
    repetitions = int(bootstrap or suite.get("bootstrap_repetitions", 2000))
    with (state / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audits = {name:audit_stage(state / "results" / name, cfg) for name,cfg in stages}
        results = {}
        with futures.ProcessPoolExecutor(max_workers=min(workers, len(stages))) as pool:
            pending = {pool.submit(stage_report, state / "results" / name, repetitions, audits[name]["expected"]): name
                       for name, cfg in stages}
            for future in futures.as_completed(pending):
                name = pending[future]
                try: results[name] = future.result()
                except Exception as exc:
                    results[name] = dict(completed_runs=audits[name]["completed"], expected_runs=audits[name]["expected"],
                                         complete=False, error=repr(exc), budget_violations=[])
                print("analysis", name, results[name], flush=True)
        complete = all(r.get("complete") and not r.get("budget_violations") and not r.get("error") for r in results.values())
        complete &= all(a["missing"] == 0 and a["unexpected"] == 0 and a["integrity"] == "ok" and
                        not a.get("bad_budget") and not a.get("bad_configuration") for a in audits.values())
        status = dict(complete=bool(complete), stages=results, inventory_audits=audits, bootstrap_repetitions=repetitions,
                      finalized_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      interpretation="QAOA gaps are to empirical references; niching attainment and diagnostic basin access are distinct.")
        report = latex_report(state, status)
        shutil.copy2(root / "manuscript.tex", report / "manuscript.tex")
        shutil.copy2(root / "references.bib", report / "references.bib")
        if (root / "manuscript.pdf").is_file(): shutil.copy2(root / "manuscript.pdf", report / "manuscript.pdf")
        # Compile when TeX is available; all LaTeX/CSV/PDF figures are always
        # exported. No system privileges or compute-node downloads are needed.
        build = subprocess.run(["bash", str(root / "cluster/build_manuscripts.sh"), str(report)],
                               capture_output=True, text=True)
        (report / "latex_build.txt").write_text(build.stdout + build.stderr)
        status["latex_pdf_compilation_available"] = build.returncode == 0
        write_json(state / "final_status.json", status)
        target = archive_results(root, state, status, state / "exports/gnvqa_manuscript_results.zip")
        print("RESULTS_ZIP", target, "COMPLETE", complete, flush=True)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="cluster/suite.json")
    parser.add_argument("--state", default="work/full_study")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bootstrap", type=int)
    a = parser.parse_args(); status = finalize(a.suite, a.state, a.workers, a.bootstrap)
    if not status["complete"]: raise SystemExit(2)


if __name__ == "__main__": main()
