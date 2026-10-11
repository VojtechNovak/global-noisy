"""Single-writer transactional storage, suitable for shared cluster filesystems.

Workers write temporary traces on node-local storage; the coordinator commits
record and trace together. WAL is intentionally avoided on network filesystems.
"""
from __future__ import annotations
import io
import json
import sqlite3
from pathlib import Path
import numpy as np


def open_store(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, payload TEXT NOT NULL, trace BLOB NOT NULL)")
    db.commit()
    return db


def insert_record(db, record, trace):
    record = dict(record)
    record_id = Path(record["trace_file"]).stem
    record.update(record_id=record_id, trace_file="sqlite:" + record_id)
    db.execute("INSERT INTO records VALUES (?,?,?)", (record_id,
               json.dumps(record, separators=(",", ":"), allow_nan=False), sqlite3.Binary(trace)))


def records(out):
    path = Path(out) / "records.sqlite"
    if path.exists():
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=60) as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM records ORDER BY id")]
    return [json.loads(p.read_text()) for p in sorted((Path(out) / "runs").glob("*.json"))
            if not p.name.endswith(".events.json")]


def load_trace(out, record):
    filename = record["trace_file"]
    if filename.startswith("sqlite:"):
        with sqlite3.connect(f"file:{(Path(out) / 'records.sqlite').resolve()}?mode=ro", uri=True) as db:
            row = db.execute("SELECT trace FROM records WHERE id=?", (filename[7:],)).fetchone()
        if row is None: raise FileNotFoundError(filename)
        with np.load(io.BytesIO(row[0]), allow_pickle=False) as z:
            return {key: z[key] for key in z.files}
    with np.load(Path(out) / filename, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def curve_at(trace, metric, grid):
    values = trace[metric]
    checkpoints = trace.get("checkpoint_fes", np.arange(1, len(values) + 1))
    indices = np.searchsorted(checkpoints, grid, side="right") - 1
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices]
