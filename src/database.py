"""SQLite storage: uploaded retraining images, retraining runs and a prediction log."""
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from src.preprocessing import BASE_DIR, CLASS_NAMES

DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "data" / "maizeguard.db"))

STALE_AFTER_S = 2 * 3600        # a run "running" for longer than this is treated as dead

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL, label TEXT NOT NULL, image BLOB NOT NULL,
    uploaded_at TEXT NOT NULL, used_in_run INTEGER);
CREATE TABLE IF NOT EXISTS retrain_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL, triggered_by TEXT, started_at TEXT, finished_at TEXT,
    n_images INTEGER, accuracy_before REAL, accuracy_after REAL,
    promoted INTEGER, message TEXT);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT, confidence REAL, latency_ms REAL, created_at TEXT);
"""


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with connect() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)


# -------------------------------------------------------------------- uploads
def add_upload(filename, label, image_bytes):
    with connect() as con:
        con.execute("INSERT INTO uploads (filename, label, image, uploaded_at) VALUES (?,?,?,?)",
                    (filename, label, image_bytes, _now()))


def pending_counts():
    with connect() as con:
        rows = con.execute("SELECT label, COUNT(*) n FROM uploads WHERE used_in_run IS NULL "
                           "GROUP BY label").fetchall()
    counts = {c: 0 for c in CLASS_NAMES}
    counts.update({r["label"]: r["n"] for r in rows})
    return counts


def export_pending(target_dir):
    """Write all not-yet-used uploads to target_dir/<class>/ and return (ids, paths, labels)."""
    target_dir = Path(target_dir)
    ids, paths, labels = [], [], []
    with connect() as con:
        rows = con.execute("SELECT id, label, image FROM uploads WHERE used_in_run IS NULL").fetchall()
    for r in rows:
        if r["label"] not in CLASS_NAMES:
            continue
        d = target_dir / r["label"]
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"upload_{r['id']}.jpg"
        p.write_bytes(r["image"])
        ids.append(r["id"]); paths.append(str(p)); labels.append(CLASS_NAMES.index(r["label"]))
    return ids, paths, labels


def mark_used(ids, run_id):
    with connect() as con:
        con.executemany("UPDATE uploads SET used_in_run=? WHERE id=?", [(run_id, i) for i in ids])


# ------------------------------------------------------------------------ runs
def start_run(trigger):
    """Create a 'running' row. Returns None if another run is already in progress."""
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        stale = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - STALE_AFTER_S))
        con.execute("UPDATE retrain_runs SET status='failed', message='timed out', finished_at=? "
                    "WHERE status='running' AND started_at < ?", (_now(), stale))
        if con.execute("SELECT 1 FROM retrain_runs WHERE status='running'").fetchone():
            return None
        cur = con.execute("INSERT INTO retrain_runs (status, triggered_by, started_at) VALUES "
                          "('running', ?, ?)", (trigger, _now()))
        return cur.lastrowid


def finish_run(run_id, status, message="", n_images=0, before=None, after=None, promoted=None):
    with connect() as con:
        con.execute("UPDATE retrain_runs SET status=?, finished_at=?, n_images=?, "
                    "accuracy_before=?, accuracy_after=?, promoted=?, message=? WHERE id=?",
                    (status, _now(), n_images, before, after,
                     None if promoted is None else int(promoted), message, run_id))


def list_runs(limit=10):
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (limit,))]


# ----------------------------------------------------------------- predictions
def log_prediction(label, confidence, latency_ms):
    try:
        with connect() as con:
            con.execute("INSERT INTO predictions (label, confidence, latency_ms, created_at) "
                        "VALUES (?,?,?,?)", (label, confidence, latency_ms, _now()))
    except sqlite3.OperationalError:
        pass                                    # never fail a prediction because of logging


def prediction_stats():
    with connect() as con:
        total = con.execute("SELECT COUNT(*) n, AVG(latency_ms) lat, AVG(confidence) conf "
                            "FROM predictions").fetchone()
        by_class = con.execute("SELECT label, COUNT(*) n, AVG(confidence) conf FROM predictions "
                               "GROUP BY label").fetchall()
    return {"total": total["n"], "avg_latency_ms": total["lat"], "avg_confidence": total["conf"],
            "by_class": {r["label"]: {"count": r["n"], "avg_confidence": r["conf"]} for r in by_class}}
