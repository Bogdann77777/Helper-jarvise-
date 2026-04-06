"""
db.py — SQLite storage for the task manager.

Tables:
  tasks          — all tasks with project, text, due_date, priority, done
  vacation       — vacation periods (start_date, end_date)
  urgent_alerts  — tracks how many times urgent tasks were sent today
"""
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "tasks.db"


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")  # safe concurrent access
    return con


def init_db():
    con = _conn()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project     TEXT NOT NULL,
            text        TEXT NOT NULL,
            due_date    TEXT,                         -- ISO "2026-04-15" or NULL
            done        INTEGER DEFAULT 0,
            priority    TEXT DEFAULT 'normal',        -- 'high' or 'normal'
            created_at  TEXT DEFAULT (datetime('now', 'localtime')),
            updated_at  TEXT DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS vacation (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            start_date  TEXT NOT NULL,
            end_date    TEXT NOT NULL,
            note        TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS urgent_alerts (
            task_id     INTEGER NOT NULL,
            sent_date   TEXT NOT NULL,               -- ISO date
            sent_count  INTEGER DEFAULT 0,
            confirmed   INTEGER DEFAULT 0,
            PRIMARY KEY (task_id, sent_date)
        );

        CREATE INDEX IF NOT EXISTS idx_project  ON tasks(project);
        CREATE INDEX IF NOT EXISTS idx_due_date ON tasks(due_date);
        CREATE INDEX IF NOT EXISTS idx_done     ON tasks(done);
    """)
    con.commit()
    con.close()


# ── Tasks ─────────────────────────────────────────────────────────────────────

def add_task(project: str, text: str,
             due_date: Optional[str] = None,
             priority: str = "normal") -> int:
    """Add task, return new id."""
    con = _conn()
    cur = con.execute(
        "INSERT INTO tasks (project, text, due_date, priority) VALUES (?, ?, ?, ?)",
        (project.strip(), text.strip(), due_date, priority)
    )
    task_id = cur.lastrowid
    con.commit()
    con.close()
    return task_id


def get_tasks(project: Optional[str] = None, include_done: bool = False) -> list[dict]:
    """Get tasks, optionally filtered by project."""
    con = _conn()
    where = [] if include_done else ["done = 0"]
    params = []
    if project:
        where.append("project = ?")
        params.append(project.strip())
    sql = "SELECT * FROM tasks"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY due_date ASC NULLS LAST, priority DESC, id ASC"
    rows = con.execute(sql, params).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_todays_report_tasks() -> list[dict]:
    """Tasks for morning report: due today/past, due in next 3 days, undated."""
    con = _conn()
    rows = con.execute("""
        SELECT * FROM tasks
        WHERE done = 0
          AND (due_date IS NULL OR due_date <= date('now', 'localtime', '+3 days'))
        ORDER BY
          CASE WHEN due_date IS NULL THEN 1 ELSE 0 END ASC,
          due_date ASC,
          priority DESC
    """).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_urgent_pending() -> list[dict]:
    """Urgent tasks not done — for hourly alarm."""
    con = _conn()
    rows = con.execute("""
        SELECT * FROM tasks
        WHERE done = 0 AND priority = 'high'
        ORDER BY due_date ASC NULLS LAST
    """).fetchall()
    con.close()
    return [dict(r) for r in rows]


def mark_done(task_id: int) -> bool:
    con = _conn()
    cur = con.execute(
        "UPDATE tasks SET done=1, updated_at=datetime('now','localtime') WHERE id=?",
        (task_id,)
    )
    changed = cur.rowcount > 0
    con.commit()
    con.close()
    return changed


def delete_task(task_id: int) -> bool:
    con = _conn()
    cur = con.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    changed = cur.rowcount > 0
    con.commit()
    con.close()
    return changed


def update_task(task_id: int, **kwargs) -> bool:
    """Update fields: text, due_date, priority, project."""
    allowed = {"text", "due_date", "priority", "project"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return False
    fields["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = "UPDATE tasks SET " + ", ".join(f"{k}=?" for k in fields) + " WHERE id=?"
    con = _conn()
    cur = con.execute(sql, list(fields.values()) + [task_id])
    changed = cur.rowcount > 0
    con.commit()
    con.close()
    return changed


def get_task(task_id: int) -> Optional[dict]:
    con = _conn()
    row = con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    con.close()
    return dict(row) if row else None


# ── Vacation ──────────────────────────────────────────────────────────────────

def set_vacation(start: str, end: str, note: str = "") -> int:
    con = _conn()
    cur = con.execute(
        "INSERT INTO vacation (start_date, end_date, note) VALUES (?,?,?)",
        (start, end, note)
    )
    vid = cur.lastrowid
    con.commit()
    con.close()
    return vid


def is_vacation_today() -> bool:
    today = date.today().isoformat()
    con = _conn()
    row = con.execute(
        "SELECT id FROM vacation WHERE start_date <= ? AND end_date >= ?",
        (today, today)
    ).fetchone()
    con.close()
    return row is not None


def list_vacations() -> list[dict]:
    con = _conn()
    rows = con.execute(
        "SELECT * FROM vacation WHERE end_date >= date('now','localtime') ORDER BY start_date"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def cancel_vacation(vacation_id: int) -> bool:
    con = _conn()
    cur = con.execute("DELETE FROM vacation WHERE id=?", (vacation_id,))
    changed = cur.rowcount > 0
    con.commit()
    con.close()
    return changed


# ── Urgent alert tracking ─────────────────────────────────────────────────────

def get_urgent_alert_count(task_id: int) -> int:
    today = date.today().isoformat()
    con = _conn()
    row = con.execute(
        "SELECT sent_count FROM urgent_alerts WHERE task_id=? AND sent_date=?",
        (task_id, today)
    ).fetchone()
    con.close()
    return row["sent_count"] if row else 0


def increment_urgent_alert(task_id: int):
    today = date.today().isoformat()
    con = _conn()
    con.execute("""
        INSERT INTO urgent_alerts (task_id, sent_date, sent_count)
        VALUES (?, ?, 1)
        ON CONFLICT(task_id, sent_date) DO UPDATE SET sent_count = sent_count + 1
    """, (task_id, today))
    con.commit()
    con.close()


def confirm_urgent_alert(task_id: int):
    today = date.today().isoformat()
    con = _conn()
    con.execute(
        "UPDATE urgent_alerts SET confirmed=1 WHERE task_id=? AND sent_date=?",
        (task_id, today)
    )
    con.commit()
    con.close()


def is_urgent_confirmed_today(task_id: int) -> bool:
    today = date.today().isoformat()
    con = _conn()
    row = con.execute(
        "SELECT confirmed FROM urgent_alerts WHERE task_id=? AND sent_date=?",
        (task_id, today)
    ).fetchone()
    con.close()
    return bool(row and row["confirmed"])
