"""
Campaign Database — SQLite schema for outbound call management.

Tables:
  contacts    — who to call (phone, name, company, status, retry info)
  campaigns   — grouped call lists with shared script/persona
  call_log    — every call attempt with outcome
  dnc_list    — Do Not Call registry (internal)
  ab_variants — A/B test script variants
  ab_results  — A/B test outcomes per call

TCPA Compliance built-in:
  - Calling hours enforced in scheduler (not here)
  - DNC check before every dial
  - Opt-out logged immediately
"""
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "outputs" / "sales_agent.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS campaigns (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            persona_file  TEXT NOT NULL DEFAULT 'template.yaml',
            status        TEXT NOT NULL DEFAULT 'draft',  -- draft|active|paused|completed
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes         TEXT
        );

        CREATE TABLE IF NOT EXISTS contacts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id     INTEGER NOT NULL REFERENCES campaigns(id),
            phone           TEXT NOT NULL,
            first_name      TEXT,
            last_name       TEXT,
            company         TEXT,
            title           TEXT,
            email           TEXT,
            timezone        TEXT DEFAULT 'America/New_York',
            status          TEXT NOT NULL DEFAULT 'pending',
            -- statuses: pending|dialing|answered|converted|rejected|voicemail|dnc|max_retries
            attempt_count   INTEGER DEFAULT 0,
            max_attempts    INTEGER DEFAULT 3,
            last_attempt_at TIMESTAMP,
            next_retry_at   TIMESTAMP,
            notes           TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_contacts_campaign ON contacts(campaign_id);
        CREATE INDEX IF NOT EXISTS idx_contacts_status ON contacts(status);
        CREATE INDEX IF NOT EXISTS idx_contacts_next_retry ON contacts(next_retry_at);

        CREATE TABLE IF NOT EXISTS call_log (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id          INTEGER NOT NULL REFERENCES contacts(id),
            campaign_id         INTEGER NOT NULL,
            call_control_id     TEXT,           -- Telnyx call ID
            call_session_id     TEXT,           -- our internal session ID
            started_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            answered_at         TIMESTAMP,
            ended_at            TIMESTAMP,
            duration_seconds    INTEGER DEFAULT 0,
            outcome             TEXT,           -- converted|rejected|voicemail|no_answer|failed
            amd_result          TEXT,           -- human_residence|machine|etc
            transcript_path     TEXT,           -- path to JSON transcript
            audio_path          TEXT,           -- path to WAV recording
            ab_variant          TEXT,           -- which A/B script variant was used
            score               REAL,           -- post-call LLM score 0-10
            score_feedback      TEXT,           -- LLM feedback on the call
            fsm_history         TEXT,           -- state machine path
            talk_ratio          REAL,           -- agent talk ratio (target < 0.4)
            objection_count     INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS dnc_list (
            phone       TEXT PRIMARY KEY,
            added_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reason      TEXT    -- opt_out|manual|federal_dnc
        );

        CREATE TABLE IF NOT EXISTS ab_variants (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id     INTEGER REFERENCES campaigns(id),
            name            TEXT NOT NULL,      -- 'variant_a', 'variant_b', etc.
            persona_file    TEXT NOT NULL,
            description     TEXT,
            is_control      INTEGER DEFAULT 0,
            weight          REAL DEFAULT 0.5,   -- traffic split weight
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
    print(f"[DB] Initialized: {DB_PATH}")


# ── Contact CRUD ──────────────────────────────────────────────────────────────

def add_contact(
    campaign_id: int,
    phone: str,
    first_name: str = "",
    last_name: str = "",
    company: str = "",
    title: str = "",
    email: str = "",
    timezone: str = "America/New_York",
) -> int:
    """Add a contact to a campaign. Returns contact_id."""
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO contacts (campaign_id, phone, first_name, last_name, company, title, email, timezone)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (campaign_id, phone, first_name, last_name, company, title, email, timezone))
        return cur.lastrowid


def get_contacts_to_call(campaign_id: int, limit: int = 10) -> list[dict]:
    """
    Get contacts ready to be called right now.
    Respects: status=pending, retry timing, max attempts.
    """
    now = datetime.now().isoformat()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT c.*, campaign.persona_file
            FROM contacts c
            JOIN campaigns campaign ON c.campaign_id = campaign.id
            WHERE c.campaign_id = ?
              AND c.status = 'pending'
              AND c.attempt_count < c.max_attempts
              AND (c.next_retry_at IS NULL OR c.next_retry_at <= ?)
              AND NOT EXISTS (SELECT 1 FROM dnc_list d WHERE d.phone = c.phone)
            ORDER BY c.next_retry_at ASC NULLS FIRST
            LIMIT ?
        """, (campaign_id, now, limit)).fetchall()
        return [dict(r) for r in rows]


def update_contact_status(contact_id: int, status: str, next_retry_minutes: int | None = None):
    """Update contact status after a call attempt."""
    next_retry = None
    if next_retry_minutes:
        from datetime import timedelta
        next_retry = (datetime.now() + timedelta(minutes=next_retry_minutes)).isoformat()

    with get_conn() as conn:
        conn.execute("""
            UPDATE contacts
            SET status = ?,
                attempt_count = attempt_count + 1,
                last_attempt_at = CURRENT_TIMESTAMP,
                next_retry_at = ?
            WHERE id = ?
        """, (status, next_retry, contact_id))


def add_dnc(phone: str, reason: str = "opt_out"):
    """Add phone to Do Not Call list."""
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO dnc_list (phone, reason) VALUES (?, ?)
        """, (phone, reason))


def is_dnc(phone: str) -> bool:
    """Check if phone is on DNC list."""
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM dnc_list WHERE phone = ?", (phone,)).fetchone()
        return row is not None


# ── Campaign CRUD ─────────────────────────────────────────────────────────────

def create_campaign(name: str, persona_file: str = "template.yaml", notes: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO campaigns (name, persona_file, notes)
            VALUES (?, ?, ?)
        """, (name, persona_file, notes))
        return cur.lastrowid


def get_campaigns() -> list[dict]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM campaigns ORDER BY created_at DESC").fetchall()]


# ── Call Log ──────────────────────────────────────────────────────────────────

def log_call(
    contact_id: int,
    campaign_id: int,
    outcome: str,
    call_session_id: str = "",
    call_control_id: str = "",
    duration_seconds: int = 0,
    transcript_path: str = "",
    audio_path: str = "",
    score: float | None = None,
    score_feedback: str = "",
    fsm_history: str = "",
    talk_ratio: float | None = None,
    objection_count: int = 0,
    amd_result: str = "",
    ab_variant: str = "",
) -> int:
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO call_log (
                contact_id, campaign_id, call_control_id, call_session_id,
                ended_at, duration_seconds, outcome, amd_result,
                transcript_path, audio_path, ab_variant, score, score_feedback,
                fsm_history, talk_ratio, objection_count
            ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            contact_id, campaign_id, call_control_id, call_session_id,
            duration_seconds, outcome, amd_result,
            transcript_path, audio_path, ab_variant, score, score_feedback,
            fsm_history, talk_ratio, objection_count,
        ))
        return cur.lastrowid


def get_campaign_stats(campaign_id: int) -> dict:
    """Get conversion stats for a campaign."""
    with get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_calls,
                SUM(CASE WHEN outcome='converted' THEN 1 ELSE 0 END) as converted,
                SUM(CASE WHEN outcome='rejected' THEN 1 ELSE 0 END) as rejected,
                SUM(CASE WHEN outcome='voicemail' THEN 1 ELSE 0 END) as voicemail,
                SUM(CASE WHEN outcome='no_answer' THEN 1 ELSE 0 END) as no_answer,
                AVG(duration_seconds) as avg_duration,
                AVG(score) as avg_score,
                AVG(talk_ratio) as avg_talk_ratio
            FROM call_log WHERE campaign_id = ?
        """, (campaign_id,)).fetchone()
        total = row["total_calls"] or 1
        return {
            **dict(row),
            "conversion_rate": round((row["converted"] or 0) / total * 100, 1),
            "answer_rate": round((1 - (row["no_answer"] or 0) / total) * 100, 1),
        }
