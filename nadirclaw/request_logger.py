"""
SQLite-based request logging for NadirClaw.

Logs every API call with timestamp, model, tokens, cost, latency to a local SQLite database.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from nadirclaw.settings import settings

logger = logging.getLogger("nadirclaw")

_db_lock = Lock()
_db_path: Optional[Path] = None
_db_initialized = False


def _get_db_path() -> Path:
    """Get the path to the SQLite database."""
    global _db_path
    if _db_path is None:
        log_dir = settings.LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        _db_path = log_dir / "requests.db"
    return _db_path


def _init_db() -> None:
    """Initialize the SQLite database schema if it doesn't exist."""
    global _db_initialized
    if _db_initialized:
        return

    db_path = _get_db_path()
    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()

            # Enable WAL mode for better concurrent read/write performance
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA wal_autocheckpoint=1000")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    request_id TEXT,
                    type TEXT,
                    status TEXT,
                    prompt TEXT,
                    selected_model TEXT,
                    provider TEXT,
                    tier TEXT,
                    confidence REAL,
                    complexity_score REAL,
                    classifier_latency_ms INTEGER,
                    total_latency_ms INTEGER,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    cost REAL,
                    daily_spend REAL,
                    response_preview TEXT,
                    fallback_used TEXT,
                    error TEXT,
                    tool_count INTEGER,
                    has_images INTEGER,
                    has_tools INTEGER,
                    max_context_tokens INTEGER
                )
            """)

            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON requests(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model
                ON requests(selected_model)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON requests(status)
            """)
            # Additional indexes for dashboard/reporting queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tier
                ON requests(tier)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cost
                ON requests(cost)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp_model
                ON requests(timestamp, selected_model)
            """)
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_request_id
                ON requests(request_id)
            """)
            
            # Migrate: add optimization columns (idempotent)
            for col, col_type in [
                ("optimization_mode", "TEXT"),
                ("original_tokens", "INTEGER"),
                ("optimized_tokens", "INTEGER"),
                ("tokens_saved", "INTEGER"),
                ("optimizations_applied", "TEXT"),
                ("system_prompt", "TEXT"),
            ]:
                try:
                    cursor.execute(f"ALTER TABLE requests ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            # Create labeled_prompts table for classifier retraining
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS labeled_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    prompt TEXT,
                    system_prompt TEXT,
                    predicted_tier TEXT,
                    correct_tier TEXT,
                    confidence REAL,
                    labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_labeled_request_id
                ON labeled_prompts(request_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_labeled_at
                ON labeled_prompts(labeled_at)
            """)

            conn.commit()
            _db_initialized = True
            logger.debug("SQLite request log initialized at %s", db_path)
        finally:
            conn.close()


def log_request(entry: Dict[str, Any]) -> None:
    """
    Log a request to the SQLite database.
    
    Args:
        entry: Dictionary containing request metadata (timestamp, model, tokens, cost, etc.)
    """
    _init_db()
    
    db_path = _get_db_path()
    
    # Ensure timestamp is present
    if "timestamp" not in entry:
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Extract fields for SQLite (handle missing fields gracefully)
    timestamp = entry.get("timestamp")
    request_id = entry.get("request_id")
    req_type = entry.get("type")
    status = entry.get("status", "ok")
    prompt = entry.get("prompt")
    selected_model = entry.get("selected_model")
    provider = entry.get("provider")
    tier = entry.get("tier")
    confidence = entry.get("confidence")
    complexity_score = entry.get("complexity_score")
    classifier_latency_ms = entry.get("classifier_latency_ms")
    total_latency_ms = entry.get("total_latency_ms")
    prompt_tokens = entry.get("prompt_tokens")
    completion_tokens = entry.get("completion_tokens")
    total_tokens = entry.get("total_tokens")
    cost = entry.get("cost")
    daily_spend = entry.get("daily_spend")
    response_preview = entry.get("response_preview")
    fallback_used = entry.get("fallback_used")
    error = entry.get("error")
    tool_count = entry.get("tool_count")
    has_images = 1 if entry.get("has_images") else 0
    has_tools = 1 if entry.get("has_tools") else 0
    max_context_tokens = entry.get("max_context_tokens")
    optimization_mode = entry.get("optimization_mode")
    original_tokens = entry.get("original_tokens")
    optimized_tokens = entry.get("optimized_tokens")
    tokens_saved = entry.get("tokens_saved")
    optimizations_applied = (
        json.dumps(entry["optimizations_applied"])
        if entry.get("optimizations_applied")
        else None
    )
    system_prompt = entry.get("system_prompt_text") or entry.get("system_prompt")

    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        try:
            # Ensure WAL pragmas on every connection (they persist per-database
            # but setting them is cheap and defensive)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")

            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO requests (
                    timestamp, request_id, type, status, prompt, selected_model,
                    provider, tier, confidence, complexity_score, classifier_latency_ms,
                    total_latency_ms, prompt_tokens, completion_tokens, total_tokens,
                    cost, daily_spend, response_preview, fallback_used, error,
                    tool_count, has_images, has_tools, max_context_tokens,
                    optimization_mode, original_tokens, optimized_tokens,
                    tokens_saved, optimizations_applied, system_prompt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, request_id, req_type, status, prompt, selected_model,
                provider, tier, confidence, complexity_score, classifier_latency_ms,
                total_latency_ms, prompt_tokens, completion_tokens, total_tokens,
                cost, daily_spend, response_preview, fallback_used, error,
                tool_count, has_images, has_tools, max_context_tokens,
                optimization_mode, original_tokens, optimized_tokens,
                tokens_saved, optimizations_applied, system_prompt,
            ))
            conn.commit()
        except Exception as e:
            logger.error("Failed to log request to SQLite: %s", e, exc_info=True)
        finally:
            conn.close()


def get_request_count() -> int:
    """Get the total number of logged requests."""
    _init_db()
    db_path = _get_db_path()

    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM requests")
            return cursor.fetchone()[0]
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Labeled prompts for classifier retraining
# ---------------------------------------------------------------------------

VALID_LABEL_TIERS = ("simple", "medium", "complex")


def store_label(
    request_id: str,
    correct_tier: str,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    predicted_tier: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Store a human-labeled tier correction for a request.

    If prompt/predicted_tier/confidence are not provided, they are looked up
    from the requests table automatically.

    Returns dict with stored label data or error.
    """
    _init_db()
    db_path = _get_db_path()

    if correct_tier not in VALID_LABEL_TIERS:
        return {"error": f"correct_tier must be one of: {', '.join(VALID_LABEL_TIERS)}"}

    # Look up original request data if not provided
    if prompt is None or predicted_tier is None:
        with _db_lock:
            conn = sqlite3.connect(str(db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT prompt, system_prompt, tier, confidence FROM requests WHERE request_id = ? LIMIT 1",
                    (request_id,),
                )
                row = cursor.fetchone()
            finally:
                conn.close()

        if row is None:
            return {"error": f"Request {request_id} not found in logs."}

        prompt = prompt or row[0]
        system_prompt = system_prompt if system_prompt is not None else row[1]
        predicted_tier = predicted_tier or row[2]
        confidence = confidence if confidence is not None else row[3]

    now = datetime.now(timezone.utc).isoformat()

    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO labeled_prompts
                    (request_id, prompt, system_prompt, predicted_tier, correct_tier, confidence, labeled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (request_id, prompt, system_prompt, predicted_tier, correct_tier, confidence, now))
            conn.commit()
            label_id = cursor.lastrowid
        except Exception as e:
            logger.error("Failed to store label: %s", e, exc_info=True)
            return {"error": str(e)}
        finally:
            conn.close()

    result = {
        "id": label_id,
        "request_id": request_id,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "predicted_tier": predicted_tier,
        "correct_tier": correct_tier,
        "confidence": confidence,
        "labeled_at": now,
    }
    logger.info(
        "Label stored: request=%s predicted=%s correct=%s",
        request_id, predicted_tier, correct_tier,
    )
    return result


def get_recent_requests(limit: int = 20) -> list:
    """Get recent requests from SQLite for interactive labeling."""
    _init_db()
    db_path = _get_db_path()

    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT request_id, prompt, system_prompt, tier, confidence,
                       complexity_score, selected_model, timestamp
                FROM requests
                WHERE type = 'completion' AND prompt IS NOT NULL AND prompt != ''
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def export_labeled_prompts(
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> list:
    """
    Export labeled prompts as a list of dicts.

    Args:
        since: ISO timestamp or date string for start of range (inclusive)
        until: ISO timestamp or date string for end of range (inclusive)

    Returns:
        List of dicts with prompt, system_prompt, predicted_tier, correct_tier,
        confidence, labeled_at, request_id.
    """
    _init_db()
    db_path = _get_db_path()

    query = """
        SELECT request_id, prompt, system_prompt, predicted_tier, correct_tier,
               confidence, labeled_at
        FROM labeled_prompts
        WHERE 1=1
    """
    params: list = []

    if since:
        query += " AND labeled_at >= ?"
        params.append(since)
    if until:
        query += " AND labeled_at <= ?"
        params.append(until)

    query += " ORDER BY labeled_at DESC"

    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
