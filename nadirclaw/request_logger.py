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
                    fallback_reasons TEXT,
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
            
            # Migrate: add optimization columns (idempotent)
            for col, col_type in [
                ("optimization_mode", "TEXT"),
                ("original_tokens", "INTEGER"),
                ("optimized_tokens", "INTEGER"),
                ("tokens_saved", "INTEGER"),
                ("optimizations_applied", "TEXT"),
                ("fallback_reasons", "TEXT"),
            ]:
                try:
                    cursor.execute(f"ALTER TABLE requests ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

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
    fallback_reasons = (
        json.dumps(entry["fallback_reasons"])
        if entry.get("fallback_reasons")
        else None
    )
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

    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO requests (
                    timestamp, request_id, type, status, prompt, selected_model,
                    provider, tier, confidence, complexity_score, classifier_latency_ms,
                    total_latency_ms, prompt_tokens, completion_tokens, total_tokens,
                    cost, daily_spend, response_preview, fallback_used, fallback_reasons, error,
                    tool_count, has_images, has_tools, max_context_tokens,
                    optimization_mode, original_tokens, optimized_tokens,
                    tokens_saved, optimizations_applied
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, request_id, req_type, status, prompt, selected_model,
                provider, tier, confidence, complexity_score, classifier_latency_ms,
                total_latency_ms, prompt_tokens, completion_tokens, total_tokens,
                cost, daily_spend, response_preview, fallback_used, fallback_reasons, error,
                tool_count, has_images, has_tools, max_context_tokens,
                optimization_mode, original_tokens, optimized_tokens,
                tokens_saved, optimizations_applied,
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
