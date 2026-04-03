"""
Feedback and quality scoring engine for NadirClaw.

Handles:
- User feedback storage (ratings, misroute flags)
- Passive quality scoring after every completion
- Feedback statistics and reporting
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from nadirclaw.settings import settings

logger = logging.getLogger("nadirclaw")

_db_lock = Lock()
_tables_initialized = False

# ---------------------------------------------------------------------------
# Valid values
# ---------------------------------------------------------------------------

VALID_REASONS = (
    "misrouted", "slow", "bad_quality", "good", "other",
    "hallucinated", "refused", "cost_spike", "wrong_language", "incomplete",
)
VALID_TIERS = ("simple", "mid", "medium", "complex", "reasoning", "direct", "free")


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

def _get_db_path() -> Path:
    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "requests.db"


def _init_feedback_tables() -> None:
    """Create feedback and quality_scores tables if they don't exist."""
    global _tables_initialized
    if _tables_initialized:
        return

    db_path = _get_db_path()
    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
                    reason TEXT,
                    correct_tier TEXT,
                    correct_model TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_request_id
                ON feedback(request_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_created_at
                ON feedback(created_at)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    score_source TEXT NOT NULL,
                    quality_score REAL,
                    details TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_quality_request_id
                ON quality_scores(request_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_quality_created_at
                ON quality_scores(created_at)
            """)

            conn.commit()
            _tables_initialized = True
            logger.debug("Feedback tables initialized at %s", db_path)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

def request_exists(request_id: str) -> bool:
    """Check whether a request_id exists in the requests table."""
    _init_feedback_tables()
    db_path = _get_db_path()

    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM requests WHERE request_id = ? LIMIT 1",
                (request_id,),
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Feedback storage
# ---------------------------------------------------------------------------

def store_feedback(
    request_id: str,
    rating: Optional[int] = None,
    reason: Optional[str] = None,
    correct_tier: Optional[str] = None,
    correct_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Store user feedback for a request.

    Returns dict with stored feedback data or error.
    """
    _init_feedback_tables()
    db_path = _get_db_path()

    # Validate rating
    if rating is not None and not (1 <= rating <= 5):
        return {"error": "Rating must be between 1 and 5"}

    # Validate reason
    if reason is not None and reason not in VALID_REASONS:
        return {"error": f"Reason must be one of: {', '.join(VALID_REASONS)}"}

    # Validate tier
    if correct_tier is not None and correct_tier not in VALID_TIERS:
        return {"error": f"Tier must be one of: {', '.join(VALID_TIERS)}"}

    now = datetime.now(timezone.utc).isoformat()

    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback (request_id, rating, reason, correct_tier, correct_model, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (request_id, rating, reason, correct_tier, correct_model, now))
            conn.commit()
            feedback_id = cursor.lastrowid
        finally:
            conn.close()

    result = {
        "id": feedback_id,
        "request_id": request_id,
        "rating": rating,
        "reason": reason,
        "correct_tier": correct_tier,
        "correct_model": correct_model,
        "created_at": now,
    }
    logger.info(
        "Feedback stored: request=%s rating=%s reason=%s",
        request_id, rating, reason,
    )
    return result


# ---------------------------------------------------------------------------
# Passive quality scoring
# ---------------------------------------------------------------------------

def compute_quality_score(
    request_id: str,
    status: str,
    response_content: str,
    fallback_used: Optional[str],
    total_latency_ms: int,
) -> float:
    """
    Compute a passive quality score from completion signals and store it.

    Scoring rules:
    - error: score = 0.0
    - normal completion: base 1.0
    - empty response: -0.5
    - fallback used: -0.2
    - very slow (>10s): -0.1

    Returns the computed score.
    """
    _init_feedback_tables()

    details: Dict[str, Any] = {}

    if status == "error":
        score = 0.0
        details["signal"] = "error"
    else:
        score = 1.0
        details["signal"] = "normal_completion"

        if not response_content or not response_content.strip():
            score -= 0.5
            details["empty_response"] = True

        if fallback_used:
            score -= 0.2
            details["fallback_used"] = fallback_used

        if total_latency_ms > 10_000:
            score -= 0.1
            details["very_slow"] = True
            details["latency_ms"] = total_latency_ms

    score = max(0.0, min(1.0, score))
    details["final_score"] = score

    # Store in DB
    db_path = _get_db_path()
    now = datetime.now(timezone.utc).isoformat()

    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO quality_scores (request_id, score_source, quality_score, details, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (request_id, "passive", score, json.dumps(details), now))
            conn.commit()
        except Exception as e:
            logger.error("Failed to store quality score: %s", e)
        finally:
            conn.close()

    return score


# ---------------------------------------------------------------------------
# Feedback statistics
# ---------------------------------------------------------------------------

def get_feedback_stats() -> Dict[str, Any]:
    """
    Compute feedback statistics for reporting.

    Returns:
        - total_feedback: total feedback entries
        - average_rating: mean rating across all feedback
        - reason_counts: {reason: count} breakdown
        - misroute_rate: fraction of feedback with reason='misrouted'
        - avg_quality_7d: average passive quality score over last 7 days
        - total_requests_7d: total requests in the last 7 days
    """
    _init_feedback_tables()
    db_path = _get_db_path()

    stats: Dict[str, Any] = {
        "total_feedback": 0,
        "average_rating": None,
        "reason_counts": {},
        "misroute_rate": None,
        "avg_quality_7d": None,
        "total_requests_7d": 0,
    }

    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()

            # Total feedback count
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total = cursor.fetchone()[0]
            stats["total_feedback"] = total

            # Average rating (only non-null ratings)
            cursor.execute("SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL")
            avg = cursor.fetchone()[0]
            stats["average_rating"] = round(avg, 2) if avg is not None else None

            # Reason breakdown
            cursor.execute("""
                SELECT reason, COUNT(*) FROM feedback
                WHERE reason IS NOT NULL
                GROUP BY reason
                ORDER BY COUNT(*) DESC
            """)
            reason_counts = {}
            for row in cursor.fetchall():
                reason_counts[row[0]] = row[1]
            stats["reason_counts"] = reason_counts

            # Misroute rate
            misrouted = reason_counts.get("misrouted", 0)
            stats["misroute_rate"] = round(misrouted / total, 4) if total > 0 else 0.0

            # Average quality score over last 7 days
            cursor.execute("""
                SELECT AVG(quality_score), COUNT(*)
                FROM quality_scores
                WHERE score_source = 'passive'
                  AND created_at >= datetime('now', '-7 days')
            """)
            row = cursor.fetchone()
            if row and row[0] is not None:
                stats["avg_quality_7d"] = round(row[0], 4)
                stats["total_requests_7d"] = row[1]

        except sqlite3.OperationalError as e:
            # Tables might not exist yet if no requests have been made
            logger.debug("Feedback stats query failed (tables may not exist): %s", e)
        finally:
            conn.close()

    return stats
