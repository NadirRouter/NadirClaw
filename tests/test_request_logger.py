"""
Tests for the SQLite request logger - basic smoke test.
"""

import json
import sqlite3
import tempfile
from pathlib import Path

from nadirclaw import request_logger


def test_basic_logging_works():
    """Smoke test: verify logging creates a database and writes records."""
    # Create a temp directory manually
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db = Path(tmpdir) / "test_requests.db"
        
        # Override the db path in the module
        original_path = request_logger._db_path
        original_initialized = request_logger._db_initialized
        
        try:
            request_logger._db_path = temp_db
            request_logger._db_initialized = False
            
            # Log a request
            entry = {
                "request_id": "test-123",
                "type": "completion",
                "status": "ok",
                "prompt": "Hello world",
                "selected_model": "gpt-3.5-turbo",
                "provider": "openai",
                "tier": "simple",
                "confidence": 0.85,
                "complexity_score": 0.2,
                "classifier_latency_ms": 5,
                "total_latency_ms": 1200,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost": 0.0015,
                "daily_spend": 0.45,
                "response_preview": "Hello! How can I help?",
                "fallback_reasons": [{
                    "model": "model-primary",
                    "error_type": "RateLimitExhausted",
                    "message": "Rate limit exhausted for model-primary (retry in 60s)",
                }],
            }
            
            request_logger.log_request(entry)
            
            # Verify it was logged
            assert temp_db.exists()
            
            conn = sqlite3.connect(str(temp_db))
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT request_id, selected_model, cost, fallback_reasons FROM requests WHERE request_id = ?",
                ("test-123",),
            )
            row = cursor.fetchone()
            
            assert row is not None
            assert row[0] == "test-123"
            assert row[1] == "gpt-3.5-turbo"
            assert row[2] == 0.0015
            assert json.loads(row[3]) == entry["fallback_reasons"]
            
            conn.close()
            
        finally:
            # Restore original state
            request_logger._db_path = original_path
            request_logger._db_initialized = original_initialized


def test_imports_cleanly():
    """Verify the module imports without errors."""
    from nadirclaw import request_logger
    assert hasattr(request_logger, 'log_request')
    assert hasattr(request_logger, 'get_request_count')
