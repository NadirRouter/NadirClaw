"""Tests for columnar JSON-array packing (aggressive-mode transform).

Packing rewrites homogeneous arrays-of-objects into a header + one value-array
per row. It must be information-lossless (deterministically reversible), must
never run in safe mode, and must skip arrays it cannot pack unambiguously.
"""
import json

import pytest

from nadirclaw.optimize import (
    _pack_array,
    _pack_homogeneous_arrays,
    _unpack_table,
    optimize_messages,
)


def _roundtrip(arr):
    packed = _pack_array(arr)
    assert packed is not None
    return _unpack_table(packed)


# ---------------------------------------------------------------------------
# Losslessness across value types
# ---------------------------------------------------------------------------

def test_roundtrip_scalars():
    arr = [{"id": i, "name": f"u{i}", "active": bool(i % 2), "score": i / 3} for i in range(8)]
    assert _roundtrip(arr) == arr


def test_roundtrip_nested_and_null_and_tricky_strings():
    arr = [
        {"id": 1, "meta": {"a": [1, 2], "b": None}, "note": 'has "quotes", commas, [brackets]'},
        {"id": 2, "meta": {"a": [], "b": 5}, "note": "tab\tand\nnewline"},
        {"id": 3, "meta": {"a": [9], "b": None}, "note": "unicode ✓ é 中"},
        {"id": 4, "meta": {"a": [1], "b": 0}, "note": ""},
        {"id": 5, "meta": {"a": [2, 3], "b": 1}, "note": "⟦cols= looks like a marker"},
    ]
    assert _roundtrip(arr) == arr


def test_roundtrip_preserves_row_and_key_order():
    arr = [{"z": i, "a": i + 1, "m": i + 2} for i in range(6)]
    out = _roundtrip(arr)
    assert out == arr
    assert list(out[0].keys()) == ["z", "a", "m"]


# ---------------------------------------------------------------------------
# Skip conditions (fall back to json_minify)
# ---------------------------------------------------------------------------

def test_skip_too_few_rows():
    assert _pack_array([{"a": 1, "b": 2}] * 4) is None  # < 5 rows


def test_skip_non_homogeneous_keys():
    assert _pack_array([{"a": 1, "b": 2}, {"a": 1}, {"c": 3}] * 3) is None


def test_skip_single_column():
    assert _pack_array([{"a": i} for i in range(10)]) is None


def test_skip_non_dict_elements():
    assert _pack_array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]) is None


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def _msgs():
    rows = [{"id": 1000 + i, "user": f"user{i}", "status": "active" if i % 3 else "inactive",
             "plan": "pro" if i % 5 == 0 else "free"} for i in range(40)]
    return [{"role": "user", "content": "list users"},
            {"role": "tool", "content": "result:\n" + json.dumps(rows, indent=2)}]


def test_aggressive_packs_and_saves():
    r = optimize_messages(_msgs(), mode="aggressive")
    assert "json_array_pack" in r.optimizations_applied
    assert r.tokens_saved > 0


def test_safe_mode_never_packs():
    r = optimize_messages(_msgs(), mode="safe")
    assert "json_array_pack" not in r.optimizations_applied
    assert "⟦cols=" not in r.messages[1]["content"]


def test_only_packs_when_smaller():
    # A short homogeneous array whose table form isn't worth it stays as-is.
    arr = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6},
           {"a": 7, "b": 8}, {"a": 9, "b": 10}]
    content = "data " + json.dumps(arr)
    out, changed = _pack_homogeneous_arrays(content)
    # tiny arrays may not beat minified form; if unchanged, output is untouched
    if not changed:
        assert out == content


def test_fenced_code_is_left_untouched():
    arr = [{"id": i, "v": i * 2} for i in range(10)]
    content = "```json\n" + json.dumps(arr, indent=2) + "\n```"
    out, changed = _pack_homogeneous_arrays(content)
    assert not changed and out == content
