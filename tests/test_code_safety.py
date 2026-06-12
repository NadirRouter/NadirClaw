"""Regression tests: compression must not corrupt source code.

Raw (unfenced) code arrives in coding-agent traffic as file-read tool outputs.
Whitespace normalization must preserve leading indentation, or it flattens
Python/YAML/diffs into invalid syntax. Fenced code must stay byte-identical.
"""
import ast
import textwrap

import pytest

from nadirclaw.optimize import optimize_messages

PY_SRC = textwrap.dedent('''\
    import json


    def process(record, config):
        result = {}
        for key, spec in config.items():
            value = record.get(key)
            if value is None:
                if spec.get("required"):
                    raise ValueError(key)
                continue
            result[key] = value
        return result


    class Validator:
        def __init__(self, schema):
            self.schema = schema

        def check(self, data):
            for field in self.schema:
                if field not in data:
                    return False
            return True
''')


@pytest.mark.parametrize("mode", ["safe", "aggressive"])
def test_raw_code_stays_valid_python(mode):
    """Unfenced source code in a tool message must remain parseable."""
    msgs = [{"role": "tool", "content": PY_SRC}]
    out = optimize_messages(msgs, mode=mode).messages[0]["content"]
    ast.parse(out)  # raises SyntaxError if indentation was flattened


@pytest.mark.parametrize("mode", ["safe", "aggressive"])
def test_leading_indentation_preserved(mode):
    msgs = [{"role": "tool", "content": PY_SRC}]
    out = optimize_messages(msgs, mode=mode).messages[0]["content"]
    # The deepest line is indented 16 spaces; it must keep its indentation.
    line = next(ln for ln in out.split("\n") if "raise ValueError" in ln)
    assert line.startswith("                raise ValueError")


def test_fenced_code_is_byte_identical():
    snippet = "def f(x):\n    if x:\n        return  x  +  1\n    return 0"
    content = "Here:\n```python\n" + snippet + "\n```"
    out = optimize_messages([{"role": "assistant", "content": content}], mode="safe").messages[0]["content"]
    assert snippet in out  # fenced block untouched, including its interior spacing


def test_interior_spaces_still_collapse_in_prose():
    # The fix only protects leading indentation; prose double-spaces still collapse.
    content = "this   sentence   has   wide   gaps and is long enough to process"
    out = optimize_messages([{"role": "user", "content": content}], mode="safe").messages[0]["content"]
    assert "   " not in out
