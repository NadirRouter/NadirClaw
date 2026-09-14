"""Serve a structural view for a Claude Code Read the host would truncate.

Claude Code returns at most ~25,000 tokens per `Read` and truncates the rest,
so a large module arrives as a partial page and everything below the cut is
silently missing. This hook answers those reads with a declaration-level view
of the file instead: every definition, decorator, docstring excerpt and module
constant, each tagged with its ORIGINAL source lines.

No model call, no network, no archive. The source file is untouched on disk,
so recovery is a narrowed re-read rather than a copy. Unlike `optimize.py` and
`compress.py`, which rewrite the messages array in the proxy, this runs in the
agent before the file ever enters the conversation.
"""

import ast
import json
from pathlib import Path

# Claude Code serves one Read page up to this many tokens and truncates the
# rest. A file past it never reaches the conversation whole.
READ_TOKEN_CAP = 25000
# Parsing is linear but a hook is given seconds, and ~10 MiB of Python takes
# ~6s to parse. Decline past this rather than stall every read of a huge file.
MAX_VIEW_BYTES = 4 * 1024 * 1024
PYTHON_SUFFIXES = {".py", ".pyi"}


def python_view(text, *, lean=False):
    """A source view, never replacement code. Declarations stay verbatim.

    Returns None when the file cannot be parsed, so the caller preserves the
    original rather than serving a confident view of nothing.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    if not tree.body:
        return None
    lines = text.splitlines(keepends=True)
    rows = []

    def emit(first, last):
        if last >= first:
            location = f"[L{first}-{last}]" if lean else f"[source lines {first}-{last}]"
            rows.append(location + "\n" + "".join(lines[first - 1:last]).rstrip("\r\n"))

    def is_doc(node):
        return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str))

    def emit_doc(node):
        if lean:
            return
        last = min(node.end_lineno, node.lineno + 2)
        emit(node.lineno, last)
        if last < node.end_lineno:
            rows.append(f"[docstring remainder omitted; source lines {last + 1}-{node.end_lineno}]")

    def walk(nodes, prefix=""):
        for node in nodes:
            if is_doc(node):
                emit_doc(node)
                continue
            named = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            if named:
                child = node.body[0]
                body_start = min([child.lineno] + [d.lineno for d in getattr(child, "decorator_list", [])])
                # One-line definitions include their body; never split a line.
                emit(first, max(node.lineno, body_start - 1))
                if body_start == node.lineno:
                    continue
                if isinstance(node, ast.ClassDef):
                    walk(node.body, prefix + node.name + ".")
                else:
                    # Docstrings describe the contract; retain them in the view.
                    first_stmt = node.body[0]
                    if is_doc(first_stmt):
                        emit_doc(first_stmt)
                        body_start = first_stmt.end_lineno + 1
                    if body_start <= node.end_lineno:
                        name = prefix + node.name
                        rows.append(f"[body omitted L{body_start}-{node.end_lineno}]" if lean else
                                    f"[body omitted: {name}; source lines {body_start}-{node.end_lineno}]")
            elif (lean and isinstance(node, (ast.Assign, ast.AnnAssign))
                    and sum(len(line) for line in lines[first - 1:node.end_lineno]) > 256):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = ", ".join(ast.unparse(target) for target in targets)
                rows.append(f"[data omitted: {names}; source lines {first}-{node.end_lineno}]")
            else:
                emit(first, node.end_lineno)

    walk(tree.body)
    header = ("[Lean Python view: L = original source lines; docstrings omitted. "
              "Recover exact source before editing.]\n\n" if lean else "")
    return header + "\n\n".join(rows)


def decide(event, *, read_token_cap=READ_TOKEN_CAP):
    """Map a PreToolUse event to a hook decision.

    Returns (output, report). An empty output means "let the read through";
    every rejection names its reason so `nadirclaw claude read-hook --explain`
    and the tests can assert on it.
    """
    if not isinstance(event, dict) or event.get("hook_event_name") != "PreToolUse":
        return {}, {"reason": "unsupported_event"}
    if event.get("tool_name") != "Read":
        return {}, {"reason": "unsupported_tool"}
    args = event.get("tool_input")
    if not isinstance(args, dict):
        return {}, {"reason": "unsupported_shape"}
    # A narrowed read is already the recovery path; never intercept one.
    if args.get("offset") is not None or args.get("limit") is not None:
        return {}, {"reason": "targeted_read_preserved"}
    path = args.get("file_path")
    if not isinstance(path, str) or Path(path).suffix.lower() not in PYTHON_SUFFIXES:
        return {}, {"reason": "unsupported_language"}
    try:
        if Path(path).stat().st_size > MAX_VIEW_BYTES:
            return {}, {"reason": "oversized_preserved"}
        raw = Path(path).read_bytes()
        # chars/4 under-counts dense source, so this fires only well past the
        # cap. A file just under it keeps normal behavior rather than losing
        # exact bodies the reader could have had.
        if len(raw) // 4 <= read_token_cap:
            return {}, {"reason": "below_read_cap"}
        original = raw.decode("utf-8")
    except (OSError, ValueError):
        return {}, {"reason": "unreadable_preserved"}  # let Read report it
    view = python_view(original)
    if view is not None and len(view) // 4 > read_token_cap:
        view = python_view(original, lean=True) or view
    # A view no smaller than the page the host would serve buys nothing.
    if view is None or len(view) >= len(original):
        return {}, {"reason": "code_structure_unavailable"}
    lines = original.count("\n") + 1
    reason = (
        f"NadirClaw served a structural view of {path} ({lines} lines) instead of a full "
        f"read, which this host would have truncated mid-file at {read_token_cap} tokens.\n"
        "Line numbers in the [source lines N-M] markers are the ORIGINAL file, not this "
        "text. Bodies are omitted: this is not valid replacement code. Re-read "
        f"{path} with offset/limit for the exact lines before editing or quoting them.\n\n"
        + view
    )
    return ({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                    "permissionDecision": "deny",
                                    "permissionDecisionReason": reason}},
            {"reason": "structural_view_served", "served": True,
             "original_chars": len(original), "returned_chars": len(reason)})


def run(stream_in, stream_out):
    """Read one hook event from stdin and write the decision to stdout."""
    try:
        raw = stream_in.read(64 * 1024 * 1024 + 1)
        if len(raw) > 64 * 1024 * 1024:
            return {"reason": "oversized_event"}
        output, report = decide(json.loads(raw))
    except (OSError, ValueError, TypeError, RecursionError):
        # A failed local transform must never block the read it was inspecting.
        return {"reason": "hook_error_read_preserved"}
    if output:
        stream_out.write(json.dumps(output) + "\n")
    return report
