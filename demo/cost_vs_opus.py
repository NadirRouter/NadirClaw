"""
Demo: show Nadir routing in action, comparing per-prompt cost vs always-Opus.

Designed for screen recording. Runs in ~5 seconds, no API keys needed
(uses `nadirclaw classify` which is local, embedding-only).

Usage:
    pip install nadirclaw
    python demo/cost_vs_opus.py
"""

import json
import os
import shutil
import subprocess
import sys

from rich.console import Console
from rich.table import Table
from rich.text import Text


PROMPTS = [
    "What is 2+2?",
    "Format this JSON: {'name':'alice','age':30,'roles':['admin','user']}",
    "Add a one-line docstring to this function",
    "Write a Python function that deduplicates a list of dicts by key",
    "Refactor this auth module to use dependency injection and add unit tests",
    "Design a distributed system for real-time trading with exactly-once semantics",
]

DEMO_ENV = {
    "NADIRCLAW_SIMPLE_MODEL": "haiku",
    "NADIRCLAW_MID_MODEL": "sonnet",
    "NADIRCLAW_COMPLEX_MODEL": "opus",
    "NADIRCLAW_TIER_THRESHOLDS": "0.35,0.65",
}

ALIASES = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
    "flash": "gemini-2.5-flash",
    "gemini-pro": "gemini-2.5-pro",
    "gpt4": "gpt-4.1",
    "gpt5": "gpt-5.2",
}

PRICING_PER_M = {
    "claude-opus-4-6": (15.0, 75.0),
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3-flash-preview": (0.30, 2.50),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5.2": (5.0, 15.0),
    "openai-codex/gpt-5.3-codex": (5.0, 15.0),
}
BENCHMARK_MODEL = "claude-opus-4-6"
ASSUMED_OUTPUT_TOKENS = 180


def resolve_model(model: str) -> str:
    if model in ALIASES:
        return ALIASES[model]
    return model


def price_for(model: str) -> tuple[float, float]:
    model = resolve_model(model)
    if model in PRICING_PER_M:
        return PRICING_PER_M[model]
    for key, val in PRICING_PER_M.items():
        if model.startswith(key) or key.startswith(model):
            return val
    return PRICING_PER_M[BENCHMARK_MODEL]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = price_for(model)
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


def classify(prompt: str) -> dict:
    env = {**os.environ, **DEMO_ENV}
    raw = subprocess.check_output(
        ["nadirclaw", "classify", "--format", "json", prompt],
        text=True,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    return json.loads(raw)


def estimate_input_tokens(prompt: str) -> int:
    return max(8, int(len(prompt.split()) * 1.3))


def main() -> int:
    console = Console()

    if shutil.which("nadirclaw") is None:
        console.print("[red]nadirclaw is not on PATH. Run: pip install nadirclaw[/red]")
        return 1

    console.print()
    console.print(Text("Nadir routing demo", style="bold cyan"))
    console.print(
        Text(
            f"Benchmark: always route everything to {BENCHMARK_MODEL}.\n"
            f"Compare: Nadir routes each prompt to the right tier.",
            style="dim",
        )
    )
    console.print()

    table = Table(show_lines=False, header_style="bold")
    table.add_column("Prompt", max_width=52, overflow="fold")
    table.add_column("Tier", style="cyan")
    table.add_column("Routed to", style="green")
    table.add_column("Nadir cost", justify="right")
    table.add_column("Opus-only", justify="right", style="dim")

    total_nadir = 0.0
    total_opus = 0.0

    for prompt in PROMPTS:
        result = classify(prompt)
        routed_model = result.get("model", "unknown")
        tier = result.get("tier", "?")
        in_tok = estimate_input_tokens(prompt)
        n_cost = cost_usd(routed_model, in_tok, ASSUMED_OUTPUT_TOKENS)
        o_cost = cost_usd(BENCHMARK_MODEL, in_tok, ASSUMED_OUTPUT_TOKENS)
        total_nadir += n_cost
        total_opus += o_cost
        table.add_row(
            prompt,
            tier,
            routed_model,
            f"${n_cost:.5f}",
            f"${o_cost:.5f}",
        )

    table.add_section()
    saved = total_opus - total_nadir
    pct = (saved / total_opus) * 100 if total_opus else 0
    table.add_row(
        Text(f"Total across {len(PROMPTS)} prompts", style="bold"),
        "",
        "",
        Text(f"${total_nadir:.4f}", style="bold green"),
        Text(f"${total_opus:.4f}", style="bold dim"),
    )

    console.print(table)
    console.print()
    console.print(
        f"[bold green]Saved ${saved:.4f} ({pct:.0f}%) vs always-Opus.[/bold green] "
        f"[dim]Extrapolated to 10k requests/day: ~${saved * 2000:.0f}/mo.[/dim]"
    )
    console.print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
