#!/usr/bin/env python3
"""
Re-run LLM-as-judge on saved benchmark results.
Reads benchmark_cs50_results.json, judges non-baseline entries, updates file.
"""

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import litellm
litellm.suppress_debug_info = True
os.environ.setdefault("LITELLM_LOG", "ERROR")

import warnings
warnings.filterwarnings("ignore")

# Load Google API key
env_path = os.path.join(os.path.dirname(__file__), "..", "Horizen", ".env.local")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("GOOGLE_API_KEY="):
                os.environ.setdefault("GOOGLE_API_KEY", line.split("=", 1)[1])

JUDGE_MODEL = "gemini/gemini-2.0-flash"
BASELINE_SHORT = "gemini-2.5-pro"


def extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM output."""
    text = text.strip()
    # Remove markdown code fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try finding JSON object in text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


async def judge_one(ticket: str, response_nadir: str, response_baseline: str) -> dict:
    """LLM-as-judge: compare NadirClaw response vs baseline."""
    judge_prompt = f"""You are an expert quality evaluator for customer support responses.

A customer sent this ticket:
<ticket>
{ticket}
</ticket>

Two AI customer support agents responded. Rate EACH response on a 1-5 scale across these dimensions:
1. Accuracy — Is the information correct and relevant?
2. Completeness — Does it fully address the customer's question?
3. Actionability — Does it give clear, actionable steps?
4. Tone — Is it professional, empathetic, and appropriate?
5. Efficiency — Is it concise without being too terse?

Response A (cost-optimized model):
<response_a>
{response_nadir[:3000]}
</response_a>

Response B (premium baseline model):
<response_b>
{response_baseline[:3000]}
</response_b>

Return ONLY valid JSON with this structure:
{{
  "response_a": {{"accuracy": 4, "completeness": 4, "actionability": 4, "tone": 4, "efficiency": 4, "overall": 4}},
  "response_b": {{"accuracy": 4, "completeness": 4, "actionability": 4, "tone": 4, "efficiency": 4, "overall": 4}},
  "verdict": "A_better",
  "reasoning": "Brief explanation"
}}

verdict must be exactly one of: "A_better", "B_better", "tie"
All scores must be integers 1-5."""

    try:
        response = await asyncio.to_thread(
            litellm.completion,
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=1024,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        result = extract_json(text)
        if "verdict" not in result:
            return {"error": "No verdict in response", "raw": text[:200], "verdict": "error"}
        return result
    except Exception as e:
        return {"error": str(e)[:200], "verdict": "error"}


async def main():
    results_file = os.path.join(os.path.dirname(__file__), "benchmark_cs50_results.json")
    with open(results_file) as f:
        data = json.load(f)

    results = data["results"]

    # Find entries that used a different model than baseline (need judging)
    judgeable = [
        r for r in results
        if r["routed_model"] != BASELINE_SHORT
        and r["nadir"]["success"]
        and r["opus"]["success"]
    ]

    print(f"Found {len(judgeable)} prompts to judge (routed to cheaper model)")
    print(f"Skipping {len(results) - len(judgeable)} prompts (same model or failed)\n")

    judge_results = []
    for r in judgeable:
        idx = r["index"]
        ticket = r["ticket"]
        nadir_out = r["nadir"]["output"]
        opus_out = r["opus"]["output"]

        print(f"  Judging #{idx} ({r['expected_tier']}, routed to {r['routed_model']})...", end=" ", flush=True)
        judge = await judge_one(ticket, nadir_out, opus_out)

        verdict = judge.get("verdict", "error")
        a_score = judge.get("response_a", {}).get("overall", "?")
        b_score = judge.get("response_b", {}).get("overall", "?")
        print(f"NadirClaw={a_score}/5, Baseline={b_score}/5 → {verdict}")

        judge_results.append({
            "index": idx,
            "tier": r["expected_tier"],
            "routed_model": r["routed_model"],
            "judge": judge,
        })

        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

    # Update the results file
    data["judge_results"] = judge_results
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Print summary
    valid = [j for j in judge_results if j["judge"].get("verdict") != "error"]
    print(f"\n{'='*70}")
    print(f"  LLM-AS-JUDGE RESULTS ({len(valid)} valid judgments)")
    print(f"{'='*70}")

    if not valid:
        print("  No valid judgments.")
        return

    a_wins = sum(1 for j in valid if j["judge"]["verdict"] == "A_better")
    b_wins = sum(1 for j in valid if j["judge"]["verdict"] == "B_better")
    ties = sum(1 for j in valid if j["judge"]["verdict"] == "tie")
    print(f"\n  NadirClaw wins: {a_wins}  |  Baseline wins: {b_wins}  |  Ties: {ties}")

    avg_a = sum(j["judge"]["response_a"]["overall"] for j in valid) / len(valid)
    avg_b = sum(j["judge"]["response_b"]["overall"] for j in valid) / len(valid)
    print(f"  Avg quality: NadirClaw={avg_a:.2f}/5, Baseline={avg_b:.2f}/5")

    quality_gap = avg_b - avg_a
    print(f"  Quality gap: {quality_gap:+.2f} (negative = NadirClaw better)")

    # Per-dimension
    dims = ["accuracy", "completeness", "actionability", "tone", "efficiency"]
    print(f"\n  {'Dimension':20s} {'NadirClaw':>10s} {'Baseline':>10s} {'Gap':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for dim in dims:
        a_avg = sum(j["judge"]["response_a"][dim] for j in valid) / len(valid)
        b_avg = sum(j["judge"]["response_b"][dim] for j in valid) / len(valid)
        gap = b_avg - a_avg
        print(f"  {dim:20s} {a_avg:10.2f} {b_avg:10.2f} {gap:+10.2f}")

    # Cost saved on judged prompts
    judged_indices = {j["index"] for j in valid}
    judged_results = [r for r in results if r["index"] in judged_indices]
    nadir_cost = sum(r["nadir"]["cost"] for r in judged_results)
    baseline_cost = sum(r["opus"]["cost"] for r in judged_results)
    saved = baseline_cost - nadir_cost
    pct = saved / baseline_cost * 100 if baseline_cost > 0 else 0

    print(f"\n  Cost on judged prompts:")
    print(f"    Baseline: ${baseline_cost:.4f}")
    print(f"    NadirClaw: ${nadir_cost:.4f}")
    print(f"    Saved: ${saved:.4f} ({pct:.1f}%)")
    print(f"    Quality/$ ratio: NadirClaw={avg_a/nadir_cost:.0f}/$ vs Baseline={avg_b/baseline_cost:.0f}/$" if nadir_cost > 0 and baseline_cost > 0 else "")

    print(f"\n  Results saved to: {results_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
