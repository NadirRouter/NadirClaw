"""Real-data benchmark: optimizer backends on public coding + chat datasets.

- Chat: allenai/WildChat-1M (real multi-turn user<->assistant conversations)
- Coding/tools: glaiveai/glaive-function-calling-v2 (tool schemas + function calls + JSON)

Compares: native-safe (lossless, ships today), Pro-aggressive (native ceiling),
headroom (new opt-in backend). Single tiktoken estimator for all => fair.
"""
import json, os, re, sys, time, collections, urllib.request

# Resolve the NadirClaw repo root from this file, and the sibling Nadir package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_NADIRCLAW = os.path.dirname(_HERE)
sys.path.insert(0, _NADIRCLAW)
_NADIR = os.path.join(os.path.dirname(_NADIRCLAW), "Nadir")
if os.path.isdir(_NADIR):
    sys.path.insert(0, _NADIR)

import nadirclaw.optimize as claw
try:
    import nadir.optimize as pro
except Exception:                       # Nadir Pro not on path — fall back to native
    pro = claw

est = claw._estimate_tokens_messages

N = 200          # conversations per dataset
CACHE = os.environ.get("BENCH_CACHE_DIR", "/tmp")


def _fetch(dataset, config, split, dest, total=N):
    """Fetch rows from the HF datasets-server (no full dataset download). Cached to disk."""
    if os.path.exists(dest):
        return
    rows = []
    for off in range(0, total, 100):
        url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}"
               f"&config={config}&split={split}&offset={off}&length=100")
        for _ in range(3):
            try:
                with urllib.request.urlopen(url, timeout=40) as r:
                    rows += [x["row"] for x in json.load(r).get("rows", [])]
                break
            except Exception:
                time.sleep(2)
    json.dump(rows, open(dest, "w"))


_WILDCHAT = os.path.join(CACHE, "ds_wildchat.json")
_GLAIVE = os.path.join(CACHE, "ds_glaive.json")
_fetch("allenai/WildChat-1M", "default", "train", _WILDCHAT)
_fetch("glaiveai/glaive-function-calling-v2", "default", "train", _GLAIVE)


def load_wildchat():
    rows = json.load(open(_WILDCHAT))[:N]
    convs = []
    for r in rows:
        msgs = [{"role": t.get("role", "user"), "content": t.get("content") or ""}
                for t in (r.get("conversation") or []) if isinstance(t, dict)]
        msgs = [m for m in msgs if isinstance(m["content"], str) and m["content"]]
        if len(msgs) >= 2:
            convs.append(msgs)
    return convs


def load_glaive():
    rows = json.load(open(_GLAIVE))[:N]
    convs = []
    marker = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)", re.I)
    rolemap = {"USER": "user", "ASSISTANT": "assistant", "FUNCTION RESPONSE": "tool"}
    for r in rows:
        sysm = (r.get("system") or "").strip()
        if sysm.upper().startswith("SYSTEM:"):
            sysm = sysm[7:].strip()
        msgs = [{"role": "system", "content": sysm}] if sysm else []
        chat = r.get("chat") or ""
        parts = marker.split(chat)
        # parts: ['', 'USER:', ' ...', 'ASSISTANT:', ' ...', ...]
        i = 1
        while i < len(parts) - 0:
            lab = parts[i].rstrip(":").upper()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if lab in rolemap and content:
                msgs.append({"role": rolemap[lab], "content": content})
            i += 2
        if len(msgs) >= 2:
            convs.append(msgs)
    return convs


def bench(convs, runners):
    out = {name: [0, 0] for name in runners}          # name -> [orig, after]
    transforms = {name: collections.Counter() for name in runners}
    for msgs in convs:
        for name, fn in runners.items():
            r = fn([{**m} for m in msgs])
            out[name][0] += r.original_tokens
            out[name][1] += r.optimized_tokens
            for t in r.optimizations_applied:
                transforms[name][t.split(":")[1] if t.startswith("headroom:") else t] += 1
    return out, transforms


RUNNERS = {
    "native-safe":    lambda m: claw.optimize_messages(m, mode="safe", backend="native"),
    "pro-aggressive": lambda m: pro.optimize_messages(m, mode="aggressive", backend="native"),
    "headroom":       lambda m: claw.optimize_messages(m, mode="safe", backend="headroom"),
}

for label, loader in [("CHAT — WildChat-1M", load_wildchat), ("CODING/TOOLS — glaive-function-calling-v2", load_glaive)]:
    convs = loader()
    t0 = time.time()
    res, tf = bench(convs, RUNNERS)
    base = res["native-safe"][0]
    print(f"\n### {label}  ({len(convs)} conversations, {base:,} raw tokens, {time.time()-t0:.0f}s)")
    print(f"{'backend':<18}{'after':>10}{'saved':>9}{'%':>7}   top transforms")
    for name in RUNNERS:
        o, a = res[name]
        top = ", ".join(f"{k}:{v}" for k, v in tf[name].most_common(4))
        print(f"{name:<18}{a:>10,}{o-a:>9,}{100*(o-a)/max(1,o):>6.1f}%   {top}")
