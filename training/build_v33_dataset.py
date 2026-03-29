"""
Build blended_v33.jsonl for v3.3 training.

Fix: replace BeaverTails-style safe examples in blended_v32 with
     agent-action safe examples from Agent-SafetyBench + Gemini synth.

Sources for safe class:
  - data/safe_agent_actions.jsonl  (Agent-SafetyBench, fulfillable=1)
  - data/safe_synth.jsonl          (Gemini-generated agent-action safe)

All other 8 categories kept as-is from blended_v32.jsonl.
Target: 2500 examples per label (oversample safe if needed).

Usage:
    python -m training.build_v33_dataset
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

TARGET_PER_LABEL = 2500
SEED = 42

def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path)]

def main() -> None:
    rng = random.Random(SEED)

    # ── Load unsafe categories from blended_v32 (drop safe entirely) ──────────
    print("[v33] Loading blended_v32.jsonl (dropping safe class)...")
    v32 = load_jsonl("data/blended_v32.jsonl")
    unsafe_examples = [ex for ex in v32 if ex["label"] != "safe"]
    counts = Counter(ex["label"] for ex in unsafe_examples)
    print(f"[v33] Unsafe examples: {len(unsafe_examples)}")
    for label, n in sorted(counts.items()):
        print(f"  {label:<25} {n}")

    # ── Load agent-action safe examples ───────────────────────────────────────
    print("\n[v33] Loading safe examples...")
    safe_examples: list[dict] = []

    asb_path = "data/safe_agent_actions.jsonl"
    if Path(asb_path).exists():
        asb = load_jsonl(asb_path)
        safe_examples.extend(asb)
        print(f"  Agent-SafetyBench: {len(asb)}")

    synth_path = "data/safe_synth.jsonl"
    if Path(synth_path).exists():
        synth = load_jsonl(synth_path)
        safe_examples.extend(synth)
        print(f"  Gemini synth: {len(synth)}")

    print(f"  Total safe before oversample: {len(safe_examples)}")

    # ── Oversample safe to TARGET_PER_LABEL ───────────────────────────────────
    if len(safe_examples) < TARGET_PER_LABEL:
        extra = TARGET_PER_LABEL - len(safe_examples)
        safe_examples += rng.choices(safe_examples, k=extra)
        print(f"  Oversampled to: {len(safe_examples)} (+{extra})")
    else:
        rng.shuffle(safe_examples)
        safe_examples = safe_examples[:TARGET_PER_LABEL]
        print(f"  Capped to: {len(safe_examples)}")

    # ── Cap each unsafe label to TARGET_PER_LABEL ─────────────────────────────
    by_label: dict[str, list] = {}
    for ex in unsafe_examples:
        by_label.setdefault(ex["label"], []).append(ex)

    balanced_unsafe: list[dict] = []
    for label, examples in by_label.items():
        rng.shuffle(examples)
        balanced_unsafe.extend(examples[:TARGET_PER_LABEL])

    # ── Combine and shuffle ───────────────────────────────────────────────────
    all_examples = balanced_unsafe + safe_examples
    rng.shuffle(all_examples)

    output = "data/blended_v33.jsonl"
    with open(output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    final_counts = Counter(ex["label"] for ex in all_examples)
    print(f"\n[v33] ══ blended_v33.jsonl ══")
    print(f"  Total: {len(all_examples)}")
    for label, n in sorted(final_counts.items()):
        bar = "█" * (n // 100)
        print(f"  {label:<25} {n:>4}  {bar}")
    print(f"\n  Saved → {output}")


if __name__ == "__main__":
    main()
