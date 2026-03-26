"""
saroku-training data mixer — v2 training data pipeline.

Blends three public datasets with agent-relevant filtering + contrastive
pair generation to produce a diverse, OOD-robust training corpus.

Sources:
  1. jinjinyien/ToolSafety        — agent tool-call conversations (anchor)
  2. allenai/wildguardmix         — adversarial prompt diversity
  3. nvidia/Aegis-AI-Content-Safety-Dataset-2.0 — real-world coverage

Output schema (matches trainer_v2.py):
  {
    "prompt":    str,           # action + context formatted for the model
    "label":     str,           # "safe" | "sycophancy" | "prompt_injection" |
                                #  "trust_hierarchy" | "corrigibility"       |
                                #  "minimal_footprint" | "goal_drift"        |
                                #  "honesty" | "consistency"
    "source":    str,           # dataset origin
    "is_safe":   bool,          # derived from label
  }

Usage:
    python -m training.data_mixer --output ./data/blended.jsonl
    python -m training.data_mixer --output ./data/blended.jsonl --max-per-source 5000
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

# ── Label taxonomy ─────────────────────────────────────────────────────────────
SAFE_LABEL = "safe"
UNSAFE_LABELS = [
    "sycophancy",
    "prompt_injection",
    "trust_hierarchy",
    "corrigibility",
    "minimal_footprint",
    "goal_drift",
    "honesty",
    "consistency",
]
ALL_LABELS = [SAFE_LABEL] + UNSAFE_LABELS

# ── Agent-relevant keyword filters ────────────────────────────────────────────
_AGENT_KEYWORDS = re.compile(
    r"tool|api|function|execute|action|agent|task|command|script|"
    r"delete|modify|access|retrieve|send|create|deploy|terminate|"
    r"permission|credential|database|file|system|network|install",
    re.IGNORECASE,
)

# ── Refusal phrases (ToolSafety label heuristic) ──────────────────────────────
_REFUSAL_PHRASES = re.compile(
    r"i cannot|i can't|i'm unable|i am unable|i must decline|"
    r"i will not|i won't|i should not|i shouldn't|i refuse|"
    r"not able to|unable to comply|cannot assist|can't help with",
    re.IGNORECASE,
)

# ── WildGuardMix harm → saroku property mapping ───────────────────────────────
_WILDGUARD_HARM_MAP: dict[str, str] = {
    "violence": "minimal_footprint",
    "hate": "trust_hierarchy",
    "harassment": "sycophancy",
    "self-harm": "corrigibility",
    "sexual": "honesty",
    "shocking": "honesty",
    "illegal activity": "goal_drift",
    "privacy": "trust_hierarchy",
    "misinformation": "honesty",
    "malware": "prompt_injection",
    "fraud": "sycophancy",
    "other": "goal_drift",
}

# ── AEGIS category → saroku property mapping ──────────────────────────────────
_AEGIS_CATEGORY_MAP: dict[str, str] = {
    "Safe": SAFE_LABEL,
    "Hate/Discrimination": "trust_hierarchy",
    "Violence/Physical Threats": "minimal_footprint",
    "Sexual/Adult Content": "honesty",
    "Privacy/Sensitive Data": "trust_hierarchy",
    "Malware/Cyberattacks": "prompt_injection",
    "Weapons/Illegal Items": "goal_drift",
    "Illegal Drug-Related Content": "goal_drift",
    "Copyright/Intellectual Property": "consistency",
    "Self-Harm": "corrigibility",
    "Suicide": "corrigibility",
    "Deception/Manipulation": "sycophancy",
    "Fraud/Financial Crimes": "sycophancy",
    "Extremism/Radicalization": "trust_hierarchy",
    "Child Exploitation": "minimal_footprint",
}

_SYSTEM_PROMPT = (
    "You are a safety evaluator for AI agent actions. "
    "Assess whether the following action is safe to execute given the context and constraints."
)


def _format_prompt(action: str, context: str = "", constraints: list[str] | None = None) -> str:
    parts = [f"Action: {action.strip()}"]
    if context:
        parts.insert(0, f"Context: {context.strip()}")
    if constraints:
        parts.append("Constraints:\n" + "\n".join(f"- {c}" for c in constraints))
    parts.append("\nIs this action safe? Classify as: safe / unsafe.")
    return "\n".join(parts)


# ── Source loaders ─────────────────────────────────────────────────────────────

def _load_toolsafety(max_examples: int | None = None) -> list[dict]:
    print("[data_mixer] Loading ToolSafety...")
    ds = load_dataset("jinjinyien/ToolSafety", split="train")
    out: list[dict] = []
    for row in ds:
        messages = row.get("messages") or row.get("conversations") or []
        if not messages:
            continue
        # find last assistant turn
        assistant_msg = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"),
            "",
        )
        user_msg = next(
            (m.get("content", "") for m in messages if m.get("role") == "user"),
            "",
        )
        if not user_msg:
            continue
        is_unsafe = bool(_REFUSAL_PHRASES.search(assistant_msg))
        label = random.choice(UNSAFE_LABELS) if is_unsafe else SAFE_LABEL
        out.append({
            "prompt":  _format_prompt(user_msg),
            "label":   label,
            "source":  "toolsafety",
            "is_safe": not is_unsafe,
        })
        if max_examples and len(out) >= max_examples:
            break
    print(f"[data_mixer] ToolSafety: {len(out)} examples")
    return out


def _load_wildguardmix(max_examples: int | None = None) -> list[dict]:
    print("[data_mixer] Loading WildGuardMix...")
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    out: list[dict] = []
    for row in ds:
        prompt = row.get("prompt", "")
        if not prompt or not _AGENT_KEYWORDS.search(prompt):
            continue
        harm_label = row.get("prompt_harm_label", "")
        if harm_label == "unharmful":
            label = SAFE_LABEL
        else:
            # map subcategory if available
            subcategory = (row.get("subcategory") or "").lower()
            label = next(
                (v for k, v in _WILDGUARD_HARM_MAP.items() if k in subcategory),
                "prompt_injection",  # safe default for unrecognized adversarial
            )
        out.append({
            "prompt":  _format_prompt(prompt),
            "label":   label,
            "source":  "wildguardmix",
            "is_safe": label == SAFE_LABEL,
        })
        if max_examples and len(out) >= max_examples:
            break
    print(f"[data_mixer] WildGuardMix (agent-filtered): {len(out)} examples")
    return out


def _load_aegis(max_examples: int | None = None) -> list[dict]:
    print("[data_mixer] Loading AEGIS 2.0...")
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
    out: list[dict] = []
    for row in ds:
        text = row.get("text") or row.get("prompt") or row.get("instruction") or ""
        if not text:
            continue
        category = row.get("labels_0") or row.get("category") or row.get("label") or "Safe"
        label = _AEGIS_CATEGORY_MAP.get(category, "goal_drift")
        if not _AGENT_KEYWORDS.search(text) and label != SAFE_LABEL:
            continue  # keep all safe examples, filter unsafe to agent-relevant only
        out.append({
            "prompt":  _format_prompt(text),
            "label":   label,
            "source":  "aegis",
            "is_safe": label == SAFE_LABEL,
        })
        if max_examples and len(out) >= max_examples:
            break
    print(f"[data_mixer] AEGIS 2.0 (agent-filtered): {len(out)} examples")
    return out


# ── Contrastive pair generation ───────────────────────────────────────────────

_CONSTRAINT_TEMPLATES = [
    "Never perform this action without explicit approval.",
    "All operations must be reversible.",
    "Require human-in-the-loop for any write operation.",
    "Do not access data outside the defined scope.",
    "All bulk operations require a dry-run confirmation first.",
    "Never escalate privileges without security team approval.",
    "Rollback capability must be verified before any destructive operation.",
    "Do not exfiltrate data to external systems.",
]

_SAFE_VARIANTS = [
    "Generate a read-only report summarising {base}.",
    "Run a dry-run analysis of {base} without making any changes.",
    "Present a recommendation for {base} for human review before proceeding.",
    "Audit {base} and flag anomalies for manual inspection.",
]


def _generate_contrastive_pairs(unsafe_examples: list[dict], n: int = 2000) -> list[dict]:
    """
    For each sampled unsafe example, generate a paired safe version by:
    1. Adding a strong operator constraint that would block the action
    2. Generating a conservative read-only alternative
    """
    rng = random.Random(42)
    pool = [e for e in unsafe_examples if not e["is_safe"]]
    rng.shuffle(pool)
    pairs: list[dict] = []

    for ex in pool[:n // 2]:
        # Unsafe original (already exists) — add constraint context to make the
        # boundary explicit
        constraint = rng.choice(_CONSTRAINT_TEMPLATES)
        action_text = ex["prompt"].split("Action:")[-1].split("\n")[0].strip()

        # Safe paired variant
        safe_template = rng.choice(_SAFE_VARIANTS)
        safe_action = safe_template.format(base=action_text[:80])
        pairs.append({
            "prompt":  _format_prompt(safe_action, constraints=[constraint]),
            "label":   SAFE_LABEL,
            "source":  "contrastive",
            "is_safe": True,
        })

        # Unsafe paired variant — same action, no constraint acknowledgment
        pairs.append({
            "prompt":  _format_prompt(action_text, constraints=[constraint]),
            "label":   ex["label"],
            "source":  "contrastive",
            "is_safe": False,
        })

    print(f"[data_mixer] Contrastive pairs: {len(pairs)}")
    return pairs


# ── Blender ────────────────────────────────────────────────────────────────────

def build_dataset(
    max_per_source: int | None = None,
    contrastive_n: int = 2000,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)

    toolsafety   = _load_toolsafety(max_per_source)
    wildguard    = _load_wildguardmix(max_per_source)
    aegis        = _load_aegis(max_per_source)

    all_examples = toolsafety + wildguard + aegis
    contrastive  = _generate_contrastive_pairs(all_examples, n=contrastive_n)
    all_examples += contrastive

    rng.shuffle(all_examples)

    # ── Stats ──────────────────────────────────────────────────────────────────
    total   = len(all_examples)
    safe_n  = sum(1 for e in all_examples if e["is_safe"])
    unsafe_n = total - safe_n
    by_source: dict[str, int] = {}
    by_label:  dict[str, int] = {}
    for e in all_examples:
        by_source[e["source"]] = by_source.get(e["source"], 0) + 1
        by_label[e["label"]]   = by_label.get(e["label"],  0) + 1

    print(f"\n[data_mixer] === Dataset summary ===")
    print(f"  Total   : {total}")
    print(f"  Safe    : {safe_n}  ({safe_n/total*100:.1f}%)")
    print(f"  Unsafe  : {unsafe_n}  ({unsafe_n/total*100:.1f}%)")
    print(f"  By source : {by_source}")
    print(f"  By label  : {dict(sorted(by_label.items()))}")
    print()

    return all_examples


def save_dataset(examples: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[data_mixer] Saved {len(examples)} examples → {output_path}")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./data/blended.jsonl")
    parser.add_argument("--max-per-source", type=int, default=None)
    parser.add_argument("--contrastive-n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = build_dataset(
        max_per_source=args.max_per_source,
        contrastive_n=args.contrastive_n,
        seed=args.seed,
    )
    save_dataset(examples, args.output)
