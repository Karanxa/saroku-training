"""
saroku-training data mixer — v3 training data pipeline.

Sources:
  1. jinjinyien/ToolSafety            — safe agent conversations (safe only, no random labels)
  2. deepset/prompt-injections        — ground-truth prompt injection examples
  3. lmsys/toxic-chat                 — human-annotated jailbreak attempts (prompt_injection)
  4. EleutherAI/sycophancy            — sycophancy eval examples
  5. truthfulqa/truthful_qa           — adversarial truthfulness (honesty)
  6. nvidia/Aegis-AI-Content-Safety-Dataset-2.0 — real-world safety coverage
  7. Synthetic examples               — trust_hierarchy, corrigibility, minimal_footprint,
                                        goal_drift, consistency (no public data exists)
  8. Contrastive pairs                — safe read-only variants of unsafe examples

Output schema:
  {
    "prompt":    str,
    "label":     str,   # safe | sycophancy | prompt_injection | trust_hierarchy |
                        # corrigibility | minimal_footprint | goal_drift | honesty | consistency
    "source":    str,
    "is_safe":   bool,
  }

Usage:
    python -m training.data_mixer --output ./data/blended.jsonl
    python -m training.data_mixer --output ./data/blended.jsonl --target-per-label 2500
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

# ── AEGIS category → saroku property mapping ──────────────────────────────────
_AEGIS_CATEGORY_MAP: dict[str, str] = {
    "Safe": SAFE_LABEL,
    "Hate/Discrimination": "trust_hierarchy",
    "Violence/Physical Threats": "goal_drift",
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
    "Child Exploitation": "trust_hierarchy",
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
    """Load ToolSafety — safe examples only. Unsafe examples are dropped because
    ToolSafety has no per-category labels; random assignment was polluting training."""
    print("[data_mixer] Loading ToolSafety (safe examples only)...")
    ds = load_dataset("jinjinyien/ToolSafety", split="train")
    out: list[dict] = []
    for row in ds:
        convs = row.get("conversations") or []
        user_msg = next(
            (m.get("value", "") for m in convs if m.get("from") == "user"), ""
        )
        assistant_msg = next(
            (m.get("value", "") for m in reversed(convs) if m.get("from") == "assistant"), ""
        )
        if not user_msg:
            continue
        is_unsafe = bool(_REFUSAL_PHRASES.search(assistant_msg))
        if is_unsafe:
            continue  # drop — no ground-truth category label available
        out.append({
            "prompt":  _format_prompt(user_msg),
            "label":   SAFE_LABEL,
            "source":  "toolsafety",
            "is_safe": True,
        })
        if max_examples and len(out) >= max_examples:
            break
    print(f"[data_mixer] ToolSafety (safe only): {len(out)} examples")
    return out


def _load_deepset_injections(max_examples: int | None = None) -> list[dict]:
    """deepset/prompt-injections — ground-truth binary labels, 662 examples."""
    print("[data_mixer] Loading deepset/prompt-injections...")
    ds = load_dataset("deepset/prompt-injections", split="train")
    out: list[dict] = []
    for row in ds:
        text = row.get("text", "")
        label_id = row.get("label", 0)
        if not text or label_id != 1:
            continue  # only injection examples (label=1)
        out.append({
            "prompt":  _format_prompt(text),
            "label":   "prompt_injection",
            "source":  "deepset_injections",
            "is_safe": False,
        })
        if max_examples and len(out) >= max_examples:
            break
    print(f"[data_mixer] deepset/prompt-injections: {len(out)} examples")
    return out


def _load_toxic_chat_injections(max_examples: int | None = None) -> list[dict]:
    """lmsys/toxic-chat — use jailbreaking=1 rows as prompt_injection examples."""
    print("[data_mixer] Loading lmsys/toxic-chat (jailbreaking examples)...")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    out: list[dict] = []
    for row in ds:
        if not row.get("jailbreaking"):
            continue
        text = row.get("user_input", "")
        if not text:
            continue
        out.append({
            "prompt":  _format_prompt(text),
            "label":   "prompt_injection",
            "source":  "toxic_chat",
            "is_safe": False,
        })
        if max_examples and len(out) >= max_examples:
            break
    print(f"[data_mixer] toxic-chat jailbreaking: {len(out)} examples")
    return out


def _load_sycophancy_eval(max_examples: int | None = None) -> list[dict]:
    """EleutherAI/sycophancy — 30k examples across 3 subsets.
    answer_matching_behavior = the sycophantic (unsafe) response."""
    print("[data_mixer] Loading EleutherAI/sycophancy...")
    out: list[dict] = []
    configs = [
        "sycophancy_on_nlp_survey",
        "sycophancy_on_philpapers2020",
        "sycophancy_on_political_typology_quiz",
    ]
    for config in configs:
        try:
            ds = load_dataset("EleutherAI/sycophancy", config, split="validation")
        except Exception:
            continue
        for row in ds:
            question = row.get("question", "")
            sycophantic_answer = row.get("answer_matching_behavior", "")
            if not question or not sycophantic_answer:
                continue
            # Frame as: agent is asked a question and gives the sycophantic answer
            prompt_text = f"{question.strip()}\n\nAgent response: {sycophantic_answer.strip()}"
            out.append({
                "prompt":  _format_prompt(prompt_text),
                "label":   "sycophancy",
                "source":  "sycophancy_eval",
                "is_safe": False,
            })
            if max_examples and len(out) >= max_examples:
                break
        if max_examples and len(out) >= max_examples:
            break
    print(f"[data_mixer] EleutherAI/sycophancy: {len(out)} examples")
    return out


def _load_truthfulqa(max_examples: int | None = None) -> list[dict]:
    """truthfulqa/truthful_qa — adversarial questions where models tend to be dishonest."""
    print("[data_mixer] Loading truthful_qa...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    out: list[dict] = []
    for row in ds:
        question = row.get("question", "")
        incorrect = row.get("incorrect_answers", [])
        if not question or not incorrect:
            continue
        # Frame as: agent answers a question dishonestly (gives an incorrect/misleading answer)
        prompt_text = f"{question.strip()}\n\nAgent response: {incorrect[0].strip()}"
        out.append({
            "prompt":  _format_prompt(prompt_text),
            "label":   "honesty",
            "source":  "truthfulqa",
            "is_safe": False,
        })
        if max_examples and len(out) >= max_examples:
            break
    print(f"[data_mixer] truthful_qa: {len(out)} examples")
    return out


# BeaverTails category → saroku property mapping
_BEAVERTAILS_MAP: dict[str, str] = {
    "violence,aiding_and_abetting,incitement":   "goal_drift",
    "hate_speech,offensive_language":            "trust_hierarchy",
    "financial_crime,property_crime,theft":      "sycophancy",
    "privacy_violation":                         "trust_hierarchy",
    "malware,hacking,computer_crime":            "prompt_injection",
    "misinformation_regarding_ethics":           "honesty",
    "weapon_descriptions":                       "goal_drift",
    "drug_abuse,weapons,banned_substance":       "goal_drift",
    "terrorism,organized_crime":                 "trust_hierarchy",
    "non_violent_unethical_behavior":            "corrigibility",
    "controversial_topics,politics":             "consistency",
    "discrimination,stereotype,injustice":       "consistency",
    "sexually_explicit,adult_content":           "honesty",
}


def _load_beavertails(max_examples: int | None = None) -> list[dict]:
    print("[data_mixer] Loading BeaverTails...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    out: list[dict] = []
    for row in ds:
        prompt = row.get("prompt", "")
        if not prompt:
            continue
        if not _AGENT_KEYWORDS.search(prompt) and not row.get("is_safe", True):
            continue  # filter unsafe to agent-relevant only; keep all safe
        is_safe = bool(row.get("is_safe", True))
        if is_safe:
            label = SAFE_LABEL
        else:
            category_dict = row.get("category", {})
            # category is {category_name: bool} — pick first True key
            active = next((k for k, v in category_dict.items() if v), None)
            label = _BEAVERTAILS_MAP.get(active, "goal_drift") if active else "goal_drift"
        out.append({
            "prompt":  _format_prompt(prompt),
            "label":   label,
            "source":  "beavertails",
            "is_safe": is_safe,
        })
        if max_examples and len(out) >= max_examples:
            break
    print(f"[data_mixer] BeaverTails (agent-filtered): {len(out)} examples")
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


# ── Rebalancer ─────────────────────────────────────────────────────────────────

def _rebalance_dataset(
    examples: list[dict],
    target_per_label: int,
    seed: int = 42,
) -> list[dict]:
    """Cap over-represented labels and oversample (with replacement) under-represented ones."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for ex in examples:
        by_label.setdefault(ex["label"], []).append(ex)

    result: list[dict] = []
    for label, exs in by_label.items():
        if len(exs) >= target_per_label:
            result.extend(rng.sample(exs, target_per_label))
        else:
            result.extend(exs)
            result.extend(rng.choices(exs, k=target_per_label - len(exs)))

    rng.shuffle(result)
    print(f"[data_mixer] Rebalanced to {target_per_label} per label → {len(result)} total")
    return result


# ── Blender ────────────────────────────────────────────────────────────────────

def build_dataset(
    max_per_source: int | None = None,
    contrastive_n: int = 2000,
    target_per_label: int | None = None,
    synth_path: str | None = None,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)

    # Safe examples
    toolsafety   = _load_toolsafety(max_per_source)
    beavertails  = _load_beavertails(max_per_source)
    aegis        = _load_aegis(max_per_source)

    # Ground-truth labeled unsafe examples
    injections   = _load_deepset_injections(max_per_source)
    toxic        = _load_toxic_chat_injections(max_per_source)
    sycophancy   = _load_sycophancy_eval(max_per_source)
    truthfulqa   = _load_truthfulqa(max_per_source)

    all_examples = toolsafety + beavertails + aegis + injections + toxic + sycophancy + truthfulqa

    # Synthetic examples for categories with no public data
    if synth_path and Path(synth_path).exists():
        synth = load_jsonl(synth_path)
        print(f"[data_mixer] Loaded {len(synth)} synthetic examples from {synth_path}")
        all_examples += synth

    contrastive  = _generate_contrastive_pairs(all_examples, n=contrastive_n)
    all_examples += contrastive

    if target_per_label is not None:
        all_examples = _rebalance_dataset(all_examples, target_per_label, seed=seed)
    else:
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
    parser.add_argument("--target-per-label", type=int, default=None,
                        help="Resample each label to exactly this many examples.")
    parser.add_argument("--synth-path", type=str, default=None,
                        help="Path to synthetic examples JSONL (for uncovered categories).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = build_dataset(
        max_per_source=args.max_per_source,
        contrastive_n=args.contrastive_n,
        target_per_label=args.target_per_label,
        synth_path=args.synth_path,
        seed=args.seed,
    )
    save_dataset(examples, args.output)
