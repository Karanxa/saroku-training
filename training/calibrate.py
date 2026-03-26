"""
Post-training temperature scaling calibration for saroku-safety-0.5b.

Finds the optimal temperature T that minimises Expected Calibration Error (ECE)
on a held-out labelled set, then saves it alongside the model weights so
inference can apply it automatically.

Usage:
    python -m training.calibrate \
        --model-path ./models/saroku-safety-0.5b-v2/model \
        --eval-data  ./data/blended.jsonl \
        --n-samples  500

The script writes  <model-path>/calibration.json:
    {
        "temperature": 1.23,
        "ece_before":  0.18,
        "ece_after":   0.07,
        "n_samples":   500
    }
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.data_mixer import ALL_LABELS, SAFE_LABEL, load_jsonl

LABEL_TOKEN = {label: f"<|{label}|>" for label in ALL_LABELS}


def _get_label_logits(model, tokenizer, prompt: str, device: str) -> dict[str, float]:
    """
    Returns raw logits for each label token at the first generated position.
    """
    messages = [
        {"role": "system", "content": (
            "You are a safety evaluator for AI agent actions. "
            "Classify the action using exactly one label: "
            + ", ".join(LABEL_TOKEN.values())
        )},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs)

    # logits at the last input position — next token prediction
    next_token_logits = out.logits[0, -1, :]

    label_logits = {}
    for label, token in LABEL_TOKEN.items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            label_logits[label] = next_token_logits[ids[0]].item()

    return label_logits


def _ece(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = [[] for _ in range(n_bins)]
    for conf, cor in zip(confidences, correct):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, cor))
    ece = 0.0
    total = len(confidences)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(x[0] for x in b) / len(b)
        avg_acc  = sum(x[1] for x in b) / len(b)
        ece += (len(b) / total) * abs(avg_conf - avg_acc)
    return round(ece, 4)


def _apply_temperature(logits: dict[str, float], temperature: float) -> dict[str, float]:
    scaled = {k: v / temperature for k, v in logits.items()}
    max_val = max(scaled.values())
    exp_vals = {k: math.exp(v - max_val) for k, v in scaled.items()}
    total = sum(exp_vals.values())
    return {k: v / total for k, v in exp_vals.items()}


def calibrate(
    model_path: str,
    eval_data_path: str,
    n_samples: int = 500,
    seed: int = 42,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[calibrate] Model : {model_path}")
    print(f"[calibrate] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rng = random.Random(seed)
    data = load_jsonl(eval_data_path)
    sample = rng.sample(data, min(n_samples, len(data)))

    # ── Collect logits ────────────────────────────────────────────────────────
    all_logits: list[dict[str, float]] = []
    all_labels: list[str] = []
    print(f"[calibrate] Collecting logits for {len(sample)} examples...")
    for i, ex in enumerate(sample):
        logits = _get_label_logits(model, tokenizer, ex["prompt"], device)
        all_logits.append(logits)
        all_labels.append(ex["label"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)}")

    # ── ECE before calibration ────────────────────────────────────────────────
    def _metrics_at_temp(T: float) -> tuple[float, float]:
        confidences, correct = [], []
        for logits, gold in zip(all_logits, all_labels):
            probs = _apply_temperature(logits, T)
            pred  = max(probs, key=probs.get)
            confidences.append(probs[pred])
            correct.append(pred == gold)
        acc = sum(correct) / len(correct)
        ece = _ece(confidences, correct)
        return acc, ece

    acc_before, ece_before = _metrics_at_temp(1.0)
    print(f"[calibrate] Before: accuracy={acc_before:.4f}  ECE={ece_before:.4f}")

    # ── Grid search for optimal temperature ──────────────────────────────────
    best_T, best_ece = 1.0, ece_before
    for T in [t / 10 for t in range(5, 31)]:  # 0.5 → 3.0, step 0.1
        _, ece = _metrics_at_temp(T)
        if ece < best_ece:
            best_ece = ece
            best_T   = T

    acc_after, ece_after = _metrics_at_temp(best_T)
    print(f"[calibrate] After : accuracy={acc_after:.4f}  ECE={ece_after:.4f}  T={best_T}")
    print(f"[calibrate] ECE reduction: {ece_before:.4f} → {ece_after:.4f}  ({(ece_before-ece_after)/ece_before*100:.1f}% improvement)")

    result = {
        "temperature": best_T,
        "ece_before":  ece_before,
        "ece_after":   ece_after,
        "acc_before":  acc_before,
        "acc_after":   acc_after,
        "n_samples":   len(sample),
    }

    out_path = Path(model_path) / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[calibrate] Saved → {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",  required=True)
    parser.add_argument("--eval-data",   required=True, help="JSONL from data_mixer.py")
    parser.add_argument("--n-samples",   type=int, default=500)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    calibrate(
        model_path=args.model_path,
        eval_data_path=args.eval_data,
        n_samples=args.n_samples,
        seed=args.seed,
    )
