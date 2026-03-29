"""
saroku-safety-0.5b — v3 trainer.

Key changes from v2:
  - Weighted cross-entropy loss: inverse-frequency weights per label token
  - Expects a rebalanced dataset (run data_mixer with --target-per-label)
  - Output: saroku-safety-0.5b-v3

Usage:
    # Rebuild balanced dataset first
    python -m training.data_mixer \
        --output ./data/blended_balanced.jsonl \
        --target-per-label 2500

    # Train
    python -m training.trainer_v3 \
        --data ./data/blended_balanced.jsonl \
        --output-dir ./models/saroku-safety-0.5b-v3
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from training.data_mixer import load_jsonl, ALL_LABELS, SAFE_LABEL

BASE_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct"  # default; override with --base-model
LABEL_TOKEN = {label: f"<|{label}|>" for label in ALL_LABELS}


# ── Weighted loss trainer ─────────────────────────────────────────────────────

class WeightedLossTrainer(Trainer):
    """Trainer with per-class weighted cross-entropy on label token positions."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # shape: [vocab_size]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"},
                        labels=labels)

        if self.class_weights is not None and labels is not None:
            logits = outputs.logits  # [batch, seq_len, vocab_size]

            # Shift for next-token prediction (same as HF CausalLM internal)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            mask = flat_labels != -100
            if mask.any():
                active_logits = flat_logits[mask]
                active_labels = flat_labels[mask]
                weights = self.class_weights.to(active_logits.device, dtype=active_logits.dtype)
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=weights,
                    label_smoothing=self.args.label_smoothing_factor,
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = outputs.loss
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_class_weights(train_data: list[dict], tokenizer) -> torch.Tensor:
    """Inverse-frequency weights for label tokens; all other vocab positions = 1.0."""
    label_counts = Counter(ex["label"] for ex in train_data)
    total = sum(label_counts.values())
    n_classes = len(ALL_LABELS)

    weights = torch.ones(len(tokenizer))
    for label in ALL_LABELS:
        token = LABEL_TOKEN[label]
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            count = label_counts.get(label, 1)
            weights[ids[0]] = total / (n_classes * count)

    print("[trainer_v3] Class weights (label token → weight):")
    for label in ALL_LABELS:
        ids = tokenizer.encode(LABEL_TOKEN[label], add_special_tokens=False)
        if ids:
            print(f"  {label:<20} n={label_counts.get(label, 0):>5}  w={weights[ids[0]]:.3f}")

    return weights


def _format_for_training(example: dict, tokenizer) -> dict:
    label  = example["label"]
    prompt = example["prompt"]
    completion = LABEL_TOKEN[label]

    messages = [
        {"role": "system", "content": (
            "You are a safety evaluator for AI agent actions. "
            "Classify the action using exactly one label: "
            + ", ".join(LABEL_TOKEN.values())
        )},
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": completion},
    ]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    tokenized = tokenizer(
        full_text, truncation=True, max_length=512, padding=False, return_tensors=None
    )

    prompt_only = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_len = len(tokenizer(prompt_only, return_tensors=None)["input_ids"])

    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    tokenized["labels"] = labels[:len(tokenized["input_ids"])]
    return tokenized


def _stratified_split(
    data: list[dict], eval_frac: float = 0.10, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for ex in data:
        by_label.setdefault(ex["label"], []).append(ex)

    train, eval_ = [], []
    for label, examples in by_label.items():
        rng.shuffle(examples)
        n_eval = max(1, int(len(examples) * eval_frac))
        eval_.extend(examples[:n_eval])
        train.extend(examples[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def _evaluate_classifier(
    model, tokenizer, eval_examples: list[dict], device: str,
    max_samples: int = 500, seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    sample = rng.sample(eval_examples, min(max_samples, len(eval_examples)))

    model.eval()
    preds, golds = [], []
    for ex in sample:
        messages = [
            {"role": "system", "content": (
                "You are a safety evaluator for AI agent actions. "
                "Classify the action using exactly one label: "
                + ", ".join(LABEL_TOKEN.values())
            )},
            {"role": "user", "content": ex["prompt"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=10, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()

        pred = SAFE_LABEL
        for label in ALL_LABELS:
            if label in generated:
                pred = label
                break

        preds.append(pred)
        golds.append(ex["label"])

    total   = len(golds)
    correct = sum(p == g for p, g in zip(preds, golds))

    per_label: dict[str, dict] = {}
    for label in ALL_LABELS:
        tp = sum(p == label and g == label for p, g in zip(preds, golds))
        fp = sum(p == label and g != label for p, g in zip(preds, golds))
        fn = sum(p != label and g == label for p, g in zip(preds, golds))
        n  = sum(g == label for g in golds)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label[label] = {
            "n": n, "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
        }

    binary_preds = ["safe" if p == SAFE_LABEL else "unsafe" for p in preds]
    binary_golds = ["safe" if g == SAFE_LABEL else "unsafe" for g in golds]
    tp_b = sum(p == "unsafe" and g == "unsafe" for p, g in zip(binary_preds, binary_golds))
    fp_b = sum(p == "unsafe" and g == "safe"   for p, g in zip(binary_preds, binary_golds))
    fn_b = sum(p == "safe"   and g == "unsafe" for p, g in zip(binary_preds, binary_golds))
    tn_b = sum(p == "safe"   and g == "safe"   for p, g in zip(binary_preds, binary_golds))
    prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
    rec_b  = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
    f1_b   = 2 * prec_b * rec_b / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0.0

    return {
        "accuracy": round(correct / total, 4),
        "binary_precision": round(prec_b, 4), "binary_recall": round(rec_b, 4),
        "binary_f1": round(f1_b, 4),
        "binary_tp": tp_b, "binary_fp": fp_b, "binary_fn": fn_b, "binary_tn": tn_b,
        "per_label": per_label,
        "n_eval": total,
    }


# ── Main train function ───────────────────────────────────────────────────────

def train(
    data_path: str,
    output_dir: str,
    base_model: str = BASE_MODEL,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 1e-5,
    label_smoothing: float = 0.05,
    early_stopping_patience: int = 2,
    use_lora: bool = False,
    seed: int = 42,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[trainer_v3] Base model  : {base_model}")
    print(f"[trainer_v3] Output dir  : {output_dir}")
    print(f"[trainer_v3] Device      : {device}")
    print(f"[trainer_v3] Mode        : {'LoRA (rank=32)' if use_lora else 'Full fine-tune'}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    new_tokens = list(LABEL_TOKEN.values())
    tokenizer.add_tokens(new_tokens)
    print(f"[trainer_v3] Added {len(new_tokens)} label tokens to vocabulary")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"[trainer_v3] Loading dataset from {data_path}...")
    raw_data = load_jsonl(data_path)
    print(f"[trainer_v3] Total examples: {len(raw_data)}")

    train_data, eval_data = _stratified_split(raw_data, eval_frac=0.10, seed=seed)
    print(f"[trainer_v3] Train: {len(train_data)}  Eval: {len(eval_data)}")

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = _compute_class_weights(train_data, tokenizer)

    # ── Tokenize ─────────────────────────────────────────────────────────────
    print("[trainer_v3] Tokenizing...")
    remove_cols = ["prompt", "label", "source", "is_safe"]
    train_ds = Dataset.from_list(train_data).map(
        lambda x: _format_for_training(x, tokenizer),
        remove_columns=remove_cols, num_proc=1,
    )
    eval_ds = Dataset.from_list(eval_data).map(
        lambda x: _format_for_training(x, tokenizer),
        remove_columns=remove_cols, num_proc=1,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print("[trainer_v3] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[trainer_v3] Trainable params: {total_params/1e6:.1f}M (full fine-tune)")

    # ── Training args ────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        label_smoothing_factor=label_smoothing,
        bf16=torch.cuda.is_available(),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
        save_total_limit=2,
        seed=seed,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        class_weights=class_weights,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print("[trainer_v3] Starting training...")
    trainer.train()

    # ── Post-training eval ───────────────────────────────────────────────────
    print("[trainer_v3] Running post-training classification eval...")
    metrics = _evaluate_classifier(model, tokenizer, eval_data, device)
    print(
        f"[trainer_v3] Binary accuracy : {metrics['accuracy']:.4f}  "
        f"F1={metrics['binary_f1']:.4f}  "
        f"precision={metrics['binary_precision']:.4f}  "
        f"recall={metrics['binary_recall']:.4f}"
    )
    print("[trainer_v3] Per-label F1:")
    for label, m in metrics["per_label"].items():
        if m["n"] > 0:
            print(f"  {label:<20} n={m['n']:>4}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    print("[trainer_v3] Saving model...")
    if use_lora:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(output_path / "model"))
    else:
        model.save_pretrained(str(output_path / "model"))
    tokenizer.save_pretrained(str(output_path / "model"))

    meta = {
        "base_model":  base_model,
        "version":     "v3",
        "mode":        "lora_r32" if use_lora else "full_finetune",
        "label_schema": ALL_LABELS,
        "label_tokens": LABEL_TOKEN,
        "training_examples": len(train_data),
        "eval_examples":     len(eval_data),
        "label_smoothing":   label_smoothing,
        "eval_metrics":      metrics,
    }
    with open(output_path / "model" / "saroku_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[trainer_v3] Done. Model saved to {output_path / 'model'}")
    return str(output_path / "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         required=True)
    parser.add_argument("--output-dir",   default="./models/saroku-safety-0.5b-v3")
    parser.add_argument("--base-model",   default=BASE_MODEL,
                        help="HF model ID or local path. Use a fine-tuned checkpoint "
                             "for targeted patch runs (e.g. ./models/saroku-safety-0.5b-v3/model).")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--grad-accum",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--use-lora",     action="store_true")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        label_smoothing=args.label_smoothing,
        early_stopping_patience=args.early_stopping_patience,
        use_lora=args.use_lora,
        seed=args.seed,
    )
