"""
saroku-safety-0.5b — v2 trainer.

Key changes from v1:
  - Full fine-tuning (no LoRA) for better OOD generalization
  - Multi-task label schema: 9-way classification across all behavioral properties
  - Label smoothing 0.05
  - Loads blended dataset from data_mixer.py

Usage:
    # Build dataset first
    python -m training.data_mixer --output ./data/blended.jsonl

    # Train
    python -m training.trainer_v2 \
        --data ./data/blended.jsonl \
        --output-dir ./models/saroku-safety-0.5b-v2

    # Full fine-tune requires ~8GB VRAM in bfloat16.
    # Set --use-lora to fall back to LoRA (rank=32) if memory-constrained.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
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

BASE_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct"
LABEL_TOKEN = {label: f"<|{label}|>" for label in ALL_LABELS}
# Completion for training is just the label token — one token output
# e.g. "<|safe|>" or "<|prompt_injection|>"


def _format_for_training(example: dict, tokenizer, label_smoothing: float = 0.0) -> dict:
    label = example["label"]
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
        full_text,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # Mask prompt tokens — only train on the label token
    prompt_only = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_len = len(tokenizer(prompt_only, return_tensors=None)["input_ids"])

    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    labels = labels[:len(tokenized["input_ids"])]
    tokenized["labels"] = labels
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
    model,
    tokenizer,
    eval_examples: list[dict],
    device: str,
    max_samples: int = 500,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    sample = rng.sample(eval_examples, min(max_samples, len(eval_examples)))

    # Build reverse map: token_id → label
    label_token_ids = {}
    for label, token in LABEL_TOKEN.items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            label_token_ids[ids[0]] = label

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
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()

        # Match generated text to a label
        pred = SAFE_LABEL
        for label in ALL_LABELS:
            if label in generated:
                pred = label
                break

        preds.append(pred)
        golds.append(ex["label"])

    total   = len(golds)
    correct = sum(p == g for p, g in zip(preds, golds))

    # Per-label accuracy
    per_label: dict[str, dict] = {}
    for label in ALL_LABELS:
        tp = sum(p == label and g == label for p, g in zip(preds, golds))
        fp = sum(p == label and g != label for p, g in zip(preds, golds))
        fn = sum(p != label and g == label for p, g in zip(preds, golds))
        n  = sum(g == label for g in golds)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label[label] = {"n": n, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    # Binary safe/unsafe
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
        "accuracy":         round(correct / total, 4),
        "binary_precision": round(prec_b, 4),
        "binary_recall":    round(rec_b,  4),
        "binary_f1":        round(f1_b,   4),
        "binary_tp": tp_b, "binary_fp": fp_b,
        "binary_fn": fn_b, "binary_tn": tn_b,
        "per_label": per_label,
        "n_eval":    total,
    }


def train(
    data_path: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 1e-5,       # lower LR for full fine-tune
    label_smoothing: float = 0.05,
    early_stopping_patience: int = 2,
    use_lora: bool = False,             # full fine-tune by default
    seed: int = 42,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[trainer_v2] Base model  : {BASE_MODEL}")
    print(f"[trainer_v2] Output dir  : {output_dir}")
    print(f"[trainer_v2] Device      : {device}")
    print(f"[trainer_v2] Mode        : {'LoRA (rank=32)' if use_lora else 'Full fine-tune'}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Add label tokens to vocabulary
    new_tokens = list(LABEL_TOKEN.values())
    tokenizer.add_tokens(new_tokens)
    print(f"[trainer_v2] Added {len(new_tokens)} label tokens to vocabulary")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"[trainer_v2] Loading dataset from {data_path}...")
    raw_data = load_jsonl(data_path)
    print(f"[trainer_v2] Total examples: {len(raw_data)}")

    train_data, eval_data = _stratified_split(raw_data, eval_frac=0.10, seed=seed)
    print(f"[trainer_v2] Train: {len(train_data)}  Eval: {len(eval_data)}")

    # ── Tokenize ─────────────────────────────────────────────────────────────
    print("[trainer_v2] Tokenizing...")
    remove_cols = ["prompt", "label", "source", "is_safe"]
    train_ds = Dataset.from_list(train_data).map(
        lambda x: _format_for_training(x, tokenizer, label_smoothing),
        remove_columns=remove_cols,
        num_proc=1,
    )
    eval_ds = Dataset.from_list(eval_data).map(
        lambda x: _format_for_training(x, tokenizer, label_smoothing),
        remove_columns=remove_cols,
        num_proc=1,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print("[trainer_v2] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[trainer_v2] Trainable params: {total_params/1e6:.1f}M (full fine-tune)")

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"[trainer_v2] Starting training...")
    trainer.train()

    # ── Post-training eval ───────────────────────────────────────────────────
    print("[trainer_v2] Running post-training classification eval...")
    metrics = _evaluate_classifier(model, tokenizer, eval_data, device)
    print(
        f"[trainer_v2] Binary accuracy : {metrics['accuracy']:.4f}  "
        f"F1={metrics['binary_f1']:.4f}  "
        f"precision={metrics['binary_precision']:.4f}  "
        f"recall={metrics['binary_recall']:.4f}"
    )
    print("[trainer_v2] Per-label F1:")
    for label, m in metrics["per_label"].items():
        if m["n"] > 0:
            print(f"  {label:<20} n={m['n']:>4}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    print("[trainer_v2] Saving model...")
    if use_lora:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(output_path / "model"))
    else:
        model.save_pretrained(str(output_path / "model"))
    tokenizer.save_pretrained(str(output_path / "model"))

    meta = {
        "base_model":     BASE_MODEL,
        "version":        "v2",
        "mode":           "lora_r32" if use_lora else "full_finetune",
        "label_schema":   ALL_LABELS,
        "label_tokens":   LABEL_TOKEN,
        "training_examples": len(train_data),
        "eval_examples":     len(eval_data),
        "label_smoothing":   label_smoothing,
        "eval_metrics":      metrics,
    }
    with open(output_path / "model" / "saroku_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[trainer_v2] Done. Model saved to {output_path / 'model'}")
    return str(output_path / "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         required=True,  help="Path to blended.jsonl from data_mixer.py")
    parser.add_argument("--output-dir",   default="./models/saroku-safety-0.5b-v2")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--grad-accum",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--use-lora",     action="store_true",
                        help="Use LoRA (rank=32) instead of full fine-tune")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        label_smoothing=args.label_smoothing,
        early_stopping_patience=args.early_stopping_patience,
        use_lora=args.use_lora,
        seed=args.seed,
    )
