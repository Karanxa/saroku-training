"""
saroku safety model trainer.

Fine-tunes Qwen2.5-0.5B-Instruct with LoRA to produce a specialized
safety classifier for agent actions.

Usage:
    python -m saroku.training.trainer \
        --output-dir ./models/saroku-safety-0.5b \
        --epochs 5

The output is a merged model ready for inference via saroku's local_judge.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)

from saroku.training.dataset_loader import load_toolsafety

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)


def _format_for_training(example: dict, tokenizer) -> dict:
    """
    Format a prompt+completion pair into input_ids + labels.
    Labels are -100 for prompt tokens (not trained on) and real ids for completion.
    """
    prompt = example["prompt"]
    completion = example["completion"]

    messages = [
        {"role": "user", "content": prompt},
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

    prompt_only = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_only, return_tensors=None)["input_ids"]
    prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    # Truncate labels to match input_ids length — long system prompts can push
    # prompt_len > max_length, leaving labels longer than the truncated input.
    labels = labels[:len(tokenized["input_ids"])]
    tokenized["labels"] = labels
    return tokenized


def _stratified_split(
    data: list[dict], eval_frac: float = 0.10, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """
    Split into train/eval preserving the SAFE/UNSAFE class ratio in both sets.
    Returns (train_data, eval_data).
    """
    rng = random.Random(seed)
    safe = [x for x in data if x["verdict"] == "SAFE"]
    unsafe = [x for x in data if x["verdict"] == "UNSAFE"]
    rng.shuffle(safe)
    rng.shuffle(unsafe)

    safe_eval_n = max(1, int(len(safe) * eval_frac))
    unsafe_eval_n = max(1, int(len(unsafe) * eval_frac))

    train = safe[safe_eval_n:] + unsafe[unsafe_eval_n:]
    eval_ = safe[:safe_eval_n] + unsafe[:unsafe_eval_n]
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def _evaluate_classifier(
    model,
    tokenizer,
    eval_examples: list[dict],
    device: str,
    max_samples: int = 300,
    seed: int = 42,
) -> dict:
    """
    Run greedy inference on a stratified sample of eval_examples.
    Returns accuracy, precision, recall, F1 (UNSAFE as positive class).
    """
    rng = random.Random(seed)
    safe_ex = [x for x in eval_examples if x["verdict"] == "SAFE"]
    unsafe_ex = [x for x in eval_examples if x["verdict"] == "UNSAFE"]

    n_per_class = min(max_samples // 2, len(safe_ex), len(unsafe_ex))
    sample = rng.sample(safe_ex, n_per_class) + rng.sample(unsafe_ex, n_per_class)
    rng.shuffle(sample)

    model.eval()
    preds: list[str] = []
    labels: list[str] = []

    for ex in sample:
        messages = [{"role": "user", "content": ex["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().upper()

        # "UNSAFE" contains "SAFE", so check UNSAFE first
        pred = "UNSAFE" if "UNSAFE" in generated else "SAFE"
        preds.append(pred)
        labels.append(ex["verdict"])

    total = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    tp = sum(p == "UNSAFE" and l == "UNSAFE" for p, l in zip(preds, labels))
    fp = sum(p == "UNSAFE" and l == "SAFE"   for p, l in zip(preds, labels))
    fn = sum(p == "SAFE"   and l == "UNSAFE" for p, l in zip(preds, labels))
    tn = sum(p == "SAFE"   and l == "SAFE"   for p, l in zip(preds, labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy":  round(correct / total, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_eval": total,
    }


def train(
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    max_external: int | None = None,
    early_stopping_patience: int = 2,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[saroku trainer] Base model : {BASE_MODEL}")
    print(f"[saroku trainer] Output dir : {output_dir}")
    print(f"[saroku trainer] Device     : {device}")

    # ── Tokenizer ───────────────────────────────────────────────────────────────
    print("[saroku trainer] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Build dataset ───────────────────────────────────────────────────────────
    print("[saroku trainer] Loading ToolSafety dataset...")
    raw_data = load_toolsafety(max_examples=max_external)
    data_sources = ["toolsafety"]
    print(f"[saroku trainer] Total examples : {len(raw_data)}")

    # ── Stratified split ────────────────────────────────────────────────────────
    train_data, eval_data = _stratified_split(raw_data, eval_frac=0.10)

    train_safe   = sum(1 for x in train_data if x["verdict"] == "SAFE")
    train_unsafe = len(train_data) - train_safe
    eval_safe    = sum(1 for x in eval_data  if x["verdict"] == "SAFE")
    eval_unsafe  = len(eval_data) - eval_safe

    print(
        f"[saroku trainer] Train : {len(train_data)} "
        f"({train_safe} SAFE / {train_unsafe} UNSAFE)"
    )
    print(
        f"[saroku trainer] Eval  : {len(eval_data)} "
        f"({eval_safe} SAFE / {eval_unsafe} UNSAFE)"
    )

    # ── Tokenize ────────────────────────────────────────────────────────────────
    print("[saroku trainer] Tokenizing...")
    train_dataset = Dataset.from_list(train_data).map(
        lambda x: _format_for_training(x, tokenizer),
        remove_columns=["prompt", "completion", "verdict", "property", "severity"],
        num_proc=1,
    )
    eval_dataset = Dataset.from_list(eval_data).map(
        lambda x: _format_for_training(x, tokenizer),
        remove_columns=["prompt", "completion", "verdict", "property", "severity"],
        num_proc=1,
    )

    # ── Model ───────────────────────────────────────────────────────────────────
    print("[saroku trainer] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # ── Training arguments ──────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,              # L2 regularisation — penalises large weights
        fp16=torch.cuda.is_available(),
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
        save_total_limit=2,             # keep only last 2 checkpoints
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # ── Train ───────────────────────────────────────────────────────────────────
    print(
        f"[saroku trainer] Starting training "
        f"(max_epochs={epochs}, early_stopping_patience={early_stopping_patience})..."
    )
    trainer.train()

    # ── Post-training classification eval ───────────────────────────────────────
    print("[saroku trainer] Running post-training classification eval...")
    eval_metrics = _evaluate_classifier(model, tokenizer, eval_data, device)
    print(
        f"[saroku trainer] Eval results → "
        f"accuracy={eval_metrics['accuracy']:.4f}  "
        f"F1={eval_metrics['f1']:.4f}  "
        f"precision={eval_metrics['precision']:.4f}  "
        f"recall={eval_metrics['recall']:.4f}  "
        f"(n={eval_metrics['n_eval']})"
    )

    # ── Save merged model ───────────────────────────────────────────────────────
    print("[saroku trainer] Merging LoRA weights and saving...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_path / "model"))
    tokenizer.save_pretrained(str(output_path / "model"))

    meta = {
        "base_model": BASE_MODEL,
        "lora_rank": LORA_CONFIG.r,
        "training_examples": len(train_data),
        "eval_examples": len(eval_data),
        "epochs_requested": epochs,
        "early_stopping_patience": early_stopping_patience,
        "weight_decay": 0.01,
        "saroku_version": "0.3.0",
        "model_type": "safety_classifier",
        "data_sources": data_sources,
        "eval_metrics": eval_metrics,
    }
    with open(output_path / "model" / "saroku_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[saroku trainer] Done. Model saved to {output_path / 'model'}")
    return str(output_path / "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train saroku safety classifier")
    parser.add_argument("--output-dir", default="./models/saroku-safety-0.5b")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap on examples loaded from ToolSafety (default: all)")
    parser.add_argument("--early-stopping-patience", type=int, default=2,
                        help="Stop if eval_loss doesn't improve for N epochs")
    args = parser.parse_args()

    train(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        max_external=args.max_examples,
        early_stopping_patience=args.early_stopping_patience,
    )
