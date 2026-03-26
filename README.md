# saroku-training

Training pipeline and model weights for `saroku-safety-0.5b` — the local safety classifier used in saroku's 3-layer SafetyGuard.

## Structure

```
training/        # Fine-tuning pipeline (Qwen2.5-0.5B + LoRA)
  trainer.py     # Main training script
  dataset_loader.py  # ToolSafety dataset loader
  benchmark.py   # OOD evaluation (ATBench, Agent-SafetyBench)
dashboards/
  dashboard.py        # Live training progress dashboard (port 7860)
  benchmark_dashboard.py  # Benchmark results dashboard (port 7861)
models/
  saroku-safety-0.5b/   # Trained model weights + checkpoints
```

## Train

```bash
pip install saroku[train]

python -m training.trainer \
    --output-dir ./models/saroku-safety-0.5b \
    --epochs 5
```

## Model on HuggingFace

`karanxa/saroku-safety-0.5b` — [huggingface.co/karanxa/saroku-safety-0.5b](https://huggingface.co/karanxa/saroku-safety-0.5b)
