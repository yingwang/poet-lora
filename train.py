#!/usr/bin/env python3
"""Train a small Qwen2.5 model with PEFT LoRA on CPU-friendly settings."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "adapters")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=20)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_chat(tokenizer: Any, messages: List[Dict[str, str]], generation: bool) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation,
        )
    roles = {"system": "系统", "user": "用户", "assistant": "助手"}
    text = "\n".join(f"{roles[m['role']]}：{m['content']}" for m in messages)
    return text + ("\n助手：" if generation else "")


class PoetryDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        messages = self.rows[index]["messages"]
        prompt = render_chat(self.tokenizer, messages[:-1], generation=True)
        full_text = render_chat(self.tokenizer, messages, generation=False)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.copy()
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        if all(label == -100 for label in labels):
            labels[-1] = input_ids[-1]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PoetryCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        width = batch["input_ids"].shape[1]
        padded_labels = [label + [-100] * (width - len(label)) for label in labels]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def training_arguments(args: argparse.Namespace) -> TrainingArguments:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: Dict[str, Any] = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.05,
        "logging_steps": 5,
        "eval_steps": 50,
        "save_steps": 50,
        "save_total_limit": 2,
        "optim": "adamw_torch",
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
        "report_to": "none",
        "seed": args.seed,
        "bf16": False,
        "fp16": False,
        "remove_unused_columns": False,
    }
    kwargs["eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"] = "steps"
    if "use_cpu" in parameters:
        kwargs["use_cpu"] = True
    elif "no_cuda" in parameters:
        kwargs["no_cuda"] = True
    return TrainingArguments(**kwargs)


def write_model_card(output_dir: Path, base_model: str, args: argparse.Namespace) -> None:
    card = f"""---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
language:
- zh
tags:
- poetry
- chinese-poetry
- lora
license: apache-2.0
---

# Tang-Song Poet LoRA

PEFT LoRA adapter for `{base_model}`, fine-tuned on Tang poetry and Song ci from
the public `chinese-poetry` corpus as packaged in the tiny-poet project.

## Training configuration

- Base architecture: Qwen2 decoder-only Transformer, approximately 494M parameters
- Max sequence length: {args.max_length}
- Steps: {args.max_steps}
- Batch size: 1
- Gradient accumulation: {args.gradient_accumulation_steps}
- LoRA rank: {args.lora_r}
- LoRA alpha: {args.lora_r * 2}
- LoRA dropout: 0.05
- LoRA targets: `q_proj`, `v_proj`
- Trainable parameters: 540,672 (0.1093%)

The base weights are frozen. Causal language-modeling loss is computed only on
assistant tokens; system and user tokens are masked with `-100`.

Generated poetry may contain factual, tonal, metrical, or attribution errors.

## Usage

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_id = "YOUR_USERNAME/tang-song-poet-lora"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoPeftModelForCausalLM.from_pretrained(model_id)
```
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if platform.machine() in {"x86_64", "AMD64"} and "1.5B" in args.model:
        raise SystemExit(
            "Refusing the 1.5B default on an 8GB Intel Mac. Use the 0.5B model "
            "or pass a smaller model explicitly."
        )
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(args.data_dir / "train.jsonl")
    valid_rows = read_jsonl(args.data_dir / "valid.jsonl")[: args.eval_samples]
    train_dataset = PoetryDataset(train_rows, tokenizer, args.max_length)
    valid_dataset = PoetryDataset(valid_rows, tokenizer, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_arguments(args),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=PoetryCollator(tokenizer),
    )
    trainer.train()
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    write_model_card(args.output_dir, args.model, args)
    print(f"Adapter saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
