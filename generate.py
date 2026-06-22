#!/usr/bin/env python3
"""Generate poetry locally with a trained PEFT adapter."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "你是一位擅长唐诗宋词的诗人。请严格按照题目或起句创作，"
    "只输出诗词正文，不要解释，不要使用现代白话文。"
)


def clean_output(text: str) -> str:
    """Drop explanatory tails occasionally emitted by the small base model."""
    text = re.split(r"\n\s*(?:（?注(?:释)?[：:]|说明[：:]|解释[：:])", text, maxsplit=1)[0]
    return text.strip()


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="以《秋夜》为题写一首七言绝句")
    parser.add_argument("--adapter", default=str(project_dir / "adapters"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    print(clean_output(tokenizer.decode(generated, skip_special_tokens=True)))


if __name__ == "__main__":
    main()
