# Tang–Song Poetry LoRA

[中文](README_ZH.md) | English

Fine-tune a Qwen2.5 instruction model on Tang poetry and Song ci so it can generate
classical Chinese poetry from a title or opening line, then deploy it on Hugging Face Spaces.

- [Hugging Face model](https://huggingface.co/xingqiwang/tang-song-poet-lora)
- [Live Gradio demo](https://huggingface.co/spaces/xingqiwang/poet-demo)

## Model and technology

| Item | Configuration |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct`, decoder-only Transformer |
| Base model size | Approximately 494M parameters, 24 layers, hidden size 896 |
| Attention | 14 query heads and 2 KV heads (GQA) |
| Original context | 32,768 tokens; fine-tuning sequences are truncated to 256 tokens |
| Fine-tuning method | PEFT LoRA injected into attention `q_proj` and `v_proj` only |
| LoRA configuration | Rank 8, alpha 16, dropout 0.05 |
| Trainable parameters | 540,672, or 0.1093% of all parameters |
| Training precision/device | float32 on an Intel CPU |
| Deployment | Transformers + PEFT + Gradio on a Hugging Face CPU Space |

All Qwen backbone parameters remain frozen; only the low-rank matrices are updated.
Training produces an `adapter_model.safetensors` file of approximately 2.1MB, which is
combined with the public base model at load time. Unlike MLX-specific weights, this standard
PEFT adapter runs on macOS, Linux CPU, and CUDA environments.

Causal language-modeling loss is calculated only on assistant responses. Labels for the
system prompt and user instruction are set to `-100`, so the model learns to write the poem
instead of reproducing the prompt template.

## Why Transformers instead of MLX

The target machine is an 8GB Intel Mac, which cannot run Apple MLX. This project therefore
uses Transformers and PEFT to train a standard LoRA adapter. The default base is
`Qwen/Qwen2.5-0.5B-Instruct`; training the 1.5B model is not recommended with this memory
limit. The resulting adapter loads directly on a Linux x86 Hugging Face Space without an MLX
fuse or conversion step.

## Comparison with tiny-poet

[`tiny-poet`](https://github.com/yingwang/tiny-poet) is a small GPT implemented and trained
from scratch. This project demonstrates the modern alternative: adapting a pretrained model
with parameter-efficient fine-tuning.

| Comparison | tiny-poet | poet-lora |
|---|---|---|
| Method | GPT trained from scratch | Pretrained Qwen + LoRA |
| Total parameters | 7.72M | Approximately 494M |
| Trained parameters | All 7.72M | 540,672 (0.1093%) |
| Tokenizer | Character-level, vocabulary 11,601 | Qwen subword, vocabulary 151,936 |
| Data | Full corpus of approximately 76,000 poems | Balanced sample of 8,000 poems |
| Input | A few characters followed by free continuation | Natural-language title or opening line |
| Training | 6,000 steps, approximately 90 minutes | 300 steps, approximately 24 minutes |
| Standalone size | Approximately 31.6MB | 2.1MB adapter plus an approximately 1GB base model |
| Deployment | Custom model code | Standard Transformers/PEFT/Gradio stack |

`poet-lora` provides better overall generation quality and instruction following. `tiny-poet`
is smaller, fully implemented from scratch, and better suited to learning how a Transformer
works internally.

## 1. Environment

Python 3.10 or 3.11 is recommended. The project has also been verified with the system
Python 3.9 available on the target machine.

```bash
cd poet-lora
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Data

By default, the preprocessing script reads chinese-poetry-formatted files from
`../tiny-poet/data/raw`. It normalizes whitespace, filters records that are too short or too
long, and deduplicates exact bodies. It then draws 4,000 examples from 55,645 Tang candidates
and 4,000 from 20,566 Song ci candidates. With a fixed random seed, the result is 7,600
training examples and 400 validation examples.

Each work is randomly converted into one of two tasks:

- Title generation: the user supplies a title or tune pattern and the assistant outputs the
  complete work.
- Opening-line continuation: the user supplies the first line and the assistant outputs only
  the continuation, avoiding repetition of the opening line.

The output is system/user/assistant JSONL compatible with the Qwen chat template. Metadata
retains genre, title, author, and task type for auditing, but is not included in training.

```bash
python data_prep.py
```

For a small smoke-test dataset:

```bash
python data_prep.py --max-samples 100 --output-dir data-smoke
```

## 3. Training

The default settings for an 8GB Intel machine are:

- Batch size 1, gradient accumulation 2, effective batch size 2
- 300 optimizer steps and maximum sequence length 256
- AdamW, initial learning rate `2e-4`, 5% warmup, and linear decay
- Save a checkpoint every 50 steps and evaluate on 20 fixed validation samples
- Gradient checkpointing enabled and zero DataLoader workers

```bash
python train.py
```

To verify the entire pipeline first:

```bash
python train.py --data-dir data-smoke --output-dir adapters-smoke --max-steps 1 --eval-samples 2
```

CPU training is substantially slower than GPU training. The same script and data can be used
on a Linux GPU; the output remains a standard, portable PEFT adapter.

## 4. Local generation

```bash
python generate.py "以《秋夜》为题写一首七言绝句"
```

The completed local run used 300 steps, took 24 minutes 14 seconds, reached a training loss of
4.319 and a final validation loss of 4.286. The adapter weights are approximately 2.1MB,
excluding tokenizer files. One generated example was:

```text
空山一枕梦难寻，月落霜天露滴频。
野水清流知几许，不知身在何处人。
```

## 5. Uploading the model and Space

Log in to the target Hugging Face account, then run the upload script:

```bash
hf auth login
python upload.py
```

The script creates and uploads:

- Model: `<current-account>/tang-song-poet-lora`
- Space: `<current-account>/poet-demo`

Before uploading, the script verifies the authenticated account and required adapter files.
At runtime, the Space uses `SPACE_AUTHOR_NAME` to derive the model ID in the same namespace,
so the Python application does not hard-code a personal account. The first startup downloads
the approximately 1GB base model; later starts use the Space cache.

## Limitations

- The 0.5B base is an engineering tradeoff for an 8GB Intel machine and has a lower quality
  ceiling than the 1.5B model.
- The free Space uses CPU inference. Initial startup and generation can take tens of seconds.
- Outputs are not guaranteed to follow strict classical prosody and may contain incorrect
  allusions or attributions.
