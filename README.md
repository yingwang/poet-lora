# 唐诗宋词 LoRA

用 Qwen2.5 指令模型和唐诗宋词语料训练一个“给题目或起句生成古诗词”的模型，
并部署到 Hugging Face Spaces。

- [Hugging Face 模型](https://huggingface.co/xingqiwang/tang-song-poet-lora)
- [在线 Gradio Demo](https://huggingface.co/spaces/xingqiwang/poet-demo)

## 模型与技术

| 项目 | 配置 |
|---|---|
| 底座 | `Qwen/Qwen2.5-0.5B-Instruct`，decoder-only Transformer |
| 底座规模 | 约 494M 参数，24 层，hidden size 896 |
| Attention | 14 个 query heads、2 个 KV heads（GQA） |
| 原始上下文 | 32,768 tokens；微调时截断到 256 tokens |
| 微调方法 | PEFT LoRA，只注入 attention 的 `q_proj`、`v_proj` |
| LoRA 配置 | rank 8、alpha 16、dropout 0.05 |
| 可训练参数 | 540,672，占总参数 0.1093% |
| 训练精度/设备 | float32、Intel CPU |
| 部署 | Transformers + PEFT + Gradio，Hugging Face CPU Space |

Qwen 主干参数在训练中全部冻结，只更新低秩矩阵。训练产物是约 2.1MB 的
`adapter_model.safetensors`，加载时再与公开底座组合。与 MLX 专用权重不同，这个标准
PEFT adapter 可在 macOS、Linux CPU 或 CUDA 环境使用。

训练只对 assistant 回复计算 causal language-modeling loss：system prompt 和用户指令的
label 全部置为 `-100`。这样模型学习的是“如何写诗”，而不是复述输入模板。

## Intel 机器对应的技术路线

目标环境是 8GB Intel Mac，不能运行 Apple MLX。项目因此使用 Transformers + PEFT
直接训练标准 LoRA adapter。默认底座是 `Qwen/Qwen2.5-0.5B-Instruct`；在该环境中
不建议训练 1.5B。adapter 可直接在 Linux x86 的 Hugging Face Space 中加载，无需
MLX fuse 或格式转换。

## 与 tiny-poet 对比

[`tiny-poet`](https://github.com/yingwang/tiny-poet) 是从零实现和训练的小型 GPT；
本项目则展示预训练模型加参数高效微调的现代工作流。

| 对比 | tiny-poet | poet-lora |
|---|---|---|
| 方法 | 从零训练 GPT | 预训练 Qwen + LoRA |
| 总参数 | 7.72M | 约 494M |
| 实际训练参数 | 全部 7.72M | 540,672（0.1093%） |
| Tokenizer | 字符级，11,601 词表 | Qwen subword，151,936 词表 |
| 数据 | 约 7.6 万首全部语料 | 平衡抽样 8,000 首 |
| 输入方式 | 给几个字，继续生成 | 自然语言题目或起句 |
| 训练 | 6,000 steps，约 90 分钟 | 300 steps，约 24 分钟 |
| 独立运行 | 约 31.6MB | adapter 2.1MB，但依赖约 1GB 底座 |
| 部署 | 自定义代码 | 标准 Transformers/PEFT/Gradio |

整体生成和指令理解水平以 `poet-lora` 更高；`tiny-poet` 的优势是完全从零实现、体积小，
更适合理解 Transformer 的内部结构。

## 1. 环境

推荐 Python 3.10 或 3.11；本项目也已在这台机器自带的 Python 3.9 上验证。

```bash
cd ~/claude/poet-lora
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. 数据

默认从 `../tiny-poet/data/raw` 读取 chinese-poetry 格式的语料。预处理会规范空白、过滤
过短/过长记录、按正文去重，再从 55,645 首唐诗与 20,566 首宋词候选中各抽 4,000 条。
固定随机种子后生成 7,600 条训练样本和 400 条验证样本。

每首作品随机转换为两类任务之一：

- 题目生成：输入题目或词牌，assistant 输出完整正文。
- 起句续写：输入第一句，assistant 只输出后续内容，避免重复起句。

输出采用 Qwen chat template 对应的 system/user/assistant JSONL，metadata 保留体裁、题目、
作者和任务类型，方便审计但不参与训练。

```bash
python data_prep.py
```

先做烟雾测试可使用：

```bash
python data_prep.py --max-samples 100 --output-dir data-smoke
```

## 3. 训练

8GB Intel 默认参数如下：

- batch size 1，梯度累积 2，effective batch size 2
- 300 optimizer steps，max sequence length 256
- AdamW，初始学习率 `2e-4`，5% warmup，linear decay
- 每 50 steps 保存 checkpoint，并在 20 条固定验证样本上评估
- gradient checkpointing 开启，dataloader worker 为 0

```bash
python train.py
```

先验证完整流程：

```bash
python train.py --data-dir data-smoke --output-dir adapters-smoke --max-steps 1 --eval-samples 2
```

CPU 训练会明显慢于 GPU。若本机速度不可接受，保持相同脚本和数据，在 Linux GPU
环境运行即可；产物仍是完全相同的标准 PEFT adapter。

## 4. 本地生成

```bash
python generate.py "以《秋夜》为题写一首七言绝句"
```

实测正式训练完成 300 steps，耗时 24 分 14 秒，训练 loss 4.319，最终验证 loss 4.286。
adapter 权重约 2.1MB（tokenizer 文件另计）。一次实测输出：

```text
空山一枕梦难寻，月落霜天露滴频。
野水清流知几许，不知身在何处人。
```

## 5. 上传模型与 Space

先登录目标 Hugging Face 账号：

```bash
hf auth login
python upload.py
```

脚本会创建并上传：

- 模型：`<当前登录账号>/tang-song-poet-lora`
- Space：`<当前登录账号>/poet-demo`

上传前脚本会检查登录账号和 adapter 文件，避免把未训练模型或错误账号推到 Hub。
Space 运行时通过 `SPACE_AUTHOR_NAME` 自动拼出同一账号下的模型 ID，因此源码不硬编码
个人账号。首次启动会下载约 1GB 的底座；后续从 Space 缓存加载。

## 限制

- 0.5B 是为 8GB Intel 机器做出的工程取舍，效果上限低于 1.5B。
- 免费 Space 是 CPU 推理，首次启动要下载底座模型，生成可能需要几十秒。
- 输出不保证严格符合格律，也可能出现错误的典故或归属。
