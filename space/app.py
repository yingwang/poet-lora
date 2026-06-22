import os
import re

import gradio as gr
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


MODEL_ID = os.getenv("MODEL_ID") or f"{os.getenv('SPACE_AUTHOR_NAME', 'YOUR_USERNAME')}/tang-song-poet-lora"
SYSTEM_PROMPT = (
    "你是一位擅长唐诗宋词的诗人。请严格按照题目或起句创作，"
    "只输出诗词正文，不要解释，不要使用现代白话文。"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()


def clean_output(text: str) -> str:
    text = re.split(r"\n\s*(?:（?注(?:释)?[：:]|说明[：:]|解释[：:])", text, maxsplit=1)[0]
    return text.strip()


def generate_poem(text: str, mode: str, temperature: float) -> str:
    text = text.strip()
    if not text:
        return "请输入题目或起句。"
    prompt = f"以《{text}》为题写一首古诗" if mode == "题目" else f"续写下面的诗词起句：{text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return clean_output(tokenizer.decode(generated, skip_special_tokens=True))


demo = gr.Interface(
    fn=generate_poem,
    inputs=[
        gr.Textbox(label="题目或起句", placeholder="例如：秋夜 / 明月照高楼"),
        gr.Radio(["题目", "起句"], value="题目", label="输入类型"),
        gr.Slider(0.2, 1.2, value=0.8, step=0.1, label="灵感温度"),
    ],
    outputs=gr.Textbox(label="生成的诗词", lines=10),
    title="唐诗宋词生成器",
    description="输入题目或起句，由微调后的 Qwen2.5 模型生成古诗词。CPU 推理可能需要几十秒。",
    examples=[["秋夜", "题目", 0.8], ["明月照高楼", "起句", 0.7]],
)


if __name__ == "__main__":
    demo.launch()
