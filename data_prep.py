#!/usr/bin/env python3
"""Prepare Tang poetry and Song ci as instruction-tuning JSONL data."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SYSTEM_PROMPT = (
    "你是一位擅长唐诗宋词的诗人。请严格按照题目或起句创作，"
    "只输出诗词正文，不要解释，不要使用现代白话文。"
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=project_dir.parent / "tiny-poet" / "data" / "raw",
        help="Directory containing poet.tang.*.json and ci.song.*.json",
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--max-samples", type=int, default=8000)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_line(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    return text.strip()


def iter_poems(raw_dir: Path, pattern: str, genre: str) -> Iterable[Dict[str, Any]]:
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {raw_dir / pattern}")

    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            continue
        for record in records:
            paragraphs = [clean_line(p) for p in record.get("paragraphs", [])]
            paragraphs = [p for p in paragraphs if p]
            if len(paragraphs) < 2:
                continue
            body = "\n".join(paragraphs)
            if not 12 <= len(body) <= 700:
                continue
            title = clean_line(record.get("title"))
            if genre == "宋词":
                title = clean_line(record.get("rhythmic")) or title
            if not title:
                title = "无题"
            yield {
                "title": title,
                "author": clean_line(record.get("author")) or "佚名",
                "paragraphs": paragraphs,
                "body": body,
                "genre": genre,
            }


def deduplicate(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for record in records:
        key = record["body"]
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def allocate_samples(
    tang: Sequence[Dict[str, Any]], song: Sequence[Dict[str, Any]], limit: int
) -> Tuple[int, int]:
    if limit <= 0:
        return len(tang), len(song)
    half = limit // 2
    tang_count = min(len(tang), half + limit % 2)
    song_count = min(len(song), half)
    remaining = limit - tang_count - song_count
    if remaining:
        tang_count += min(remaining, len(tang) - tang_count)
        remaining = limit - tang_count - song_count
        song_count += min(remaining, len(song) - song_count)
    return tang_count, song_count


def make_example(record: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    title = record["title"]
    paragraphs = record["paragraphs"]
    if rng.random() < 0.5:
        if record["genre"] == "宋词":
            instruction = f"请依《{title}》词牌写一首词"
        else:
            instruction = f"以《{title}》为题写一首古诗"
        output = record["body"]
        task = "title"
    else:
        instruction = f"续写下面的诗词起句：{paragraphs[0]}"
        output = "\n".join(paragraphs[1:])
        task = "continuation"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ],
        "metadata": {
            "genre": record["genre"],
            "title": title,
            "author": record["author"],
            "task": task,
        },
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not 0 < args.valid_ratio < 0.5:
        raise ValueError("--valid-ratio must be between 0 and 0.5")
    if args.max_samples == 1:
        raise ValueError("--max-samples must be 0 or at least 2")

    rng = random.Random(args.seed)
    tang = deduplicate(iter_poems(args.raw_dir, "poet.tang.*.json", "唐诗"))
    song = deduplicate(iter_poems(args.raw_dir, "ci.song.*.json", "宋词"))
    rng.shuffle(tang)
    rng.shuffle(song)
    tang_count, song_count = allocate_samples(tang, song, args.max_samples)
    selected = tang[:tang_count] + song[:song_count]
    examples = [make_example(record, rng) for record in selected]
    rng.shuffle(examples)

    valid_count = max(1, round(len(examples) * args.valid_ratio))
    valid = examples[:valid_count]
    train = examples[valid_count:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)

    print(
        json.dumps(
            {
                "raw_tang": len(tang),
                "raw_song": len(song),
                "selected_tang": tang_count,
                "selected_song": song_count,
                "train": len(train),
                "valid": len(valid),
                "output_dir": str(args.output_dir.resolve()),
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

