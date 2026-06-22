#!/usr/bin/env python3
"""Upload the trained adapter and Gradio Space to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-repo", default="tang-song-poet-lora")
    parser.add_argument("--space-repo", default="poet-demo")
    parser.add_argument("--adapter-dir", type=Path, default=project_dir / "adapters")
    parser.add_argument("--space-dir", type=Path, default=project_dir / "space")
    args = parser.parse_args()

    required = ["adapter_config.json", "adapter_model.safetensors", "README.md"]
    missing = [name for name in required if not (args.adapter_dir / name).exists()]
    if missing:
        raise SystemExit(f"Adapter is incomplete; missing: {', '.join(missing)}")

    api = HfApi()
    user = api.whoami()["name"]
    adapter_repo = args.adapter_repo if "/" in args.adapter_repo else f"{user}/{args.adapter_repo}"
    space_repo = args.space_repo if "/" in args.space_repo else f"{user}/{args.space_repo}"
    for repo_id in (adapter_repo, space_repo):
        if repo_id.split("/", 1)[0] != user:
            raise SystemExit(
                f"Authenticated account does not own {repo_id!r}. "
                "Use a repo name without a namespace, or log in as its owner."
            )

    api.create_repo(adapter_repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=adapter_repo,
        repo_type="model",
        folder_path=args.adapter_dir,
        ignore_patterns=["checkpoints/**"],
    )
    api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        repo_id=space_repo,
        repo_type="space",
        folder_path=args.space_dir,
    )
    print(f"Model: https://huggingface.co/{adapter_repo}")
    print(f"Space: https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
