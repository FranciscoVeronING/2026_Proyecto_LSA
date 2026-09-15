"""Sube un GGUF o adaptador LoRA a Hugging Face."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publica el modelo español → glosas LSA en Hugging Face.")
    parser.add_argument("--folder", required=True, help="Carpeta local (LoRA o *_gguf).")
    parser.add_argument("--repo-id", required=True, help="ej. usuario/lsa-oyente")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", default="Release: modelo español → glosas LSA")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"No existe {folder}")

    api = HfApi()
    create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.message,
    )
    print(f"Publicado en https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
