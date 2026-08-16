#!/usr/bin/env python3
"""Confirm a toy_validation training run actually landed on HuggingFace Hub.

Run this on the TPU VM (same env as training, so `huggingface_hub` is
already installed and `hf auth login` is already done) right after
train_text_to_image_xla.py finishes:

    python3 verify_upload.py <hf-username>/<repo-name>
"""

import sys

from huggingface_hub import HfApi

# Files a successful StableDiffusionPipeline.save_pretrained() + upload should
# always produce, regardless of exact checkpoint filenames.
EXPECTED_SUFFIXES = ["model_index.json", "README.md"]
EXPECTED_SUBFOLDER_MARKERS = ["unet/", "vae/", "text_encoder/"]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python3 {sys.argv[0]} <hf-username>/<repo-name>")
    repo_id = sys.argv[1]

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id)
    except Exception as e:
        sys.exit(f"FAIL: could not read repo {repo_id!r} -- {e}")

    print(f"{repo_id}: {len(files)} files")
    for f in sorted(files):
        print(f"  {f}")

    missing = [name for name in EXPECTED_SUFFIXES if name not in files]
    missing_subfolders = [prefix for prefix in EXPECTED_SUBFOLDER_MARKERS if not any(f.startswith(prefix) for f in files)]

    if missing or missing_subfolders:
        print("\nFAIL: repo exists but is missing expected content:")
        for name in missing:
            print(f"  missing file: {name}")
        for prefix in missing_subfolders:
            print(f"  missing subfolder: {prefix}")
        sys.exit(1)

    print("\nOK: repo has model_index.json, README.md, and unet/vae/text_encoder subfolders.")


if __name__ == "__main__":
    main()
