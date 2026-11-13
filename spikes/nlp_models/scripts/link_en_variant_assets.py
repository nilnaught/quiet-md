#!/usr/bin/env python
"""Point an English variant directory at the shared, non-weight assets.

`en-upos-base/onnx-int8` and `en-upos-large/onnx-int8` contain only the two
int8 graphs. The tokenizer / label maps / word vocab are identical to the fp32
sibling (same model card, same esupar export conventions), so we *symlink*
them instead of copying. Symlinks cost no disk and keep a single source of
truth.

Usage:
  /workspace/.venv-nlp/bin/python scripts/link_en_variant_assets.py \
      --variant large-int8 --from-dir /workspace/models/en-upos-large/onnx \
      --to-dir /workspace/models/en-upos-large/onnx-int8
"""
import argparse
import os

ASSETS = [
    "tagger_labels.json",
    "parser_labels.json",
    "parser_word_vocab.json",
]
ASSET_DIRS = ["tokenizer"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", required=True, dest="src")
    ap.add_argument("--to-dir", required=True, dest="dst")
    ap.add_argument("--force", action="store_true", help="replace existing entries")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    missing = [a for a in ASSETS if not os.path.exists(os.path.join(args.src, a))]
    if missing:
        print(f"missing assets in {args.src}: {missing}")
        return 2
    for a in ASSETS + ASSET_DIRS:
        target = os.path.abspath(os.path.join(args.src, a))
        link = os.path.join(args.dst, a)
        if os.path.islink(link):
            if os.path.realpath(link) == target:
                print(f"ok   {link}")
                continue
            os.remove(link)
        elif os.path.exists(link):
            if not args.force:
                print(f"skip {link} (exists, not a symlink; use --force)")
                continue
            if os.path.isdir(link):
                import shutil

                shutil.rmtree(link)
            else:
                os.remove(link)
        os.symlink(target, link)
        print(f"link {link} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
