#!/usr/bin/env python
"""Stage `/workspace/models/en-upos-base` from the HF snapshot using symlinks.

The task forbids copying the big weights (`pytorch_model.bin` ~478 MB,
`esupar.model` ~527 MB per model). `transformers` / `esupar` resolve relative
symlinks fine, so we link every file of the snapshot into a flat model dir that
`export_english_upos.py --model` can consume.

Usage:
  HF_HOME=/workspace/models/hf-cache /workspace/.venv-nlp/bin/python \
      scripts/stage_en_base.py
"""
import argparse
import glob
import os

DEFAULT_SNAPSHOT = (
    "/workspace/models/hf-cache/hub/models--KoichiYasuoka--roberta-base-english-upos"
    "/snapshots/a2e272a7949ecc23269cd5df2eadde045cfa90fd"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--out", default="/workspace/models/en-upos-base")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not os.path.isdir(args.snapshot):
        # fall back to whatever snapshot exists (name may change after re-download)
        cands = glob.glob(os.path.dirname(os.path.dirname(args.snapshot)) + "/snapshots/*")
        if not cands:
            raise SystemExit(f"no snapshot under {args.snapshot}")
        snap = sorted(cands)[-1]
        print(f"snapshot fallback: {snap}")
    else:
        snap = args.snapshot
    os.makedirs(args.out, exist_ok=True)
    for name in sorted(os.listdir(snap)):
        if name.startswith("."):
            continue
        target = os.path.realpath(os.path.join(snap, name))
        link = os.path.join(args.out, name)
        if os.path.islink(link) and os.path.realpath(link) == target:
            print(f"ok   {link}")
            continue
        if os.path.lexists(link):
            if not args.force:
                print(f"skip {link} (exists; --force to relink)")
                continue
            os.remove(link)
        os.symlink(target, link)  # absolute: survives dir moves of the flat dir
        print(f"link {link} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
