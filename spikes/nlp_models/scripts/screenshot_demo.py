#!/usr/bin/env python
"""Render `demo.html` to a cropped `demo.png` with headless Chrome.

Chrome's `--screenshot` only captures the viewport, so we

  1. ask Chrome for the full document height (`--dump-dom` + a tiny probe page
     that writes the height into `<title>` is unreliable), so instead we take a
     tall screenshot and crop the trailing uniform background with Pillow, and
  2. also trim uniform rows at the top/bottom/left/right.

Usage:
  /workspace/.venv-nlp/bin/python scripts/screenshot_demo.py \
      --html /workspace/spike-out/demo.html --out /workspace/spike-out/demo.png
"""
import argparse
import os
import subprocess
import sys
import tempfile

CHROME = "/usr/bin/google-chrome-stable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="/workspace/spike-out/demo.html")
    ap.add_argument("--out", default="/workspace/spike-out/demo.png")
    ap.add_argument("--width", type=int, default=1180)
    ap.add_argument("--height", type=int, default=4200)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "raw.png")
        cmd = [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--window-size={args.width},{args.height}",
            f"--screenshot={raw}",
            "file://" + os.path.abspath(args.html),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from PIL import Image, ImageChops

        im = Image.open(raw).convert("RGB")
        bg = Image.new("RGB", im.size, im.getpixel((1, 1)))
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()
        if not bbox:
            raise SystemExit("empty screenshot")
        # a small margin so descenders of emphasis dots are not clipped
        left, top, right, bottom = bbox
        m = 8
        box = (
            max(0, left - m),
            max(0, top - m),
            min(im.width, right + m),
            min(im.height, bottom + m),
        )
        im.crop(box).save(args.out, optimize=True)
        print(f"{args.out}: {box[2]-box[0]}x{box[3]-box[1]} (from {im.size})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
