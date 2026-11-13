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
    ap.add_argument("--content-width", type=int, default=1148,
                    help="right edge of the page body (body max-width 1080 + 2*34 padding); "
                         "crops the empty viewport gutter")
    ap.add_argument("--max-width", type=int, default=1600,
                    help="hard cap; the image is never wider than this")
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
        # Sample the background at the origin: `getpixel((1,1))` would read the
        # first text or underline once the whole page is full-column width.
        bg = Image.new("RGB", im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        full = diff.getbbox()
        if not full:
            raise SystemExit("empty screenshot")
        # The page sets `max-width:1080px`; Chrome still renders a wider viewport
        # with empty background on the right. Crop to the *page body box*
        # (measured with Chrome) instead of the viewport width.
        bbox = full
        if args.content_width:
            body_right = args.content_width
            side = max(0, min(im.width, body_right) - full[2])
            if side <= 4:
                # content really extends past our guess: fall back to the bbox
                body_right = full[2]
            bbox = (full[0], full[1], min(body_right, full[2]), full[3])
        # a small margin so descenders of emphasis dots / the core dot are not
        # clipped
        left, top, right, bottom = bbox
        m = 18
        box = (
            max(0, left - m),
            max(0, top - m),
            min(im.width, right + m),
            min(im.height, bottom + m),
        )
        cropped = im.crop(box)
        if cropped.width > args.max_width:
            h = round(cropped.height * args.max_width / cropped.width)
            cropped = cropped.resize((args.max_width, h), Image.LANCZOS)
        cropped.save(args.out, optimize=True)
        print(f"{args.out}: {box[2]-box[0]}x{box[3]-box[1]} (from {im.size})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
