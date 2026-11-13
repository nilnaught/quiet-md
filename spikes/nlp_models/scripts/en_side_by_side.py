#!/usr/bin/env python
"""Side-by-side rendering of base-int8 vs large-int8 on a few sentences.

Reads the CLI outputs of two benchmark runs (`/workspace/spike-out/bench/<v>/
tokens.json`, produced by `bench_en_variants.py`), picks the requested
sentences and writes a small self-contained HTML page + PNG (headless Chrome,
width <= 1600, cropped to the content).

Both variants share the marker logic; the page shows, per sentence, the two
parses with the same tags the demo uses (solid underline = connective, dotted
underline = English non-finite, one dot below = core predicate). Rows where the
two variants differ are highlighted *and* listed in a short diff line
(`UPOS` / `deprel` / marker differences), so the comparison is machine-checked,
not only visual.

Usage:
  timeout 300 /workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/en_side_by_side.py
"""
import argparse
import json
import os
import subprocess
import tempfile

OUT = "/workspace/spike-out"
SENTENCES = [
    "I want to go home before it gets dark.",
    "She enjoys reading novels late at night.",
    "Having finished dinner, they started a movie.",
    "The book which I borrowed is still on the table.",
    "While he was cooking, she set the table carefully.",
    "Unless you hurry, we will miss the last bus.",
]


def load(variant):
    d = json.load(open(os.path.join(OUT, "bench", variant, "tokens.json")))
    return {g["text"]: [t for s in g["sentences"] for t in s["tokens"]] for g in d["en_compare"]}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def coarse(pos):
    return {"NOUN": "noun", "VERB": "verb", "ADJ": "adj", "ADV": "adv"}.get(pos, "")


def render_tokens(toks):
    out = []
    for t in toks:
        cls = ["tok"]
        if coarse(t["pos"]):
            cls.append(coarse(t["pos"]))
        if t["is_connective"]:
            cls.append("conn")
        if t["is_nonfinite"]:
            cls.append("nonfinite")
        if t["is_core_verb"]:
            cls.append("core")
        upos = t["upos_or_ltp_tag"].split("|")[0]
        title = f"{t['upos_or_ltp_tag']} · {t['deprel']}"
        inner = esc(t["text"])
        out.append(f'<span class="{" ".join(cls)}" title="{esc(title)}">{inner}</span>')
    return " ".join(out)


def row_diff(a, b):
    if len(a) != len(b):
        return [f"token count {len(a)} vs {len(b)}"]
    diffs = []
    for x, y in zip(a, b):
        up_x, up_y = x["upos_or_ltp_tag"].split("|")[0], y["upos_or_ltp_tag"].split("|")[0]
        bits = []
        if up_x != up_y:
            bits.append(f"UPOS {x['text']}: {up_x} vs {up_y}")
        elif x["upos_or_ltp_tag"] != y["upos_or_ltp_tag"]:
            bits.append(f"FEATS {x['text']}")
        if x["deprel"] != y["deprel"]:
            bits.append(f"deprel {x['text']}: {x['deprel']} vs {y['deprel']}")
        for flag, name in (("is_core_verb", "core"), ("is_connective", "conn"), ("is_nonfinite", "nonfin")):
            if x[flag] != y[flag]:
                bits.append(f"{name} {x['text']}: {x[flag]} vs {y[flag]}")
        diffs.extend(bits)
    return diffs


CSS = """
body { margin:0; padding:20px 26px 26px; background:#fbfaf7; color:#1c1b19; width:1180px;
       font-family:-apple-system,"Segoe UI","Noto Sans CJK SC","Noto Sans",sans-serif; }
h1 { font-size:18px; margin:0 0 2px; }
.sub { color:#6b6a66; font-size:12.5px; margin:0 0 12px; line-height:1.45; }
table { border-collapse:collapse; width:100%; }
td, th { padding:9px 10px; vertical-align:top; border-bottom:1px solid #e6e3dc; }
th { text-align:left; font-size:12.5px; color:#3b3a37; font-weight:600; }
td.lbl { width:120px; font-size:12px; color:#6b6a66; font-family:ui-monospace,Menlo,monospace; }
td.idx { width:26px; font-size:12px; color:#98968e; font-family:ui-monospace,Menlo,monospace; }
.diff { font-size:11.5px; color:#8a3b12; margin-top:5px; font-family:ui-monospace,Menlo,monospace; }
.same { font-size:11.5px; color:#6b8a4a; margin-top:5px; font-family:ui-monospace,Menlo,monospace; }
.legend { font-size:12px; color:#333; margin:0 0 10px; display:flex; gap:18px; }
.sent { font-size:20px; line-height:2.3; padding-bottom:3px; }
.tok { white-space:pre-wrap; }
.noun{color:#2456AE} .verb{color:#8E1F2C} .adj{color:#00783E} .adv{color:#7A4A1E}
.conn { border-bottom:2px solid #333; padding-bottom:1px; }
.nonfinite { border-bottom:2px dotted #7b4fbf; padding-bottom:1px; }
.core { position:relative; }
.core::after { content:""; position:absolute; left:50%; transform:translateX(-50%);
               bottom:-11px; width:6px; height:6px; border-radius:50%; background:#c02a2a; }
.core.conn::after, .core.nonfinite::after { bottom:-14px; }
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="base-int8")
    ap.add_argument("--b", default="large-int8")
    ap.add_argument("--html", default=os.path.join(OUT, "en_side_by_side.html"))
    ap.add_argument("--out", default=os.path.join(OUT, "en_side_by_side.png"))
    ap.add_argument("--width", type=int, default=1180)
    ap.add_argument("--height", type=int, default=2400)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    rows = []
    for i, s in enumerate(SENTENCES, 1):
        a, b = A[s], B[s]
        d = row_diff(a, b)
        note = ("<div class=\"diff\">差异：" + esc("；".join(d)) + "</div>") if d else \
               "<div class=\"same\">两变体标记完全一致</div>"
        rows.append(
            f'<tr><td class="idx">{i}</td><td class="lbl">{esc(args.a)}</td>'
            f'<td><div class="sent">{render_tokens(a)}</div></td></tr>'
            f'<tr><td class="idx"></td><td class="lbl">{esc(args.b)}</td>'
            f'<td><div class="sent">{render_tokens(b)}</div>{note}</td></tr>'
        )
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>EN side-by-side · {esc(args.a)} vs {esc(args.b)}</title><style>{CSS}</style></head><body>
<h1>英文 int8 小样对照：{esc(args.a)} vs {esc(args.b)}</h1>
<p class="sub">同一套规则（<code>pipeline.rs</code>），只换 ONNX 图。选句覆盖 to 不定式、动名词、
分词状语、缩略关系从句与从句标记（before / which / while / unless）。鼠标悬停看 UPOS+FEATS 与 deprel。</p>
<div class="legend"><span>连词 / 从句标记&nbsp;<span class="conn">实线</span></span>
<span>英文非谓语&nbsp;<span class="nonfinite">下方虚线</span></span>
<span>核心谓语&nbsp;<span class="core">下方圆点</span></span></div>
<table><tr><th>#</th><th>variant</th><th>句子</th></tr>
{''.join(rows)}
</table></body></html>
"""
    with open(args.html, "w", encoding="utf-8") as f:
        f.write(html)

    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "raw.png")
        subprocess.run([
            "/usr/bin/google-chrome-stable", "--headless=new", "--disable-gpu", "--no-sandbox",
            "--hide-scrollbars", "--force-device-scale-factor=1",
            f"--window-size={args.width},{args.height}", f"--screenshot={raw}",
            "file://" + os.path.abspath(args.html),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from PIL import Image, ImageChops

        im = Image.open(raw).convert("RGB")
        bg = Image.new("RGB", im.size, im.getpixel((0, 0)))
        bbox = ImageChops.difference(im, bg).getbbox()
        if not bbox:
            raise SystemExit("empty screenshot")
        # page body is 1180 + 2*26 px = 1232; crop the empty gutter
        body_right = min(bbox[2], 1232)
        m = 14
        box = (max(0, bbox[0] - m), max(0, bbox[1] - m),
               min(im.width, body_right + m), min(im.height, bbox[3] + m))
        out = im.crop(box)
        if out.width > 1600:
            h = round(out.height * 1600 / out.width)
            out = out.resize((1600, h), Image.LANCZOS)
        out.save(args.out, optimize=True)
        print(f"{args.out}: {out.size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
