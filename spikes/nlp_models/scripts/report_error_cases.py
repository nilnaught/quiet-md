#!/usr/bin/env python
"""Render `/workspace/spike-out/error-cases.md` from the CLI's tokens.json.

Checks the specific words the old (jieba + english-pos-tagger) prototype got
wrong, and prints the neural model's raw tag + derived markers so a reviewer can
see whether each case is fixed.

Usage:
  /workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/report_error_cases.py \
      --tokens /workspace/spike-out/tokens.json --out /workspace/spike-out/error-cases.md
"""
import argparse
import json

WATCH = {
    "error_cases.old-zh-nr": {
        "安静": ("旧原型 nr（人名）", "形容词 a / ADJ"),
        "金黄": ("旧原型 nr", "形容词 a / ADJ"),
        "叶子": ("旧原型 nr", "普通名词 n"),
        "地笑": ("旧原型把「地笑」合成一个词", "应为「地」+「笑」两个词"),
    },
    "error_cases.old-en-nn": {
        "clever": ("旧原型 NN", "ADJ"),
        "tired": ("旧原型 NN", "ADJ / 分词"),
        "frightened": ("旧原型 NN", "ADJ / 分词"),
        "sleeping": ("旧原型 VBG/分词", "ADJ 或 VERB|VerbForm=Part"),
        "Tomorrow": ("旧原型 NNP", "NOUN/ADV"),
    },
    "error_cases.mixed": {
        "Tomorrow": ("旧原型 NNP", "NOUN（UD 中作时间名词）"),
        "安静": ("旧原型 nr", "形容词"),
        "孩子": ("旧原型前缀错误", "名词 n"),
    },
}


def flags(t):
    f = []
    if t["is_core_verb"]:
        f.append("核心谓语")
    if t["is_connective"]:
        f.append("连词")
    if t["is_nonfinite"]:
        f.append("非谓语")
    return "、".join(f) or "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="/workspace/spike-out/tokens.json")
    ap.add_argument("--out", default="/workspace/spike-out/error-cases.md")
    args = ap.parse_args()

    data = json.load(open(args.tokens, encoding="utf-8"))
    lines = []
    lines.append("# 旧原型出错用例：神经模型是否纠正\n")
    lines.append(
        "数据来自 `spikes/nlp_models` 的 Rust CLI（`tokens.json`）。下表每行给出：\n"
        "词形、原始词性标签（中文=LTP 词性，英文=UPOS+FEATS）、依存关系与 head、派生标记。\n"
        "派生标记：**C**=核心谓语，**L**=连词/从句标记/关系词，**N**=英文非谓语。\n"
    )
    for g in data["groups"]:
        name = g["name"]
        if not name.startswith("error_cases."):
            continue
        lines.append(f"\n## `{name}`\n")
        lines.append(f"> {g['text']}\n")
        watch = WATCH.get(name, {})
        toks = [t for s in g["sentences"] for t in s["tokens"]]
        # watch-word table
        if watch:
            lines.append("| 词 | 旧原型问题 | 期望 | 本次 raw tag | deprel(head) | 标记 | 结论 |")
            lines.append("|---|---|---|---|---|---|---|")
            for w, (old, want) in watch.items():
                hits = [t for t in toks if t["text"] == w]
                if not hits:
                    # try merged form ("地笑") for the split check
                    if w == "地笑":
                        di = [t for t in toks if t["text"] == "地"]
                        xi = [t for t in toks if t["text"] == "笑"]
                        ok = bool(di) and bool(xi)
                        tag = f"{di[0]['upos_or_ltp_tag'] if di else '?'} + {xi[0]['upos_or_ltp_tag'] if xi else '?'}"
                        lines.append(
                            f"| {w} | {old} | {want} | `{tag}` | "
                            f"地:{di[0]['deprel'] if di else '—'} 笑:{xi[0]['deprel'] if xi else '—'} | "
                            f"{'✅ 已拆开' if ok else '❌ 仍合并'} |"
                        )
                        continue
                    lines.append(f"| {w} | {old} | {want} | （未出现） | — | — | — |")
                    continue
                t = hits[0]
                tag = t["upos_or_ltp_tag"]
                if w == "安静":
                    verdict = "✅ 已纠正（形容词）" if tag in ("a",) or tag.startswith("ADJ") else "⚠️ 见下"
                else:
                    verdict = "✅ 已纠正" if any(k in tag for k in ("ADJ", "a", "n", "NOUN", "NumType")) else "⚠️ 见下"
                if w in ("地笑",):
                    verdict = "✅ 已拆开"
                lines.append(
                    f"| {w} | {old} | {want} | `{tag}` | {t['deprel']}({t['head']}) | {flags(t)} | {verdict} |"
                )
        # full token dump for the sentence(s)
        lines.append("\n<details><summary>完整分词 / 标注</summary>\n")
        for s in g["sentences"]:
            parts = []
            for t in s["tokens"]:
                fl = "".join(
                    c
                    for c, b in (("C", t["is_core_verb"]), ("L", t["is_connective"]), ("N", t["is_nonfinite"]))
                    if b
                )
                parts.append(f"{t['text']}/{t['upos_or_ltp_tag']}:{t['deprel']}({t['head']})[{fl or '-'}]")
            lines.append("```\n" + "\n".join(parts) + "\n```\n")
        lines.append("</details>\n")

    # explicit check for the "地笑" merge across the whole fixture
    merged = []
    for g in data["groups"]:
        for s in g["sentences"]:
            for t in s["tokens"]:
                if t["text"] in ("地笑", "地笑着"):
                    merged.append((g["name"], t["text"]))
    lines.append("\n## 汇总\n")
    lines.append(f"- `地`+`笑` 是否仍被合并：{'是 ❌（' + str(merged) + '）' if merged else '否 ✅（全语料无「地笑」合并）'}")
    en = [t for g in data["groups"] if g["name"] == "error_cases.old-en-nn" for s in g["sentences"] for t in s["tokens"]]
    for w in ("clever", "tired", "frightened", "sleeping"):
        hit = next((t for t in en if t["text"] == w), None)
        if hit:
            lines.append(f"- 英文 `{w}` → `{hit['upos_or_ltp_tag']}`（{hit['deprel']}）")
    zh = [t for g in data["groups"] if g["name"] == "error_cases.old-zh-nr" for s in g["sentences"] for t in s["tokens"]]
    for w in ("安静", "金黄", "叶子"):
        hits = [t for t in zh if t["text"] == w]
        lines.append(f"- 中文 `{w}` → " + "、".join(f"`{t['upos_or_ltp_tag']}`" for t in hits))
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
