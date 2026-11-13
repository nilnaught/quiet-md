#!/usr/bin/env python
"""Dump tokenizer golden cases so the Rust pipeline can be diffed against Python.

Writes JSON to stdout with:
  * en: word tokens (esupar `mapping`: AddPrefix " " + RoBERTa BPE) and the
        per-word piece ids that the parser graph consumes
  * zh: LTP char ids for the fixture chunks (no padding)
The Rust CLI has an `--emit-tokens` debug mode that prints the same structure so
`cargo test` can assert equality without loading weights.
"""
import json
import sys

MODEL = "/workspace/models/en-upos-large"
sys.path.insert(0, MODEL)


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    from tokenizers import Tokenizer
    zh_tok = Tokenizer.from_file("/workspace/models/ltp-small/tokenizer.json")

    en = ["The clever young fox quietly crossed the dark river.",
          "Suddenly a loud dog barked, and the frightened animal quickly ran into the dense forest.",
          "Having finished dinner, they left the restaurant.",
          "It's 3.5 km—won't they stop?",
          "quietly opened the",
          "Tom and Mary opened the door and walked inside."]
    zh = ["秋天的傍晚，明亮的月光轻轻地洒在安静的湖面上。",
          "金黄的叶子轻轻地落在清澈的水里。",
          "她 quietly opened the 旧书"]

    out = {"en": {}, "zh": {}}
    for s in en:
        words = [t for t in tok.tokenize(" " + s) if t and not t.startswith(" ") ] if False else tok.tokenize(" " + s)
        # esupar: for every word, `tokenize(" " + word)`; a word cannot contain spaces.
        word_tokens = s.split(" ")
        words = []
        pieces = []
        for w in word_tokens:
            p = tok.tokenize(" " + w)
            words.append(p)
            pieces.append(tok.convert_tokens_to_ids(p))
        out["en"][s] = {"words": words, "pieces": pieces,
                        "ids": tok(s)["input_ids"]}
    for s in zh:
        enc = zh_tok.encode(s)
        out["zh"][s] = {"tokens": enc.tokens, "ids": enc.ids}
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
