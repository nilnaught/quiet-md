#!/usr/bin/env python
"""Sanity-check an English variant (fp32 or int8) with ONNX Runtime *alone*.

For every sentence of the fixture's `en_compare` set:
  * run the tagger, take the argmax label per subword, compare with the fp32
    reference dir (subword agreement + per-word UPOS agreement);
  * run the parser with the same inputs the Rust CLI builds, compare the
    argmax heads (`arc_logits[:, 1:, :]`) and the selected rel ids;
  * report whether the argmax head graph is a valid single-rooted tree (the
    Rust CLI falls back to its MST repair otherwise);
  * print the first mismatches so a reviewer sees *where* int8 drifts.

Usage (one variant per process):
  timeout 600 /workspace/.venv-nlp/bin/python scripts/check_en_variants.py \
      --ref /workspace/models/en-upos-base/onnx \
      --variant base-int8=/workspace/models/en-upos-base/onnx-int8 \
      --sentences fixtures/sentences.json --out /workspace/spike-out/check_en_base_int8.json
"""
import argparse
import json
import os
import sys


def build_parser_inputs(tok, sub_vocab, words):
    """Word-level BPE pieces + word-vocab ids, exactly like english.rs."""
    pieces = []
    for w in words:
        ids = tok.encode(" " + w, add_special_tokens=False).ids
        pieces.append(ids if ids else [1])
    F = max(len(p) for p in pieces)
    rows = len(words) + 2
    sub_ids = [[1] * F for _ in range(rows)]
    sub_ids[0][0] = 0
    for i, p in enumerate(pieces):
        for k, x in enumerate(p):
            sub_ids[i + 1][k] = x
    order = sorted(range(rows * F), key=lambda i: (1 if sub_ids[i // F][i % F] == 1 else 0, i))
    words_flat = [x for row in sub_ids for x in row]
    word_ids = [2] + [sub_vocab.get(w.lower(), 1) for w in words] + [0]
    return words_flat, word_ids, order, rows, F


def valid_tree(heads):
    """heads: list with index 0 == ROOT, 1..n words; returns (ok, reason)."""
    n = len(heads) - 1
    if sum(1 for h in heads[1:] if h == 0) != 1:
        return False, "not single-rooted"
    for d in range(1, n + 1):
        seen = set()
        cur = d
        while cur != 0:
            if cur in seen:
                return False, f"cycle at {d}"
            seen.add(cur)
            cur = heads[cur]
            if cur > n:
                return False, f"bad head {cur}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="fp32 dir (tagger/parser/assets)")
    ap.add_argument("--variant", required=True, help="name=dir")
    ap.add_argument("--sentences", default="fixtures/sentences.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-report", type=int, default=8)
    ap.add_argument("--vocab-from", default="", help="dir holding parser_word_vocab.json if absent")
    ap.add_argument("--tokens-dir", default="", help="CLI --compare-dir output to cross-check tokenisation")
    ap.add_argument("--dump-golden", default="", help="write golden tagger/parser inputs as JSON")
    args = ap.parse_args()
    name, vdir = args.variant.split("=", 1)

    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.ref, "tokenizer", "tokenizer.json"))
    tl = json.load(open(os.path.join(args.ref, "tagger_labels.json")))["id2label"]
    id2label = {int(k): v for k, v in tl.items()}
    # the base export predates `parser_word_vocab.json`; both model cards share
    # the same 15721-entry word vocab (checked: parser word_embed shapes equal),
    # so fall back to the large export's dump.
    wv_path = os.path.join(args.ref, "parser_word_vocab.json")
    if not os.path.exists(wv_path):
        wv_path = os.path.join(args.vocab_from or "/workspace/models/en-upos-large/onnx",
                               "parser_word_vocab.json")
    pv = json.load(open(wv_path))["itos"]
    sub_vocab = {}
    for i, w in enumerate(pv):
        sub_vocab.setdefault(w, i)
    rels = json.load(open(os.path.join(args.ref, "parser_labels.json")))["labels"]

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    tag_v = ort.InferenceSession(os.path.join(vdir, "en_upos_tagger.onnx"), so, providers=["CPUExecutionProvider"])
    par_v = ort.InferenceSession(os.path.join(vdir, "en_supar_parser.onnx"), so, providers=["CPUExecutionProvider"])
    have_ref = os.path.realpath(vdir) != os.path.realpath(args.ref)
    if have_ref:
        tag_r = ort.InferenceSession(os.path.join(args.ref, "en_upos_tagger.onnx"), so, providers=["CPUExecutionProvider"])
        par_r = ort.InferenceSession(os.path.join(args.ref, "en_supar_parser.onnx"), so, providers=["CPUExecutionProvider"])

    import re

    def words_of(text):
        out = []
        for w in text.split():
            # split edge punctuation like english.rs::split_punct
            m = re.findall(r"[^\s.,!?;:\"'()\[\]{}“”‘’…—–]+|[.,!?;:\"'()\[\]{}“”‘’…—–]", w)
            out.extend(m)
        return out

    fx = json.load(open(args.sentences))
    sents = fx.get("en_compare", [])

    res = {
        "variant": name, "dir": vdir, "sentences": len(sents),
        "tagger_subword_agree": 0, "tagger_subword_total": 0,
        "word_upos_agree": 0, "word_total": 0,
        "head_agree": 0, "head_total": 0,
        "rel_agree": 0, "rel_total": 0,
        "valid_trees": 0, "invalid_trees": [],
        "max_arc_absdiff": 0.0, "max_logit_absdiff": 0.0,
        "mismatches": [],
        "word_mismatch_list": [],
        "tagger_pathology": [],
    }
    import numpy as np

    # ---- golden inputs for the Rust CLI (reused by --tokens-dir mode) ------
    golden = {"tagger": [], "parser": []}

    def tagger_input(sent):
        # Mirrors english.rs: flat = <s> + BPE(" " + word) per word + </s>, padded
        # to 40 with <pad>, mask = id != 1. (Tokenising the *sentence* lets
        # RoBERTa merge "stamps." differently — the CLI never does that.)
        words = words_of(sent)
        flat = [0]
        for w in words:
            flat.extend(tok.encode(" " + w, add_special_tokens=False).ids)
        flat.append(2)
        flat = flat[:512]
        flat += [1] * max(0, 40 - len(flat))
        return words, np.array([flat], dtype=np.int64), np.array([[int(x != 1) for x in flat]], dtype=np.int64)

    for sent in sents:
        words, ids, mask = tagger_input(sent)
        lengths = [len(tok.encode(" " + w, add_special_tokens=False).ids) or 1 for w in words]
        lv = tag_v.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        lv = np.ascontiguousarray(lv, dtype=np.float32)
        # per-word argmax follows english.rs: the last non-I- subword label wins
        pv_idx = 1
        v_words = []
        for L in lengths:
            cur = ""
            for k in range(L):
                lab = id2label[int(lv[0, pv_idx + k].argmax())]
                if not lab.startswith("I-"):
                    cur = lab.replace("B-", "", 1) if lab.startswith("B-") else lab
            v_words.append(cur)
            pv_idx += L
        sub_ids, word_ids, order, rows, F = build_parser_inputs(tok, sub_vocab, words)
        n = len(words)
        golden["tagger"].append({"text": sent, "words": words,
                                 "input_ids": ids[0].tolist(), "mask": mask[0].tolist()})
        golden["parser"].append({"text": sent, "words": words, "word_ids": word_ids,
                                 "subword_ids": sub_ids, "order": order, "rows": rows, "F": F})
        ao = par_v.run(None, {
            "subword_ids": np.array([sub_ids], dtype=np.int64).reshape(1, rows, F),
            "word_ids": np.array([word_ids], dtype=np.int64),
            "piece_order": np.array([order], dtype=np.int64),
        })
        heads = ao[0][0, 1:, :].argmax(-1).tolist()
        ok, why = valid_tree([0] + heads)
        if ok:
            res["valid_trees"] += 1
        else:
            res["invalid_trees"].append({"text": sent, "reason": why, "heads": heads})
        if not have_ref:
            continue
        lr = np.ascontiguousarray(tag_r.run(None, {"input_ids": ids, "attention_mask": mask})[0],
                                  dtype=np.float32)
        ar = par_r.run(None, {
            "subword_ids": np.array([sub_ids], dtype=np.int64).reshape(1, rows, F),
            "word_ids": np.array([word_ids], dtype=np.int64),
            "piece_order": np.array([order], dtype=np.int64),
        })
        res["max_logit_absdiff"] = max(res["max_logit_absdiff"], float(np.abs(lv - lr).max()))
        if int((lv[0].argmax(-1) != lr[0].argmax(-1)).sum()) > len(words):
            res["tagger_pathology"].append(sent)
        res["max_arc_absdiff"] = max(res["max_arc_absdiff"], float(np.abs(ao[0] - ar[0]).max()))
        av = lv[0].argmax(-1).tolist()
        ar_ = lr[0].argmax(-1).tolist()
        res["tagger_subword_agree"] += sum(int(a == b) for a, b in zip(av, ar_))
        res["tagger_subword_total"] += len(av)
        # ref words for comparison
        pv_idx = 1
        r_words = []
        for L in lengths:
            cur = ""
            for k in range(L):
                lab = id2label[int(lr[0, pv_idx + k].argmax())]
                if not lab.startswith("I-"):
                    cur = lab.replace("B-", "", 1) if lab.startswith("B-") else lab
            r_words.append(cur)
            pv_idx += L
        for w, a, b in zip(words, v_words, r_words):
            res["word_total"] += 1
            if a.split("|")[0] == b.split("|")[0]:
                res["word_upos_agree"] += 1
            if a != b:
                res["word_mismatch_list"].append({"sentence": sent, "word": w,
                                                  "variant": a, "fp32": b})
                if len(res["mismatches"]) < args.max_report:
                    res["mismatches"].append({"word": w, "variant": a, "fp32": b})
        rheads = ar[0][0, 1:, :].argmax(-1).tolist()
        for a, b in zip(heads, rheads):
            res["head_total"] += 1
            res["head_agree"] += int(a == b)
        # rel ids: pick the label on the chosen arc
        rcls = ao[1].shape[-1]
        for i, h in enumerate(heads):
            ra = int(ao[1][0, i + 1, h].argmax())
            rb = int(ar[1][0, i + 1, rheads[i]].argmax())
            res["rel_total"] += 1
            res["rel_agree"] += int(ra == rb)

    def rate(a, b):
        return round(a / b, 4) if b else None

    if args.dump_golden:
        json.dump(golden, open(args.dump_golden, "w"))
        print(f"golden inputs -> {args.dump_golden}", file=sys.stderr)
    if args.tokens_dir:
        import glob as _glob
        ours = {}
        for f in _glob.glob(os.path.join(args.tokens_dir, "*.json")):
            d = json.load(open(f))
            ours[d["text"]] = [t for s in d["sentences"] for t in s["tokens"]]
        idx = {s["text"]: i for i, s in enumerate(golden["parser"])}
        # tagger: the 2nd output tensor logits -> labels; parser: len(heads) must
        # be words+1 (i.e. the ORT graph ran, not a stub)
        tok_ok = 0
        tok_bad = []
        for g in golden["tagger"]:
            toks = ours.get(g["text"], [])
            got = [t["text"] for t in toks]
            if got == g["words"] or [w for w in got if w not in ("",)] == g["words"]:
                tok_ok += 1
            else:
                tok_bad.append({"text": g["text"], "expected": g["words"], "got": got})
        res["cli_tokenization_matches"] = {"ok": tok_ok, "total": len(golden["tagger"]),
                                            "bad": tok_bad[:3]}

    out = {
        **res,
        "tagger_subword_agreement": rate(res["tagger_subword_agree"], res["tagger_subword_total"]),
        "word_upos_agreement": rate(res["word_upos_agree"], res["word_total"]),
        "head_agreement": rate(res["head_agree"], res["head_total"]),
        "rel_agreement_vs_fp32": rate(res["rel_agree"], res["rel_total"]),
        "valid_tree_rate": rate(res["valid_trees"], len(sents)),
    }
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if args.out:
        open(args.out, "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
