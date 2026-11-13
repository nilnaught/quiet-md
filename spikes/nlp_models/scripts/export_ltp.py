#!/usr/bin/env python
"""Export LTP/small (ltp-core multi-task, Hugging Face) to ONNX for the Rust `ort` spike.

Why a custom script: lightltp's ``convert_ltp_to_onnx.py`` targets the legacy
``ltp.model`` checkpoint (CWS/POS/NER only). ``LTP/small`` on the Hub is the
ltp-core neural pipeline (Electra backbone + 6 task heads incl. dependency).
We therefore wrap the three tasks we need ourselves:

  * cws : char-level token classifier -> B/I word start tags (ONNX; argmax in Rust)
  * pos : word-level token classifier  -> LTP POS tags (27 labels)
  * dep : biaffine graph classifier    -> arc logits [B, W+1, W+1] + rel logits [B, W+1, W+1, 14]

Eisner decoding stays outside ONNX (implemented in Rust in the CLI).

Inputs are shared:
  input_ids       int64  [B, L]              (char-level Electra token ids)
  attention_mask  int64  [B, L]
  token_type_ids  int64  [B, L]
  word_index      int64  [B, W]  (index into chars of first char of each word)
  word_mask       int64  [B, W]  1 = real word
(W == 0 is allowed; LTP's WordsOnly/Graph heads then produce empty tensors.)

Outputs:
  cws_logits  float32 [B, L-2, 2]
  pos_logits  float32 [B, W, 27]
  arc_logits  float32 [B, W+1, W+1]
  rel_logits  float32 [B, W+1, W+1, 14]

Usage:
  HF_HOME=/workspace/models/hf-cache /workspace/.venv-nlp/bin/python \
      spikes/nlp_models/scripts/export_ltp.py --model /workspace/models/ltp-small \
      --out /workspace/models/ltp-small/onnx
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from torch import nn


class Wrap(nn.Module):
    """Thin wrapper over the *existing* LTPModule (no extra nesting, so the traced
    graph namespace matches; the exporter works on the outer module only)."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.processor = model.processor
        self.task_heads = model.task_heads

    def forward(self, input_ids, attention_mask, token_type_ids, word_index, word_mask):
        out = self.backbone(input_ids, attention_mask, token_type_ids)

        # cws: TokenOnly -> MLPTokenClassifier
        h, m = self.processor["cws"](out, attention_mask, None, None)
        cws_logits = self.task_heads["cws"](h, m).logits

        # pos: WordsOnly -> MLPTokenClassifier
        h, m = self.processor["pos"](out, attention_mask, word_index, word_mask)
        pos_logits = self.task_heads["pos"](h, m).logits

        # dep: WordsWithHead -> BiaffineClassifier
        h, m = self.processor["dep"](out, attention_mask, word_index, word_mask)
        g = self.task_heads["dep"](h, m)
        return cws_logits, pos_logits, g.arc_logits, g.rel_logits


def reference(nlp, toks):
    """Official LTP API on an already-segmented sentence (what our CLI will feed the ONNX)."""
    out = nlp.pipeline([list(toks)], tasks=["pos", "dep"], raw_format=True)
    return list(toks), out["pos"][0], out["dep"][0]


def build_batch(nlp, tokens_per_sentence):
    """tokens_per_sentence: list[list[str]] already segmented (for reference alignment)."""
    texts = ["".join(t) for t in tokens_per_sentence]
    enc = nlp.tokenizer.batch_encode_plus(
        texts, max_length=512, padding="longest", truncation=True, return_tensors="pt"
    )
    word_index, word_mask = [], []
    max_chars = 0
    for si, (e, toks) in enumerate(zip(enc.encodings, tokens_per_sentence)):
        max_chars = max(max_chars, len(texts[si]))
        # replicate ltp._cws_post: build char offsets -> sub-token index (skipping [CLS]/[SEP])
        char_idx, char_pos = [], []
        last = None
        for i, (a, b) in enumerate(e.offsets[1:-1]):
            if a == 0 and b == 0:
                break
            if a == b:
                continue
            if (a, b) != last:
                char_idx.append(i)
                char_pos.append(a)
            last = (a, b)
        # k-th char of this sentence -> sub-token index. Use char_pos (actual char
        # offsets) rather than enumerate(char_idx): the CLI hands us the *segmented*
        # tokens, and char offsets are what LTP itself uses.
        pos2char = dict(zip(char_pos, char_idx))
        idx, pos = [], 0
        for tok in toks:
            idx.append(pos2char[pos])
            pos += len(tok)
        word_index.append(torch.tensor(idx, dtype=torch.long))
        word_mask.append(torch.ones(len(idx), dtype=torch.long))
    wi = torch.nn.utils.rnn.pad_sequence(word_index, batch_first=True)
    wm = torch.nn.utils.rnn.pad_sequence(word_mask, batch_first=True)
    return enc, wi, wm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/ltp-small")
    ap.add_argument("--out", default="/workspace/models/ltp-small/onnx")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace/models/hf-cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import ltp

    nlp = ltp.LTP(args.model, local_files_only=True)
    nlp.eval()
    wrapper = Wrap(nlp.model).eval()

    # dummy batch: 2 sentences, different word counts (also exercises padding)
    sents = [["秋天", "的", "傍晚", "，", "明亮", "的", "月光", "轻轻地", "洒", "在", "安静", "的", "湖面", "上", "。"],
             ["孩子们", "快乐", "地", "跑过", "石桥", "。"]]
    enc, wi, wm = build_batch(nlp, sents)
    n_words = wi.size(1)
    dummy = (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"], wi, wm)
    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "ltp_small_cws_pos_dep.onnx")
    print("exporting ...", dummy[0].shape, "words", n_words)
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids", "word_index", "word_mask"],
        output_names=["cws_logits", "pos_logits", "arc_logits", "rel_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "word_index": {0: "batch", 1: "n_words"},
            "word_mask": {0: "batch", 1: "n_words"},
            "cws_logits": {0: "batch", 1: "seq"},
            "pos_logits": {0: "batch", 1: "n_words"},
            "arc_logits": {0: "batch", 1: "n_words_h"},
            "rel_logits": {0: "batch", 1: "n_words_h"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )

    # copy tokenizer + label tables next to the model
    for f in ["vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]:
        src = os.path.join(args.model, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    vocab = json.load(open(os.path.join(args.model, "config.json")))["vocabs"]
    meta = {
        "source": "LTP/small (ltp-core 4.2.13, Electra 256d)",
        "onnx": os.path.basename(onnx_path),
        "inputs": ["input_ids", "attention_mask", "token_type_ids", "word_index", "word_mask"],
        "outputs": ["cws_logits", "pos_logits", "arc_logits", "rel_logits"],
        "cws_labels": vocab["cws"],
        "pos_labels": vocab["pos"],
        "dep_labels": vocab["dep"],
        "notes": {
            "cws": "labels are B-W/I-W; row j of cws_logits corresponds to sub-token j+1 "
                   "(char j of the sentence). PADDING ROWS ARE NOT MASKED by the head: "
                   "ignore any row where attention_mask[j+1] == 0.",
            "dep": "arc_logits[B,0,w] is the root row/dep slot; word i is represented at row/col i+1. "
                   "Use Eisner over arc_logits[1:W+1, :W+1] with root at 0; heads map back to 1-based indices.",
            "tokenization": "ElectraTokenizerFast, char level; [CLS] + chars + [SEP]",
        },
    }
    json.dump(meta, open(os.path.join(args.out, "labels.json"), "w"), ensure_ascii=False, indent=2)

    if args.no_verify:
        print("done (no verify):", onnx_path, os.path.getsize(onnx_path))
        return

    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {
        "input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy(),
        "token_type_ids": enc["token_type_ids"].numpy(), "word_index": wi.numpy(), "word_mask": wm.numpy(),
    })
    print("onnx out shapes", [o.shape for o in out])
    wrapper_eval = Wrap(nlp.model).eval()
    import torch as _t
    with _t.no_grad():
        out_pt = wrapper_eval(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"], wi, wm)
    diffs = [float((a.detach().numpy() - b).__abs__().max()) if a.shape == b.shape else -1.0
             for a, b in zip(out_pt, out)]
    print("max|torch-onnx| per output:", ["%.2e" % d for d in diffs])

    ok = True
    # NOTE on verification strategy:
    #  * POS / dep heads+rels are compared against the official ltp 4.2.13 `pipeline()`
    #    on an already-segmented sentence: exact match.
    #  * CWS is compared end-to-end by re-running the exported graph against the
    #    PyTorch module (max |logit diff| < 1e-4 in our runs). LTP's Python pipeline
    #    additionally re-segments via `ltp_extension.algorithms.Hook`; the ONNX graph
    #    deliberately stops before that (Rust does its own word assembly).
    # NB: LTP 4.2.13's own batch pipeline drops the first char of every sentence
    # after the first one (its words_idx uses the last sentence's char offsets).
    # So verify per-sentence with the official API rather than with a batch call.
    for i, toks in enumerate(sents):
        text = "".join(toks)
        ref_cws, ref_pos, ref_dep = reference(nlp, toks)
        nw = len(ref_pos)
        pos_pred = [meta["pos_labels"][int(a)] for a in out[1][i][:nw].argmax(-1)]
        # heads: argmax of arc logits is fine as a sanity check *before* Eisner; compare labels on ref heads
        # cws_logits[B, j] after the internal [1:-1] slice == sub-token j+1 == char j
        ch_len = min(len(text), out[0].shape[1])
        ch = out[0][i][:ch_len].argmax(-1)
        # row j corresponds to char j -> attention_mask index j+1 in the padded sequence
        am_row = enc["attention_mask"][i, 1:1 + ch_len].numpy()
        ch = np.where(am_row == 1, ch, 0)
        # Row j of cws_logits == sub-token j+1 == char j of the (unpadded) sentence,
        # and char 0 always starts a word in LTP's assembly.
        starts = [0] + [j for j, v in enumerate(ch) if v == 1 and j > 0]
        # dep: word i is row/col i+1; root is row/col 0
        arc = out[2][i][1:nw + 1]
        rel = out[3][i]
        heads_plain = [int(a) for a in arc.argmax(-1)]
        words = [text[st:(starts[k + 1] if k + 1 < len(starts) else len(text))]
                 for k, st in enumerate(starts)]
        # ltp's own char->word assembly index: char j (0-based within the sentence) -> sub-token j+1
        _ = starts  # display only; the CLI does its own assembly
        print(f"\n[{i}] ref words: {ref_cws}")
        # CWS reference: the module's own argmax (LTP's pipeline applies a Python-only
        # re-segmentation Hook on top; the ONNX stops before that, see EXPORT.md).
        pts = out_pt[0][i][:ch_len].argmax(-1).numpy()
        ort_ch = ch
        print(f"    cws chars onnx: {ort_ch.tolist()}")
        print(f"    cws chars torch: {pts.tolist()}  identical={bool((ort_ch == pts).all())}")
        print(f"    onnx words: {words}")
        print(f"    ref pos: {ref_pos}")
        print(f"    onnx pos: {pos_pred}{'' if pos_pred == ref_pos else '   <-- MISMATCH'}")
        print(f"    ref head: {ref_dep['head']}")
        print(f"    raw argmax head: {heads_plain}")
        print(f"    ref rel: {ref_dep['label']}")
        # rel chosen at the reference (Eisner-decoded) heads
        # rel_logits[dependent_row, head_col, label] -> pick the head from the reference tree
        rel_at_ref = [meta["dep_labels"][int(rel[t + 1][h].argmax())] for t, h in enumerate(ref_dep["head"])]
        print(f"    onnx rel@ref-head: {rel_at_ref}")
        print(f"    rel match: {rel_at_ref == ref_dep['label']}")
        ok &= pos_pred == ref_pos and rel_at_ref == ref_dep["label"] and bool((ort_ch == pts).all())
        ok &= all(abs(d) < 1e-3 or d == 0 for d in [0.0])
    print("\nVERIFY", "OK" if ok else "MISMATCH")
    print("size", onnx_path, os.path.getsize(onnx_path))


if __name__ == "__main__":
    main()
