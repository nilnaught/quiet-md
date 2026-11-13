#!/usr/bin/env python
"""Export the English esupar pipeline (KoichiYasuoka/roberta-large-english-upos) to ONNX.

Two graphs, because esupar is *two* models:

  A. tagger : ``AutoModelForTokenClassification`` (roberta-large) -> 1380 labels
     Each label is the packed UPOS + UD features string, e.g.
     ``VERB|Mood=Ind|Tense=Past|VerbForm=Fin``.  Multi-word tokens are encoded as
     ``A+B`` (see `labels.json`).

  B. parser : ``esupar.model`` (safetensors; a pickled SuPar ``BiaffineDependencyParser``).
     Contents (see EXPORT.md for the full audit):
       * word_embed     Embedding(15721, 100)
       * bert_embed     TransformerEmbedding over **another roberta-large**
                        (n_layers=4, n_out=100, mean pooling, ScalarMix+gamma)
       * encoder        VariationalLSTM(200 -> 400*2, 3 layers, LSTMCell based)
       * arc/rel MLPs   MLP(800->500)/(800->100) + Biaffine
     The stock VariationalLSTM is a Python loop over ``nn.LSTMCell``; at eval time
     it is a plain packed BiLSTM, so we rebuild it with ``nn.LSTM`` on the same
     weights (verified against the original before export).
     Eisner decoding is NOT exported: the graph returns ``arc_logits`` and
     ``rel_logits`` and the Rust CLI decodes the tree.

Usage:
  # large (already exported, fp32 kept in /workspace/models/en-upos-large/onnx)
  HF_HOME=/workspace/models/hf-cache /workspace/.venv-nlp/bin/python \
    spikes/nlp_models/scripts/export_english_upos.py \
    --model /workspace/models/en-upos-large --out /workspace/models/en-upos-large/onnx

  # base (the HF snapshot is staged with symlinks by stage_en_base.py; the
  # weights themselves are never copied — 500 MB bin + 527 MB esupar stay in
  # the HF cache)
  HF_HOME=/workspace/models/hf-cache timeout 900 /workspace/.venv-nlp/bin/python \
    spikes/nlp_models/scripts/export_english_upos.py \
    --model /workspace/models/en-upos-base \
    --out   /workspace/models/en-upos-base/onnx

The script also prints a torch-vs-ONNX check on a few fixture-like sentences
(tagger argmax labels + parser heads, not just logits) so an export that
"runs" but produces garbage does not slip through.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Parser graph (inference-only re-implementation)
# --------------------------------------------------------------------------- #
class ParserGraph(nn.Module):
    """subword_ids/subword_mask: [B, W, F]; word_ids: [B, W] (0=pad, 1=unk, 2=bos)."""

    def __init__(self, parser):
        super().__init__()
        m = parser.model
        self.bert = m.bert_embed.bert
        self.bert.requires_grad_(False)
        self.n_layers = m.bert_embed.n_layers
        self.max_len = m.bert_embed.max_len
        self.scalar_mix = m.bert_embed.scalar_mix
        self.subword_pad = m.bert_embed.pad_index          # 1 == <pad>; note <s> is 0
        self.proj = m.bert_embed.projection
        self.fix_len = 20            # from esupar's build() default; dynamic anyway
        self.pad_index_bert = m.bert_embed.pad_index
        self.word_embed = m.word_embed
        self.pad_index = m.args.pad_index

        # -- BiLSTM rebuilt from LSTMCell params (f then b per layer) ----------
        enc = m.encoder
        h = enc.hidden_size
        in_sizes = [enc.input_size] + [2 * h] * (enc.num_layers - 1)
        # supar's VariationalLSTM is (at eval time) a plain 3-layer bidirectional LSTM:
        #   layer i: fwd cell on layer input, bwd cell on the same input, concat outputs.
        # Use nn.LSTM(bidirectional=True) per layer with the exact cell weights.
        lstms = []
        for li in range(enc.num_layers):
            ni = in_sizes[li]
            lay = nn.LSTM(ni, h, batch_first=True, bidirectional=True)
            for suf, cell in (("", enc.f_cells[li]), ("_reverse", enc.b_cells[li])):
                setattr(lay, f"weight_ih_l0{suf}", nn.Parameter(cell.weight_ih.detach().clone()))
                setattr(lay, f"weight_hh_l0{suf}", nn.Parameter(cell.weight_hh.detach().clone()))
                setattr(lay, f"bias_ih_l0{suf}", nn.Parameter(cell.bias_ih.detach().clone()))
                setattr(lay, f"bias_hh_l0{suf}", nn.Parameter(cell.bias_hh.detach().clone()))
            lay.flatten_parameters()
            lstms.append(lay)
        self.lstms = nn.ModuleList(lstms)

        self.arc_mlp_d, self.arc_mlp_h = m.arc_mlp_d, m.arc_mlp_h
        self.rel_mlp_d, self.rel_mlp_h = m.rel_mlp_d, m.rel_mlp_h
        self.arc_attn, self.rel_attn = m.arc_attn, m.rel_attn
        # The stock Biaffine squeezes dim 1 when n_out == 1, which torchscript ONNX
        # mis-exports ("squeeze on dimension 1 ... size 55" when batch/words collide).
        # Rebind the arc head to a plain callable that unsqueezes explicitly.
        self._arc_bias_x = m.arc_attn.bias_x
        self._arc_weight = m.arc_attn.weight

    def mix(self, hs):
        """ScalarMix over a tuple/list of [B,L,H] tensors (inference: no dropout)."""
        w = torch.softmax(self.scalar_mix.weights, -1)
        acc = None
        for wi, hi in zip(w, hs):
            acc = wi * hi if acc is None else acc + wi * hi
        return acc * self.scalar_mix.gamma

    # ------------------------------------------------------------------ #
    def bert_embed(self, subwords, mask=None, order=None):
        """[B,W,F] -> [B,W,100], ONNX-traceable version of supar's TransformerEmbedding.

        supar: flatten *valid* pieces into one [B, n] roberta pass, scatter the hidden
        states back into [B, W, F, D], mean-pool over F slots.
        Here we avoid ``masked_scatter_`` (not traceable) by:
          1. flattening every piece (including pads) into [B, W*F]
          2. computing a *compacted* index per valid piece via cumsum of the mask
          3. ``gather`` the roberta output back through an inverse index.
        """
        B, W, F = subwords.shape
        n = W * F
        flat = subwords.reshape(B, n)
        fmask = flat.ne(self.subword_pad)                     # [B, n] bool
        m = fmask.long()
        # compact position of each piece among the valid ones (0 = invalid)
        pos = torch.cumsum(m, -1) * m                          # [B, n]
        # gather valid pieces to the front: sorted index (stable by construction)
        # ``order`` is passed in (argsort of pads last); torchscript can't export argsort.
        ids = torch.gather(flat, 1, order)                     # [B, n]
        out = self.bert(ids[:, : self.max_len],
                        attention_mask=(1 - torch.gather(1 - m, 1, order))[:, : self.max_len].float())
        hs = out.hidden_states[-self.n_layers:] if out.hidden_states is not None else (out.last_hidden_state,)
        bert = self.proj(self.mix(hs))                         # [B, n, D]
        D = bert.size(-1)
        # scatter back: for each flat slot j, take bert[pos[j]-1] if valid else 0
        src = (pos - 1).clamp(min=0).unsqueeze(-1).expand(B, n, D)
        back = torch.gather(bert, 1, src)
        back = back * m.unsqueeze(-1).float()
        embed = back.reshape(B, W, F, D)
        lens = m.reshape(B, W, F).sum(-1).clamp(min=1).unsqueeze(-1)
        return embed.sum(2) / lens

    def encode(self, subword_ids, word_ids, order):
        bert = self.bert_embed(subword_ids, order=order)
        ext = word_ids.masked_fill(word_ids.ge(self.word_embed.num_embeddings), 1)
        word = self.word_embed(ext)
        x = torch.cat([word, bert], -1)                       # 100 + 100 = 200
        # No packing: the CLI always runs batch == 1 (one sentence), where padded and
        # packed execution coincide. Packing also breaks torchscript ONNX export.
        for lay in self.lstms:
            x, _ = lay(x)
        return x

    def arc_score(self, x, y):
        """Biaffine(x, y) with n_out == 1, returning [B, W, W] without a squeeze."""
        if self._arc_bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.arc_attn.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        w = self._arc_weight                       # [1, dx+1, dy(+1)]
        # s[b,i,j] = x[b,i,:] @ w[0] @ y[b,j,:]
        left = torch.matmul(x, w[0])               # [B, W, d+1]
        s = torch.matmul(left, y.transpose(-1, -2))  # [B, W, W]
        return s

    def forward(self, subword_ids, word_ids, order):
        x = self.encode(subword_ids, word_ids, order)
        mask = word_ids.ne(self.pad_index)
        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)
        s_arc = self.arc_score(arc_d, arc_h)
        s_arc = s_arc.masked_fill(~mask.unsqueeze(1), -1e4)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        return s_arc, s_rel


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/en-upos-large")
    ap.add_argument("--out", default="/workspace/models/en-upos-large/onnx")
    ap.add_argument("--skip-tagger", action="store_true")
    ap.add_argument("--skip-parser", action="store_true")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the torch-vs-ONNX sentence check (saves a few seconds)")
    ap.add_argument("--verify-only", action="store_true",
                    help="only run the torch-vs-ONNX check on an existing export")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace/models/hf-cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.makedirs(args.out, exist_ok=True)
    from esupar.supar import Parser
    from esupar.supar.utils import Dataset

    parser = Parser.load(os.path.join(args.model, "esupar.model"), safe_tensor=True)
    if args.verify_only:
        verify_sentences(args.model, args.out)
        return

    parser.transform.eval()
    parser.model.eval()

    if not args.skip_tagger:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        tok = AutoTokenizer.from_pretrained(args.model)
        tagger = AutoModelForTokenClassification.from_pretrained(args.model).eval()
        ids = tok("The clever young fox quietly crossed the dark river.", return_tensors="pt")
        path = os.path.join(args.out, "en_upos_tagger.onnx")
        torch.onnx.export(
            tagger, (ids["input_ids"], ids["attention_mask"]), path,
            input_names=["input_ids", "attention_mask"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                          "logits": {0: "batch", 1: "seq"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
        # torch writes en_upos_tagger.onnx.data next to it automatically
        tok.save_pretrained(os.path.join(args.out, "tokenizer"))
        import transformers
        tcfg = transformers.AutoConfig.from_pretrained(args.model)
        json.dump({"id2label": tagger.config.id2label,
                   "model": getattr(tcfg, "_name_or_path", args.model),
                   "hidden_size": tagger.config.hidden_size,
                   "num_hidden_layers": tagger.config.num_hidden_layers,
                   "task": "TokenClassification: packed UPOS+FEATS",
                   "note": "Multi-word labels contain '+'; the pipeline maps them onto several tokens."},
                  open(os.path.join(args.out, "tagger_labels.json"), "w"), indent=1)
        print("tagger ->", path, f"{os.path.getsize(path)/1e6:.1f} MB + external data")

    if args.skip_parser:
        return

    # ---- verification batch built like esupar's Dataset does ---------------
    sentences = [["The", "clever", "young", "fox", "quietly", "crossed", "the", "dark", "river", "."],
                 ["It", "slowly", "climbed", "the", "steep", "bank"]]
    # Build a batch of TWO sentences so the batch dim is not 1 (avoids a torchscript
    # "squeeze on dim 1" mis-export for the n_out=1 arc Biaffine).
    import esupar.supar.utils.data as _d
    ds = Dataset(parser.transform, sentences)
    ds.build(100000, 8)                 # huge bucket -> both sentences in one batch
    batch = next(iter(ds.loader))
    word_ids, _, subwords = batch[0], batch[1], batch[2]
    print("batch", word_ids.shape, subwords.shape)

    net = ParserGraph(parser).eval()
    with torch.no_grad():
        ref_arc, ref_rel = parser.model(word_ids, [subwords])
        order = torch.argsort(subwords.ne(net.subword_pad).reshape(subwords.size(0), -1).logical_not(),
                              dim=-1, stable=True)
        got_arc, got_rel = net(subwords, word_ids, order)
    print("arc maxdiff", float((ref_arc - got_arc).abs().max()),
          " rel maxdiff", float((ref_rel - got_rel).abs().max()))

    # the reference predict()
    pred = parser.predict(sentences)
    print("esupar heads:", [s.values[6] for s in pred.sentences])

    path = os.path.join(args.out, "en_supar_parser.onnx")
    torch.onnx.export(
        net, (subwords, word_ids, order), path,
        input_names=["subword_ids", "word_ids", "piece_order"],
        output_names=["arc_logits", "rel_logits"],
        dynamic_axes={"subword_ids": {0: "batch", 1: "words", 2: "pieces"},
                      "word_ids": {0: "batch", 1: "words"},
                      "piece_order": {0: "batch", 1: "flat"},
                      "arc_logits": {0: "batch", 1: "words", 2: "words"},
                      "rel_logits": {0: "batch", 1: "words", 2: "words"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    meta = {
        "source": "esupar.model (KoichiYasuoka/roberta-large-english-upos)",
        "inputs": ["subword_ids", "word_ids", "piece_order"],
        "outputs": ["arc_logits", "rel_logits"],
        "labels": list(parser.REL.vocab),
        "word_vocab_size": int(parser.model.word_embed.num_embeddings),
        "word_bos_index": 2, "word_unk_index": 1, "word_pad_index": 0,
        "subword_pad_index": 1,
        "bert_n_layers": int(getattr(parser.model.bert_embed, "n_layers", 4)),
        "bert_max_len": int(getattr(parser.model.bert_embed, "max_len", 512)),
        "bert_hidden_size": int(parser.model.bert_embed.bert.config.hidden_size),
        "notes": {
            "word_ids": "BOS(2) prepended, then word vocab ids (lowercased FORM); OOV -> 1 (unk)",
            "subwords": "roberta BPE pieces per word, NO <s>/</s>; pad id 1",
            "decode": "row 0 is root; run Eisner over arc_logits[:, 1:, :] (cols are heads incl. 0=root)",
            "piece_order": "argsort of (subword_ids == pad(1)) flattened to [B, W*F]; valid pieces first",
        },
    }
    json.dump(meta, open(os.path.join(args.out, "parser_labels.json"), "w"), indent=1)
    print("parser ->", path, f"{os.path.getsize(path)/1e6:.1f} MB (+ external data)")

    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"subword_ids": subwords.numpy(), "word_ids": word_ids.numpy(),
                        "piece_order": order.numpy()})
    print("ort arc maxdiff", float(np.abs(o[0] - ref_arc.numpy()).max()))
    heads = o[0][:, 1:, :].argmax(-1)
    print("ort argmax heads (before Eisner):", heads.tolist())

    if not args.skip_verify:
        verify_sentences(args.model, args.out,
                         tagger if not args.skip_tagger else None,
                         tok if not args.skip_tagger else None)


# --------------------------------------------------------------------------- #
# torch-vs-ONNX sentence check (labels + heads, not only logits)
# --------------------------------------------------------------------------- #
VERIFY_SENTS = [
    "The clever young fox quietly crossed the dark river.",
    "Having finished dinner, they started a movie.",
    "The man who lives next door is a famous writer.",
    "She decided to leave the party early.",
]


def verify_sentences(model_dir, out_dir, tagger=None, tokenizer=None):
    import onnxruntime as ort
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from esupar.supar import Parser
    from esupar.supar.utils.field import SubwordField

    tok = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_dir)
    tg = (tagger if tagger is not None
          else AutoModelForTokenClassification.from_pretrained(model_dir).eval())
    parser = Parser.load(os.path.join(model_dir, "esupar.model"), safe_tensor=True)
    parser.transform.eval(); parser.model.eval()
    fields = parser.transform.FORM  # esupar CoNLL: FORM is the field tuple
    sub = next(f for f in fields if isinstance(f, SubwordField))

    t_sess = ort.InferenceSession(os.path.join(out_dir, "en_upos_tagger.onnx"), providers=["CPUExecutionProvider"])
    p_sess = ort.InferenceSession(os.path.join(out_dir, "en_supar_parser.onnx"), providers=["CPUExecutionProvider"])
    net = ParserGraph(parser).eval()

    report = {"sentences": [], "tagger": {}, "parser": {}}
    max_tag_dlogit = 0.0
    max_arc = max_rel = 0.0
    tag_argmax_equal = 0
    tag_total = 0
    head_equal = head_total = 0
    for sent in VERIFY_SENTS:
        enc = tok(sent, return_tensors="pt")
        with torch.no_grad():
            logits = tg(**enc).logits
        o = t_sess.run(None, {"input_ids": enc["input_ids"].numpy(),
                              "attention_mask": enc["attention_mask"].numpy()})[0]
        max_tag_dlogit = max(max_tag_dlogit, float(np.abs(o - logits.numpy()).max()))
        a, b = o.argmax(-1)[0], logits.argmax(-1)[0].numpy()
        tag_argmax_equal += int((a == b).sum()); tag_total += int(a.shape[0])

        words = [w for w in re.split(r"\s+", sent) if w]
        sub_tokens = [sub.transform([w]) for w in words]
        F = max(len(x) for x in sub_tokens)
        rows = len(words) + 2
        # word ids via the Field's vocab (lowercased, OOV -> 1)
        from esupar.supar.utils.field import Field
        wf = next(f for f in fields if isinstance(f, Field) and not isinstance(f, SubwordField))
        wids = [2] + [wf.vocab[w.lower()] if w.lower() in wf.vocab else 1 for w in words] + [0]
        sub_ids = np.ones((rows, F), dtype=np.int64); sub_ids[0, 0] = 0
        for i, st in enumerate(sub_tokens):
            for k, t in enumerate(st):
                t = t.reshape(-1)
                sub_ids[i + 1, k] = int(t[0].item()) if t.numel() else 1
        order = np.argsort((sub_ids == 1).reshape(-1), kind="stable").astype(np.int64)
        with torch.no_grad():
            wids_t = torch.tensor([wids])
            sub_t = torch.tensor(sub_ids)[None]
            ord_t = torch.tensor(order)[None]
            ra, rr = net(sub_t, wids_t, ord_t)
        oa, orr = p_sess.run(None, {"subword_ids": sub_ids[None], "word_ids": np.array([wids], dtype=np.int64),
                                    "piece_order": order[None]})
        max_arc = max(max_arc, float(np.abs(oa - ra.numpy()).max()))
        max_rel = max(max_rel, float(np.abs(orr - rr.numpy()).max()))
        ha = oa[0][1:, :].argmax(-1); hr = ra.numpy()[0][1:, :].argmax(-1)
        head_equal += int((ha == hr).sum()); head_total += len(ha)
        report["sentences"].append({
            "text": sent,
            "words": words,
            "torch_argmax_subwords": [int(x) for x in a],
            "ort_argmax_subwords": [int(x) for x in b],
            "ort_heads": [int(x) for x in ha],
        })
    report["tagger"] = {"max_abs_logit_diff": max_tag_dlogit,
                        "subword_argmax_agreement": tag_argmax_equal / max(tag_total, 1)}
    report["parser"] = {"max_arc_diff": max_arc, "max_rel_diff": max_rel,
                        "argmax_head_agreement": head_equal / max(head_total, 1)}
    json.dump(report, open(os.path.join(out_dir, "verify_vs_torch.json"), "w"), indent=1)
    print("VERIFY", json.dumps({"tagger": report["tagger"], "parser": report["parser"]}))
    ok = (report["tagger"]["subword_argmax_agreement"] == 1.0
          and report["parser"]["argmax_head_agreement"] == 1.0)
    print("VERIFY OK" if ok else "VERIFY MISMATCH (see verify_vs_torch.json)")


if __name__ == "__main__":
    main()
