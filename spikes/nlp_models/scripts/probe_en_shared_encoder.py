#!/usr/bin/env python
"""Probe: does the SuPar parser's `bert_embed.bert` reuse the tagger's RoBERTa?

The plan sketched in the task was "if the two encoders are (almost) the same,
export one shared RoBERTa and feed both heads". This script decides that
empirically for one model dir (`large` or `base`):

  1. load `esupar.model` (pickled SuPar parser, safetensors) and the tagger
     snapshot (`pytorch_model.bin` / safetensors in the same dir);
  2. for every parameter that exists in both (embeddings, per-layer attention
     + FFN, layer norms), report max|Δ| / relative Δ;
  3. report structure facts that matter for sharing: how many encoder layers
     the parser keeps (`n_layers`), whether it uses ScalarMix, a projection,
     and a BiLSTM on top.

Output: JSON on stdout (also written with `--out`), plus a one-line verdict.

Usage:
  HF_HOME=/workspace/models/hf-cache timeout 600 /workspace/.venv-nlp/bin/python \
      scripts/probe_en_shared_encoder.py --model /workspace/models/en-upos-large \
      --out /workspace/spike-out/en_encoder_share_large.json
"""
import argparse
import json
import os
import sys

import torch


def tagger_state(model_dir: str) -> dict:
    from transformers import AutoModelForTokenClassification

    m = AutoModelForTokenClassification.from_pretrained(model_dir)
    sd = m.state_dict()
    return {k[len("roberta."):]: v for k, v in sd.items() if k.startswith("roberta.")}


def parser_encoder(model_dir: str):
    from esupar.supar import Parser

    p = Parser.load(os.path.join(model_dir, "esupar.model"), safe_tensor=True)
    p.model.eval()
    emb = p.model.bert_embed
    return p, emb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/en-upos-large")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace/models/hf-cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    parser, emb = parser_encoder(args.model)
    sd = tagger_state(args.model)
    enc = emb.bert
    cfg = enc.config
    n_layers_parser = cfg.num_hidden_layers
    uses = getattr(emb, "n_layers", None)
    print(
        f"parser encoder: {n_layers_parser} layers, hidden {cfg.hidden_size}; "
        f"ScalarMix over last {uses} layers; projection "
        f"{'yes' if getattr(emb, 'projection', None) is not None else 'no'}; "
        f"SuPar encoder {type(parser.model.encoder).__name__}",
        file=sys.stderr,
    )

    # Compare the *first* layers + embeddings: if even layer 0 differs, the two
    # RoBERTas were fine-tuned separately and a shared graph is not free.
    rows = []
    worst = 0.0
    compared = 0
    for name, p in enc.named_parameters():
        key = name
        t = sd.get(key)
        if t is None:
            continue
        if t.shape != p.shape:
            rows.append({"param": key, "shape_mismatch": [list(t.shape), list(p.shape)]})
            continue
        d = (t.float() - p.detach().float()).abs()
        rel = (d / t.float().abs().clamp(min=1e-6)).median().item()
        md = d.max().item()
        compared += 1
        worst = max(worst, md)
        rows.append({"param": key, "max_abs_diff": md, "median_rel_diff": rel,
                     "mean_abs": t.float().abs().mean().item()})
    print(f"compared {compared} parameters, worst max|Δ| {worst:.3e}", file=sys.stderr)

    # aggregate per component
    def agg(prefixes):
        sel = [r for r in rows if "max_abs_diff" in r and r["param"].startswith(prefixes)]
        if not sel:
            return None
        return {
            "params": len(sel),
            "max_abs_diff": max(r["max_abs_diff"] for r in sel),
            "median_rel_diff": sorted(r["median_rel_diff"] for r in sel)[len(sel) // 2],
        }

    groups = {
        "embeddings": agg(("embeddings.",)),
        "layer0": agg(("encoder.layer.0.",)),
        "layer1": agg(("encoder.layer.1.",)),
        "last_layer": agg((f"encoder.layer.{n_layers_parser - 1}.",)),
    }
    out = {
        "model": args.model,
        "parser_encoder_layers": n_layers_parser,
        "parser_scalar_mix_layers": uses,
        "parser_has_projection": getattr(emb, "projection", None) is not None,
        "parser_encoder_top": type(parser.model.encoder).__name__,
        "params_compared": compared,
        "worst_max_abs_diff": worst,
        "groups": groups,
        "identical_enough": worst < 1e-6,
    }
    verdict = (
        "IDENTICAL -> a shared encoder graph is conceivable"
        if out["identical_enough"]
        else "DIFFERENT -> the parser carries its own fine-tuned encoder; sharing is not cheap"
    )
    out["verdict"] = verdict
    text = json.dumps(out, indent=2)
    if args.out:
        open(args.out, "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
