#!/usr/bin/env python
"""Benchmark the four English variants against spaCy `en_core_web_trf`.

Design constraints (16 GB box):

  * **one subprocess per variant** - `cargo run --release ... --en-variant X`
    loads LTP + exactly one English graph pair; peak RSS is read from the
    process' own `metrics.json` (`VmHWM`), so variants never stack in memory.
  * the Rust CLI writes `tokens.json` per variant into
    `/workspace/spike-out/bench/<variant>/`; this script only parses them.
  * spaCy runs once, in this process, over the 60 `en_compare` sentences.

Reported per variant:
  files (tagger / parser / int8 external-data), load time, per-sentence latency
  (p50/avg/max after warmup, from the CLI's group latencies), peak RSS, coarse
  UPOS agreement, root agreement, `is_core_verb` agreement, and non-finite
  detection quality against a spaCy-derived gold standard.

Non-finite gold (documented in the report):
  * spaCy morphology `VerbForm=Inf|Ger|Part`;
  * `TO` + `VB`/`VBG`/`VBN` when morphology is missing;
  * verbal uses of VBG/VBN that spaCy attaches as `xcomp|advcl|acl|ccomp|csubj|
    conj|relcl|ROOT`;
  * a VBG/VBN tagged ADJ and attached as `amod` is **not** gold (adjectival
    participle: "the sleeping village").

Usage:
  timeout 1500 /workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/bench_en_variants.py
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = "/workspace/quiet-md"
MANIFEST = os.path.join(ROOT, "spikes/nlp_models/Cargo.toml")
FIXTURES = os.path.join(ROOT, "spikes/nlp_models/fixtures/sentences.json")
EN_ONLY = os.path.join(ROOT, "spikes/nlp_models/fixtures/en_only.json")
OUT = "/workspace/spike-out"
VARIANTS = ["large-fp32", "large-int8", "base-fp32", "base-int8"]
NONFINITE_DEPS = {"xcomp", "advcl", "acl", "ccomp", "csubj", "conj", "relcl", "ROOT", "parataxis"}


def norm(tok: str) -> str:
    return tok.strip().lower()


def coarse(upos: str):
    return {
        "NOUN": "NOUN", "PROPN": "NOUN", "VERB": "VERB", "AUX": "VERB",
        "ADJ": "ADJ", "ADV": "ADV",
    }.get(upos)


def run_variant(variant: str, threads: int) -> dict:
    out = os.path.join(OUT, "bench", variant)
    os.makedirs(out, exist_ok=True)
    cmd = [
        "timeout", "900",
        "cargo", "run", "--release", "--manifest-path", MANIFEST, "--",
        "--models-dir", "/workspace/models",
        "--out-dir", out,
        "--input", EN_ONLY,
        "--threads", str(threads),
        "--en-variant", variant,
    ]
    t0 = time.time()
    print(f"[bench] {variant} …", file=sys.stderr, flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        raise SystemExit(f"{variant} failed ({p.returncode}):\n{p.stderr[-3000:]}")
    m = json.load(open(os.path.join(out, "metrics.json")))
    tok = json.load(open(os.path.join(out, "tokens.json")))
    m["wall_s_including_build"] = dt
    m["tokens"] = tok
    return m


def spacy_pass(sentences):
    import spacy

    nlp = spacy.load("en_core_web_trf")
    docs = []
    for s in sentences:
        doc = nlp(s)
        docs.append(doc)
    return docs


def align(ours, sp_toks):
    """Token alignment (same heuristic as compare_spacy.py)."""
    i = j = 0
    aligned = []
    while i < len(ours) and j < len(sp_toks):
        a, b = ours[i], sp_toks[j]
        if norm(a["text"]) == norm(b.text):
            aligned.append((a, b))
            i += 1
            j += 1
            continue
        if j + 1 < len(sp_toks) and norm(a["text"]) == norm(sp_toks[j + 1].text):
            j += 1
            continue
        if i + 1 < len(ours) and norm(ours[i + 1]["text"]) == norm(b.text):
            i += 1
            continue
        j += 1
    return aligned


def spacy_nonfinite(b) -> bool:
    """spaCy-derived gold for "verbally used non-finite verb".

    Deliberately stricter than "spaCy says VerbForm=Inf/Ger/Part":
      * a form without an explicit auxiliary (`will walk` -> walk:VB after
        `will:AUX`, `did not enjoy` -> enjoy:VB after `not:PART`) is finite in
        surface terms, so it is not gold;
      * participles that continue an auxiliary chain of the main verb
        (`have been working`, `is replaced`) are finite constructions, not
        clause-level non-finite verbs;
      * a participle tagged ADJ and attached as `amod` is adjectival.
    The task's rule (README §4) marks the verbal non-finite uses; the auxiliary
    chain is a known, documented limitation.
    """
    if b.pos_ == "ADJ" and b.dep_ == "amod":
        return False
    def preceded_by_to() -> bool:
        prev = b.doc[b.i - 1] if b.i > 0 else None
        return prev is not None and prev.lower_ == "to" and prev.tag_ == "TO"

    vf = b.morph.get("VerbForm")
    if vf and vf[0] == "Inf":
        return preceded_by_to()
    if b.tag_ == "VB":
        return preceded_by_to()
    if (vf and vf[0] in ("Ger", "Part")) or b.tag_ in ("VBG", "VBN"):
        if b.dep_ in ("aux", "auxpass"):
            return False
        # reduced passive/adverbial clause with a fronted participle
        # ("Broken by the storm, the old bridge …") - structurally the same as
        # the gerund-participle clause the rule is written for
        if b.dep_ in ("advcl", "acl", "relcl") and b.tag_ in ("VBG", "VBN"):
            return True
        # participle hanging off a verb chain: "have been working"
        cur = b
        while cur.head is not cur and cur.head.pos_ in ("AUX", "VERB"):
            if cur.head.lemma_ == "have" or cur.dep_ in ("aux", "auxpass", "ROOT"):
                return False
            if cur.head.pos_ == "VERB" and cur.dep_ in ("xcomp", "advcl", "acl", "relcl", "ccomp"):
                cur = cur.head
                continue
            break
        return True
    return False


def evaluate(m, docs, sentences):
    per_variant = {}
    upos_h = upos_t = root_h = root_t = core_h = core_t = 0
    nf_tp = nf_fp = nf_fn = 0
    nf_gold_n = 0
    nf_examples_ok = []
    nf_examples_bad = []
    confusion = {}
    for g, doc in zip(m["tokens"]["en_compare"], docs):
        ours = [t for s in g["sentences"] for t in s["tokens"]]
        sp_toks = [t for t in doc if not t.is_space]
        aligned = align(ours, sp_toks)

        # ---- coarse UPOS ----
        for a, b in aligned:
            ca, cb = coarse(a["upos_or_ltp_tag"].split("|")[0]), coarse(b.pos_)
            if ca and cb:
                upos_t += 1
                upos_h += ca == cb
                if ca != cb:
                    k = f"{ca}->{cb}"
                    confusion[k] = confusion.get(k, 0) + 1

        # ---- root ----
        ours_root = next((i for i, (a, _) in enumerate(aligned) if a["head"] == 0), None)
        sp_root = next((i for i, (_, b) in enumerate(aligned) if b.head.i == b.i and b.dep_ == "ROOT"), None)
        if ours_root is not None and sp_root is not None:
            root_t += 1
            root_h += ours_root == sp_root

        # ---- is_core_verb vs spaCy predicate centre ----
        ours_core = {i for i, (a, _) in enumerate(aligned) if a["is_core_verb"]}
        sp_core = set()
        for i, (_, b) in enumerate(aligned):
            if b.head.i == b.i and b.dep_ == "ROOT":
                sp_core.add(i)
                if b.pos_ in ("VERB", "AUX"):
                    for k, (_, c) in enumerate(aligned):
                        if c.head.i == b.i and c.dep_ in ("acomp", "attr", "oprd"):
                            sp_core.add(k)
        core_t += 1
        core_h += int(bool(ours_core & sp_core) and len(ours_core) <= len(sp_core) + 1)

        # ---- non-finite ----
        for a, b in aligned:
            gold = spacy_nonfinite(b)
            pred = bool(a["is_nonfinite"])
            if gold:
                nf_gold_n += 1
            if gold and pred:
                nf_tp += 1
                if len(nf_examples_ok) < 4:
                    nf_examples_ok.append({"word": a["text"], "sentence": g["text"],
                                           "ours": a["upos_or_ltp_tag"], "spacy": b.tag_})
            elif pred and not gold:
                nf_fp += 1
                if len(nf_examples_bad) < 4:
                    nf_examples_bad.append({"word": a["text"], "sentence": g["text"],
                                            "ours": a["upos_or_ltp_tag"], "spacy_tag": b.tag_,
                                            "spacy_upos": b.pos_, "spacy_dep": b.dep_,
                                            "error": "false positive"})
            elif gold and not pred:
                nf_fn += 1
                if len(nf_examples_bad) < 8:
                    nf_examples_bad.append({"word": a["text"], "sentence": g["text"],
                                            "ours": a["upos_or_ltp_tag"], "spacy_tag": b.tag_,
                                            "spacy_upos": b.pos_, "spacy_dep": b.dep_,
                                            "error": "missed"})
    lat = m["latency_ms_per_input"]
    ev = m["en_variant"]
    per_variant = {
        "variant": m["en_variant"]["name"],
        "files": {
            "tagger": ev["tagger_bytes"],
            "parser": ev["parser_bytes"],
            "total_mib": round(ev["total_mib"], 1),
        },
        "load_ms": ev["load_ms"],
        "latency_ms": {
            "note": "one en_compare sentence per annotate() call, after warmup",
            "p50": round(m["en_latency_ms"]["p50"], 1),
            "avg": round(m["en_latency_ms"]["avg"], 1),
            "max": round(m["en_latency_ms"]["max"], 1),
            "n": m["en_latency_ms"]["samples"],
        },
        "peak_rss_mib": round(m["peak_rss_mib"] or 0, 0),
        "infer_ms_per_run": m["inference"],
    }
    n = len(sentences)
    metrics = {
        "upos_coarse": round(upos_h / upos_t, 4) if upos_t else None,
        "root": round(root_h / root_t, 4) if root_t else None,
        "core": round(core_h / core_t, 4) if core_t else None,
        "nonfinite": {
            "gold_tokens": nf_gold_n,
            "tp": nf_tp, "fp": nf_fp, "fn": nf_fn,
            "precision": round(nf_tp / (nf_tp + nf_fp), 4) if nf_tp + nf_fp else None,
            "recall": round(nf_tp / (nf_tp + nf_fn), 4) if nf_tp + nf_fn else None,
            "agreement": round((nf_tp) / (nf_tp + nf_fp + nf_fn), 4)
            if nf_tp + nf_fp + nf_fn else None,
            "ok_examples": nf_examples_ok,
            "bad_examples": nf_examples_bad,
        },
        "upos_confusion": dict(sorted(confusion.items(), key=lambda kv: -kv[1])[:10]),
        "sentences": n,
    }
    per_variant["metrics"] = metrics
    return per_variant


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-json", default=os.path.join(OUT, "en_compare_bench.json"))
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]

    fixtures = json.load(open(FIXTURES))
    sentences = fixtures["en_compare"]
    print(f"[bench] {len(sentences)} en_compare sentences; spaCy pass …", file=sys.stderr, flush=True)
    docs = spacy_pass(sentences)

    results = {}
    for v in variants:
        m = run_variant(v, args.threads)
        # english-only latency: the CLI records per-group latency; en_compare
        # groups are the last 60 samples. Re-derive from tokens' sentence texts.
        results[v] = evaluate(m, docs, sentences)
        results[v]["wall_s_including_build"] = round(m["wall_s_including_build"], 1)
        print(f"[bench] {v}: {json.dumps(results[v]['metrics']['nonfinite'])[:120]}", file=sys.stderr)
    json.dump(results, open(args.out_json, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {args.out_json}", file=sys.stderr)
    for v, r in results.items():
        print(v, json.dumps({"files": r["files"], "load": r["load_ms"], "lat": r["latency_ms"],
                             "rss": r["peak_rss_mib"], "m": {k: r["metrics"][k] for k in
                             ("upos_coarse", "root", "core")},
                             "nf": {k: r["metrics"]["nonfinite"][k] for k in
                                    ("precision", "recall", "agreement", "tp", "fp", "fn")}},
                            ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
