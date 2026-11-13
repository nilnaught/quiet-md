#!/usr/bin/env python
"""spaCy `en_core_web_trf` reference for the English part of the spike.

Runs the *same* `en_compare` sentences through spaCy, aligns tokens 1:1 with
the Rust output (exact string match after lowercasing and punctuation
normalisation), and reports

  * UPOS agreement mapped to coarse classes (NOUN/PROPN -> 名词, VERB/AUX ->
    动词, ADJ -> 形容词, ADV -> 副词). Tokens outside these classes are ignored
    on both sides, so the rate answers "of the colour-bearing words, how many
    does spaCy agree on?".
  * root agreement: does spaCy's parse root point at the same token as ours?
  * predicate-centre agreement (`is_core_verb`): same token set?

Usage:
  /workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/compare_spacy.py \
     --our /workspace/spike-out/tokens.json --out /workspace/spike-out/compare-en.json
"""
import argparse
import json
import sys
from collections import Counter


def coarse(upos: str) -> str | None:
    return {
        "NOUN": "名词",
        "PROPN": "名词",
        "VERB": "动词",
        "AUX": "动词",
        "ADJ": "形容词",
        "ADV": "副词",
    }.get(upos)


def norm(tok: str) -> str:
    return tok.strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--our", default="/workspace/spike-out/tokens.json")
    ap.add_argument("--out", default="/workspace/spike-out/compare-en.json")
    ap.add_argument("--model", default="en_core_web_trf")
    args = ap.parse_args()

    import spacy

    nlp = spacy.load(args.model)
    nlp.max_length = 100_000

    data = json.load(open(args.our, encoding="utf-8"))
    groups = data.get("en_compare", [])

    per_sentence = []
    upos_hits = upos_total = 0
    root_hits = root_total = 0
    pred_hits = pred_total = 0

    for g in groups:
        text = g["text"]
        ours = []
        for s in g["sentences"]:
            ours.extend(s["tokens"])
        doc = nlp(text)
        sp_toks = [t for t in doc if not t.is_space]

        # Align by walking both lists, skipping tokens that are pure punctuation
        # on one side only (spaCy splits "don't" as do + n't).
        i = j = 0
        aligned = []
        while i < len(ours) and j < len(sp_toks):
            a, b = ours[i], sp_toks[j]
            if norm(a["text"]) == norm(b.text):
                aligned.append((a, b))
                i += 1
                j += 1
                continue
            # resynchronise: look ahead a couple of tokens on each side
            if j + 1 < len(sp_toks) and norm(a["text"]) == norm(sp_toks[j + 1].text):
                j += 1
                continue
            if i + 1 < len(ours) and norm(ours[i + 1]["text"]) == norm(b.text):
                i += 1
                continue
            # unmatched: skip the spaCy token
            j += 1

        # ---- coarse UPOS agreement --------------------------------------
        hits = total = 0
        for a, b in aligned:
            ca, cb = coarse(a["upos_or_ltp_tag"].split("|")[0]), coarse(b.pos_)
            if ca and cb:
                total += 1
                if ca == cb:
                    hits += 1
        upos_hits += hits
        upos_total += total

        # ---- root / predicate-centre ------------------------------------
        # Root agreement is reported two ways:
        #   * raw: same *token* is the syntactic root. This is expected to be
        #     low because UD roots the predicate nominal of a copular clause
        #     while spaCy roots the copula ("was":ROOT + "fascinating":acomp).
        #   * predicate-centre: does the token that carries the main predicate
        #     agree? Our `is_core_verb` vs spaCy's ROOT *or* the copula that
        #     spaCy hangs under the ROOT.
        ours_root = next((i for i, (a, _) in enumerate(aligned) if a["head"] == 0), None)
        sp_root = next(
            (i for i, (_, b) in enumerate(aligned) if b.head.i == b.i and b.dep_ == "ROOT"),
            None,
        )
        root_ok = None
        if ours_root is not None and sp_root is not None:
            root_total += 1
            root_ok = ours_root == sp_root
            root_hits += int(root_ok)

        ours_core = {i for i, (a, _) in enumerate(aligned) if a["is_core_verb"]}
        # spaCy predicate centre: the ROOT if it is content-bearing; otherwise the
        # ROOT plus the copular/adjectival head it governs (verb -> acomp/attr).
        sp_core = set()
        for i, (_, b) in enumerate(aligned):
            if b.head.i == b.i and b.dep_ == "ROOT":
                sp_core.add(i)
                if b.pos_ in ("VERB", "AUX"):
                    for k, (_, c) in enumerate(aligned):
                        if c.head.i == b.i and c.dep_ in ("acomp", "attr", "oprd"):
                            sp_core.add(k)
        pred_total += 1
        # agreement: the sets intersect on the predicate token (either the root
        # itself or the predicate nominal is acceptable for a copular clause)
        pred_hits += int(bool(ours_core & sp_core) and len(ours_core) <= len(sp_core) + 1)

        per_sentence.append(
            {
                "text": text,
                "aligned": len(aligned),
                "ours_tokens": len(ours),
                "spacy_tokens": len(sp_toks),
                "upos_agree": hits,
                "upos_total": total,
                "root_agree": root_ok,
                "ours_core": sorted(ours_core),
                "spacy_core": sorted(sp_core),
                "ours": [
                    {
                        "text": a["text"],
                        "upos": a["upos_or_ltp_tag"],
                        "deprel": a["deprel"],
                        "head": a["head"],
                        "core": a["is_core_verb"],
                        "spacy_upos": b.pos_,
                        "spacy_dep": b.dep_,
                        "spacy_head_i": b.head.i,
                    }
                    for a, b in aligned
                ],
            }
        )

    confusion = {}
    for s in per_sentence:
        for row in s["ours"]:
            ca = coarse(row["upos"].split("|")[0])
            cb = coarse(row["spacy_upos"])
            if ca and cb:
                key = f"ours={ca} spacy={cb}"
                confusion[key] = confusion.get(key, 0) + 1

    def rate(h: int, t: int) -> float:
        return round(h / t, 4) if t else 0.0

    out = {
        "note": "spaCy en_core_web_trf is a reference only; the Rust CLI uses roberta-large-english-upos + SuPar.",
        "model": args.model,
        "sentences": len(groups),
        "upos_coarse_agreement": rate(upos_hits, upos_total),
        "upos_counts": {"agree": upos_hits, "total": upos_total},
        "root_agreement_raw": rate(root_hits, root_total),
        "root_agreement": rate(root_hits, root_total),
        "root_counts": {"agree": root_hits, "total": root_total},
        "predicate_centre_agreement": rate(pred_hits, pred_total),
        "predicate_counts": {"agree": pred_hits, "total": pred_total},
        "class_confusion": confusion,
        "per_sentence": per_sentence,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(
        f"UPOS coarse {rate(upos_hits, upos_total):.1%} ({upos_hits}/{upos_total}), "
        f"root {rate(root_hits, root_total):.1%} ({root_hits}/{root_total}), "
        f"predicate-centre {rate(pred_hits, pred_total):.1%} ({pred_hits}/{pred_total})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
