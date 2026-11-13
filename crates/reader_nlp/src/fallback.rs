//! Fallback analysis: `jieba-rs` 0.11 (Chinese segmentation + POS) and
//! `english-pos-tagger` 0.2 (Penn tags). No dependency parse, so the marks are
//! deliberately rough (and conservative: a missed mark is fine, a wrong one is
//! not).
//!
//! The Chinese side reuses `jieba`'s own dictionary tags and maps them onto the
//! LTP tag names the neural path emits, so `zh_coarse` works unchanged.

use std::sync::OnceLock;

use jieba_rs::Jieba;

use crate::derive::{derive_english_coarse, ZH_CONNECTIVES};
use crate::neural::zh_coarse;
use crate::text::ScriptKind;
use crate::types::TokenSpan;

fn jieba() -> &'static Jieba {
    static JIEBA: OnceLock<Jieba> = OnceLock::new();
    JIEBA.get_or_init(Jieba::new)
}

fn en_tagger() -> &'static english_pos_tagger::Tagger {
    static TAGGER: OnceLock<english_pos_tagger::Tagger> = OnceLock::new();
    TAGGER.get_or_init(english_pos_tagger::Tagger::new)
}

/// Chinese raw tags that identify a connective even when jieba does not tag
/// the word `c` (it tags 因为/所以/但是 as `c`, but 同时/接着 as `d`/`v`).
fn zh_tag_for(word: &str, jieba_tag: &str) -> String {
    if ZH_CONNECTIVES.contains(&word) && jieba_tag != "p" {
        return "c".to_string();
    }
    let coarse = match jieba_tag {
        // jieba's flag set: n, nr, ns, nt, nz, v, vn, a, ad, an, d, r, p, c,
        // u, m, q, x, eng, f, s, t, z, …
        "n" | "nr" | "ns" | "nt" | "nz" | "nl" | "ng" | "j" | "s" | "f" | "t" => "n",
        "v" | "vn" | "vd" | "vg" => "v",
        "a" | "ad" | "an" | "ag" | "b" | "z" => "a",
        "d" | "df" | "dg" => "d",
        "r" | "rr" | "rz" | "ry" | "rg" => "r",
        "p" => "p",
        "c" | "cc" => "c",
        // jieba tags Chinese prepositions that head a clause (`因为`, `为了`)
        // as `p`; keep them connectives when they are in the table.
        _ if ZH_CONNECTIVES.contains(&word) => "c",
        other => other,
    };
    coarse.to_string()
}

/// Tokenize one Han run with jieba tags.
pub fn zh_tokens(run: &str) -> Vec<(String, usize, usize, String)> {
    let mut out = Vec::new();
    let mut pad = 0usize;
    // jieba's `tag` keeps spaces/its own normalization out of the offsets only
    // for ASCII; Han runs are contiguous, but guard anyway by re-anchoring
    // every word inside the run.
    for t in jieba().tag(run, false) {
        if t.word.trim().is_empty() {
            continue;
        }
        let Some(off) = run[pad.min(run.len())..].find(t.word) else {
            continue;
        };
        let s = pad + off;
        let e = s + t.word.len();
        pad = e;
        out.push((
            t.word.to_string(),
            s,
            e,
            zh_tag_for(t.word, t.tag),
        ));
    }
    out
}

/// Tokenize one Latin batch with the Brill-style Penn tagger.
pub fn en_tokens(batch: &str) -> Vec<(String, usize, usize, String)> {
    let tagger = en_tagger();
    let mut out = Vec::new();
    let mut pad = 0usize;
    for tok in tagger.tag_sentence(batch) {
        let value = tok.value.as_str();
        if value.trim().is_empty() {
            continue;
        }
        // The tagger splits contractions (`don't` -> `do` + `n't`); tokens may
        // not be separated by spaces there, so search from the running cursor,
        // falling back to the raw location of `n't` inside `don't`.
        let Some(off) = batch[pad.min(batch.len())..].find(value) else {
            continue;
        };
        let s = pad + off;
        let e = s + value.len();
        pad = e;
        let tag = tok.pos.unwrap_or_default();
        out.push((value.to_string(), s, e, tag));
    }
    out
}

/// Build token spans for one script run (offsets inside `run`), adding the
/// run's start offset to every range.
pub fn tokens_for_run(kind: ScriptKind, run: &str, base: usize, mixed: bool) -> Vec<TokenSpan> {
    let mut out = Vec::new();
    match kind {
        ScriptKind::Han => {
            for (_, s, e, tag) in zh_tokens(run) {
                out.push(TokenSpan {
                    range: base + s..base + e,
                    upos: zh_coarse(&tag).to_string(),
                    is_core: false,
                    is_connective: tag == "c",
                    is_nonfinite: false,
                });
            }
        }
        ScriptKind::Latin => {
            let words = en_tokens(run);
            let flat: Vec<(&str, &str)> = words
                .iter()
                .map(|(w, _, _, t)| (w.as_str(), t.as_str()))
                .collect();
            let flags = derive_english_coarse(&flat);
            for ((_, s, e, tag), (core, conn, nonfinite)) in words.into_iter().zip(flags) {
                out.push(TokenSpan {
                    range: base + s..base + e,
                    upos: en_coarse(&tag).to_string(),
                    // In a mixed sentence an English fragment's adjective is
                    // never the core predicate (same rule as the neural path).
                    is_core: core && (!mixed || en_coarse(&tag) == "VERB"),
                    is_connective: conn,
                    is_nonfinite: nonfinite,
                });
            }
        }
    }
    out
}

/// Penn tags / UD UPOS (case-insensitive) -> coarse class.
pub fn en_coarse(tag: &str) -> &'static str {
    match tag.to_ascii_uppercase().as_str() {
        "NN" | "NNS" | "NNP" | "NNPS" | "NOUN" | "PROPN" => "NOUN",
        "PRP" | "PRP$" | "WP" | "WP$" | "PRON" => "PRON",
        "VB" | "VBD" | "VBG" | "VBN" | "VBP" | "VBZ" | "MD" | "VERB" | "AUX" => "VERB",
        "JJ" | "JJR" | "JJS" | "ADJ" => "ADJ",
        "RB" | "RBR" | "RBS" | "WRB" | "ADV" => "ADV",
        _ => "OTHER",
    }
}
