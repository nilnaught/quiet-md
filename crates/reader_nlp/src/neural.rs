//! Neural pipeline: LTP/small (Chinese) and roberta-UPOS + SuPar (English) via
//! ONNX Runtime, assembling one token list with byte offsets into the original
//! input.
//!
//! Derived flags are computed per *clause* (the span of one model call, i.e.
//! one Han run / one joined Latin batch), because dependency heads are only
//! comparable inside one parse.

use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::derive::{derive_chinese_ctx, derive_english_ctx};
use crate::english::English;
use crate::ltp::Ltp;
use crate::text::{self, LatinBatch, ScriptKind, Span};
use crate::types::TokenSpan;

/// The loaded ONNX models. Analysis holds the engine lock for the duration of
/// a call (sessions are not Send-friendly to bounce in and out of the slot).
pub struct Neural {
    pub(crate) ltp: Option<Ltp>,
    pub(crate) en: Option<English>,
}

impl Neural {
    /// Everything that exists under `models_dir`; `None` when nothing does.
    pub fn load(models_dir: &std::path::Path, variant: crate::EnVariant, threads: usize) -> Result<Option<Self>> {
        let ltp = Ltp::load(models_dir, threads).ok();
        let en = English::load_variant(models_dir, variant.into(), threads).ok();
        if ltp.is_none() && en.is_none() {
            return Ok(None);
        }
        Ok(Some(Self { ltp, en }))
    }

    /// Analyse one paragraph. Returns the per-sentence token spans, using byte
    /// offsets into `base` (the paragraph's start in the caller's text).
    pub fn analyze_paragraph(
        &mut self,
        text: &str,
        base: usize,
    ) -> Vec<(std::ops::Range<usize>, Vec<TokenSpan>)> {
        let mut out = Vec::new();
        for sent in text::sentences(text) {
            let sent_start = base + sent.start;
            let mut toks: Vec<TokenSpan> = Vec::new();
            let mut latin_runs: Vec<Span> = Vec::new();
            let runs = text::split_scripts(&sent.text);
            let mixed = runs.iter().any(|(k, _)| *k == ScriptKind::Han)
                && runs.iter().any(|(k, _)| *k == ScriptKind::Latin);
            for (kind, run) in runs {
                match kind {
                    ScriptKind::Latin => latin_runs.push(run),
                    ScriptKind::Han => {
                        self.flush_latin(&mut latin_runs, sent_start, mixed, &mut toks);
                        let Some(ltp) = self.ltp.as_mut() else {
                            continue;
                        };
                        let parsed = match ltp
                            .segment(&run.text)
                            .and_then(|w| ltp.parse(&run.text, &w))
                        {
                            Ok(p) => p,
                            Err(_) => continue,
                        };
                        let flags = derive_chinese_ctx(&parsed, mixed);
                        for (t, (core, conn)) in parsed.into_iter().zip(flags) {
                            let s = sent_start + run.start + t.start;
                            let e = sent_start + run.start + t.end;
                            toks.push(TokenSpan {
                                range: s..e,
                                upos: zh_coarse(&t.pos).to_string(),
                                is_core: core,
                                is_connective: conn,
                                is_nonfinite: false,
                            });
                        }
                    }
                }
            }
            self.flush_latin(&mut latin_runs, sent_start, mixed, &mut toks);
            toks.sort_by_key(|t| (t.range.start, t.range.end));
            toks.dedup_by(|a, b| a.range == b.range);
            out.push((sent_start..sent_start + sent.text.len(), toks));
        }
        out
    }

    fn flush_latin(
        &mut self,
        runs: &mut Vec<Span>,
        sent_start: usize,
        mixed: bool,
        toks: &mut Vec<TokenSpan>,
    ) {
        if runs.is_empty() {
            return;
        }
        let batch = LatinBatch::from_runs(runs);
        runs.clear();
        let Some(en) = self.en.as_mut() else {
            return;
        };
        let Ok(ents) = en.annotate(&batch.text) else {
            return;
        };
        let flags = derive_english_ctx(&ents, mixed);
        for (t, (core, conn, nonfinite)) in ents.into_iter().zip(flags) {
            let Some((s, e)) = batch.map_span(t.lstart, t.lend) else {
                continue;
            };
            toks.push(TokenSpan {
                range: sent_start + s..sent_start + e,
                upos: en_coarse(&t.upos).to_string(),
                is_core: core,
                is_connective: conn,
                is_nonfinite: nonfinite,
            });
        }
    }
}

/// Map LTP's 27 Penn-Chinese tags onto the reader's colour classes.
///
/// `nh` is a person name ("人名" in LTP's own docs) and `z`/`b` are the
/// status/other-noun modifiers that behave like adjectives in running text
/// (轻轻地 → `z`, 金黄 → `z`, 雪白 → `b`). We keep `z` as ADJ so colour words
/// are coloured; the raw tag is not part of the public result.
pub fn zh_coarse(tag: &str) -> &'static str {
    match tag {
        "n" | "ns" | "nt" | "nh" | "ni" | "nl" | "nz" | "nd" | "j" => "NOUN",
        "v" => "VERB",
        "a" | "b" | "z" => "ADJ",
        // `r` is LTP's pronoun class (代词: 她/我们/谁…). It used to be mapped to
        // ADV (co-listed with `d`), which wrongly coloured pronouns as adverbs.
        "r" => "PRON",
        "d" => "ADV",
        _ => "OTHER",
    }
}

/// Coarse class of one UD UPOS tag (the English neural path).
pub fn en_coarse(upos: &str) -> &'static str {
    match upos {
        "NOUN" | "PROPN" => "NOUN",
        "VERB" | "AUX" => "VERB",
        "ADJ" => "ADJ",
        "ADV" => "ADV",
        "PRON" => "PRON",
        _ => "OTHER",
    }
}

/// `Arc<Mutex<Neural>>` shared with the background loader.
pub type SharedNeural = Arc<Mutex<Option<Neural>>>;
