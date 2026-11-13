//! Chinese pipeline: LTP/small (ltp-core) exported to ONNX.
//!
//! Graph: `ltp_small_cws_pos_dep.onnx` with
//!   inputs  input_ids[int64, 1×L], attention_mask, token_type_ids,
//!           word_index[int64, 1×W], word_mask[int64, 1×W]
//!   outputs cws_logits[1, L-2, 2], pos_logits[1, W, 27],
//!           arc_logits[1, W+1, W+1], rel_logits[1, W+1, W+1, 14]
//!
//! `word_index` is *only* needed for POS/dep, which run on already-segmented
//! words. The CLI therefore runs CWS first (one dummy word), assembles words
//! from the B-W/I-W tags, then runs the graph a second time with real word
//! indices.
//!
//! Word indices are offsets in the tokenizer's *normalized* string (offsets from
//! `tokenizers` with `NormalizedString` removed); we borrow the normalized text
//! so the returned spans slice the original chunk exactly.

use anyhow::{anyhow, Context, Result};
use ort::session::Session;
use serde::Deserialize;
use std::path::Path;

use crate::eisner;
use crate::ortutil;

#[derive(Deserialize)]
struct LabelsFile {
    cws_labels: Vec<String>,
    pos_labels: Vec<String>,
    dep_labels: Vec<String>,
}

pub struct Ltp {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
    cws_labels: Vec<String>,
    pos_labels: Vec<String>,
    dep_labels: Vec<String>,
}

/// One segmented Chinese word.
#[derive(Debug, Clone)]
pub struct ZhWord {
    pub text: String,
    /// byte offsets inside the (original) chunk
    pub start: usize,
    pub end: usize,
}

/// One parsed Chinese word with POS + dependency.
#[derive(Debug, Clone)]
pub struct ZhToken {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub pos: String,
    pub head: usize, // 1-based word index within the chunk, 0 == ROOT
    pub deprel: String,
}

/// Raw char-level encoding of a chunk.
struct Encoded {
    ids: Vec<i64>,
    mask: Vec<i64>,
    types: Vec<i64>,
    offsets: Vec<(usize, usize)>,
    /// normalized text (what `offsets` index into)
    text: String,
}

impl Ltp {
    pub fn load(models_dir: &Path, threads: usize) -> Result<Self> {
        let dir = models_dir.join("ltp-small").join("onnx");
        let onnx = dir.join("ltp_small_cws_pos_dep.onnx");
        let session = ortutil::session(&onnx, threads)?;
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("ltp tokenizer: {e}"))?;
        let labels: LabelsFile =
            serde_json::from_str(&std::fs::read_to_string(dir.join("labels.json"))?)
                .context("ltp labels.json")?;
        Ok(Self {
            session,
            tokenizer,
            cws_labels: labels.cws_labels,
            pos_labels: labels.pos_labels,
            dep_labels: labels.dep_labels,
        })
    }

    fn encode(&self, chunk: &str) -> Result<Encoded> {
        let enc = self
            .tokenizer
            .encode(chunk, true)
            .map_err(|e| anyhow!("ltp encode: {e}"))?;
        let offsets = enc.get_offsets().to_vec();
        Ok(Encoded {
            ids: enc.get_ids().iter().map(|&x| x as i64).collect(),
            mask: enc.get_attention_mask().iter().map(|&x| x as i64).collect(),
            types: enc.get_type_ids().iter().map(|&x| x as i64).collect(),
            offsets,
            text: chunk.to_string(),
        })
    }

    fn run(&mut self, e: &Encoded, words: &[i64]) -> Result<ort::session::SessionOutputs<'_>> {
        let l = e.ids.len();
        let w = words.len().max(1);
        let wi: Vec<i64> = if words.is_empty() { vec![0] } else { words.to_vec() };
        let wm: Vec<i64> = vec![1; wi.len()];
        let ids = ortutil::i64_tensor(&e.ids, [1, l])?;
        let mask = ortutil::i64_tensor(&e.mask, [1, l])?;
        let types = ortutil::i64_tensor(&e.types, [1, l])?;
        let wi_t = ortutil::i64_tensor(&wi, [1, w])?;
        let wm_t = ortutil::i64_tensor(&wm, [1, w])?;
        let out = self.session.run(ort::inputs![
            "input_ids" => ids,
            "attention_mask" => mask,
            "token_type_ids" => types,
            "word_index" => wi_t,
            "word_mask" => wm_t,
        ])?;
        Ok(out)
    }

    /// Segment `chunk` into words using the CWS head (first inference pass).
    pub(crate) fn segment(&mut self, chunk: &str) -> Result<Vec<ZhWord>> {
        let e = self.encode(chunk)?;
        let (dims, data) = {
            let out = self.run(&e, &[])?;
            ortutil::take_f32(&out, "cws_logits")?
        };
        let rows = *dims.get(1).unwrap_or(&0);
        let classes = *dims.get(2).unwrap_or(&2);
        let mut words: Vec<ZhWord> = Vec::new();
        let mut cur: Option<ZhWord> = None;
        for j in 0..rows {
            // attention_mask for row j is mask[j+1]
            if j + 1 >= e.mask.len() || e.mask[j + 1] == 0 {
                continue;
            }
            if j + 1 >= e.offsets.len() {
                continue;
            }
            let (s, t) = e.offsets[j + 1];
            if s == t {
                continue;
            }
            // The exported head emits `[logit(B-W), logit(I-W)]` per char; a new
            // word starts where `logit(B-W) > logit(I-W)` (verified against
            // onnxruntime + ltp_core in scripts/check_ltp_cws.py).
            let bi = self
                .cws_labels
                .iter()
                .position(|l| l.starts_with('B'))
                .unwrap_or(0);
            let is_start = data[j * classes + bi]
                > data[j * classes + (if bi == 0 { 1 } else { 0 })];
            match (&mut cur, is_start) {
                (None, _) | (Some(_), true) => {
                    if let Some(w) = cur.take() {
                        words.push(w);
                    }
                    cur = Some(ZhWord {
                        text: e.text[s..t].to_string(),
                        start: s,
                        end: t,
                    });
                }
                (Some(w), false) => {
                    w.end = t;
                    w.text = e.text[w.start..t].to_string();
                }
            }
        }
        if let Some(w) = cur {
            words.push(w);
        }
        if std::env::var_os("NLP_SPIKE_DEBUG_CWS").is_some() {
            eprintln!(
                "CWS chunk={chunk:?} words={:?}",
                words.iter().map(|w| (&w.text, w.start, w.end)).collect::<Vec<_>>()
            );
        }
        Ok(words)
    }

    /// POS + dependency (second inference pass) over `words` of `chunk`.
    pub(crate) fn parse(&mut self, chunk: &str, words: &[ZhWord]) -> Result<Vec<ZhToken>> {
        let e = self.encode(chunk)?;
        if words.is_empty() {
            return Ok(Vec::new());
        }
        // char byte offset -> index into `hidden[:, 1:-1]` (LTP's word_index space:
        // sub-token index j in the full sequence corresponds to j-1 after dropping
        // [CLS] and [SEP]).
        let mut off2sub: std::collections::HashMap<usize, usize> = Default::default();
        for (j, &(s, t)) in e.offsets.iter().enumerate() {
            if s == t || j == 0 {
                continue;
            }
            off2sub.entry(s).or_insert(j - 1);
        }
        let mut wi: Vec<i64> = Vec::with_capacity(words.len());
        for w in words {
            let sub = *off2sub.get(&w.start).with_context(|| {
                format!("no sub-token for word {} at byte {}", w.text, w.start)
            })?;
            wi.push(sub as i64);
        }
        let nw = wi.len();
        let out = self.run(&e, &wi)?;
        let (pdims, pdata) = ortutil::take_f32(&out, "pos_logits")?;
        let (adims, adata) = ortutil::take_f32(&out, "arc_logits")?;
        let (rdims, rdata) = ortutil::take_f32(&out, "rel_logits")?;
        drop(out);
        let pclass = *pdims.get(2).unwrap_or(&0);
        let aw = *adims.get(1).unwrap_or(&0);
        let rclass = *rdims.get(3).unwrap_or(&0);
        let _ = rdims;

        // POS
        let mut pos = Vec::with_capacity(nw);
        for w in 0..nw {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for c in 0..pclass {
                let v = pdata[w * pclass + c];
                if v > bv {
                    bv = v;
                    best = c;
                }
            }
            pos.push(self.pos_labels.get(best).cloned().unwrap_or_default());
        }

        // dep: arc_logits[0] is ROOT; word i is row/col i+1.
        // Restrict to the (nw+1) square, then run Eisner with ROOT at 0.
        let n = nw + 1;
        if aw < n {
            return Err(anyhow!("arc_logits smaller than expected: {aw} < {n}"));
        }
        let mut scores = vec![eisner::NEG_INF; n * n];
        for d in 0..n {
            for h in 0..n {
                scores[d * n + h] = adata[d * aw + h];
            }
        }
        // Decoding strategy (see EXPORT.md):
        //   1. take the bare argmax heads;
        //   2. if they already form a tree, use them (matches LTP's own argmax);
        //   3. otherwise fall back to the projective Eisner decoder.
        let mut argmax_heads = vec![0usize; nw + 1];
        for d in 0..nw {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for h in 0..n {
                let v = adata[(d + 1) * aw + h];
                if v > bv {
                    bv = v;
                    best = h;
                }
            }
            argmax_heads[d + 1] = best;
        }
        let heads = if eisner::is_tree(&argmax_heads, nw) {
            argmax_heads.clone()
        } else {
            eisner::decode_eisner(&scores, nw)
        };
        let mut toks = Vec::with_capacity(nw);
        for (d, w) in words.iter().enumerate() {
            let head = heads[d + 1];
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for c in 0..rclass {
                let v = rdata[((d + 1) * aw + head) * rclass + c];
                if v > bv {
                    bv = v;
                    best = c;
                }
            }
            toks.push(ZhToken {
                text: w.text.clone(),
                start: w.start,
                end: w.end,
                pos: pos[d].clone(),
                head,
                deprel: self.dep_labels.get(best).cloned().unwrap_or_default(),
            });
        }
        Ok(toks)
    }
}
