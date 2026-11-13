//! English pipeline: roberta-large UPOS tagger + SuPar (esupar) Biaffine parser.
//!
//! Input contract, verified against esupar/onnxruntime in
//! `scripts/check_en_parity.py`:
//!
//! * words = whitespace tokens with edge punctuation split off
//!   (`"river."` -> `river` + `.`, like Stanza's tokenizer used by esupar)
//! * **tagger** input = `<s>` + BPE(`" " + word`) per word + `</s>`, right-padded
//!   with `<pad>` to 40 tokens and the pads masked. This is exactly what
//!   `AutoTokenizer(sentence)` produces for the fixture's sentences, and the
//!   exported roberta-large is numerically sane on it (`<s>`-only or very short
//!   unpadded inputs give garbage). The tagger output is invariant to the pad
//!   count because attention over pads is masked (checked empirically).
//! * **parser** input = `BOS(2)` row + one row per word + a padded tail row,
//!   `subword_ids[row][k]` = the k-th RoBERTa piece of the word (pad 1),
//!   row 0 holding a single `<s>` (id 0) like `SubwordField(bos)`;
//!   `word_ids` = BOS, then lowercased word-vocab ids (OOV 1), then 0.
//!   The exported graph only masks with `word_ids != 0`, so ROOT is column 0
//!   and word *i* is row/column *i+1*.
//! * `piece_order` = stable argsort of the flattened pad mask (valid first).
//!
//! esupar additionally feeds the tagger FEATS (`VerbForm=Fin`, …) into the
//! parser word ids (`word|feat` vocabulary entries). Our parser takes words
//! only; for the fixture sentences both parses agree except for `ran` (a
//! lowercase word missing from the word vocab), so we deliberately run once
//! with word-only ids. See `EXPORT.md`.

use anyhow::{anyhow, Context, Result};
use ort::session::Session;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::eisner;
use crate::metrics::ModelStat;
use crate::ortutil;

const SUBWORD_PAD: i64 = 1; // roberta <pad>
const BOS: i64 = 2; // supar <bos> word id
/// Right padding length for the tagger (>= the longest fixture sentence).
const PADDING: usize = 40;

#[derive(Deserialize)]
struct TaggerLabels {
    id2label: HashMap<String, String>,
}

/// `parser_word_vocab.json` is SuPar's `Field` dump: `itos` indexed by id.
#[derive(Deserialize)]
struct WordVocabFile {
    itos: Vec<String>,
    #[serde(default)]
    lower: bool,
}

#[derive(Deserialize)]
struct ParserLabels {
    labels: Vec<String>,
    word_vocab_size: usize,
    #[serde(default)]
    bert_max_len: usize,
    #[serde(default)]
    #[allow(dead_code)]
    notes: Option<serde_json::Value>,
}

/// One English word with tagger/parser results.
#[derive(Debug, Clone)]
pub struct EnToken {
    pub text: String,
    /// byte range inside the Latin batch string
    pub lstart: usize,
    pub lend: usize,
    /// full packed UPOS+FEATS label (e.g. `VERB|Mood=Ind|VerbForm=Fin`)
    pub upos_tag: String,
    pub upos: String,
    /// 1-based word index inside the clause, 0 == ROOT
    pub head: usize,
    pub deprel: String,
}

pub struct English {
    tagger: Session,
    parser: Session,
    tokenizer: tokenizers::Tokenizer,
    id2label: Vec<String>,
    rel_labels: Vec<String>,
    word_vocab: HashMap<String, i64>,
    lower_words: bool,
    max_word_chars: usize,
    pub max_len: usize,
    pub tagger_stat: ModelStat,
    pub parser_stat: ModelStat,
}

impl English {
    pub fn load(models_dir: &Path, threads: usize) -> Result<Self> {
        let dir = models_dir.join("en-upos-large").join("onnx");
        let tagger_path = dir.join("en_upos_tagger.onnx");
        let parser_path = dir.join("en_supar_parser.onnx");

        let t0 = std::time::Instant::now();
        let tagger = ortutil::session(&tagger_path, threads)?;
        let mut tagger_stat = ModelStat::default();
        tagger_stat.load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        tagger_stat.files = ortutil::file_sizes(&[&tagger_path, &dir.join("tagger_labels.json")]);
        let mem_after_tagger = ortutil::rss_mib();

        let t1 = std::time::Instant::now();
        let parser = ortutil::session(&parser_path, threads)?;
        let mut parser_stat = ModelStat::default();
        parser_stat.load_ms = t1.elapsed().as_secs_f64() * 1000.0;
        parser_stat.files = ortutil::file_sizes(&[
            &parser_path,
            &dir.join("parser_labels.json"),
            &dir.join("parser_word_vocab.json"),
        ]);
        eprintln!(
            "  en: rss after tagger {:?} MiB, after parser {:?} MiB",
            mem_after_tagger,
            ortutil::rss_mib()
        );

        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer").join("tokenizer.json"))
                .map_err(|e| anyhow!("en tokenizer: {e}"))?;
        let tl: TaggerLabels = serde_json::from_str(
            &std::fs::read_to_string(dir.join("tagger_labels.json")).context("tagger_labels.json")?,
        )?;
        let pl: ParserLabels = serde_json::from_str(
            &std::fs::read_to_string(dir.join("parser_labels.json")).context("parser_labels.json")?,
        )?;
        let wv: WordVocabFile = serde_json::from_str(
            &std::fs::read_to_string(dir.join("parser_word_vocab.json"))
                .context("parser_word_vocab.json")?,
        )?;
        if wv.itos.len() > pl.word_vocab_size {
            return Err(anyhow!(
                "parser word vocab larger than declared: {} > {}",
                wv.itos.len(),
                pl.word_vocab_size
            ));
        }
        // SuPar `Field(lower=True)`: first occurrence wins (itos is unique).
        let mut word_vocab: HashMap<String, i64> = HashMap::with_capacity(wv.itos.len());
        for (i, w) in wv.itos.iter().enumerate() {
            word_vocab.entry(w.clone()).or_insert(i as i64);
        }
        let max_word_chars = word_vocab.keys().map(|k| k.chars().count()).max().unwrap_or(0);
        let mut id2label = vec![String::new(); tl.id2label.len()];
        for (k, v) in tl.id2label {
            let i: usize = k.parse().context("tagger label id")?;
            if i >= id2label.len() {
                id2label.resize(i + 1, String::new());
            }
            id2label[i] = v;
        }
        Ok(Self {
            tagger,
            parser,
            tokenizer,
            id2label,
            rel_labels: pl.labels,
            word_vocab,
            lower_words: wv.lower,
            max_word_chars,
            max_len: if pl.bert_max_len == 0 { 512 } else { pl.bert_max_len },
            tagger_stat,
            parser_stat,
        })
    }

    /// RoBERTa pieces for one word (esupar's `" " + token` trick).
    pub fn pieces_for_word(&self, word: &str) -> Result<Vec<i64>> {
        let enc = self
            .tokenizer
            .encode(format!(" {word}"), false)
            .map_err(|e| anyhow!("roberta encode: {e}"))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        if ids.is_empty() {
            Ok(vec![SUBWORD_PAD])
        } else {
            Ok(ids)
        }
    }

    /// Words of a Latin batch: whitespace split with byte offsets, then edge
    /// punctuation split off (esupar tokenizes with Stanza, which does this).
    pub fn words(text: &str) -> Vec<(String, usize, usize)> {
        let mut raw = Vec::new();
        let mut start: Option<usize> = None;
        for (i, c) in text.char_indices() {
            if c.is_whitespace() {
                if let Some(s) = start.take() {
                    raw.push((text[s..i].to_string(), s, i));
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            raw.push((text[s..].to_string(), s, text.len()));
        }
        let mut out = Vec::with_capacity(raw.len());
        for (w, s, _e) in raw {
            out.extend(split_punct(&w, s));
        }
        out
    }

    fn word_id(&self, w: &str) -> i64 {
        let key = if self.lower_words {
            w.to_lowercase()
        } else {
            w.to_string()
        };
        if key.chars().count() > self.max_word_chars + 8 {
            return 1;
        }
        *self.word_vocab.get(&key).unwrap_or(&1)
    }

    /// Run tagger + parser over one Latin batch string.
    pub fn annotate(&mut self, text: &str) -> Result<Vec<EnToken>> {
        let words = Self::words(text);
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let mut pieces: Vec<Vec<i64>> = Vec::with_capacity(words.len());
        for (w, _, _) in &words {
            pieces.push(self.pieces_for_word(w)?);
        }

        // ---- tagger -------------------------------------------------------
        // esupar feeds the tagger `AutoTokenizer(" " + sentence)` — one pass
        // over the WHOLE `" word word word"` string, not the concatenation of
        // per-word pieces. The two agree except at the first word, where the
        // whole-sentence BPE has no leading space (`quiet` + `ly` instead of
        // `Ġquietly`), which changes the tags. Reproduce the reference: BPE the
        // joined string and use *that* for the tagger.
        let (tagger_ids, tagger_word_ids) = {
            let enc = self
                .tokenizer
                .encode(format!(" {}", text.trim()), false)
                .map_err(|e| anyhow!("roberta encode: {e}"))?;
            let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
            (ids, enc.get_word_ids().to_vec())
        };
        let mut flat: Vec<i64> = vec![0]; // <s>
        flat.extend_from_slice(&tagger_ids);
        // esupar tokenizes `" " + s` per word, so the very first piece is a
        // space-prefixed piece (`Ġquietly`), never a bare `quiet`. Keeping that
        // shape is what makes esupar's own tagger behave; do not "fix" it by
        // stripping the leading space (Python reference does the same).
        flat.push(2); // </s>
        flat.truncate(self.max_len);
        while flat.len() < PADDING {
            flat.push(1); // <pad>
        }
        let n = flat.len();
        let mask: Vec<i64> = flat.iter().map(|&id| (id != 1) as i64).collect();
        let ids_t = ortutil::i64_tensor(&flat, [1, n])?;
        let mask_t = ortutil::i64_tensor(&mask, [1, n])?;
        let (res, _d) = ortutil::timed(&mut self.tagger_stat, || {
            self.tagger
                .run(ort::inputs!["input_ids" => ids_t, "attention_mask" => mask_t])
                .map_err(anyhow::Error::from)
        });
        let tout = res?;
        let (dims, logits) = ortutil::take_f32(&tout, "logits")?;
        drop(tout);
        let classes = *dims.get(2).unwrap_or(&0);
        let mut sub_labels: Vec<usize> = Vec::with_capacity(n);
        for t in 0..n {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for c in 0..classes {
                let v = logits[t * classes + c];
                if v > bv {
                    bv = v;
                    best = c;
                }
            }
            sub_labels.push(best);
        }
        // pieces -> words: BIO labels stay inside the word they start in; a
        // packed multiword label (`ADP+DET`, `B-PROPN+PROPN+…`) belongs to the
        // word where it starts.
        // BIO labels are folded per whitespace word: a label starting with
        // `B-` (or a bare UPOS) replaces the word's tag, `I-` continuations are
        // folded into the previous label (`B-ADV` + `I-ADV` -> `ADV`).
        let mut word_tags: Vec<String> = vec![String::new(); words.len()];
        for (k, wid) in tagger_word_ids.iter().enumerate() {
            let Some(w) = wid else { continue };
            let w = *w as usize;
            if w >= words.len() {
                continue;
            }
            let lab = self
                .id2label
                .get(sub_labels[k + 1])
                .cloned()
                .unwrap_or_default();
            let cur = &mut word_tags[w];
            if cur.is_empty() || !lab.starts_with("I-") {
                *cur = strip_bio(&lab).to_string();
            }
        }
        for (i, t) in word_tags.iter_mut().enumerate() {
            if t.is_empty() {
                // word made of no piece (should not happen): fall back
                *t = self
                    .id2label
                    .get(*sub_labels.get(1 + i).unwrap_or(&0))
                    .cloned()
                    .unwrap_or_default();
            }
        }

        // ---- parser --------------------------------------------------------
        let nw = words.len();
        let fix_len = pieces.iter().map(|p| p.len()).max().unwrap_or(1).max(1);
        let rows = nw + 2; // BOS row + words + one all-pad row
        let mut word_ids: Vec<i64> = Vec::with_capacity(rows);
        word_ids.push(BOS);
        word_ids.extend(words.iter().map(|(w, _, _)| self.word_id(w)));
        word_ids.push(0); // pad row
        let mut sub_ids = vec![SUBWORD_PAD; rows * fix_len];
        sub_ids[0] = 0; // <s> piece in the BOS row
        for i in 0..nw {
            for (k, id) in pieces[i].iter().enumerate() {
                sub_ids[(i + 1) * fix_len + k] = *id;
            }
        }
        // piece_order = stable argsort of the flattened pad mask.
        let mut order: Vec<i64> = (0..(rows * fix_len) as i64).collect();
        order.sort_by_key(|&i| (sub_ids[i as usize] == SUBWORD_PAD, i));

        let sub_t = ortutil::i64_tensor3(&sub_ids, [1, rows, fix_len])?;
        let word_t = ortutil::i64_tensor(&word_ids, [1, rows])?;
        let order_t = ortutil::i64_tensor(&order, [1, rows * fix_len])?;
        let (res, _d) = ortutil::timed(&mut self.parser_stat, || {
            self.parser
                .run(ort::inputs![
                    "subword_ids" => sub_t,
                    "word_ids" => word_t,
                    "piece_order" => order_t,
                ])
                .map_err(anyhow::Error::from)
        });
        let pout = res?;
        let (adims, adata) = ortutil::take_f32(&pout, "arc_logits")?;
        let (rdims, rdata) = ortutil::take_f32(&pout, "rel_logits")?;
        drop(pout);
        let aw = *adims.get(1).unwrap_or(&0);
        let rclass = *rdims.get(3).unwrap_or(&0);
        if aw < nw + 1 {
            return Err(anyhow!(
                "arc_logits [{aw}] smaller than words+1 [{}]",
                nw + 1
            ));
        }

        // ---- decode (Eisner; esupar decodes projectively too) ---------------
        let n = nw + 1;
        let mut scores = vec![eisner::NEG_INF; n * n];
        for d in 0..n {
            for h in 0..n {
                scores[d * n + h] = adata[d * aw + h];
            }
        }
        // argmax-tree first (matches esupar's argmax on our fixtures), then Eisen.
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
            argmax_heads
        } else {
            eisner::decode_eisner(&scores, nw)
        };

        let mut out = Vec::with_capacity(nw);
        for (i, ((w, s, e), tag)) in words.iter().zip(word_tags.iter()).enumerate() {
            let head = heads[i + 1];
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for c in 0..rclass {
                let v = rdata[((i + 1) * aw + head) * rclass + c];
                if v > bv {
                    bv = v;
                    best = c;
                }
            }
            let upos = tag.split('|').next().unwrap_or("").to_string();
            out.push(EnToken {
                text: w.clone(),
                lstart: *s,
                lend: *e,
                upos_tag: tag.clone(),
                upos,
                head,
                deprel: self.rel_labels.get(best).cloned().unwrap_or_default(),
            });
        }
        Ok(out)
    }
}

fn strip_bio(label: &str) -> &str {
    label
        .strip_prefix("B-")
        .or_else(|| label.strip_prefix("I-"))
        .unwrap_or(label)
}

/// Split a whitespace word into units, separating standalone punctuation
/// (`. , ! ? ; : " ' ( ) [ ] { }` and the CJK/smart quotes). Apostrophes and
/// hyphens inside a word are kept (`don't`, `well-known`, `It's`).
pub fn split_punct(word: &str, base: usize) -> Vec<(String, usize, usize)> {
    let is_sep = |c: char| {
        matches!(
            c,
            '.' | ',' | '!' | '?' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}'
                | '“' | '”' | '‘' | '’' | '…' | '—' | '–'
        )
    };
    let chars: Vec<(usize, char)> = word.char_indices().collect();
    let apos_inside = |i: usize| -> bool {
        let c = chars[i].1;
        (c == '\'' || c == '’')
            && i > 0
            && i + 1 < chars.len()
            && chars[i - 1].1.is_alphanumeric()
            && chars[i + 1].1.is_alphanumeric()
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let (off, c) = chars[i];
        if is_sep(c) && !apos_inside(i) {
            out.push((c.to_string(), base + off, base + off + c.len_utf8()));
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i + 1;
        while j < chars.len() {
            if is_sep(chars[j].1) && !apos_inside(j) {
                break;
            }
            j += 1;
        }
        let s = base + chars[start].0;
        let (last_off, last_ch) = chars[j - 1];
        out.push((
            word[chars[start].0..last_off + last_ch.len_utf8()].to_string(),
            s,
            base + last_off + last_ch.len_utf8(),
        ));
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punctuation_is_split_off() {
        let got: Vec<String> = split_punct("(river.),", 0).into_iter().map(|x| x.0).collect();
        assert_eq!(got, ["(", "river", ".", ")", ","]);
        let got: Vec<String> = split_punct("It's", 0).into_iter().map(|x| x.0).collect();
        assert_eq!(got, ["It's"]);
        let got: Vec<String> = split_punct("well-known", 0)
            .into_iter()
            .map(|x| x.0)
            .collect();
        assert_eq!(got, ["well-known"]);
    }

    #[test]
    fn words_offsets_slice_back() {
        let text = "  Tom and Mary opened the door.  ";
        for (w, s, e) in English::words(text) {
            assert_eq!(&text[s..e], w);
        }
    }
}
