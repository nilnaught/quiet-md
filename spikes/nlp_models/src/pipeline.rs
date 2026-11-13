//! Local pipeline: ICU sentences -> Han/Latin script runs -> LTP (zh) and
//! roberta-large tagger + esupar (en) -> one flat token list with byte offsets
//! into the original input.
//!
//! Derived flags (`is_core_verb`, `is_connective`, `is_nonfinite`) are computed
//! per *clause* (the span of one model call, i.e. one script run / one joined
//! Latin segment), because dependency heads are only comparable inside one
//! parse.

use anyhow::Result;

use crate::english::English;
use crate::ltp::Ltp;
use crate::text::{self, LatinBatch, ScriptKind, Span};

#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub start: usize,
    pub end: usize,
    /// coarse class for colouring: NOUN / VERB / ADJ / ADV / OTHER
    pub pos: String,
    /// LTP tag (zh, e.g. `n`, `a`, `c`) or packed UPOS+FEATS (en)
    pub upos_or_ltp_tag: String,
    /// 1-based token index inside its clause; 0 == ROOT
    pub head: usize,
    pub deprel: String,
    pub is_core_verb: bool,
    pub is_connective: bool,
    pub is_nonfinite: bool,
    pub lang: &'static str,
    pub sentence: usize,
    /// 1-based word index inside the clause (after `tokens[*].clause_index`)
    pub clause_index: usize,
    /// script run index inside the sentence
    pub run: usize,
}

#[derive(Debug, Clone)]
pub struct SentenceOut {
    pub index: usize,
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub tokens: Vec<Token>,
}

pub struct Pipeline {
    pub ltp: Option<Ltp>,
    pub en: Option<English>,
}

impl Pipeline {
    pub fn annotate(&mut self, text: &str) -> Result<Vec<SentenceOut>> {
        let mut out = Vec::new();
        for (si, sent) in text::sentences(text).into_iter().enumerate() {
            let mut toks: Vec<Token> = Vec::new();
            let mut latin_runs: Vec<Span> = Vec::new();
            let mut run_id = 0usize;
            for (kind, run) in text::split_scripts(&sent.text) {
                match kind {
                    ScriptKind::Latin => latin_runs.push(run),
                    ScriptKind::Han => {
                        self.flush_latin(&mut latin_runs, si, sent.start, run_id, &mut toks)?;
                        run_id += 1;
                        let Some(ltp) = self.ltp.as_mut() else {
                            continue;
                        };
                        let words = ltp.segment(&run.text)?;
                        let parsed = ltp.parse(&run.text, &words)?;
                        let n = parsed.len();
                        let flags = derive_chinese(&parsed);
                        for (i, t) in parsed.into_iter().enumerate() {
                            toks.push(Token {
                                text: t.text,
                                start: sent.start + run.start + t.start,
                                end: sent.start + run.start + t.end,
                                pos: zh_coarse(&t.pos).to_string(),
                                upos_or_ltp_tag: t.pos,
                                head: t.head,
                                deprel: t.deprel,
                                is_core_verb: flags[i].0,
                                is_connective: flags[i].1,
                                is_nonfinite: false,
                                lang: "zh",
                                sentence: si,
                                clause_index: i + 1,
                                run: run_id,
                            });
                        }
                        let _ = n;
                    }
                }
            }
            self.flush_latin(&mut latin_runs, si, sent.start, run_id, &mut toks)?;
            toks.sort_by_key(|t| (t.start, t.end));
            toks.dedup_by(|a, b| a.start == b.start && a.end == b.end && a.lang == b.lang);
            out.push(SentenceOut {
                index: si,
                start: sent.start,
                end: sent.end,
                text: sent.text.clone(),
                tokens: toks,
            });
        }
        Ok(out)
    }

    fn flush_latin(
        &mut self,
        runs: &mut Vec<Span>,
        si: usize,
        sent_start: usize,
        run_id: usize,
        toks: &mut Vec<Token>,
    ) -> Result<()> {
        if runs.is_empty() {
            return Ok(());
        }
        let first_start = runs[0].start;
        let batch = LatinBatch::from_runs(runs);
        let ents = match self.en.as_mut() {
            Some(e) => e.annotate(&batch.text)?,
            None => Vec::new(),
        };
        let flags = derive_english(&ents);
        for (i, t) in ents.into_iter().enumerate() {
            let (s, e) = batch
                .map_span(t.lstart, t.lend)
                .map(|(a, b)| (sent_start + a, sent_start + b))
                .unwrap_or((sent_start + first_start, sent_start + first_start));
            toks.push(Token {
                text: t.text,
                start: s,
                end: e,
                pos: en_coarse(&t.upos).to_string(),
                upos_or_ltp_tag: t.upos_tag,
                head: t.head,
                deprel: t.deprel,
                is_core_verb: flags[i].0,
                is_connective: flags[i].1,
                is_nonfinite: flags[i].2,
                lang: "en",
                sentence: si,
                clause_index: i + 1,
                run: run_id,
            });
        }
        runs.clear();
        Ok(())
    }
}

pub fn zh_coarse(tag: &str) -> &'static str {
    match tag {
        "n" | "ns" | "nt" | "nh" | "ni" | "nl" | "nz" | "nd" | "j" => "NOUN",
        "v" => "VERB",
        "a" | "b" | "z" => "ADJ",
        "d" | "r" => "ADV",
        _ => "OTHER",
    }
}

pub fn en_coarse(upos: &str) -> &'static str {
    match upos {
        "NOUN" | "PROPN" => "NOUN",
        "VERB" | "AUX" => "VERB",
        "ADJ" => "ADJ",
        "ADV" => "ADV",
        _ => "OTHER",
    }
}

// ------------------------------------------------------------------------- //
// Derived flags
// ------------------------------------------------------------------------- //

/// Chinese connectives that LTP does not always tag as `c`.
pub const ZH_CONNECTIVES: &[&str] = &[
    "因为", "所以", "因此", "因而", "于是", "虽然", "但是", "但", "可是", "然而", "不过", "如果",
    "假如", "要是", "那么", "不仅", "而且", "并且", "尽管", "即使", "无论", "不管", "只要",
    "只有", "除非", "否则", "或者", "还是", "以及", "不但", "同时", "然后", "接着", "首先", "其次",
    "最后", "既然", "由于", "以便", "以免", "与其", "宁可", "之所以", "何况", "另外", "此外",
];

/// `(is_core_verb, is_connective)` per token of one Chinese clause.
pub fn derive_chinese(parsed: &[crate::ltp::ZhToken]) -> Vec<(bool, bool)> {
    let n = parsed.len();
    let mut core = vec![false; n];
    if n > 0 {
        let root = (0..n).find(|&i| parsed[i].head == 0).unwrap_or(0);
        let mut cands: Vec<usize> = Vec::new();
        // 1) the root itself if predicative
        if is_zh_predicate(parsed, root) {
            cands.push(root);
        }
        // 2) coordinated predicates hanging off the root
        for i in 0..n {
            if i == root {
                continue;
            }
            if parsed[i].deprel == "COO" && is_zh_predicate(parsed, i) {
                let mut h = parsed[i].head;
                let mut guard = 0;
                while h != 0 && h - 1 != root && guard < n {
                    h = parsed[h - 1].head;
                    guard += 1;
                }
                if h == 0 || h - 1 == root {
                    cands.push(i);
                }
            }
        }
        // 3) root is a copula / particle ('是', '的', …): the real predicate is
        //    its verb/adj dependent.
        let root_is_copula = parsed[root].pos == "v" && is_copula(&parsed[root].text);
        if cands.is_empty() && root_is_copula {
            for i in 0..n {
                if parsed[i].head == root + 1 && matches!(parsed[i].pos.as_str(), "v" | "a") {
                    cands.push(i);
                }
            }
        }
        if cands.is_empty() {
            // 4) fall back: top-most verb in the tree
            let mut best: Option<(usize, usize)> = None; // (depth, idx)
            for i in 0..n {
                if parsed[i].pos != "v" {
                    continue;
                }
                let mut d = 0usize;
                let mut h = parsed[i].head;
                while h != 0 && d < n {
                    h = parsed[h - 1].head;
                    d += 1;
                }
                if h == 0 && best.map(|(bd, _)| d < bd).unwrap_or(true) {
                    best = Some((d, i));
                }
            }
            if let Some((_, i)) = best {
                cands.push(i);
            }
        }
        for c in cands {
            core[c] = true;
        }
    }
    parsed
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let conn = t.pos == "c"
                || ZH_CONNECTIVES.contains(&t.text.as_str())
                || matches!(t.deprel.as_str(), "LAD" | "RAD") && t.pos == "c";
            (core[i], conn)
        })
        .collect()
}

fn is_copula(t: &str) -> bool {
    matches!(t, "是" | "为" | "系" | "乃")
}

fn is_zh_predicate(parsed: &[crate::ltp::ZhToken], i: usize) -> bool {
    let t = &parsed[i];
    // An adjective only counts as the predicate when it is the clause root or a
    // coordinated predicate; attributive `a` (deprel ATT) is a modifier.
    if t.pos == "a" {
        return t.head == 0 || t.deprel == "COO" || t.deprel == "HED";
    }
    if t.pos != "v" {
        return false;
    }
    !is_copula(&t.text)
}

/// `(is_core_verb, is_connective, is_nonfinite)` per token of one English clause.
pub fn derive_english(toks: &[crate::english::EnToken]) -> Vec<(bool, bool, bool)> {
    let n = toks.len();
    let mut out = vec![(false, false, false); n];
    if n == 0 {
        return out;
    }
    let root = (0..n).find(|&i| toks[i].head == 0).unwrap_or(0);

    // UD attaches the copula *to* the predicate nominal ("She is tired" ->
    // tired:root, is:cop). spaCy instead roots the copula. For the reading
    // marker "core predicate" the real predicate is:
    //   * the root itself when it is a VERB/ADJ/Noun predicate;
    //   * when the root is a verb but there is an AUX/cop chain, still the root.
    // (see README: "root is copula/particle -> fall through to the real predicate")
    let mut core = vec![false; n];
    let mut candidates: Vec<usize> = Vec::new();
    if is_verb_like(&toks[root]) || matches!(toks[root].upos.as_str(), "ADJ" | "NOUN" | "PROPN") {
        candidates.push(root);
    }
    // a clause whose root is a copula (ModelForTokenClassification may root "is")
    if is_verb_like(&toks[root])
        && toks
            .iter()
            .enumerate()
            .any(|(i, t)| i != root && (t.deprel == "cop" || t.deprel == "aux") && is_content_pos(&t.upos))
        && toks.iter().any(|t| t.deprel == "cop" && t.head == root + 1)
    {
        // root is the copula: predicate = its copular complement
        for i in 0..n {
            if toks[i].head == root + 1 && toks[i].deprel != "cop" && is_content_pos(&toks[i].upos) {
                candidates.push(i);
            }
        }
    }
    // coordinated / paratactic predicates attached to the root
    for i in 0..n {
        if i == root || !is_verb_like(&toks[i]) {
            continue;
        }
        if toks[i].deprel == "conj" || toks[i].deprel == "parataxis" {
            let mut h = toks[i].head;
            let mut guard = 0;
            while h != 0 && h - 1 != root && guard < n {
                h = toks[h - 1].head;
                guard += 1;
            }
            if h == 0 || h - 1 == root {
                candidates.push(i);
            }
        }
    }
    for c in candidates {
        core[c] = true;
    }

    for i in 0..n {
        let t = &toks[i];
        let lower = t.text.to_lowercase();
        let finite = t.upos_tag.contains("VerbForm=Fin")
            || (t.upos == "VERB" && !t.upos_tag.contains("VerbForm=") && t.deprel != "xcomp");
        let vf = t
            .upos_tag
            .split('|')
            .find_map(|f| f.strip_prefix("VerbForm="))
            .unwrap_or("");
        let mut nonfinite = matches!(vf, "Inf" | "Ger" | "Part");
        if !nonfinite && t.upos == "VERB" && t.deprel == "xcomp" && !t.upos_tag.contains("VerbForm=Fin") {
            // TO + VB: esupar sometimes leaves off VerbForm on the infinitive
            let h = t.head;
            if h != 0 && (h - 1 == root || h - 1 < n) {
                let hd = &toks[h - 1];
                if hd.deprel == "aux" || hd.text.eq_ignore_ascii_case("to") {
                    nonfinite = true;
                }
            }
        }
        if core[i] && finite {
            nonfinite = false;
        }
        if core[i] {
            nonfinite = false;
        }
        let connective = matches!(t.upos.as_str(), "CCONJ" | "SCONJ")
            || matches!(t.deprel.as_str(), "mark" | "cc" | "cc:preconj")
            || (matches!(t.upos.as_str(), "PRON" | "DET" | "ADV")
                && (t.upos_tag.contains("PronType=Rel")
                    || t.upos_tag.contains("ExtPos=SCONJ")
                    || t.upos_tag.contains("ExtPos=CCONJ")));
        out[i] = (core[i], connective, nonfinite);
        let _ = lower;
    }
    out
}

fn has_nominal_child(toks: &[crate::english::EnToken], root: usize) -> bool {
    toks.iter().any(|t| {
        t.head == root + 1
            && matches!(t.upos.as_str(), "NOUN" | "PROPN" | "ADJ" | "PRON" | "NUM" | "VERB")
    })
}

fn is_verb_like(t: &crate::english::EnToken) -> bool {
    matches!(t.upos.as_str(), "VERB" | "AUX")
}

fn is_content_pos(upos: &str) -> bool {
    matches!(
        upos,
        "VERB" | "AUX" | "NOUN" | "PROPN" | "ADJ" | "ADV" | "PRON" | "NUM"
    )
}
