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
            let runs = text::split_scripts(&sent.text);
            let mixed_sentence = runs.iter().any(|(k, _)| *k == ScriptKind::Han)
                && runs.iter().any(|(k, _)| *k == ScriptKind::Latin);
            for (kind, run) in runs {
                match kind {
                    ScriptKind::Latin => {
                        debug_assert_eq!(run.text, sent.text[run.start..run.end]);
                        latin_runs.push(run);
                    }
                    ScriptKind::Han => {
                        self.flush_latin(&mut latin_runs, si, sent.start, run_id, mixed_sentence, &mut toks)?;
                        run_id += 1;
                        let Some(ltp) = self.ltp.as_mut() else {
                            continue;
                        };
                        debug_assert_eq!(run.text, sent.text[run.start..run.end], "han run offsets");
                        let words = ltp.segment(&run.text)?;
                        let parsed = ltp.parse(&run.text, &words)?;
                        let n = parsed.len();
                        let flags = derive_chinese_ctx(&parsed, mixed_sentence);
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
            self.flush_latin(&mut latin_runs, si, sent.start, run_id, mixed_sentence, &mut toks)?;
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
        mixed_sentence: bool,
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
        let flags = derive_english_ctx(&ents, mixed_sentence);
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

/// Map LTP's 27 Penn-Chinese tags onto the four colour classes.
///
/// `nh` is a person name ("人名" in LTP's own docs) and `z`/`b` are the
/// status/other-noun modifiers that behave like adjectives in running text
/// (轻轻地 → `z`, 金黄 → `z`, 雪白 → `b`). We keep `z` as ADJ so colour words
/// are coloured; the raw tag is preserved in `upos_or_ltp_tag`.
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
    derive_chinese_ctx(parsed, false)
}

/// Same, but `mixed` says the surrounding *sentence* also has Latin script.
///
/// In a mixed CN/EN sentence the Chinese clause is often a fragment
/// ("…的 安静的河边，") that LTP roots on an attributive-ish adjective. Marking
/// that as the core predicate would show 《安静》 as the sentence's predicate,
/// so in mixed sentences only a root/coordinated predicate that is a verb or a
/// predicative adjective (not `ATT`) counts - and the copula fallback is off.
pub fn derive_chinese_ctx(parsed: &[crate::ltp::ZhToken], mixed: bool) -> Vec<(bool, bool)> {
    let n = parsed.len();
    let mut core = vec![false; n];
    if n > 0 {
        let root = (0..n).find(|&i| parsed[i].head == 0).unwrap_or(0);
        let mut cands: Vec<usize> = Vec::new();
        // 1) the root itself if predicative
        if is_zh_predicate(parsed, root)
            && (!mixed || (parsed[root].deprel != "ATT" && is_zh_mixed_core(parsed, root)))
        {
            cands.push(root);
        }
        // 2) coordinated predicates hanging off the root
        for i in 0..n {
            if i == root {
                continue;
            }
            if parsed[i].deprel == "COO"
                && is_zh_predicate(parsed, i)
                && (!mixed || is_zh_mixed_core(parsed, i))
            {
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
        if cands.is_empty() && root_is_copula && !mixed {
            for i in 0..n {
                if parsed[i].head == root + 1 && matches!(parsed[i].pos.as_str(), "v" | "a") {
                    cands.push(i);
                }
            }
        }
        if cands.is_empty() && !mixed {
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
        .map(|(i, t)| (core[i], is_zh_connective(t)))
        .collect()
}

/// True for conjunctions (`c`) and for words from the association table.
///
/// Deliberately *not* conditional on the dependency label: LTP tags almost all
/// Chinese prepositions as `p` and a few verbs as `v`, and both `c` and the
/// table entries are already the right signal. A lone `以` (a preposition or a
/// bound morpheme in 以便/以免) is not a connective by itself.
pub fn is_zh_connective(t: &crate::ltp::ZhToken) -> bool {
    if t.pos == "c" {
        return true;
    }
    // ordinary prepositions are never connectives, even if they appear in the
    // table (none do today, but keep the guard explicit)
    if t.pos == "p" {
        return false;
    }
    ZH_CONNECTIVES.contains(&t.text.as_str())
}

/// In a mixed CN/EN sentence, a Chinese core predicate must be a verb or a
/// predicative adjective - an attributive adjective (`ATT`, e.g. 安静的河边)
/// is a modifier of the fragment's head noun and never the predicate.
fn is_zh_mixed_core(parsed: &[crate::ltp::ZhToken], i: usize) -> bool {
    let t = &parsed[i];
    match t.pos.as_str() {
        "v" => !is_copula(&t.text),
        "a" => t.deprel != "ATT",
        _ => false,
    }
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

/// Relation words that mark a relative / adverbial clause even when the tagger
/// does not emit `PronType=Rel` (small models often drop it on `that`).
pub const EN_REL_WORDS: &[&str] = &["who", "whom", "whose", "which", "that"];
/// Relativisers that are adverbial when they head a clause.
pub const EN_REL_ADVERBS: &[&str] = &["when", "where", "why", "whenever", "wherever"];

/// `(is_core_verb, is_connective, is_nonfinite)` per token of one English clause.
///
/// `mixed` says whether the *sentence* also contains Han script (the caller
/// knows; a single Latin run does not). In a mixed sentence only VERB/AUX
/// predicates count as the core predicate, so that a predicative `tired` in
/// "The tired 孩子们…" is not marked as the clause's verb. In a pure English
/// sentence a copular predicate nominal / adjective root is still the core
/// predicate (UD roots "She is tired" on `tired`).
pub fn derive_english(toks: &[crate::english::EnToken]) -> Vec<(bool, bool, bool)> {
    derive_english_ctx(toks, false)
}

pub fn derive_english_ctx(toks: &[crate::english::EnToken], mixed: bool) -> Vec<(bool, bool, bool)> {
    let n = toks.len();
    let mut out = vec![(false, false, false); n];
    if n == 0 {
        return out;
    }
    let root = (0..n).find(|&i| toks[i].head == 0).unwrap_or(0);

    // Core predicate:
    //   * UD attaches the copula *to* the predicate ("She is tired" ->
    //     tired:root, is:cop). In a pure English sentence the root is the core
    //     predicate when it is VERB/AUX or a predicative ADJ/NOUN.
    //   * If the tagger/parser rooted a copula and a complement hangs off it,
    //     the complement is the predicate.
    //   * Coordinated verbs (`conj`/`parataxis` chained to the root) are too.
    //   * In a *mixed* CN/EN sentence only VERB/AUX roots count: an English
    //     fragment glued to Chinese text ("The tired 孩子们…") tends to be
    //     parsed with an ADJ root whose subject sits in the Chinese part, which
    //     would otherwise be shown as the sentence's core predicate.
    let mut core = vec![false; n];
    let mut candidates: Vec<usize> = Vec::new();
    let root_ok = if mixed {
        is_verb_like(&toks[root])
    } else {
        is_verb_like(&toks[root]) || matches!(toks[root].upos.as_str(), "ADJ" | "NOUN" | "PROPN")
    };
    if root_ok {
        candidates.push(root);
    }
    // a clause whose root is a copula (ModelForTokenClassification may root "is")
    if is_verb_like(&toks[root])
        && toks
            .iter()
            .any(|t| t.deprel == "cop" && t.head == root + 1)
    {
        for i in 0..n {
            if i == root {
                continue;
            }
            let t = &toks[i];
            if t.head == root + 1 && t.deprel != "cop" && is_content_pos(&t.upos)
                && (!mixed || is_verb_like(t))
            {
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
        let vf = t
            .upos_tag
            .split('|')
            .find_map(|f| f.strip_prefix("VerbForm="))
            .unwrap_or("");
        let finite = vf == "Fin"
            || (t.upos == "VERB" && vf.is_empty() && t.deprel != "xcomp");
        let mut nonfinite = matches!(vf, "Inf" | "Ger" | "Part");
        // The model occasionally tags the copula/auxiliary of a *passive* or
        // progressive as a participle ("can be dangerous" -> be:Part). A
        // `cop|aux`-attached be/have is the auxiliary, not a non-finite verb.
        if nonfinite && matches!(t.deprel.as_str(), "cop" | "aux") {
            nonfinite = false;
        }
        // TO + VB: esupar occasionally leaves VerbForm off the infinitive.
        if !nonfinite && !finite && !core[i] && matches!(t.upos.as_str(), "VERB" | "AUX") {
            let h = t.head;
            let has_to = h != 0
                && h <= n
                && (toks[h - 1].text.eq_ignore_ascii_case("to")
                    || (t.upos == "VERB" && t.deprel == "xcomp"));
            if has_to {
                nonfinite = true;
            }
        }
        // a finite main predicate is never "non-finite"
        if core[i] || finite {
            nonfinite = false;
        }
        // Participles used purely as adjectives carry no VerbForm=Part (the
        // model emits `ADJ|Degree=Pos` for "the sleeping village"), so they are
        // not marked. A `VerbForm=Part` tag on a verbal use ("Exhausted from
        // the trip, …") *is* marked - that is the `vf` branch above.
        out[i] = (core[i], is_en_connective(toks, i), nonfinite);
    }
    out
}

/// Connective / clause-marker test for English (tightened vs. the first spike).
///
/// Included:
///   * `CCONJ` / `SCONJ` tags;
///   * dependency `cc`, `cc:preconj`, `mark` - **except** the infinitival `to`
///     (esupar sometimes labels it `mark`; it is PART/ADP + VerbForm=Inf);
///   * relative pronouns / adverbs: `PronType=Rel`, or who/whom/whose/which/
///     that attached as `nsubj|obj|obl|acl|acl:relcl|nmod|advcl|ref` (a `that`
///     inside "the book that you lent me" starts a relative clause);
///   * when/where/why attached as `advmod|advcl|acl(.relcl)` (relative adverb).
///
/// Excluded: prepositions (`ADP`, e.g. about/in/on) even when the parser gives
/// them `mark`, and infinitival `to` always.
pub fn is_en_connective(toks: &[crate::english::EnToken], i: usize) -> bool {
    let t = &toks[i];
    let lower = t.text.to_lowercase();
    // Common prepositions must never be connectives, even when the parser
    // tags them SCONJ/mark (e.g. "excited about visiting").
    // Do not list real subordinators: after/before/since/until/as/while/than.
    const PREP_NEVER_CONN: &[&str] = &[
        "about", "of", "with", "at", "from", "into", "onto", "over", "under",
        "by", "in", "on", "through", "between", "among", "against", "without",
        "during", "per", "via",
    ];
    if PREP_NEVER_CONN.contains(&lower.as_str()) {
        return false;
    }
    // Infinitival `to` is never a connective, whatever the parser calls it.
    let infinitival_to = lower == "to"
        && (matches!(t.upos.as_str(), "PART" | "ADP" | "SCONJ")
            || toks.get(i + 1).map(|n| n.upos_tag.contains("VerbForm=Inf")).unwrap_or(false));
    if lower == "to" || infinitival_to {
        return false;
    }
    let rel_dep = matches!(
        t.deprel.as_str(),
        "nsubj" | "nsubj:pass" | "obj" | "obl" | "acl" | "acl:relcl" | "nmod" | "advcl"
            | "ref" | "csubj" | "ccomp" | "mark"
    );
    // relative pronoun / determiner
    let pron_rel = t.upos_tag.contains("PronType=Rel")
        || (matches!(t.upos.as_str(), "PRON" | "DET" | "SCONJ")
            && EN_REL_WORDS.contains(&lower.as_str())
            && rel_dep);
    // relative adverb ("the place where we met", "the reason why …")
    let adv_rel = EN_REL_ADVERBS.contains(&lower.as_str())
        && matches!(t.deprel.as_str(), "advmod" | "advcl" | "acl" | "acl:relcl" | "mark");
    // prepositions (about/in/on/as …) are not connectives, even if the parser
    // tagged them `mark`; only an explicit SCONJ role (ExtPos=SCONJ) qualifies.
    if t.upos == "ADP" {
        return t.upos_tag.contains("ExtPos=SCONJ");
    }
    matches!(t.upos.as_str(), "CCONJ" | "SCONJ")
        || t.upos_tag.contains("ExtPos=SCONJ")
        || t.upos_tag.contains("ExtPos=CCONJ")
        || matches!(t.deprel.as_str(), "cc" | "cc:preconj" | "mark")
        || pron_rel
        || adv_rel
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::english::EnToken;
    use crate::ltp::ZhToken;

    fn zh(text: &str, pos: &str, head: usize, deprel: &str) -> ZhToken {
        ZhToken {
            text: text.into(),
            start: 0,
            end: 0,
            pos: pos.into(),
            head,
            deprel: deprel.into(),
            head_plain: head,
            deprel_plain: deprel.into(),
        }
    }

    /// `upos|FEATS` notation; `head` is 1-based, 0 == root.
    fn en(text: &str, upos_feats: &str, head: usize, deprel: &str) -> EnToken {
        EnToken {
            text: text.into(),
            lstart: 0,
            lend: 0,
            upos_tag: upos_feats.into(),
            upos: upos_feats.split('|').next().unwrap_or("").into(),
            head,
            deprel: deprel.into(),
        }
    }

    fn flags(ts: &[EnToken], mixed: bool) -> Vec<(bool, bool, bool)> {
        derive_english_ctx(ts, mixed)
    }

    // ---------------- Chinese coarse class / connectives ----------------

    #[test]
    fn zh_r_is_pron_not_adv() {
        assert_eq!(zh_coarse("r"), "PRON");
        assert_eq!(zh_coarse("d"), "ADV");
        // pronouns take the no-colour path (not NOUN/VERB/ADJ/ADV)
        assert_eq!(zh_coarse("r"), "PRON");
        assert_ne!(zh_coarse("r"), "ADV");
    }

    #[test]
    fn zh_connectives_vs_prepositions() {
        // conjunction tag
        assert!(is_zh_connective(&zh("和", "c", 3, "LAD")));
        // association-table adverb-ish connective
        assert!(is_zh_connective(&zh("因为", "c", 0, "HED")));
        assert!(is_zh_connective(&zh("但是", "d", 0, "HED")));
        // ordinary preposition / verb-ish word is *not* a connective
        assert!(!is_zh_connective(&zh("在", "p", 3, "PP")));
        assert!(!is_zh_connective(&zh("以", "p", 3, "PP")));
        assert!(!is_zh_connective(&zh("他", "r", 2, "SBV")));
    }

    // ---------------- English connectives ----------------

    #[test]
    fn en_to_is_not_a_connective() {
        // "want to go": to = PART in front of an infinitive
        let ts = [
            en("I", "PRON", 2, "nsubj"),
            en("want", "VERB|VerbForm=Fin", 0, "root"),
            en("to", "PART", 2, "mark"),
            en("go", "VERB|VerbForm=Inf", 2, "xcomp"),
        ];
        let f = flags(&ts, false);
        assert!(!f[2].1, "infinitival `to` must not be a connective");
        assert!(f[1].0, "want is the core predicate");
    }

    #[test]
    fn en_about_is_not_a_connective() {
        // "about" used as a preposition, even when the parser says `mark`
        // (an explicit ExtPos=SCONJ would be a legitimate subordinator)
        let ts = [
            en("She", "PRON", 2, "nsubj"),
            en("talked", "VERB|VerbForm=Fin", 0, "root"),
            en("about", "ADP|ExtPos=ADP", 2, "mark"),
            en("it", "PRON", 3, "obl"),
        ];
        assert!(!is_en_connective(&ts, 2));
        // plain ADP: never
        let ts2 = [
            en("She", "PRON", 2, "nsubj"),
            en("talked", "VERB|VerbForm=Fin", 0, "root"),
            en("about", "ADP", 2, "obl"),
        ];
        assert!(!is_en_connective(&ts2, 2));
        // esupar often tags "about visiting" as SCONJ/mark — still a preposition
        let ts3 = [
            en("excited", "ADJ", 0, "root"),
            en("about", "SCONJ", 3, "mark"),
            en("visiting", "VERB|VerbForm=Ger", 1, "advcl"),
        ];
        assert!(!is_en_connective(&ts3, 1));
    }

    #[test]
    fn en_conjunctions_and_relativisers_are_connectives() {
        let ts = [
            en("Tom", "PROPN", 2, "nsubj"),
            en("slept", "VERB|VerbForm=Fin", 0, "root"),
            en("and", "CCONJ", 2, "cc"),
            en("Mary", "PROPN", 6, "nsubj"),
            en("read", "VERB|VerbForm=Fin", 2, "conj"),
            en("because", "SCONJ", 7, "mark"),
            en("it", "PRON", 7, "nsubj"),
            en("rained", "VERB|VerbForm=Fin", 5, "advcl"),
        ];
        let f = flags(&ts, false);
        assert!(f[2].1, "and (CCONJ/cc)");
        assert!(f[5].1, "because (SCONJ/mark)");

        let ts = [
            en("The", "DET", 2, "det"),
            en("man", "NOUN", 5, "nsubj"),
            en("who", "PRON", 5, "nsubj"),
            en("lives", "VERB|VerbForm=Fin", 2, "acl:relcl"),
            en("next", "ADV", 5, "advmod"),
            en("door", "NOUN", 0, "root"),
        ];
        assert!(is_en_connective(&ts, 2), "who (relative pronoun)");

        // `that` as a relative-clause marker (no PronType=Rel on small models)
        let ts = [
            en("The", "DET", 2, "det"),
            en("book", "NOUN", 5, "nsubj"),
            en("that", "PRON", 4, "obj"),
            en("you", "PRON", 4, "nsubj"),
            en("lent", "VERB|VerbForm=Fin", 2, "acl:relcl"),
            en("me", "PRON", 0, "root"),
        ];
        assert!(is_en_connective(&ts, 2), "that (relative)");
        // `although` / `while` as SCONJ
        let ts = [
            en("although", "SCONJ", 3, "mark"),
            en("it", "PRON", 3, "nsubj"),
            en("rained", "VERB|VerbForm=Fin", 0, "advcl"),
            en("they", "PRON", 5, "nsubj"),
            en("left", "VERB|VerbForm=Fin", 2, "advcl"),
        ];
        assert!(is_en_connective(&ts, 0));
        // when/where as relative adverbs
        let ts = [
            en("the", "DET", 2, "det"),
            en("place", "NOUN", 0, "root"),
            en("where", "ADV", 4, "advmod"),
            en("we", "PRON", 4, "nsubj"),
            en("met", "VERB|VerbForm=Fin", 2, "acl:relcl"),
        ];
        assert!(is_en_connective(&ts, 2), "where (relative adverb)");
    }

    // ---------------- core predicate ----------------

    #[test]
    fn mixed_sentence_adjective_root_is_not_core() {
        // "The tired 孩子们…": parse of the Latin fragment "The tired"
        let ts = [
            en("The", "DET", 2, "det"),
            en("tired", "ADJ|Degree=Pos", 0, "root"),
        ];
        let mixed = flags(&ts, true);
        assert!(!mixed[1].0, "ADJ root must not be core in a mixed sentence");
        let pure = flags(&ts, false);
        assert!(pure[1].0, "in a pure English copular clause the ADJ root is the predicate");
    }

    #[test]
    fn real_verbs_stay_core_in_mixed_sentences() {
        for (w, tag) in [
            ("opened", "VERB|Mood=Ind|Tense=Past|VerbForm=Fin"),
            ("closed", "VERB|Mood=Ind|Tense=Past|VerbForm=Fin"),
            ("walk", "VERB|VerbForm=Inf"),
            ("tell", "VERB|VerbForm=Inf"),
        ] {
            let ts = [
                en("I", "PRON", 2, "nsubj"),
                en("gently", "ADV", 2, "advmod"),
                en(w, tag, 0, "root"),
            ];
            let f = flags(&ts, true);
            assert!(f[2].0, "{w} must be the core predicate even in a mixed sentence");
        }
    }

    #[test]
    fn mixed_chinese_attributive_adjective_is_not_core() {
        // "安静的河边" as a fragment: root = 河边, 安静 is ATT
        let parsed = vec![
            zh("安静", "a", 3, "ATT"),
            zh("的", "u", 1, "RAD"),
            zh("河边", "n", 0, "HED"),
        ];
        let f = derive_chinese_ctx(&parsed, true);
        assert!(!f[0].0, "ATT adjective must not be core in a mixed sentence");
        assert!(!f[2].0, "nominal root is not a predicate");
        // and 孩子们已经慢慢地睡着了，so … - the real predicate 睡 is core
        let parsed = vec![
            zh("孩子", "n", 4, "SBV"),
            zh("们", "k", 1, "RAD"),
            zh("已经", "d", 4, "ADV"),
            zh("慢慢地", "z", 4, "ADV"),
            zh("睡着", "v", 0, "HED"),
            zh("了", "u", 5, "RAD"),
        ];
        let f = derive_chinese_ctx(&parsed, true);
        assert!(f[4].0);
    }

    // ---------------- non-finite ----------------

    #[test]
    fn nonfinite_inf_ger_part_are_marked() {
        let ts = [
            en("She", "PRON", 2, "nsubj"),
            en("decided", "VERB|VerbForm=Fin", 0, "root"),
            en("to", "PART", 4, "mark"),
            en("leave", "VERB|VerbForm=Inf", 2, "xcomp"),
            en("the", "DET", 6, "det"),
            en("party", "NOUN", 4, "obj"),
        ];
        let f = flags(&ts, false);
        assert!(f[3].2, "to leave = Inf");
        assert!(!f[1].2, "finite main predicate is not non-finite");
        assert!(!f[2].2, "`to` itself is not marked");

        let ts = [
            en("Swimming", "VERB|VerbForm=Ger", 6, "csubj"),
            en("is", "AUX|VerbForm=Fin", 5, "cop"),
            en("not", "PART", 5, "advmod"),
            en("a", "DET", 5, "det"),
            en("good", "ADJ", 5, "amod"),
            en("idea", "NOUN", 0, "root"),
        ];
        let f = flags(&ts, false);
        assert!(f[0].2, "gerund Swimming (csubj, not the finite predicate)");
        assert!(!f[0].0, "a gerund subject is not the core predicate");
        // verbal use of a participle ("Exhausted from the trip, …")
        let ts = [
            en("Exhausted", "VERB|Tense=Past|VerbForm=Part", 5, "advcl"),
            en("the", "DET", 2, "det"),
            en("children", "NOUN", 5, "nsubj"),
            en("fell", "VERB|VerbForm=Fin", 0, "root"),
            en("asleep", "ADV", 3, "advmod"),
        ];
        let f = flags(&ts, false);
        assert!(f[0].2, "VerbForm=Part → non-finite");
        assert!(!f[3].2);
    }

    #[test]
    fn adjectival_participle_without_verbform_is_not_nonfinite() {
        // "the sleeping village": the model gives ADJ|Degree=Pos, no VerbForm
        let ts = [
            en("the", "DET", 2, "det"),
            en("sleeping", "ADJ|Degree=Pos", 3, "amod"),
            en("village", "NOUN", 0, "root"),
        ];
        let f = flags(&ts, false);
        assert!(!f[1].2, "pure adjective use is not non-finite");
        assert!(!f[1].1, "and not a connective");
    }
}
