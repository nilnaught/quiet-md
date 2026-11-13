//! Derived reader marks: core predicate, connective, non-finite verb.
//!
//! These run on top of the dependency parses (neural) or on the fallback's
//! rough POS stream. They are deliberately conservative: a wrong underline or
//! a wrong emphasis dot is worse than a missing one.
//!
//! Two entry points exist for the English path:
//! * [`derive_english_ctx`] — the full rule set over `EnToken` (tagger UPOS +
//!   FEATS + SuPar heads/deprels);
//! * [`derive_english_coarse`] — the same public shape over a flat UPOS list
//!   with no dependencies, used by the fallback [(jieba-rs +
//!   english-pos-tagger)].
//!
//! Chinese rules are unchanged from the spike (see [`derive_chinese_ctx`]).

use crate::english::EnToken;
use crate::ltp::ZhToken;

/// `(is_core, is_connective, is_nonfinite)` per token.
pub type Flags = (bool, bool, bool);

// ------------------------------------------------------------------------- //
// Chinese
// ------------------------------------------------------------------------- //

/// Chinese connectives that LTP does not always tag as `c`.
pub const ZH_CONNECTIVES: &[&str] = &[
    "因为", "所以", "因此", "因而", "于是", "虽然", "但是", "但", "可是", "然而", "不过", "如果",
    "假如", "要是", "那么", "不仅", "而且", "并且", "尽管", "即使", "无论", "不管", "只要",
    "只有", "除非", "否则", "或者", "还是", "以及", "不但", "同时", "然后", "接着", "首先", "其次",
    "最后", "既然", "由于", "以便", "以免", "与其", "宁可", "之所以", "何况", "另外", "此外",
];

/// Same, but `mixed` says the surrounding *sentence* also has Latin script.
///
/// In a mixed CN/EN sentence the Chinese clause is often a fragment
/// ("…的 安静的河边，") that LTP roots on an attributive-ish adjective. Marking
/// that as the core predicate would show 《安静》 as the sentence's predicate,
/// so in mixed sentences only a root/coordinated predicate that is a verb or a
/// predicative adjective (not `ATT`) counts - and the copula fallback is off.
pub fn derive_chinese_ctx(parsed: &[ZhToken], mixed: bool) -> Vec<(bool, bool)> {
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
pub fn is_zh_connective(t: &ZhToken) -> bool {
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
fn is_zh_mixed_core(parsed: &[ZhToken], i: usize) -> bool {
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

fn is_zh_predicate(parsed: &[ZhToken], i: usize) -> bool {
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

// ------------------------------------------------------------------------- //
// English
// ------------------------------------------------------------------------- //

/// Relation words that mark a relative / adverbial clause even when the tagger
/// does not emit `PronType=Rel` (small models often drop it on `that`).
pub const EN_REL_WORDS: &[&str] = &["who", "whom", "whose", "which", "that"];
/// Relativisers that are adverbial when they head a clause.
pub const EN_REL_ADVERBS: &[&str] = &["when", "where", "why", "whenever", "wherever"];
/// Modal / auxiliary words whose presence makes a following verb a finite
/// (grammatical) verb, whatever the FEATS claim.
pub const EN_MODALS: &[&str] = &[
    "will", "would", "can", "could", "should", "must", "may", "might", "shall", "ought",
];
/// Have/be forms: an auxiliary chain of these keeps the main verb finite.
pub const EN_HAVE_BE: &[&str] = &[
    "have", "has", "had", "having", "be", "am", "is", "are", "was", "were", "been", "being",
];
/// English conjunction word list (fallback path; the neural path trusts CCONJ
/// and `cc`/`mark` deprels, this covers scripts where the tagger is unsure).
pub const EN_CONNECTIVES: &[&str] = &[
    "and", "or", "but", "nor", "so", "yet", "because", "although", "though", "if", "when",
    "while", "unless", "since", "whereas", "whether", "until", "after", "before", "once",
    "whenever", "wherever", "as", "than",
];

fn feats_verb_form(tag: &str) -> &str {
    tag.split('|')
        .find_map(|f| f.strip_prefix("VerbForm="))
        .unwrap_or("")
}

/// `(is_core, is_connective, is_nonfinite)` per token of one English clause.
///
/// `mixed` says whether the *sentence* also contains Han script (the caller
/// knows; a single Latin run does not). In a mixed sentence only VERB/AUX
/// predicates count as the core predicate, so that a predicative `tired` in
/// "The tired 孩子们…" is not marked as the clause's verb. In a copular
/// clause the emphasis sits on the copula (`is`/`are`/`was`…), not on the
/// predicate nominal or adjective that UD often roots.
pub fn derive_english_ctx(toks: &[EnToken], mixed: bool) -> Vec<Flags> {
    let n = toks.len();
    let mut out = vec![(false, false, false); n];
    if n == 0 {
        return out;
    }
    let root = (0..n).find(|&i| toks[i].head == 0).unwrap_or(0);
    let core = english_core(toks, root, mixed);

    for i in 0..n {
        let nonfinite = is_nonfinite(toks, i, core[i]);
        out[i] = (core[i], is_en_connective(toks, i), nonfinite);
    }
    out
}

/// Core predicate selection (see the doc on [`derive_english_ctx`]).
fn english_core(toks: &[EnToken], root: usize, mixed: bool) -> Vec<bool> {
    let n = toks.len();
    let mut core = vec![false; n];
    let mut candidates: Vec<usize> = Vec::new();
    let mut core_root = root;
    if let Some(cop) = copula_of(toks, root) {
        // "She is a famous writer" / "The cup is still on the table": UD roots
        // the predicate nominal and attaches the copula as `cop`. The emphasis
        // belongs on the copula, never on the noun/adjective it links to.
        // A verbal complement is handled by the copula-root branch below.
        core_root = cop;
    }
    let root_ok = if mixed {
        is_verb_like(&toks[core_root])
    } else {
        is_verb_like(&toks[core_root])
            || matches!(toks[core_root].upos.as_str(), "ADJ" | "NOUN" | "PROPN")
    };
    if root_ok {
        candidates.push(core_root);
    }
    // A clause whose root is a copula (`is a famous writer`, `is not a good
    // idea`): the emphasis is on the copula itself, not on the predicate
    // nominal/adjective. Only a verbal complement (a real verb hanging off the
    // copula) can carry the core instead.
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
            if t.head == root + 1 && is_verb_like(t) && (!mixed || is_verb_like(t)) {
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
    core
}

/// The `cop` child of `root`, if the root is the predicate of a copular
/// clause ("She is a famous writer": writer:root, is:cop).
fn copula_of(toks: &[EnToken], root: usize) -> Option<usize> {
    // UD often roots the predicate nominal/adjective (`writer`, `tired`) and
    // hangs the copula off it as `cop`. Look for that child regardless of the
    // root's POS; if the copula itself is already the root there is no `cop`
    // child and the caller keeps the verbal root.
    (0..toks.len()).find(|&i| {
        i != root && toks[i].head == root + 1 && toks[i].deprel == "cop" && is_verb_like(&toks[i])
    })
}

/// True when `t` is a participial clause lead-in (`Having finished…`).
fn is_participial_lead(t: &EnToken) -> bool {
    matches!(t.text.to_lowercase().as_str(), "having" | "being")
        && (matches!(t.upos.as_str(), "AUX" | "VERB")
            || feats_verb_form(&t.upos_tag) == "Ger")
}

/// True when token `i` has a modal or finite have/be auxiliary attached to it.
/// `Having`/`being` as a participial lead-in does not count: that clause is
/// non-finite, not a finite perfect/progressive chain.
fn has_aux(toks: &[EnToken], i: usize) -> bool {
    let is_finite_aux_lex = |a: &EnToken| {
        if is_participial_lead(a) {
            return false;
        }
        let lower = a.text.to_lowercase();
        EN_MODALS.contains(&lower.as_str()) || EN_HAVE_BE.contains(&lower.as_str())
    };
    let child_aux = toks.iter().any(|a| {
        a.head == i + 1
            && matches!(a.deprel.as_str(), "aux" | "cop" | "aux:pass")
            && is_finite_aux_lex(a)
    });
    // The modal/have/be word may itself be the clause root (`will come` with
    // `will:root`, `come:xcomp`); do not require its deprel to be `aux`.
    let head_aux = toks[i].head != 0
        && toks.get(toks[i].head - 1).is_some_and(|h| is_finite_aux_lex(h) && is_verb_like(h));
    child_aux || head_aux
}

/// Non-finite test for token `i` (never true for the core predicate).
fn is_nonfinite(toks: &[EnToken], i: usize, is_core: bool) -> bool {
    let t = &toks[i];
    let n = toks.len();
    let vf = feats_verb_form(&t.upos_tag);
    let finite = vf == "Fin" || (t.upos == "VERB" && vf.is_empty() && t.deprel != "xcomp");
    let mut nonfinite = matches!(vf, "Inf" | "Ger" | "Part");

    // (a) modal / auxiliary + verb: *she will come*, *can be dangerous*.
    //     The tagger often calls the governed verb VerbForm=Inf (or Part).
    if nonfinite && has_aux(toks, i) {
        nonfinite = false;
    }
    // (b) an `aux`/`cop`-attached be/have is the auxiliary itself, not a
    //     non-finite verb.
    if nonfinite && matches!(t.deprel.as_str(), "cop" | "aux" | "aux:pass") {
        nonfinite = false;
    }
    // (d) TO + VB when the tagger left VerbForm off the infinitive. Do not
    //     treat every `xcomp` as infinitival: `will come` / `have been
    //     working` are also xcomp in some parses.
    if !nonfinite && !finite && matches!(t.upos.as_str(), "VERB" | "AUX") && !has_aux(toks, i) {
        let h = t.head;
        let headed_by_to = h != 0 && h <= n && toks[h - 1].text.eq_ignore_ascii_case("to");
        let has_to_mark = toks.iter().any(|m| {
            m.head == i + 1
                && m.text.eq_ignore_ascii_case("to")
                && matches!(m.deprel.as_str(), "mark" | "aux")
        });
        if headed_by_to || has_to_mark {
            nonfinite = true;
        }
    }
    // (c) participle / gerund clause: `Having finished dinner, …`, `likes
    //     swimming`. Skip when a finite auxiliary chain is attached (so
    //     `have been working` is not marked).
    if !nonfinite
        && matches!(t.deprel.as_str(), "advcl" | "xcomp" | "acl" | "acl:relcl")
        && matches!(vf, "Part" | "Ger")
        && !has_aux(toks, i)
    {
        nonfinite = true;
    }
    // (c') …even when the model mislabels `finished` as VerbForm=Fin: if its
    // head, or a child auxiliary, is a participial lead-in ("Having",
    // "having said", …), it is part of a non-finite clause.
    let participial_clause = head_is_participial_lead(toks, i) || has_participial_lead_aux(toks, i);
    if !nonfinite && participial_clause {
        nonfinite = true;
    }
    // a finite main predicate is never "non-finite". A mislabelled Fin on a
    // participial clause verb must not clear the flag set in (c').
    if is_core || (finite && !participial_clause) {
        nonfinite = false;
    }
    nonfinite
}

fn head_is_participial_lead(toks: &[EnToken], i: usize) -> bool {
    let h = toks[i].head;
    if h == 0 || h > toks.len() {
        return false;
    }
    is_participial_lead(&toks[h - 1])
}

fn has_participial_lead_aux(toks: &[EnToken], i: usize) -> bool {
    toks.iter().any(|a| a.head == i + 1 && matches!(a.deprel.as_str(), "aux" | "aux:pass") && is_participial_lead(a))
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
pub fn is_en_connective(toks: &[EnToken], i: usize) -> bool {
    let t = &toks[i];
    let lower = t.text.to_lowercase();
    // Common prepositions must never be connectives, even when the parser
    // tags them SCONJ/mark (e.g. "excited about visiting").
    // Do not list real subordinators: after/before/since/until/as/while/than.
    const PREP_NEVER_CONN: &[&str] = &[
        "about", "of", "with", "at", "from", "into", "onto", "over", "under", "by", "in", "on",
        "through", "between", "among", "against", "without", "during", "per", "via",
    ];
    if PREP_NEVER_CONN.contains(&lower.as_str()) {
        return false;
    }
    // Infinitival `to` is never a connective, whatever the parser calls it.
    let infinitival_to = lower == "to"
        && (matches!(t.upos.as_str(), "PART" | "ADP" | "SCONJ")
            || toks
                .get(i + 1)
                .map(|n| n.upos_tag.contains("VerbForm=Inf"))
                .unwrap_or(false));
    if lower == "to" || infinitival_to {
        return false;
    }
    let rel_dep = matches!(
        t.deprel.as_str(),
        "nsubj"
            | "nsubj:pass"
            | "obj"
            | "obl"
            | "acl"
            | "acl:relcl"
            | "nmod"
            | "advcl"
            | "ref"
            | "csubj"
            | "ccomp"
            | "mark"
    );
    // relative pronoun / determiner
    let pron_rel = t.upos_tag.contains("PronType=Rel")
        || (matches!(t.upos.as_str(), "PRON" | "DET" | "SCONJ")
            && EN_REL_WORDS.contains(&lower.as_str())
            && rel_dep);
    // relative adverb ("the place where we met", "the reason why …")
    let adv_rel = EN_REL_ADVERBS.contains(&lower.as_str())
        && matches!(
            t.deprel.as_str(),
            "advmod" | "advcl" | "acl" | "acl:relcl" | "mark"
        );
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

fn is_verb_like(t: &EnToken) -> bool {
    matches!(t.upos.as_str(), "VERB" | "AUX")
}

// ------------------------------------------------------------------------- //
// English, no dependencies (fallback UPOS stream)
// ------------------------------------------------------------------------- //

/// Core / non-finite over a flat UPOS stream (Penn tags or UD UPOS, matched
/// case-insensitively). No dependency information is available, so:
///
/// * core = the sentence's first main verb in a finite/tense-bearing class
///   (`VB`, `VBD`, `VBP`, `VBZ`; `MD` is skipped), never an adjective;
/// * non-finite is conservative: only a verb immediately after infinitival
///   `to`. Modal + VB and have/be + VBG/VBN chains stay finite.
///
/// Returns flags parallel to `upos`; connectives come from [`EN_CONNECTIVES`]
/// and the CCONJ/SCONJ classes.
pub fn derive_english_coarse(words: &[(&str, &str)]) -> Vec<Flags> {
    let n = words.len();
    let mut out = vec![(false, false, false); n];
    // first finite main verb
    let mut core_idx = None;
    for (i, (_, tag)) in words.iter().enumerate() {
        let t = tag.to_ascii_uppercase();
        if matches!(t.as_str(), "VB" | "VBD" | "VBP" | "VBZ") {
            core_idx = Some(i);
            break;
        }
    }
    let mut last_was_to = false;
    for (i, (w, tag)) in words.iter().enumerate() {
        let t = tag.to_ascii_uppercase();
        let lower = w.to_lowercase();
        let is_core = core_idx == Some(i);
        let connective = matches!(t.as_str(), "CC" | "CCONJ" | "SCONJ")
            || (t == "IN" && EN_CONNECTIVES.contains(&lower.as_str()));
        // Conservative: only mark a verb right after infinitival `to`.
        let nonfinite = !is_core && last_was_to && matches!(t.as_str(), "VB" | "VBG" | "VBN" | "VERB");
        last_was_to = lower == "to";
        out[i] = (is_core, connective, nonfinite);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zh(text: &str, pos: &str, head: usize, deprel: &str) -> ZhToken {
        ZhToken {
            text: text.into(),
            start: 0,
            end: 0,
            pos: pos.into(),
            head,
            deprel: deprel.into(),
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

    fn flags(ts: &[EnToken], mixed: bool) -> Vec<Flags> {
        derive_english_ctx(ts, mixed)
    }

    fn index_of(ts: &[EnToken], word: &str) -> usize {
        ts.iter().position(|t| t.text == word).unwrap_or_else(|| panic!("no {word}"))
    }

    // ---------------- Chinese coarse class / connectives ----------------

    #[test]
    fn zh_r_is_pron_not_adv() {
        assert_eq!(crate::neural::zh_coarse("r"), "PRON");
        assert_eq!(crate::neural::zh_coarse("d"), "ADV");
        assert_ne!(crate::neural::zh_coarse("r"), "ADV");
    }

    #[test]
    fn zh_connectives_vs_prepositions() {
        assert!(is_zh_connective(&zh("和", "c", 3, "LAD")));
        assert!(is_zh_connective(&zh("因为", "c", 0, "HED")));
        assert!(is_zh_connective(&zh("但是", "d", 0, "HED")));
        assert!(!is_zh_connective(&zh("在", "p", 3, "PP")));
        assert!(!is_zh_connective(&zh("以", "p", 3, "PP")));
        assert!(!is_zh_connective(&zh("他", "r", 2, "SBV")));
    }

    #[test]
    fn mixed_chinese_attributive_adjective_is_not_core() {
        let parsed = vec![
            zh("安静", "a", 3, "ATT"),
            zh("的", "u", 1, "RAD"),
            zh("河边", "n", 0, "HED"),
        ];
        let f = derive_chinese_ctx(&parsed, true);
        assert!(!f[0].0, "ATT adjective must not be core in a mixed sentence");
        assert!(!f[2].0, "nominal root is not a predicate");
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

    // ---------------- connectives ----------------

    #[test]
    fn en_to_is_not_a_connective() {
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
        let ts = [
            en("She", "PRON", 2, "nsubj"),
            en("talked", "VERB|VerbForm=Fin", 0, "root"),
            en("about", "ADP|ExtPos=ADP", 2, "mark"),
            en("it", "PRON", 3, "obl"),
        ];
        assert!(!is_en_connective(&ts, 2));
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
    }

    // ---------------- core predicate ----------------

    #[test]
    fn copula_is_core_not_the_predicate_nominal() {
        // "She is a famous writer." — UD roots `writer`, `is` is `cop`.
        let ts = [
            en("She", "PRON", 5, "nsubj"),
            en("is", "AUX|VerbForm=Fin", 5, "cop"),
            en("a", "DET", 5, "det"),
            en("famous", "ADJ", 5, "amod"),
            en("writer", "NOUN", 0, "root"),
            en(".", "PUNCT", 5, "punct"),
        ];
        let f = flags(&ts, false);
        let is = index_of(&ts, "is");
        let writer = index_of(&ts, "writer");
        assert!(f[is].0, "the copula carries the core dot");
        assert!(!f[writer].0, "the predicate nominal is not the core");

        // "The cup is still on the table."
        let ts = [
            en("The", "DET", 2, "det"),
            en("cup", "NOUN", 3, "nsubj"),
            en("is", "AUX|VerbForm=Fin", 0, "root"),
            en("still", "ADV", 3, "advmod"),
            en("on", "ADP", 6, "case"),
            en("the", "DET", 6, "det"),
            en("table", "NOUN", 3, "obl"),
        ];
        let f = flags(&ts, false);
        let is = index_of(&ts, "is");
        let table = index_of(&ts, "table");
        assert!(f[is].0);
        assert!(!f[table].0);

        // "It is not a good idea." — copula-rooted parse
        let ts = [
            en("It", "PRON", 2, "nsubj"),
            en("is", "AUX|VerbForm=Fin", 0, "root"),
            en("not", "PART", 5, "advmod"),
            en("a", "DET", 5, "det"),
            en("good", "ADJ", 5, "amod"),
            en("idea", "NOUN", 2, "attr"),
        ];
        let f = flags(&ts, false);
        assert!(f[index_of(&ts, "is")].0);
        assert!(!f[index_of(&ts, "idea")].0);
    }

    #[test]
    fn mixed_sentence_adjective_root_is_not_core() {
        // "The tired 孩子们…": parse of the Latin fragment "The tired"
        let ts = [en("The", "DET", 2, "det"), en("tired", "ADJ|Degree=Pos", 0, "root")];
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

    // ---------------- non-finite ----------------

    #[test]
    fn modal_plus_verb_is_not_nonfinite() {
        // "She will come." — UD: come:root, will:aux; the tagger may still say Inf.
        let ts = [
            en("She", "PRON", 3, "nsubj"),
            en("will", "AUX|VerbForm=Fin", 3, "aux"),
            en("come", "VERB|VerbForm=Inf", 0, "root"),
        ];
        let f = flags(&ts, false);
        let come = index_of(&ts, "come");
        assert!(!f[come].2, "`come` after `will` is finite");
        assert!(f[come].0, "and it is the clause's main verb");

        // alternate attachment: will is the root, come hangs off it as xcomp
        let ts = [
            en("She", "PRON", 2, "nsubj"),
            en("will", "AUX|VerbForm=Fin", 0, "root"),
            en("come", "VERB|VerbForm=Inf", 2, "xcomp"),
        ];
        let f = flags(&ts, false);
        assert!(!f[index_of(&ts, "come")].2);
        for aux in ["can", "should", "must", "could", "would", "may", "might", "shall"] {
            let ts = [
                en("They", "PRON", 3, "nsubj"),
                en(aux, "AUX|VerbForm=Fin", 3, "aux"),
                en("work", "VERB|VerbForm=Inf", 0, "root"),
            ];
            let f = flags(&ts, false);
            assert!(!f[index_of(&ts, "work")].2, "{aux} + work must not be non-finite");
        }
    }

    #[test]
    fn have_been_working_chain_is_not_nonfinite() {
        // "They have been working on this project."
        let ts = [
            en("They", "PRON", 2, "nsubj"),
            en("have", "AUX|VerbForm=Fin", 0, "root"),
            en("been", "AUX|VerbForm=Part", 2, "aux"),
            en("working", "VERB|VerbForm=Ger", 2, "xcomp"),
            en("on", "ADP", 2, "case"),
            en("this", "DET", 7, "det"),
            en("project", "NOUN", 2, "obl"),
        ];
        let f = flags(&ts, false);
        let working = index_of(&ts, "working");
        assert!(!f[working].2, "`working` in a finite perfect-progressive chain is not non-finite");
        assert!(!f[working].0, "the root `have` carries the core in this parse");
        // alternate parse: working is the root with have/been as aux children
        let ts = [
            en("They", "PRON", 4, "nsubj"),
            en("have", "AUX|VerbForm=Fin", 4, "aux"),
            en("been", "AUX|VerbForm=Part", 4, "aux"),
            en("working", "VERB|VerbForm=Ger", 0, "root"),
        ];
        let f = flags(&ts, false);
        assert!(!f[3].2);
        assert!(f[3].0);
    }

    #[test]
    fn having_finished_participle_clause_is_nonfinite() {
        // "Having finished dinner, they started a movie."
        let ts = [
            en("Having", "AUX|VerbForm=Ger", 2, "aux"),
            en("finished", "VERB|VerbForm=Fin", 6, "advcl"),
            en("dinner", "NOUN", 2, "obj"),
            en(",", "PUNCT", 6, "punct"),
            en("they", "PRON", 6, "nsubj"),
            en("started", "VERB|VerbForm=Fin", 0, "root"),
            en("a", "DET", 8, "det"),
            en("movie", "NOUN", 6, "obj"),
            en(".", "PUNCT", 6, "punct"),
        ];
        let f = flags(&ts, false);
        assert!(f[index_of(&ts, "finished")].2, "finished is non-finite");
        assert!(f[index_of(&ts, "started")].0, "started is the core");
        assert!(!f[index_of(&ts, "started")].2, "started is finite");
    }

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
        let ts = [
            en("the", "DET", 2, "det"),
            en("sleeping", "ADJ|Degree=Pos", 3, "amod"),
            en("village", "NOUN", 0, "root"),
        ];
        let f = flags(&ts, false);
        assert!(!f[1].2, "pure adjective use is not non-finite");
        assert!(!f[1].1, "and not a connective");
    }

    #[test]
    fn coarse_fallback_marks() {
        // Penn tags from english-pos-tagger for "She will come."
        let words = [("She", "PRP"), ("will", "MD"), ("come", "VB"), (".", ".")];
        let f = derive_english_coarse(&words);
        assert!(!f[2].2, "come after a modal is not non-finite");
        assert!(f[2].0, "come is the main verb");
        // "They have been working on this project."
        let words = [
            ("They", "PRP"),
            ("have", "VBP"),
            ("been", "VBN"),
            ("working", "VBG"),
            ("on", "IN"),
            ("this", "DT"),
            ("project", "NN"),
        ];
        let f = derive_english_coarse(&words);
        assert!(f[1].0, "have is the first finite verb");
        assert!(!f[3].2, "working is not non-finite in a have+been chain");
        // "She is a famous writer." — the adjective is never core
        let words = [
            ("She", "PRP"),
            ("is", "VBZ"),
            ("a", "DT"),
            ("famous", "JJ"),
            ("writer", "NN"),
        ];
        let f = derive_english_coarse(&words);
        assert!(f[1].0);
        assert!(!f[3].0);
        assert!(!f[4].0);
    }
}
