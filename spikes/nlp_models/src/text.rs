//! Sentence segmentation + Han/Latin script splitting.
//!
//! The CLI works on *sentences*: `icu_segmenter` splits the raw input, then each
//! sentence is split into runs of "Chinese" (Han) vs "Latin" characters. Latin
//! runs of one sentence are joined with a single space, sent to the English
//! pipeline as one string, and every English token is mapped back to a byte
//! range of the **original** input.
//!
//! Invariant: for every emitted token, `&input[start..end] == token_text`
//! (with English whitespace tokens dropped, since the joiner space is synthetic).

use icu_segmenter::SentenceSegmenter;
use unicode_script::{Script, UnicodeScript};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptKind {
    Han,
    Latin,
}

/// A byte range `[start, end)` into the original input plus its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(text: &str, start: usize, end: usize) -> Self {
        Self {
            text: text[start..end].to_string(),
            start,
            end,
        }
    }
}

/// Split text into sentences (ICU), keeping byte offsets, trimming surrounding
/// whitespace.
pub fn sentences(text: &str) -> Vec<Span> {
    let seg = SentenceSegmenter::new(Default::default());
    let mut bounds: Vec<usize> = seg.segment_str(text).collect();
    if bounds.first() != Some(&0) {
        bounds.insert(0, 0);
    }
    if bounds.last() != Some(&text.len()) {
        bounds.push(text.len());
    }
    let mut out = Vec::new();
    let mut prev = 0usize;
    for b in bounds {
        if b <= prev {
            continue;
        }
        let lead = text[prev..b].len() - text[prev..b].trim_start().len();
        let trail = text[prev..b].len() - text[prev..b].trim_end().len();
        let (s, e) = (prev + lead, b - trail);
        if s < e {
            out.push(Span::new(text, s, e));
        }
        prev = b;
    }
    out
}

fn char_script(c: char) -> Option<ScriptKind> {
    match c.script() {
        Script::Han => Some(ScriptKind::Han),
        Script::Latin if c.is_alphabetic() => Some(ScriptKind::Latin),
        _ if c.is_ascii_alphabetic() => Some(ScriptKind::Latin),
        _ => None,
    }
}

/// Per-char script labels with punctuation/digits/space resolved to a neighbour
/// (forward fill, then backward for leading neutrals).
fn script_per_char(sentence: &str) -> Vec<Option<ScriptKind>> {
    let mut kinds: Vec<Option<ScriptKind>> = sentence.chars().map(char_script).collect();
    let mut last = None;
    for k in kinds.iter_mut() {
        match k {
            Some(s) => last = Some(*s),
            None => *k = last,
        }
    }
    let mut next = None;
    for k in kinds.iter_mut().rev() {
        match k {
            Some(s) => next = Some(*s),
            None => *k = next,
        }
    }
    kinds
}

/// Split one sentence into script runs (byte offsets inside `sentence`).
pub fn split_scripts(sentence: &str) -> Vec<(ScriptKind, Span)> {
    let kinds = script_per_char(sentence);
    let mut runs: Vec<(ScriptKind, Span)> = Vec::new();
    let mut cur: Option<(ScriptKind, usize, usize)> = None;
    for ((byte, c), kind) in sentence.char_indices().zip(kinds) {
        let Some(kind) = kind else { continue };
        let end = byte + c.len_utf8();
        match cur {
            Some((k, s, _)) if k == kind => cur = Some((k, s, end)),
            Some((k, s, e)) => {
                runs.push((k, Span::new(sentence, s, e)));
                cur = Some((kind, byte, end));
            }
            None => cur = Some((kind, byte, end)),
        }
    }
    if let Some((k, s, e)) = cur {
        runs.push((k, Span::new(sentence, s, e)));
    }
    runs
}

/// The English input for a sentence plus a map back to original byte offsets.
#[derive(Debug, Clone)]
pub struct LatinBatch {
    /// The string handed to the English tokenizer.
    pub text: String,
    /// For each byte of `text`, the original byte offset of that byte, or
    /// `None` for the synthetic joiner spaces.
    origin: Vec<Option<usize>>,
}

impl LatinBatch {
    pub fn from_runs(runs: &[Span]) -> Self {
        let mut text = String::new();
        let mut origin: Vec<Option<usize>> = Vec::new();
        let mut first = true;
        for r in runs {
            // trim the run's own edge whitespace; the joiner supplies the space
            let trimmed = r.text.trim();
            if trimmed.is_empty() {
                continue;
            }
            let lead = r.text.len() - r.text.trim_start().len();
            if !first {
                text.push(' ');
                origin.push(None);
            }
            first = false;
            text.push_str(trimmed);
            for (off, c) in trimmed.char_indices() {
                for _ in 0..c.len_utf8() {
                    origin.push(Some(r.start + lead + off));
                }
            }
        }
        // sentinel entry so `origin[e]` is always addressable
        origin.push(None);
        Self { text, origin }
    }

    /// Map a byte range `[s, e)` of `self.text` back to the original input.
    /// Returns `None` when the range contains only joiner space.
    pub fn map_span(&self, s: usize, e: usize) -> Option<(usize, usize)> {
        if s >= e || e > self.text.len() {
            return None;
        }
        let mut first = None;
        let mut last_end = None;
        for i in s..e {
            match self.origin[i] {
                Some(off) => {
                    if first.is_none() {
                        first = Some(off);
                    }
                    // original end of this byte = next byte's origin, or off+len
                    last_end = Some(off + self.text[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1));
                }
                None => {
                    if first.is_some() {
                        // token crosses a joiner: stop at the previous run end
                        break;
                    }
                }
            }
        }
        match (first, last_end) {
            (Some(a), Some(b)) if a < b => Some((a, b)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripts_offsets_slice_back() {
        let s = "她 quietly opened the 旧书，and read.";
        let runs = split_scripts(s);
        assert!(runs.iter().any(|(k, _)| *k == ScriptKind::Han));
        assert!(runs.iter().any(|(k, _)| *k == ScriptKind::Latin));
        for (_, r) in &runs {
            assert_eq!(&s[r.start..r.end], r.text);
        }
    }

    #[test]
    fn sentences_trim() {
        let text = "  第一句。 Second one!  \n\nThird?\n";
        let ss = sentences(text);
        for sp in &ss {
            assert_eq!(&text[sp.start..sp.end], sp.text);
            assert!(!sp.text.starts_with(' '));
            assert!(!sp.text.ends_with(char::is_whitespace));
        }
        assert_eq!(ss.len(), 3, "{ss:?}");
        assert!(ss[1].text.starts_with("Second"));
    }

    #[test]
    fn mixed_has_both_scripts() {
        let s = "她 quietly opened the 旧书，and carefully read every 句子。";
        let runs = split_scripts(s);
        assert!(runs.iter().any(|(k, _)| *k == ScriptKind::Han));
        assert!(runs.iter().filter(|(k, _)| *k == ScriptKind::Latin).count() >= 2);
        // every run slices back to its own text
        for (_, r) in &runs {
            assert_eq!(&s[r.start..r.end], r.text);
        }
    }

    #[test]
    fn latin_runs_join_with_space_and_map_back() {
        let s = "她 quietly opened the 旧书，and carefully read.";
        let runs: Vec<Span> = split_scripts(s)
            .into_iter()
            .filter(|(k, _)| *k == ScriptKind::Latin)
            .map(|(_, r)| r)
            .collect();
        let batch = LatinBatch::from_runs(&runs);
        assert!(batch.text.starts_with("quietly opened the"));
        assert_eq!(batch.text, "quietly opened the and carefully read."); // runs joined by one space
        let mut cursor = 0usize;
        for w in batch.text.split(' ').filter(|w| !w.is_empty()) {
            let start = batch.text[cursor..].find(w).expect(w) + cursor;
            let (x, y) = batch.map_span(start, start + w.len()).expect(w);
            assert_eq!(&s[x..y], w, "word {w:?}");
            cursor = start + w.len();
        }
    }

    /// Regression: a sentence with several Han runs must map each run-relative
    /// word offset to `sentence_start + run_start + word_offset`.
    #[test]
    fn mixed_multi_run_offsets_are_absolute() {
        let s = "她 quietly opened the 旧书，and read every 句子。";
        for sp in sentences(s) {
            for (kind, run) in split_scripts(&sp.text) {
                if kind != ScriptKind::Han {
                    continue;
                }
                // simulate two char-level words inside the run
                let mid = run.text.char_indices().nth(1).map(|(i, _)| i).unwrap_or(run.text.len());
                for (a, b) in [(0, mid), (mid, run.text.len())] {
                    let abs_a = sp.start + run.start + a;
                    let abs_b = sp.start + run.start + b;
                    assert_eq!(&s[abs_a..abs_b], &run.text[a..b], "run {:?}", run.text);
                }
            }
        }
    }

    #[test]
    fn mixed_paragraph_sentence_offsets() {
        let s = "她 quietly opened the 旧书。The tired 孩子们睡着了，so I closed the window.";
        for sp in sentences(s) {
            for (_, r) in split_scripts(&sp.text) {
                assert_eq!(&sp.text[r.start..r.end], r.text);
                assert_eq!(&s[sp.start + r.start..sp.start + r.end], r.text);
            }
        }
    }
}
