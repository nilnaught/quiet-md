//! Sentence segmentation + Han/Latin script splitting.
//!
//! The library works on *paragraphs* (blank-line separated blocks of text) and
//! *sentences* inside them. A sentence is split into runs of "Chinese" (Han)
//! vs "Latin" characters; Latin runs of one sentence are joined with a single
//! space, sent to the English pipeline as one string, and every English token
//! is mapped back to a byte range of the **original** input.
//!
//! Invariant: for every emitted token, `&input[start..end] == token_text`
//! (with English whitespace tokens dropped, since the joiner space is
//! synthetic).

use std::ops::Range;

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

    pub fn range(&self) -> Range<usize> {
        self.start..self.end
    }
}

/// Split text into paragraphs: blocks separated by a blank line (a line that
/// is empty or only whitespace). Blocks are trimmed; its byte range always
/// slices back to the block text.
pub fn paragraphs(text: &str) -> Vec<Span> {
    let mut out = Vec::new();
    let mut block_start = 0usize;
    let mut saw_content = false;
    let mut pos = 0usize;
    for line in text.split_inclusive('\n') {
        let body = line.trim_end_matches('\n').trim_end_matches('\r');
        if body.trim().is_empty() {
            if saw_content {
                out.push(Span::new(text, block_start, pos));
                saw_content = false;
            }
            block_start = pos + line.len();
        } else if !saw_content {
            block_start = pos;
            saw_content = true;
        }
        pos += line.len();
    }
    if saw_content {
        out.push(Span::new(text, block_start, pos));
    }
    // trim each block to its content
    out.into_iter()
        .filter_map(|b| {
            let lead = b.text.len() - b.text.trim_start().len();
            let trail = b.text.len() - b.text.trim_end().len();
            let (s, e) = (b.start + lead, b.end - trail);
            (s < e).then(|| Span::new(text, s, e))
        })
        .collect()
}

/// Sentence segmentation of one paragraph (pure Rust, no ICU data needed):
/// split after `.`, `!`, `?`, `。`, `！`, `？`, `…` or a newline, keeping runs
/// of terminators together and byte offsets intact, trimming the whitespace
/// around every sentence. A Latin `.` between two digits (`3.14`) is not a
/// boundary; a closing quote after a terminator stays with the sentence.
pub fn sentences(text: &str) -> Vec<Span> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let mut in_term = false;
    while i < text.len() {
        let c = text[i..].chars().next().unwrap();
        let next = i + c.len_utf8();
        let clause_end = matches!(c, '.' | '!' | '?' | '。' | '！' | '？' | '…' | '\n');
        let decimal_dot = c == '.'
            && i > 0
            && next < bytes.len()
            && bytes[i - 1].is_ascii_digit()
            && bytes[next].is_ascii_digit();
        let is_term = clause_end && !decimal_dot;

        if in_term {
            let cont = if is_term {
                true
            } else {
                matches!(c, '"' | '\'' | '”' | '’' | ')' | '）')
            };
            if cont {
                i = next;
                continue;
            }
            push_trimmed(text, start, i, &mut out);
            start = i;
            in_term = false;
        }
        if is_term {
            if c == '\n' {
                push_trimmed(text, start, next, &mut out);
                start = next;
            } else {
                in_term = true;
            }
        }
        i = next;
    }
    if in_term {
        push_trimmed(text, start, text.len(), &mut out);
        start = text.len();
    }
    if start < text.len() {
        push_trimmed(text, start, text.len(), &mut out);
    }
    out
}

fn push_trimmed(text: &str, s: usize, e: usize, out: &mut Vec<Span>) {
    if s >= e || e > text.len() {
        return;
    }
    let piece = &text[s..e];
    let lead = piece.len() - piece.trim_start().len();
    let trail = piece.len() - piece.trim_end().len();
    let (s, e) = (s + lead, e - trail);
    if s < e {
        out.push(Span::new(text, s, e));
    }
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
        for (c_off, c) in self.text[s..e].char_indices() {
            let i = s + c_off;
            match self.origin[i] {
                Some(off) => {
                    if first.is_none() {
                        first = Some(off);
                    }
                    // original end of this char = its origin + char length
                    last_end = Some(off + c.len_utf8());
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
