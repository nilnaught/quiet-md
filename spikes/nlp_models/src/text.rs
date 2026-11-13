//! Sentence segmentation + Han/Latin script splitting.
//!
//! The CLI works on *sentences*: `icu_segmenter` splits the raw input, then each
//! sentence is split into runs of "Chinese" (Han) vs "Latin" characters. Latin runs
//! are sent to the English pipeline and mapped back to original byte offsets.

use icu_segmenter::SentenceSegmenter;
use unicode_script::{Script, UnicodeScript};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptKind {
    Han,
    Latin,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// Split text into sentences using ICU, keeping byte offsets into `text`.
pub fn sentences(text: &str) -> Vec<ToolResult> {
    let seg = SentenceSegmenter::new(Default::default());
    let mut marks: Vec<usize> = seg.segment_str(text).collect();
    if marks.first() != Some(&0) {
        marks.insert(0, 0);
    }
    if marks.last() != Some(&text.len()) {
        marks.push(text.len());
    }
    let mut out = Vec::new();
    for w in marks.windows(2) {
        let (start, end) = (w[0], w[1]);
        if start == end {
            continue;
        }
        let slice = &text[start..end];
        let (s, e) = trim_ws(slice, start);
        if s < e {
            out.push(ToolResult {
                text: text[s..e].to_string(),
                start: s,
                end: e,
            });
        }
    }
    out
}

fn trim_ws(slice: &str, base: usize) -> (usize, usize) {
    let lead = slice.len() - slice.trim_start().len();
    let trail = slice.len() - slice.trim_end().len();
    (base + lead, base + slice.len() - trail)
}

/// Classify a char: Han -> Chinese pipeline, Latin letters -> English pipeline.
/// Everything else (digits, punctuation, whitespace) attaches to the *neighbouring*
/// run, preferring the run it directly follows.
fn char_script(c: char) -> Option<ScriptKind> {
    match c.script() {
        Script::Han => Some(ScriptKind::Han),
        _ if c.is_ascii_alphabetic() => Some(ScriptKind::Latin),
        // Latin-1 letters, accents, etc.
        _ if c.is_alphabetic() && c.script() == Script::Latin => Some(ScriptKind::Latin),
        _ => None,
    }
}

/// Split one sentence into script runs. Punctuation/digits/space stick to the current
/// run; a leading punctuation cluster joins the run that follows it.
pub fn split_scripts(sentence: &str) -> Vec<(ScriptKind, ToolResult)> {
    let chars: Vec<(usize, char)> = sentence.char_indices().collect();
    let mut kinds: Vec<Option<ScriptKind>> = chars.iter().map(|(_, c)| char_script(*c)).collect();

    // Forward fill: neutral chars inherit the previous explicit script.
    let mut last = None;
    for k in kinds.iter_mut() {
        match k {
            Some(s) => last = Some(*s),
            None => *k = last,
        }
    }
    // Trailing/leading neutrals: backfill from the next explicit one.
    let mut next = None;
    for k in kinds.iter_mut().rev() {
        match k {
            Some(s) => next = Some(*s),
            None => *k = next,
        }
    }
    // If everything was neutral, nothing to do.
    let mut runs: Vec<(ScriptKind, ToolResult)> = Vec::new();
    let mut cur: Option<(ScriptKind, usize, usize)> = None;
    for (i, (byte, c)) in chars.iter().enumerate() {
        let kind = match kinds[i] {
            Some(k) => k,
            None => continue,
        };
        let end = byte + c.len_utf8();
        match cur {
            Some((k, s, _)) if k == kind => cur = Some((k, s, end)),
            Some((k, s, e)) => {
                runs.push((
                    k,
                    ToolResult {
                        text: sentence[s..e].to_string(),
                        start: s,
                        end: e,
                    },
                ));
                cur = Some((kind, *byte, end));
            }
            None => cur = Some((kind, *byte, end)),
        }
    }
    if let Some((k, s, e)) = cur {
        runs.push((
            k,
            ToolResult {
                text: sentence[s..e].to_string(),
                start: s,
                end: e,
            },
        ));
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripts() {
        let runs = split_scripts("她 quietly opened the 旧书，and read.");
        for (k, r) in &runs {
            println!("{k:?} {:?}", r.text);
        }
        assert!(runs.iter().any(|(k, _)| *k == ScriptKind::Han));
        assert!(runs.iter().any(|(k, _)| *k == ScriptKind::Latin));
    }
}
