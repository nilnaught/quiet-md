//! Public result types: byte spans into the analysed text plus the derived
//! reading marks.

use std::ops::Range;

/// Where the analysis came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisSource {
    /// ONNX models (LTP/small for Chinese, roberta + SuPar for English).
    Neural,
    /// `jieba-rs` + `english-pos-tagger` (models not loaded yet or failed).
    Fallback,
}

/// One analysed text: paragraphs -> sentences -> tokens, all ranges are byte
/// offsets into the string that was passed to `analyze`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Analysis {
    pub source: AnalysisSource,
    /// Blocks separated by blank lines.
    pub paragraphs: Vec<ParagraphSpan>,
    /// All sentences of all paragraphs, in order.
    pub sentences: Vec<SentenceSpan>,
}

/// A paragraph; `range` slices the input and covers all its sentences.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParagraphSpan {
    pub range: Range<usize>,
    pub sentences: Range<usize>,
}

/// A sentence with its tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentenceSpan {
    /// Byte offset range; `text[range]` is exactly the sentence.
    pub range: Range<usize>,
    pub tokens: Vec<TokenSpan>,
}

/// One word with its coarse class and reading marks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSpan {
    pub range: Range<usize>,
    /// Coarse class for colouring: NOUN / VERB / ADJ / ADV / PRON / OTHER.
    pub upos: String,
    /// Main predicate of its clause (emphasis dot).
    pub is_core: bool,
    /// Conjunction / subordinator (underline).
    pub is_connective: bool,
    /// Non-finite verb form (dashed underline); English only.
    pub is_nonfinite: bool,
}
