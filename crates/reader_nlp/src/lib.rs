//! `reader_nlp`: offline Chinese/English POS + dependency analysis for the
//! reader, with derived reading marks (core predicate, connectives,
//! non-finite verbs, coarse colour class).
//!
//! ```no_run
//! use reader_nlp::{NlpEngine, NlpOptions};
//!
//! let engine = NlpEngine::start(NlpOptions::default());
//! let text = "她 quietly opened the 旧书。";
//! let analysis = engine.analyze(text);
//! for sentence in &analysis.sentences {
//!     for token in &sentence.tokens {
//!         println!("{:?} {}", &text[token.range.clone()], token.upos);
//!     }
//! }
//! ```
//!
//! Models live under `NlpOptions::models_dir` (`$QUIET_MD_NLP_MODELS`,
//! default `/workspace/models`): `ltp-small/onnx/` for Chinese and
//! `en-upos-base/onnx-int8/` (the default English variant) for English.
//! Nothing is loaded eagerly: [`NlpEngine::start`] returns at once, a
//! background thread loads the ONNX sessions, and until then
//! [`NlpEngine::analyze`] uses the `jieba-rs` + `english-pos-tagger` fallback.

mod derive;
mod eisner;
mod engine;
mod english;
mod fallback;
mod ltp;
mod neural;
mod ortutil;
mod text;
mod types;

pub use derive::{EN_CONNECTIVES, ZH_CONNECTIVES};
pub use engine::{NlpEngine, NlpOptions, CACHE_CAP};
pub use types::{Analysis, AnalysisSource, ParagraphSpan, SentenceSpan, TokenSpan};

/// Which exported English model to use. Default: [`EnVariant::BaseInt8`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnVariant {
    LargeFp32,
    LargeInt8,
    BaseFp32,
    #[default]
    BaseInt8,
}
