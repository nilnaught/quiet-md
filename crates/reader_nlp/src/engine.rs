//! `NlpEngine`: async model loading + a small text-keyed cache.
//!
//! `start` returns immediately; a background thread loads the ONNX models and
//! replaces the shared `Neural` slot when done. Until then (and forever if the
//! load fails) `analyze` uses the jieba/Brill fallback. The cache keys on the
//! analysed text plus the source (fallback results are dropped once the models
//! are ready), holds at most [`CACHE_CAP`] entries (FIFO) and never panics.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::fallback;
use crate::neural::{Neural, SharedNeural};
use crate::text::{self, ScriptKind};
use crate::types::{Analysis, AnalysisSource, ParagraphSpan, SentenceSpan, TokenSpan};
use crate::EnVariant;

/// Maximum cached analyses per engine.
pub const CACHE_CAP: usize = 64;

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct NlpOptions {
    /// Directory holding `ltp-small/` and `en-upos-*/` (i.e. `$MODELS`).
    pub models_dir: PathBuf,
    /// English model variant; `BaseInt8` by default.
    pub en_variant: EnVariant,
    /// Intra-op threads per ONNX session.
    pub threads: usize,
}

impl Default for NlpOptions {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            en_variant: EnVariant::BaseInt8,
            threads: 2,
        }
    }
}

/// Resolution order:
/// 1. `$QUIET_MD_NLP_MODELS`
/// 2. the models directory bundled with the app (macOS
///    `Contents/Resources/models`, or a `models/` folder next to the
///    executable on Windows / Linux)
/// 3. `/workspace/models` when that path exists (this development box, and
///    only then: debug builds, examples and tests run there)
/// 4. `<cwd>/models`
///
/// Release builds skip 3 and 4 so a shipped binary always uses its bundle (or
/// the fallback) and never models that happen to sit in the working directory.
///
/// A missing directory is fine: [`NlpEngine::analyze`] stays on the
/// jieba / english-pos-tagger fallback.
fn default_models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("QUIET_MD_NLP_MODELS") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = bundled_models_dir() {
        return dir;
    }
    // Shipped binaries: no development-only fallbacks (see the doc comment).
    if !cfg!(debug_assertions) {
        return bundled_models_dir().unwrap_or_default();
    }
    let box_default = PathBuf::from("/workspace/models");
    if box_default.is_dir() {
        return box_default;
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("models")
}

/// `Some` only when the bundled folder is already on disk.
fn bundled_models_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    // Quiet.md.app/Contents/MacOS/quiet_md → ../Resources/models
    if dir.file_name().is_some_and(|n| n == "MacOS") {
        let resources = dir.parent()?.join("Resources").join("models");
        if resources.is_dir() {
            return Some(resources);
        }
    }
    let next = dir.join("models");
    next.is_dir().then_some(next)
}

#[derive(Default)]
struct CacheInner {
    /// text -> (generation, analysis). The generation is the engine's
    /// source generation at insert time: a cached fallback result is never
    /// served once the neural models have loaded.
    entries: VecDeque<(String, u64, Analysis)>,
}

/// The cache plus the current source generation, under one lock.
struct Cache {
    inner: Mutex<CacheInner>,
    /// Bumped when the neural models become available.
    generation: AtomicU64,
}

impl Cache {
    fn get(&self, text: &str) -> Option<Analysis> {
        let gen = self.generation.load(Ordering::SeqCst);
        let inner = self.inner.lock().ok()?;
        inner
            .entries
            .iter()
            .find(|(k, egen, _)| k == text && *egen == gen)
            .map(|(_, _, a)| a.clone())
    }

    fn put(&self, text: &str, analysis: Analysis, at_generation: u64) {
        if self.generation.load(Ordering::SeqCst) != at_generation {
            return;
        }
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        if inner.entries.len() >= CACHE_CAP {
            inner.entries.pop_front();
        }
        inner
            .entries
            .push_back((text.to_string(), at_generation, analysis));
    }

    /// Drop every entry and move to the next generation (models loaded).
    fn invalidate(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut inner) = self.inner.lock() {
            inner.entries.clear();
        }
    }
}

/// The reader-facing NLP engine.
pub struct NlpEngine {
    opts: NlpOptions,
    neural: SharedNeural,
    cache: Arc<Cache>,
}

impl NlpEngine {
    /// Load the models in the background and return at once.
    pub fn start(opts: NlpOptions) -> Self {
        let neural: SharedNeural = Arc::new(Mutex::new(None));
        let cache: Arc<Cache> = Arc::new(Cache {
            inner: Mutex::new(CacheInner::default()),
            generation: AtomicU64::new(0),
        });
        let thread_neural = Arc::clone(&neural);
        let thread_cache = Arc::clone(&cache);
        let models_dir = opts.models_dir.clone();
        let variant = opts.en_variant;
        let threads = opts.threads.max(1);
        let spawned = std::thread::Builder::new()
            .name("quiet-md-nlp-load".into())
            .spawn(move || {
                let loaded = Neural::load(&models_dir, variant, threads);
                if let Ok(Some(neural)) = loaded {
                    if let Ok(mut slot) = thread_neural.lock() {
                        *slot = Some(neural);
                    }
                    // fallback entries are no longer the best answer
                    thread_cache.invalidate();
                }
            });
        if spawned.is_err() {
            // no thread: stay on the fallback path for the engine's lifetime
        }
        Self {
            opts,
            neural,
            cache,
        }
    }

    /// True once the ONNX models are loaded.
    pub fn neural_ready(&self) -> bool {
        self.neural.lock().map(|n| n.is_some()).unwrap_or(false)
    }

    /// The options this engine was created with.
    pub fn options(&self) -> &NlpOptions {
        &self.opts
    }

    /// Synchronous analysis of `text`; never changes the text.
    ///
    /// A cached analysis of the same text is returned directly. Otherwise the
    /// neural pipeline runs when the models are ready, else the fallback.
    pub fn analyze(&self, text: &str) -> Analysis {
        if let Some(hit) = self.cache.get(text) {
            return hit;
        }
        let at_generation = self.cache.generation.load(Ordering::SeqCst);
        let analysis = self.compute(text);
        self.cache.put(text, analysis.clone(), at_generation);
        analysis
    }

    fn compute(&self, text: &str) -> Analysis {
        // Keep the models in the slot. Taking them out let a second caller
        // fall back to jieba while the first was still inferring, and a panic
        // mid-analyse would drop the sessions for good.
        let Ok(mut slot) = self.neural.lock() else {
            return fallback_analysis(text);
        };
        match slot.as_mut() {
            Some(neural) => neural_analysis(neural, text),
            None => fallback_analysis(text),
        }
    }
}

fn paragraphs_of(text: &str) -> Vec<text::Span> {
    text::paragraphs(text)
}

fn neural_analysis(neural: &mut Neural, text: &str) -> Analysis {
    let mut paragraphs = Vec::new();
    let mut sentences = Vec::new();
    for para in paragraphs_of(text) {
        let start = sentences.len();
        for (range, tokens) in neural.analyze_paragraph(&para.text, para.start) {
            sentences.push(SentenceSpan { range, tokens });
        }
        paragraphs.push(ParagraphSpan {
            range: para.range(),
            sentences: start..sentences.len(),
        });
    }
    Analysis {
        source: AnalysisSource::Neural,
        paragraphs,
        sentences,
    }
}

fn fallback_analysis(text: &str) -> Analysis {
    let mut paragraphs = Vec::new();
    let mut sentences = Vec::new();
    for para in paragraphs_of(text) {
        let start = sentences.len();
        for sent in text::sentences(&para.text) {
            let base = para.start + sent.start;
            let runs = text::split_scripts(&sent.text);
            let mixed = runs.iter().any(|(k, _)| *k == ScriptKind::Han)
                && runs.iter().any(|(k, _)| *k == ScriptKind::Latin);
            let mut tokens: Vec<TokenSpan> = Vec::new();
            for (kind, run) in runs {
                tokens.extend(fallback::tokens_for_run(kind, &run.text, base + run.start, mixed));
            }
            tokens.sort_by_key(|t| (t.range.start, t.range.end));
            if !tokens.iter().any(|t| t.is_core) {
                // In a mixed sentence, do not promote an English participle
                // used as an adjective (`The tired 孩子们…`); prefer a Han verb.
                let pick = tokens.iter().position(|t| {
                    t.upos == "VERB"
                        && (!mixed
                            || text[t.range.clone()]
                                .chars()
                                .any(|c| unicode_script::UnicodeScript::script(&c)
                                    == unicode_script::Script::Han))
                });
                if let Some(i) = pick {
                    tokens[i].is_core = true;
                }
            }
            sentences.push(SentenceSpan {
                range: base..base + sent.text.len(),
                tokens,
            });
        }
        paragraphs.push(ParagraphSpan {
            range: para.range(),
            sentences: start..sentences.len(),
        });
    }
    Analysis {
        source: AnalysisSource::Fallback,
        paragraphs,
        sentences,
    }
}
