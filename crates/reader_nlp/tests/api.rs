//! Public-API tests: offline (fallback) analysis, byte ranges, caching.

use reader_nlp::{AnalysisSource, NlpEngine, NlpOptions};

/// An engine whose models never exist, i.e. always the fallback path.
fn fallback_engine() -> NlpEngine {
    NlpEngine::start(NlpOptions {
        models_dir: std::path::PathBuf::from("/nonexistent/quiet-md-models"),
        en_variant: Default::default(),
        threads: 1,
    })
}

/// Every token must slice its own text out of `text`.
fn assert_ranges_slice(text: &str, a: &reader_nlp::Analysis) {
    for s in &a.sentences {
        assert!(s.range.end <= text.len(), "sentence range {:?}", s.range);
        for t in &s.tokens {
            assert!(
                t.range.start < t.range.end && t.range.end <= text.len(),
                "token range {:?} inside {:?}",
                t.range,
                s.range
            );
            let slice = &text[t.range.clone()];
            assert!(
                !slice.trim().is_empty(),
                "token slice {slice:?} must not be blank (range {:?})",
                t.range
            );
            assert!(
                !slice.chars().any(|c| c.is_whitespace()),
                "a token must not contain whitespace: {slice:?}"
            );
            assert!(
                t.range.start >= s.range.start && t.range.end <= s.range.end,
                "token {:?} outside sentence {:?}",
                t.range,
                s.range
            );
        }
    }
}

#[test]
fn english_byte_offsets_round_trip() {
    let engine = fallback_engine();
    let text = "She will come. They have been working on this project.";
    let a = engine.analyze(text);
    assert_eq!(a.source, AnalysisSource::Fallback);
    assert_ranges_slice(text, &a);
    let words: Vec<&str> = a
        .sentences
        .iter()
        .flat_map(|s| s.tokens.iter().map(|t| &text[t.range.clone()]))
        .collect();
    for w in ["She", "will", "come", "working"] {
        assert!(words.contains(&w), "missing {w} in {words:?}");
    }
}

#[test]
fn chinese_byte_offsets_round_trip() {
    let engine = fallback_engine();
    let text = "她安静地打开了旧书。因为下雨，所以我们没有出门。";
    let a = engine.analyze(text);
    assert_ranges_slice(text, &a);
    let words: Vec<&str> = a
        .sentences
        .iter()
        .flat_map(|s| s.tokens.iter().map(|t| &text[t.range.clone()]))
        .collect();
    assert!(words.contains(&"安静"), "jieba should split 安静地: {words:?}");
    assert!(words.contains(&"因为"), "{words:?}");
    let conn: Vec<&str> = a
        .sentences
        .iter()
        .flat_map(|s| s.tokens.iter())
        .filter(|t| t.is_connective)
        .map(|t| &text[t.range.clone()])
        .collect();
    assert!(conn.contains(&"因为"), "因为 is a connective: {conn:?}");
    assert!(conn.contains(&"所以"), "所以 is a connective: {conn:?}");
}

#[test]
fn mixed_byte_offsets_round_trip_and_no_core_adjective() {
    let engine = fallback_engine();
    let text = "她 quietly opened the 旧书，and read. The tired 孩子们睡着了。";
    let a = engine.analyze(text);
    assert_ranges_slice(text, &a);
    let core: Vec<&str> = a
        .sentences
        .iter()
        .flat_map(|s| s.tokens.iter())
        .filter(|t| t.is_core)
        .map(|t| &text[t.range.clone()])
        .collect();
    assert!(!core.iter().any(|w| w.eq_ignore_ascii_case("tired")), "core = {core:?}");
}

#[test]
fn paragraphs_split_at_blank_lines() {
    let engine = fallback_engine();
    let text = "First one.\nStill first.\n\n\nSecond para.\n";
    let a = engine.analyze(text);
    assert_eq!(a.paragraphs.len(), 2, "{:?}", a.paragraphs);
    for p in &a.paragraphs {
        assert!(!text[p.range.clone()].trim().is_empty());
        assert!(p.sentences.end <= a.sentences.len());
    }
    assert_eq!(&text[a.paragraphs[0].range.clone()], "First one.\nStill first.");
    assert_eq!(&text[a.paragraphs[1].range.clone()], "Second para.");
}

#[test]
fn cache_hits_return_equal_results() {
    let engine = fallback_engine();
    let text = "She will come. 她 quietly opened the 旧书。";
    let first = engine.analyze(text);
    let second = engine.analyze(text);
    assert_eq!(first, second);
    assert_eq!(first.paragraphs, second.paragraphs);
    // a different text is not served from the first entry
    let third = engine.analyze("Another sentence.");
    assert_ne!(first.sentences.len(), third.sentences.len());
}

#[test]
fn cache_is_bounded_and_stays_correct() {
    let engine = fallback_engine();
    for i in 0..(reader_nlp::CACHE_CAP * 2) {
        let text = format!("Sentence number {i} is here.");
        let a = engine.analyze(&text);
        assert_ranges_slice(&text, &a);
    }
    // the first text was evicted; re-analysing still works and slices cleanly
    let text = "Sentence number 0 is here.";
    assert_ranges_slice(text, &engine.analyze(text));
}

#[test]
fn start_does_not_block_on_missing_models() {
    let t0 = std::time::Instant::now();
    let engine = NlpEngine::start(NlpOptions {
        models_dir: std::path::PathBuf::from("/nonexistent/quiet-md-models"),
        en_variant: Default::default(),
        threads: 1,
    });
    // even if the loader thread were slow, analyze must work immediately
    let a = engine.analyze("A short sentence.");
    assert!(!a.sentences.is_empty());
    assert!(t0.elapsed() < std::time::Duration::from_secs(5));
    assert!(!engine.neural_ready(), "no models under a missing dir");
}

/// Loads the real models when this box has them; skipped otherwise. Not part
/// of the default test run (see the task: no model dependency for `cargo test`).
#[test]
#[ignore = "needs $QUIET_MD_NLP_MODELS (or /workspace/models) with ltp-small + en-upos-base/onnx-int8"]
fn neural_integration_base_int8() {
    let opts = NlpOptions::default();
    assert!(
        opts.models_dir.join("en-upos-base/onnx-int8").is_dir(),
        "models dir {:?} has no en-upos-base/onnx-int8",
        opts.models_dir
    );
    let engine = NlpEngine::start(opts);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    while !engine.neural_ready() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    assert!(engine.neural_ready(), "models did not load in time");
    let text = "She is a famous writer. 她安静地打开了旧书。";
    let a = engine.analyze(text);
    assert_eq!(a.source, AnalysisSource::Neural);
    assert_ranges_slice(text, &a);
    let core: Vec<&str> = a
        .sentences
        .iter()
        .flat_map(|s| s.tokens.iter())
        .filter(|t| t.is_core)
        .map(|t| &text[t.range.clone()])
        .collect();
    assert!(!core.iter().any(|w| w.eq_ignore_ascii_case("writer")), "core = {core:?}");
}
