//! Offline neural CN/EN POS + dependency spike for quiet-md.
//!
//! ```text
//! cargo run --release --manifest-path spikes/nlp_models/Cargo.toml -- \
//!   --models-dir /workspace/models --out-dir /workspace/spike-out \
//!   --input spikes/nlp_models/fixtures/sentences.json
//! ```
//!
//! Writes `tokens.json`, `demo.html`, `metrics.json` into `--out-dir` and, with
//! `--compare-dir`, a `compare.json` with spaCy `en_core_web_trf` agreement over
//! the fixture's `en_compare` sentences.

mod eisner;
mod english;
mod ltp;
mod metrics;
mod ortutil;
mod pipeline;
mod text;

use anyhow::{Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

use pipeline::{Pipeline, SentenceOut, Token};

#[derive(Serialize, Clone)]
struct TokenOut {
    text: String,
    start: usize,
    end: usize,
    pos: String,
    upos_or_ltp_tag: String,
    head: usize,
    deprel: String,
    is_core_verb: bool,
    is_connective: bool,
    is_nonfinite: bool,
    lang: String,
    sentence: usize,
    clause: usize,
}

impl From<&Token> for TokenOut {
    fn from(t: &Token) -> Self {
        Self {
            text: t.text.clone(),
            start: t.start,
            end: t.end,
            pos: t.pos.clone(),
            upos_or_ltp_tag: t.upos_or_ltp_tag.clone(),
            head: t.head,
            deprel: t.deprel.clone(),
            is_core_verb: t.is_core_verb,
            is_connective: t.is_connective,
            is_nonfinite: t.is_nonfinite,
            lang: t.lang.to_string(),
            sentence: t.sentence,
            clause: t.run,
        }
    }
}

#[derive(Serialize)]
struct SentenceJson {
    index: usize,
    start: usize,
    end: usize,
    text: String,
    tokens: Vec<TokenOut>,
}

#[derive(Serialize)]
struct GroupJson {
    name: String,
    text: String,
    sentences: Vec<SentenceJson>,
}

#[derive(serde::Deserialize)]
struct Fixtures {
    #[serde(default)]
    error_cases: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    zh_extra: Vec<String>,
    #[serde(default)]
    en_compare: Vec<String>,
}

fn main() -> Result<()> {
    let mut models_dir = PathBuf::from("/workspace/models");
    let mut out_dir = PathBuf::from("/workspace/spike-out");
    let mut input = PathBuf::from("spikes/nlp_models/fixtures/sentences.json");
    let mut compare_dir: Option<PathBuf> = None;
    let mut threads = 4usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || -> Result<String> { it.next().context(format!("missing value for {a}")) };
        match a.as_str() {
            "--models-dir" => models_dir = val()?.into(),
            "--out-dir" => out_dir = val()?.into(),
            "--input" => input = val()?.into(),
            "--compare-dir" => compare_dir = Some(val()?.into()),
            "--threads" => threads = val()?.parse()?,
            "-h" | "--help" => {
                println!(
                    "usage: nlp_spike [--models-dir DIR] [--out-dir DIR] [--input JSON] \
                     [--compare-dir DIR] [--threads N]"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    std::fs::create_dir_all(&out_dir)?;

    let fx: Fixtures = serde_json::from_str(
        &std::fs::read_to_string(&input).with_context(|| format!("reading {}", input.display()))?,
    )?;

    // ---- models --------------------------------------------------------
    let t_zh = Instant::now();
    let ltp = ltp::Ltp::load(&models_dir, threads)?;
    let zh_load_ms = t_zh.elapsed().as_secs_f64() * 1000.0;
    eprintln!("zh loaded in {zh_load_ms:.0} ms, rss {:?} MiB", ortutil::rss_mib());
    let t_en = Instant::now();
    let en = english::English::load(&models_dir, threads)?;
    let en_load_ms = t_en.elapsed().as_secs_f64() * 1000.0;
    eprintln!("en loaded in {en_load_ms:.0} ms, rss {:?} MiB", ortutil::rss_mib());
    let mut pipe = Pipeline {
        ltp: Some(ltp),
        en: Some(en),
    };

    // ---- warmup (not measured) ----------------------------------------
    let t_warm = Instant::now();
    for _ in 0..2 {
        let _ = pipe.annotate("这是一个热身句子。")?;
        let _ = pipe.annotate("This is a warmup sentence.")?;
    }
    let warm_ms = t_warm.elapsed().as_secs_f64() * 1000.0;
    eprintln!("warmup {warm_ms:.0} ms");

    // ---- inputs --------------------------------------------------------
    let t_run_start = Instant::now();
    let mut groups: Vec<GroupJson> = Vec::new();
    let mut latencies_ms: Vec<f64> = Vec::new();

    let mut run_group = |name: String, text: String, lat: &mut Vec<f64>| -> Result<GroupJson> {
        let t = Instant::now();
        let sents = pipe.annotate(&text)?;
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
        Ok(GroupJson {
            name,
            text,
            sentences: to_json(&sents),
        })
    };

    for (k, v) in &fx.error_cases {
        groups.push(run_group(format!("error_cases.{k}"), v.clone(), &mut latencies_ms)?);
    }
    for (i, s) in fx.zh_extra.iter().enumerate() {
        groups.push(run_group(format!("zh_extra[{i}]"), s.clone(), &mut latencies_ms)?);
    }

    // English comparison set is annotated separately (one sentence each) so the
    // per-sentence latency numbers line up with spaCy's.
    let mut en_groups: Vec<GroupJson> = Vec::new();
    for (i, s) in fx.en_compare.iter().enumerate() {
        en_groups.push(run_group(format!("en_compare[{i}]"), s.clone(), &mut latencies_ms)?);
    }

    // ---- verify byte offsets ------------------------------------------
    let mut offset_errors = 0usize;
    for g in groups.iter().chain(en_groups.iter()) {
        for s in &g.sentences {
            for t in &s.tokens {
                match g.text.get(t.start..t.end) {
                    Some(slice) if slice == t.text => {}
                    _ => {
                        offset_errors += 1;
                        eprintln!(
                            "OFFSET MISMATCH group={} {:?} [{}..{}] -> {:?}",
                            g.name,
                            t.text,
                            t.start,
                            t.end,
                            g.text.get(t.start..t.end)
                        );
                    }
                }
            }
        }
    }

    // ---- tokens.json -----------------------------------------------------
    let tokens_path = out_dir.join("tokens.json");
    std::fs::write(
        &tokens_path,
        serde_json::to_string_pretty(&serde_json::json!({
            "note": "start/end are byte offsets into the group text; every token satisfies text[start..end] == token.text",
            "offset_errors": offset_errors,
            "groups": groups,
            "en_compare": en_groups,
        }))?,
    )?;

    // ---- demo.html -------------------------------------------------------
    let demo = render_html(&groups, &en_groups);
    std::fs::write(out_dir.join("demo.html"), &demo)?;

    // ---- metrics.json ----------------------------------------------------
    let lat_avg = if latencies_ms.is_empty() {
        0.0
    } else {
        latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64
    };
    let mut sorted = latencies_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    let p95 = sorted
        .get((sorted.len() as f64 * 0.95) as usize)
        .copied()
        .unwrap_or(0.0);
    let n_sentences: usize = groups
        .iter()
        .chain(en_groups.iter())
        .map(|g| g.sentences.len())
        .sum();
    let zh_stat = &pipe.ltp.as_ref().unwrap().stat;
    let en_tag = &pipe.en.as_ref().unwrap().tagger_stat;
    let en_par = &pipe.en.as_ref().unwrap().parser_stat;
    let metrics = serde_json::json!({
        "machine": { "cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
                     "threads_per_session": threads },
        "model_files": {
            "zh": zh_stat.files.iter().map(|(p, b)| (p.clone(), serde_json::json!({"bytes": b, "mib": *b as f64/1048576.0}))).collect::<serde_json::Map<_,_>>(),
            "en": en_tag.files.iter().chain(en_par.files.iter()).map(|(p, b)| (p.clone(), serde_json::json!({"bytes": b, "mib": *b as f64/1048576.0}))).collect::<serde_json::Map<_,_>>(),
        },
        "load_ms": { "zh_ltp_small": zh_stat.load_ms, "en_tagger": en_tag.load_ms, "en_parser": en_par.load_ms,
                     "total_measured_in_main": zh_load_ms + en_load_ms },
        "warmup_ms": warm_ms,
        "inference": {
            "zh_ltp_small": { "runs": zh_stat.runs, "avg_ms_per_run": zh_stat.avg_ms(),
                              "note": "2 runs per Chinese clause (CWS pass + POS/dep pass), warmup excluded" },
            "en_tagger": { "runs": en_tag.runs, "avg_ms_per_run": en_tag.avg_ms() },
            "en_parser": { "runs": en_par.runs, "avg_ms_per_run": en_par.avg_ms() },
        },
        "latency_ms_per_input": {
            "note": "wall clock of one group annotate() call (all sentences in that string), after warmup",
            "samples": latencies_ms.len(), "avg": lat_avg, "p50": p50, "p95": p95,
            "min": sorted.first().copied().unwrap_or(0.0), "max": sorted.last().copied().unwrap_or(0.0),
        },
        "sentences_total": n_sentences,
        "offset_errors": offset_errors,
        "peak_rss_mib": ortutil::peak_rss_mib(),
        "rss_mib_end": ortutil::rss_mib(),
        "cpu_time_s": ortutil::cpu_time(),
        "total_wall_s": t_run_start.elapsed().as_secs_f64(),
    });
    std::fs::write(
        out_dir.join("metrics.json"),
        serde_json::to_string_pretty(&metrics)?,
    )?;

    // ---- spaCy comparison ------------------------------------------------
    if let Some(dir) = compare_dir {
        std::fs::create_dir_all(&dir)?;
        for g in &en_groups {
            let name = g.name.trim_start_matches("en_compare[").trim_end_matches(']');
            let file = dir.join(format!("en_{name}.json"));
            std::fs::write(
                &file,
                serde_json::to_string_pretty(&serde_json::json!({
                    "text": g.text,
                    "sentences": g.sentences,
                }))?,
            )?;
        }
        eprintln!("wrote {} en_compare sentence files to {}", en_groups.len(), dir.display());
    }

    eprintln!(
        "done: {} groups, {} sentences, offset errors {}, peak rss {:?} MiB",
        groups.len() + en_groups.len(),
        n_sentences,
        offset_errors,
        ortutil::peak_rss_mib()
    );
    if offset_errors > 0 {
        anyhow::bail!("{offset_errors} tokens failed the start/end slice check");
    }
    Ok(())
}

fn to_json(sents: &[SentenceOut]) -> Vec<SentenceJson> {
    sents
        .iter()
        .map(|s| SentenceJson {
            index: s.index,
            start: s.start,
            end: s.end,
            text: s.text.clone(),
            tokens: s.tokens.iter().map(TokenOut::from).collect(),
            // (offsets in `tokens` are absolute into the group text)
        })
        .collect()
}

// ------------------------------------------------------------------------- //
// HTML
// ------------------------------------------------------------------------- //

fn class_for(t: &TokenOut) -> Option<&'static str> {
    match t.pos.as_str() {
        "NOUN" => Some("noun"),
        "VERB" => Some("verb"),
        "ADJ" => Some("adj"),
        "ADV" => Some("adv"),
        _ => None,
    }
}

fn render_html(groups: &[GroupJson], en_groups: &[GroupJson]) -> String {
    let mut h = String::new();
    h.push_str(
        r#"<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>quiet-md · 离线神经词性 + 依存 spike</title>
<style>
  :root { color-scheme: light; }
  body { margin:0; padding:26px 34px 40px; background:#fbfaf7; color:#1c1b19; max-width:1080px;
         font-family:-apple-system,"Segoe UI","Noto Sans CJK SC","Noto Sans",sans-serif; }
  h1 { font-size:20px; margin:0 0 4px; font-weight:650; }
  h2 { font-size:15px; margin:26px 0 6px; padding-bottom:4px; border-bottom:1px solid #e6e3dc;
       font-weight:600; color:#3b3a37; letter-spacing:.02em; }
  .sub { color:#6b6a66; font-size:13px; margin:0 0 14px; line-height:1.5; }
  .legend { display:flex; gap:12px 28px; flex-wrap:wrap; font-size:14px; color:#333;
            padding:12px 16px; background:#fff; border:1px solid #e6e3dc; border-radius:8px; }
  .sw { display:inline-block; width:11px; height:11px; border-radius:2px; vertical-align:-1px; margin-right:5px; }
  .sent { font-size:22px; line-height:2.0; margin:2px 0 8px; }
  .tok { white-space:pre-wrap; }
  .noun{color:#2456AE} .verb{color:#8E1F2C} .adj{color:#00783E} .adv{color:#7A4A1E}
  .conn { border-bottom:2px solid #333; }
  .nonfinite { border-bottom:2px dashed #8a5cd6; }
  .core { text-emphasis: filled dot; -webkit-text-emphasis: filled dot; text-emphasis-color:#c02a2a; }
  .meta { font-size:11.5px; color:#98968e; margin:0 0 16px; font-family:ui-monospace,Menlo,monospace; }
  .pill { display:inline-block; padding:0 6px; border-radius:9px; background:#efece4; color:#6b6a66;
          margin-right:6px; font-size:11px; }
</style></head><body>
<h1>quiet-md · 离线神经词性 + 依存 spike</h1>
<p class="sub">中文 <b>LTP/small</b>（分词+词性+依存）· 英文 <b>roberta-large-english-upos</b>（UPOS+SuPar 依存）·
全部 CPU / ONNX Runtime / Rust <code>ort</code>。分句 ICU，句内按汉/拉丁脚本切分后分别推理，偏移映射回原文。<br>
颜色：<b style="color:#2456AE">名词</b> <b style="color:#8E1F2C">动词</b> <b style="color:#00783E">形容词</b>
<b style="color:#7A4A1E">副词</b>；其余词性不著色。</p>
<div class="legend">
  <span><span class="sw" style="background:#2456AE"></span>名词</span>
  <span><span class="sw" style="background:#8E1F2C"></span>动词</span>
  <span><span class="sw" style="background:#00783E"></span>形容词</span>
  <span><span class="sw" style="background:#7A4A1E"></span>副词</span>
  <span><span class="conn">连词/从句标记</span>&nbsp;实线</span>
  <span><span class="nonfinite">非谓语</span>&nbsp;英文虚线</span>
  <span><span class="core">核心谓语</span>&nbsp;着重号（filled dot）</span>
</div>
"#,
    );
    let titles = [
        ("error_cases.old-zh-nr", "中文段落（旧原型把 安静/金黄/叶子 标成人名）"),
        ("error_cases.old-en-nn", "英文段落（旧原型 clever/tired/frightened 全是 NN）"),
        ("error_cases.mixed", "中英夹杂（ICU 分句 → 汉/拉丁脚本切分 → 两个模型分别推理）"),
    ];
    for g in groups {
        if g.name.starts_with("zh_extra") {
            continue;
        }
        let title = titles
            .iter()
            .find(|(n, _)| *n == g.name)
            .map(|(_, t)| (*t).to_string())
            .unwrap_or_else(|| g.name.clone());
        h.push_str(&format!("<h2>{}</h2>\n", esc(&title)));
        for s in &g.sentences {
            render_sentence(&mut h, s, &g.text);
        }
    }
    h.push_str("<h2>中文关联词（实线下划线 = 连词）</h2>\n");
    for g in groups.iter().filter(|g| g.name.starts_with("zh_extra")) {
        for s in &g.sentences {
            render_sentence(&mut h, s, &g.text);
        }
    }
    h.push_str("<h2>英文对照（20 句，与 spaCy en_core_web_trf 对比）</h2>\n");
    for g in en_groups {
        for s in &g.sentences {
            render_sentence(&mut h, s, &g.text);
        }
    }
    h.push_str("</body></html>\n");
    h
}

fn render_sentence(h: &mut String, s: &SentenceJson, group_text: &str) {
    h.push_str("<div class=\"sent\">");
    let bytes = group_text.as_bytes();
    let end = s.end.min(bytes.len());
    let mut cursor = s.start.min(bytes.len());
    for t in &s.tokens {
        let ts = t.start.min(bytes.len());
        if ts > cursor {
            if let Ok(gap) = std::str::from_utf8(&bytes[cursor..ts]) {
                h.push_str(&esc(gap));
            }
        }
        let mut cls = String::from("tok");
        if let Some(c) = class_for(t) {
            cls.push(' ');
            cls.push_str(c);
        }
        if t.is_connective {
            cls.push_str(" conn");
        }
        if t.is_nonfinite {
            cls.push_str(" nonfinite");
        }
        if t.is_core_verb {
            cls.push_str(" core");
        }
        let title = format!(
            "{} · {} · head={} {}",
            t.pos, t.upos_or_ltp_tag, t.head, t.deprel
        );
        h.push_str(&format!(
            "<span class=\"{cls}\" title=\"{}\">{}</span>",
            esc(&title),
            esc(&t.text)
        ));
        cursor = t.end.min(bytes.len());
    }
    if cursor < end {
        if let Ok(tail) = std::str::from_utf8(&bytes[cursor..end]) {
            h.push_str(&esc(tail));
        }
    }
    h.push_str("</div>\n");
    // compact annotation line: only marked tokens
    let mut marks = Vec::new();
    for t in &s.tokens {
        if t.is_core_verb || t.is_connective || t.is_nonfinite {
            let mut m = Vec::new();
            if t.is_core_verb {
                m.push("核心谓语");
            }
            if t.is_connective {
                m.push("连词");
            }
            if t.is_nonfinite {
                m.push("非谓语");
            }
            marks.push(format!("<b>{}</b> {}", esc(&t.text), m.join("+")));
        }
    }
    h.push_str(&format!(
        "<div class=\"meta\"><span class=\"pill\">#{}</span>{} {}</div>\n",
        s.index,
        esc(&format!("{} tokens", s.tokens.len())),
        marks.join(" · ")
    ));
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[allow(dead_code)]
fn _assert_paths(p: &Path) -> &Path {
    p
}
