//! Shared model plumbing: process-wide ORT environment, tensor helpers, timed
//! `session.run`, and process memory probes for `metrics.json`.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use crate::metrics::ModelStat;

static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

/// Process-wide ORT environment (created once, shared by every session).
pub fn env() -> &'static ort::environment::Environment {
    ENV.get_or_init(|| {
        ort::init()
            .with_name("quiet-md-nlp-spike")
            .with_telemetry(false)
            .build()
            .expect("ORT environment")
    })
}

/// Build a session with a fixed intra-op thread count.
pub fn session(path: &Path, threads: usize) -> Result<Session> {
    // `ort::Error<SessionBuilder>` is not Send/Sync, so these are flattened by
    // hand instead of using `?`.
    let builder = match Session::builder(env()) {
        Ok(b) => b,
        Err(e) => return Err(anyhow::anyhow!("ORT session builder: {e}")),
    };
    let mut builder = match builder
        .with_intra_threads(threads)
        .and_then(|b| b.with_inter_threads(1))
    {
        Ok(b) => b,
        Err(e) => return Err(anyhow::anyhow!("ORT session options: {e}")),
    };
    builder
        .commit_from_file(path)
        .with_context(|| format!("loading {}", path.display()))
}

/// 2-D int64 input tensor (owned, no lifetime juggling around `session.run`).
pub fn i64_tensor(t: &[i64], shape: [usize; 2]) -> Result<Tensor<i64>> {
    let data = t.to_vec().into_boxed_slice();
    Tensor::from_array(([shape[0], shape[1]], data)).map_err(|e| anyhow::anyhow!("i64 tensor: {e}"))
}

/// 3-D int64 input tensor.
pub fn i64_tensor3(t: &[i64], shape: [usize; 3]) -> Result<Tensor<i64>> {
    let data = t.to_vec().into_boxed_slice();
    Tensor::from_array(([shape[0], shape[1], shape[2]], data))
        .map_err(|e| anyhow::anyhow!("i64 3d tensor: {e}"))
}

/// Extract an f32 tensor as (shape, owned data), ending the output borrow.
pub fn take_f32(
    out: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let (shape, data) = out[name]
        .try_extract_tensor::<f32>()
        .with_context(|| format!("extracting {name}"))?;
    Ok((shape.iter().map(|&d| d as usize).collect(), data.to_vec()))
}

/// Time one `session.run` and charge it to `stat` (runs + wall milliseconds).
pub fn timed<T>(stat: &mut ModelStat, f: impl FnOnce() -> T) -> (T, std::time::Duration) {
    let t0 = Instant::now();
    let out = f();
    let d = t0.elapsed();
    stat.runs += 1;
    stat.infer_ms_total += d.as_secs_f64() * 1000.0;
    (out, d)
}

fn status_kib(key: &str) -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let kb: f64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return Some(kb / 1024.0);
        }
    }
    None
}

/// Peak RSS (VmHWM) in MiB, Linux only.
pub fn peak_rss_mib() -> Option<f64> {
    status_kib("VmHWM:")
}

/// Current RSS in MiB.
pub fn rss_mib() -> Option<f64> {
    status_kib("VmRSS:")
}

/// Process CPU time (user+sys) in seconds, Linux only.
pub fn cpu_time() -> f64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/stat") else {
        return 0.0;
    };
    let fields: Vec<&str> = s.split_whitespace().collect();
    if fields.len() > 15 {
        let ut: f64 = fields[13].parse().unwrap_or(0.0);
        let st: f64 = fields[14].parse().unwrap_or(0.0);
        return (ut + st) / 100.0; // CLK_TCK == 100 on mainstream Linux builds
    }
    0.0
}

/// File sizes for the metrics report.
pub fn file_sizes(paths: &[&Path]) -> Vec<(String, u64)> {
    paths
        .iter()
        .filter_map(|p| {
            let m = std::fs::metadata(p).ok()?;
            Some((p.display().to_string(), m.len()))
        })
        .collect()
}
