//! Shared model plumbing: process-wide ORT environment, tensor helpers, and
//! the one place a session is built.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::OnceLock;

static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

/// Process-wide ORT environment (created once, shared by every session).
pub fn env() -> &'static ort::environment::Environment {
    ENV.get_or_init(|| {
        ort::init()
            .with_name("quiet-md-reader-nlp")
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
    let builder = match builder.with_intra_threads(threads) {
        Ok(b) => b,
        Err(e) => return Err(anyhow::anyhow!("ORT session options: {e}")),
    };
    let mut builder = match builder.with_inter_threads(1) {
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
