//! Per-model inference statistics collected by the CLI (defined here so both
//! pipelines share one type without a cyclic module dependency).

#[derive(Debug, Default, Clone)]
pub struct ModelStat {
    pub files: Vec<(String, u64)>,
    pub load_ms: f64,
    pub runs: usize,
    pub infer_ms_total: f64,
}

impl ModelStat {
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|(_, b)| *b).sum()
    }
    pub fn avg_ms(&self) -> f64 {
        if self.runs == 0 {
            0.0
        } else {
            self.infer_ms_total / self.runs as f64
        }
    }
}
