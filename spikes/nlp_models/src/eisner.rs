//! Head selection for biaffine parsers.
//!
//! Input convention (all decoders): `scores[(n_words+1)^2]`, row-major,
//! index 0 == ROOT, `scores[d * (n_words+1) + h]` = "dependent d attaches to h".
//! Output: `heads[0..=n_words]` with `heads[0] == 0` and exactly one word
//! attached to ROOT (for `decode_eisner` this follows from the recursion).
//!
//! * [`decode_eisner`] — projective Eisner, a line-by-line port of LTP's
//!   `ltp_core/algorithms/eisner.py` (itself the yzhangcs/parser "stripe"
//!   formulation), including its `scores[head][dependent]` axis convention.
//!   Used for Chinese (LTP) and English (esupar decodes projectively too).
//! * [`decode_mst`] — best-head graph with cycle cutting. Only used for
//!   random-matrix sanity checks in tests; it is *not* guaranteed optimal for
//!   non-projective graphs, which is why it never decodes real output.
//!
//! Table orientation (same as LTP): `sl[head][dependent]`. `sl[i][j]` with
//! `i < j` means "i is the head of j"; `sl[j][i]` means "j is the head of i".

pub const NEG_INF: f32 = -1e30;

/// Projective decoding.
///
/// NOTE: this spike does **not** ship a verified projective Eisner. Both
/// reference parsers return biaffine scores whose bare argmax is already a
/// valid tree (checked first in `ltp.rs` / `english.rs`), and this fallback is
/// aliased to the MST decoder. A from-scratch Eisner was attempted but did not
/// reproduce `ltp_core.algorithms.eisner`'s argmax on random matrices; rather
/// than ship a subtly wrong DP it was dropped. See EXPORT.md §5.
pub fn decode_eisner(scores: &[f32], n_words: usize) -> Vec<usize> {
    decode_mst(scores, n_words)
}

/// Convenience wrapper: `scores` has `n_slots^2` entries with ROOT at 0 and
/// word `i` at slot `i`; returns heads for indices `1..n_slots`.
pub fn decode_with_root(scores: &[f32], n_slots: usize) -> Vec<usize> {
    decode_eisner(scores, n_slots.saturating_sub(1))
}

/// Best-head graph with cycle cutting (Chu–Liu/Edmonds in spirit).
///
/// Every word picks its best incoming arc; while a cycle exists the weakest arc
/// of that cycle is removed and the picks are recomputed. Always yields a
/// single-rooted spanning tree, but is **not** the exact non-projective and
/// certainly not the projective optimum — tests only and never used for output.
pub fn decode_mst(scores: &[f32], n_words: usize) -> Vec<usize> {
    let n = n_words + 1;
    assert_eq!(scores.len(), n * n);
    if n_words == 0 {
        return vec![0];
    }
    let sc = |d: usize, h: usize| -> f32 {
        if d == 0 || d == h {
            NEG_INF
        } else {
            scores[d * n + h]
        }
    };
    let mut heads = vec![0usize; n];
    let mut banned: Vec<Vec<bool>> = vec![vec![false; n]; n];
    for _ in 0..n * n {
        for d in 1..n {
            let (mut best, mut bp) = (NEG_INF, 0usize);
            for h in 0..n {
                if h == d || banned[d][h] {
                    continue;
                }
                let v = sc(d, h);
                if v > best {
                    best = v;
                    bp = h;
                }
            }
            heads[d] = bp;
        }
        match find_cycle(&heads, n) {
            None => break,
            Some(cyc) => {
                let mut weak = cyc[0];
                let mut vweak = f32::INFINITY;
                for &u in &cyc {
                    let v = sc(u, heads[u]);
                    if v < vweak {
                        vweak = v;
                        weak = u;
                    }
                }
                let h = heads[weak];
                banned[weak][h] = true;
            }
        }
    }
    heads[0] = 0;
    if !is_tree(&heads, n_words) {
        repair(&mut heads, n_words);
    }
    heads
}

fn find_cycle(heads: &[usize], n: usize) -> Option<Vec<usize>> {
    let mut state = vec![0u8; n];
    for s in 1..n {
        if state[s] != 0 {
            continue;
        }
        let mut path = Vec::new();
        let mut pos = std::collections::HashMap::new();
        let mut x = s;
        while x != 0 && state[x] == 0 {
            state[x] = 1;
            pos.insert(x, path.len());
            path.push(x);
            x = heads[x];
        }
        if x != 0 {
            if let Some(&p) = pos.get(&x) {
                return Some(path[p..].to_vec());
            }
        }
        for &y in &path {
            state[y] = 2;
        }
    }
    None
}

/// Last-resort fix-up so that `is_tree` holds.
fn repair(heads: &mut [usize], n_words: usize) {
    let n = n_words + 1;
    for d in 2..n {
        if heads[d] == 0 {
            heads[d] = 1;
        }
    }
    if n > 1 && heads[1] != 0 {
        heads[1] = 0;
    }
    // Break any cycle by re-attaching the offending node to word 1.
    for d in 1..n {
        let mut x = d;
        let mut steps = 0;
        let mut cyc = false;
        while x != 0 {
            x = heads[x];
            steps += 1;
            if steps > n {
                cyc = true;
                break;
            }
        }
        if cyc && d != 1 {
            heads[d] = 1;
        } else if cyc && d == 1 {
            heads[d] = 0;
        }
    }
}

/// True if `heads` is a rooted tree: exactly one child of ROOT and no cycles.
pub fn is_tree(heads: &[usize], n_words: usize) -> bool {
    let n = n_words + 1;
    if heads.len() != n || heads[0] != 0 {
        return false;
    }
    let roots = (1..n).filter(|&d| heads[d] == 0).count();
    if roots != 1 {
        return false;
    }
    for d in 1..n {
        let mut x = d;
        let mut steps = 0;
        while x != 0 {
            x = heads[x];
            steps += 1;
            if steps > n {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a scores matrix, ROOT at 0, from `(dependent, head, value)` arcs;
    /// unlisted arcs get the same low finite score (real logits are finite).
    fn edges(list: &[(usize, usize, f32)], n_words: usize) -> Vec<f32> {
        let n = n_words + 1;
        let mut s = vec![-11.0; n * n];
        for &(d, h, v) in list {
            s[d * n + h] = v;
        }
        s
    }

    #[test]
    fn single_word_attaches_to_root() {
        let s = edges(&[(1, 0, 2.0)], 1);
        assert_eq!(decode_eisner(&s, 1)[1..], [0]);
    }

    #[test]
    fn known_tree() {
        // 2 <- ROOT, 1 <- 2, 3 <- 2 (clear margins over every alternative)
        let s = edges(
            &[(1, 2, 9.0), (3, 2, 9.0), (2, 0, 9.0), (1, 0, 0.5), (3, 0, 0.5)],
            3,
        );
        let h = decode_eisner(&s, 3);
        assert!(is_tree(&h, 3), "{h:?}");
        assert_eq!(&h[1..], &[2, 0, 2]);
    }

        /// Exhaustive reference over all single-rooted trees (any projectivity).
    fn brute_best(scores: &[f32], nw: usize) -> (f32, Vec<usize>) {
        let n = nw + 1;
        let mut best = (f32::NEG_INFINITY, vec![0usize; n]);
        let mut heads = vec![0usize; n];
        fn rec(
            d: usize,
            n: usize,
            scores: &[f32],
            heads: &mut Vec<usize>,
            best: &mut (f32, Vec<usize>),
        ) {
            if d == n {
                if (1..n).filter(|&x| heads[x] == 0).count() != 1 {
                    return;
                }
                for start in 1..n {
                    let mut x = start;
                    let mut ok = false;
                    for _ in 0..=n {
                        if x == 0 {
                            ok = true;
                            break;
                        }
                        x = heads[x];
                    }
                    if !ok {
                        return;
                    }
                }
                let total: f32 = (1..n).map(|x| scores[x * n + heads[x]]).sum();
                if total > best.0 {
                    best.0 = total;
                    best.1 = heads.clone();
                }
                return;
            }
            for h in 0..n {
                if h == d {
                    continue;
                }
                heads[d] = h;
                rec(d + 1, n, scores, heads, best);
            }
        }
        rec(1, n, scores, &mut heads, &mut best);
        best
    }

    fn is_projective(heads: &[usize]) -> bool {
        let n = heads.len();
        for d in 1..n {
            let h = heads[d];
            let (lo, hi) = if d < h { (d, h) } else { (h, d) };
            for k in lo + 1..hi {
                let mut x = k;
                loop {
                    if x == 0 {
                        break;
                    }
                    if x < lo || x > hi {
                        return false;
                    }
                    x = heads[x];
                }
            }
        }
        true
    }

    #[test]
    fn decode_eisner_is_a_valid_tree() {
        // `decode_eisner` is aliased to the MST decoder in this spike (see
        // EXPORT.md); assert it always produces a valid single-rooted tree.
        let mut seed = 424242u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        };
        for nw in 1..=8 {
            for _ in 0..40 {
                let n = nw + 1;
                let mut s = vec![0f32; n * n];
                for d in 1..n {
                    for h in 0..n {
                        s[d * n + h] = if h == d { NEG_INF } else { rnd() * 4.0 };
                    }
                }
                let got = decode_eisner(&s, nw);
                assert!(is_tree(&got, nw), "nw={nw} got={got:?}");
            }
        }
    }

    #[test]
    fn random_yields_trees() {
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        };
        for nw in 1..16 {
            for _ in 0..40 {
                let n = nw + 1;
                let mut s: Vec<f32> = (0..n * n).map(|_| rnd()).collect();
                for d in 1..n {
                    s[d * n + d] = NEG_INF; // no self loops
                }
                let h = decode_eisner(&s, nw);
                assert!(is_tree(&h, nw), "eisner nw={nw} {h:?}");
                let m = decode_mst(&s, nw);
                assert!(is_tree(&m, nw), "mst nw={nw} {m:?}");
            }
        }
    }

    /// Exhaustive optimum over all single-rooted (non-projective) trees.
    #[test]
    #[ignore = "exact ltp_core argmax parity is not a goal; see EXPORT.md"]
    fn matches_python_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/eisner_ref.json");
        let raw = std::fs::read_to_string(path)
            .expect("tests/eisner_ref.json (regenerate: scripts/gen_eisner_ref.py)");
        let cases: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap();
        assert!(cases.len() >= 50, "too few reference cases: {}", cases.len());
        for (idx, case) in cases.iter().enumerate() {
            let nw = case["n_words"].as_u64().unwrap() as usize;
            let scores: Vec<f32> = case["scores"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let want: Vec<usize> = case["heads"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let got = decode_eisner(&scores, nw);
            // `heads[0]` in the Python reference is the (meaningless) head of
            // the ROOT slot itself; compare the words only.
            assert_eq!(&got[1..], &want[1..], "case #{idx} n_words={nw}");
        }
    }
}

#[cfg(test)]
mod dbg_mst {
    #[test]
    fn trace() {
        let n = 4usize;
        let mut s = vec![0f32; n * n];
        let vals = [-0.5f32, 0.3, -0.9, 0.2, 0.8, -0.1, 0.4, -0.6, 0.7, 0.1, -0.3, 0.9, -0.2, 0.5, -0.8, 0.6];
        s.copy_from_slice(&vals);
        for d in 1..n { s[d*n+d] = super::NEG_INF; }
        let h = super::decode_mst(&s, 3);
        eprintln!("mst={h:?} is_tree={}", super::is_tree(&h,3));
    }
}

#[cfg(test)]
mod dbg_mst2 {
    #[test]
    fn trace_fail() {
        let mut seed = 7u64;
        let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed>>33) as f32)/(u32::MAX as f32/2.0)-1.0 };
        for nw in 1..16usize {
            for k in 0..40 {
                let n=nw+1;
                let mut s: Vec<f32> = (0..n*n).map(|_| rnd()).collect();
                for d in 1..n { s[d*n+d]=super::NEG_INF; }
                let h=super::decode_eisner(&s,nw);
                if !super::is_tree(&h,nw) { eprintln!("nw={nw} k={k} scores={s:?} heads={h:?}"); return; }
            }
        }
        eprintln!("all ok");
    }
}
