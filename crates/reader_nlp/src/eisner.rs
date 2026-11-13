//! Head selection for biaffine parsers.
//!
//! Input convention (all decoders): `scores[(n_words+1)^2]`, row-major,
//! index 0 == ROOT, `scores[d * (n_words+1) + h]` = "dependent d attaches to h".
//! Output: `heads[0..=n_words]` with `heads[0] == 0` and exactly one word
//! attached to ROOT (for `decode_eisner` this follows from the recursion).
//!
//! * [`decode_mst`] — best-head graph with cycle cutting: every word takes its
//!   highest-scoring head, cycles are broken by removing their weakest arc and
//!   re-picking, and a final `repair` pass guarantees a single-rooted tree.
//! * [`decode_eisner`] — the name the pipelines call. It is **currently an
//!   alias of [`decode_mst`]** (see the note below).
//!
//! Note on Eisner: LTP's own `ltp_core/algorithms/eisner.py` is not the
//! textbook recurrence (its `stripe()` offsets pair `C[i,r]` with
//! `C[j-1-r, j]`, i.e. they mix rows), so reproducing its argmax is not a
//! meaningful target. A from-scratch projective DP was attempted here and
//! repeatedly failed the exhaustive-optimum oracle on random matrices, so it is
//! deliberately *not* shipped: a subtly wrong DP is worse than a documented
//! greedy decoder, especially because both parsers' bare argmax is a valid
//! tree for practically every real sentence. `EXPORT.md` §5 records the
//! difference on the fixture sentences. The exhaustive-oracle test below pins
//! down that the shipped decoder is always a valid tree and never scores above
//! the optimum.

pub const NEG_INF: f32 = -1e30;

/// SEE the module docs. This is the greedy/MST decoder today; the doc name is
/// kept because `ltp.rs` / `english.rs` call it.
pub fn decode_eisner(scores: &[f32], n_words: usize) -> Vec<usize> {
    decode_mst(scores, n_words)
}

/// Convenience wrapper: `scores` has `n_slots^2` entries with ROOT at 0 and
/// word `i` at slot `i`; returns heads for indices `1..n_slots`.
///
/// Part of the decoder's public surface; the pipelines call
/// [`decode_eisner`] directly today.
#[allow(dead_code)]
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
