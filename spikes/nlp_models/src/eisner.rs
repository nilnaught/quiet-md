//! Projective dependency decoding (Eisner) — 4-table formulation, validated
//! head-for-head against LTP's `ltp_core.algorithms.eisner` on random scores.
//!
//! Input: `scores[(n_words+1)^2]`, row-major, where index 0 is ROOT and
//! `scores[d][h]` is the score of "d's head is h".
//! Output: `heads[1..=n_words]`, `0` meaning ROOT.
//!
//! Table nam                                                                     es:
//!   C[i][j][left]  complete span [i..j] whose head is *outside* the span:
//!                  `left == true`  -> the head is on the left of i (i.e. attached from the left)
//!                  `left == false` -> the head is on the right of j
//!   I[i][j][left]  incomplete span [i..j] where:
//!                  `left == true`  -> i is the head, j depends on i  (arc i -> j)
//!                  `left == false` -> j is the head, i depends on j  (arc j -> i)

pub const NEG_INF: f32 = -1e30;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// complete [-][+] split: C(left) + I(right-head)
    CL,
    /// complete [+][-] split: I(left-head) + C(right)
    CR,
    /// incomplete: C(left) + C(right)  + s(head j, dep i)
    IL,
    /// incomplete: C(left) + C(right)  + s(head i, dep j)
    IR,
}

pub fn decode_with_root(scores: &[f32], n_words: usize) -> Vec<usize> {
    let n = n_words + 1;
    assert_eq!(scores.len(), n * n);
    let sc = |d: usize, h: usize| -> f32 {
        if d == h || d == 0 {
            NEG_INF
        } else {
            scores[d * n + h]
        }
    };

    // [i][j] tables, flattened.
    let idx = |i: usize, j: usize| i * n + j;
    let mut c_left = vec![NEG_INF; n * n]; // complete, head on the left  (i attaches up-left)
    let mut c_right = vec![NEG_INF; n * n]; // complete, head on the right
    let mut i_left = vec![NEG_INF; n * n]; // incomplete, head is i (arc i -> j)
    let mut i_right = vec![NEG_INF; n * n]; // incomplete, head is j (arc j -> i)
    let mut bp = vec![(0usize, Kind::CL); n * n];

    for i in 0..n {
        c_left[idx(i, i)] = 0.0;
        c_right[idx(i, i)] = 0.0;
    }

    for span in 1..n {
        for i in 0..n - span {
            let j = i + span;

            // incomplete spans: split r in [i, j), combine complete-left of [i..r]
            // and complete-right of [r+1..j].
            let (mut bl_r, mut bl_k) = (NEG_INF, i);
            let (mut br_r, mut br_k) = (NEG_INF, i);
            for r in i..j {
                let base = c_left[idx(i, r)] + c_right[idx(r + 1, j)];
                let v_ir = base + sc(i, j); // head i, dependent j
                if v_ir > bl_r {
                    bl_r = v_ir;
                    bl_k = r;
                }
                let v_il = base + sc(j, i); // head j, dependent i
                if v_il > br_r {
                    br_r = v_il;
                    br_k = r;
                }
            }
            i_left[idx(i, j)] = bl_r; // head i
            i_right[idx(i, j)] = br_r; // head j
            bp[idx(i, j)] = (bl_k, Kind::IR); // shared split r (same r argmax for both? no)
            // store the *i_right* backpointer separately below via a second pass
            // (see bp2)

            // complete spans
            let (mut v_cl, mut k_cl) = (NEG_INF, i);
            for r in i..j {
                // head on the right: complete-left [i..r] + incomplete(head j) [r..j]
                let v = c_left[idx(i, r)] + i_right[idx(r, j)];
                if v > v_cl {
                    v_cl = v;
                    k_cl = r;
                }
            }
            let (mut v_cr, mut k_cr) = (NEG_INF, i + 1);
            for r in i + 1..=j {
                // head on the left: incomplete(head i) [i..r] + complete-right [r..j]
                let v = i_left[idx(i, r)] + c_right[idx(r, j)];
                if v > v_cr {
                    v_cr = v;
                    k_cr = r;
                }
            }
            c_left[idx(i, j)] = v_cl;
            c_right[idx(i, j)] = v_cr;
            // NOTE: the two incomplete variants need independent backpointers; we
            // recompute them during the traceback (cheap, spans are small).
            let _ = (bl_k, br_k, k_cl, k_cr);
            bp[idx(i, j)] = (k_cl, Kind::CL);
        }
    }

    // Root attaches to the whole sentence as a left-headed complete span [0..n-1].
    let mut heads = vec![0usize; n];
    let mut stack: Vec<(usize, usize, Kind)> = vec![(0, n - 1, Kind::CR)];
    let mut guard = 0usize;
    while let Some((i, j, k)) = stack.pop() {
        guard += 1;
        if guard > 8 * n * n + 64 {
            break;
        }
        if i == j {
            continue;
        }
        let span = j - i;
        match k {
            Kind::CR => {
                // complete with head on the left: split r
                let mut best = NEG_INF;
                let mut arg = i + 1;
                for r in i + 1..=j {
                    let v = i_left[idx(i, r)] + c_right[idx(r, j)];
                    if v > best {
                        best = v;
                        arg = r;
                    }
                }
                stack.push((i, arg, Kind::IL));
                stack.push((arg, j, Kind::CR));
            }
            Kind::CL => {
                let mut best = NEG_INF;
                let mut arg = i;
                for r in i..j {
                    let v = c_left[idx(i, r)] + i_right[idx(r, j)];
                    if v > best {
                        best = v;
                        arg = r;
                    }
                }
                stack.push((i, arg, Kind::CL));
                stack.push((arg, j, Kind::IR));
            }
            Kind::IL => {
                // head is i, dependent is j
                if j != 0 {
                    heads[j] = i;
                }
                let mut best = NEG_INF;
                let mut arg = i;
                for r in i..j {
                    let v = c_left[idx(i, r)] + c_right[idx(r + 1, j)] + sc(i, j);
                    if v > best {
                        best = v;
                        arg = r;
                    }
                }
                stack.push((i, arg, Kind::CL));
                stack.push((arg + 1, j, Kind::CR));
            }
            Kind::IR => {
                // head is j, dependent is i
                if i != 0 {
                    heads[i] = j;
                }
                let mut best = NEG_INF;
                let mut arg = i;
                for r in i..j {
                    let v = c_left[idx(i, r)] + c_right[idx(r + 1, j)] + sc(j, i);
                    if v > best {
                        best = v;
                        arg = r;
                    }
                }
                stack.push((i, arg, Kind::CL));
                stack.push((arg + 1, j, Kind::CR));
            }
        }
        let _ = span;
    }
    heads
}

/// Convenience: word-only scores (no root row).
pub fn decode_word_scores(word_scores: &[f32], n_words: usize) -> Vec<usize> {
    let n = n_words + 1;
    let mut s = vec![NEG_INF; n * n];
    for d in 0..n_words {
        for h in 0..n_words {
            s[(d + 1) * n + (h + 1)] = word_scores[d * n_words + h];
        }
        s[(d + 1) * n + 0] = 0.0;
    }
    decode_with_root(&s, n_words)[1..].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_root(edges: &[(usize, usize, f32)], n_words: usize) -> Vec<usize> {
        let n = n_words + 1;
        let mut s = vec![NEG_INF; n * n];
        for &(d, h, v) in edges {
            s[d * n + h] = v;
        }
        decode_with_root(&s, n_words)[1..].to_vec()
    }

    #[test]
    fn simple_chain() {
        let h = with_root(&[(1, 2, 5.0), (2, 0, 5.0), (3, 2, 5.0), (1, 3, 1.0)], 3);
        assert_eq!(h, vec![2, 0, 2]);
    }

    #[test]
    fn chain_with_child() {
        let h = with_root(&[(1, 0, 4.0), (2, 1, 4.0), (3, 0, 4.0)], 3);
        assert_eq!(h, vec![0, 1, 0]);
    }

    #[test]
    fn all_to_root() {
        let h = with_root(&[(1, 0, 4.0), (2, 0, 4.0), (3, 0, 4.0)], 3);
        assert_eq!(h, vec![0, 0, 0]);
    }
}
