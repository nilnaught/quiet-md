mod eisner;
mod text;

use std::io::Read;

fn main() {
    // read score matrix (n_words+1)^2 i64-repr floats from stdin: first line n_words
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let mut it = s.split_whitespace();
    let n_words: usize = it.next().unwrap().parse().unwrap();
    let n = n_words + 1;
    let mut sc = vec![0f32; n * n];
    for i in 0..n*n { sc[i] = it.next().unwrap().parse().unwrap(); }
    let heads = eisner::decode_with_root(&sc, n_words);
    let out: Vec<String> = heads[1..].iter().map(|h| h.to_string()).collect();
    println!("{}", out.join(" "));
}
