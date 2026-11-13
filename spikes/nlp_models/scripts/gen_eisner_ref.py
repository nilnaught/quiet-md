#!/usr/bin/env python
"""Generate Eisner golden vectors from the *Python* ltp_core reference.

    /workspace/.venv-nlp/bin/python scripts/gen_eisner_ref.py > tests/eisner_ref.json

Each case is a random square score matrix (dependent x head, index 0 = ROOT)
evaluated with `ltp_core.algorithms.eisner.eisner` on a length-1 batch. Note
that ltp_core indexes scores as `[head, dependent]`, so the script transposes
before calling it; the Rust `decode_eisner` must reproduce the heads exactly.
"""
import json
import random
import sys

import torch

from ltp_core.algorithms.eisner import eisner


def main():
    rng = random.Random(20261113)
    cases = []
    for n_words in list(range(1, 13)) + [16, 21, 30]:
        for _ in range(4 if n_words < 9 else 3):
            n = n_words + 1
            scores = [[round(rng.uniform(-8, 8), 4) for _ in range(n)] for _ in range(n)]
            t = torch.tensor([scores], dtype=torch.float32)
            mask = torch.ones(1, n, dtype=torch.bool)
            # ltp_core's eisner permutes scores to [head, dependent]; we pass
            # ours as dependent x head, so transpose to feed it the same tree.
            heads = eisner(t.clone().transpose(1, 2), mask)[0].tolist()
            cases.append({"n_words": n_words, "scores": [x for row in scores for x in row],
                          "heads": heads})
    json.dump(cases, sys.stdout)


if __name__ == "__main__":
    main()
