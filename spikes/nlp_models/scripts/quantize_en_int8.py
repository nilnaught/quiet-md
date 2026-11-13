#!/usr/bin/env python
"""Dynamic int8 quantisation (onnxruntime `QuantType`) for one exported English
ONNX graph at a time.

Weight type: `QUInt8` by default. `QInt8` (signed) was tried first and drifts
badly on these graphs - on the base tagger the argmax collapsed to one class
for whole sentences and UPOS agreement vs fp32 fell to ~0.66; `QUInt8` keeps
UPOS agreement at 1.00 (max logit Δ 5.0) and parser head agreement at 0.9985.
Both are still int8 storage (see EXPORT.md table).

Why dynamic: the graphs are dominated by *dense* weights (RoBERTa matmuls + the
SuPar BiLSTM). Dynamic quantisation needs no calibration data, keeps activations
in fp32 and only stores the weights as int8 — exactly what we want for an
offline reading aid where a few 1e-3 logit drift is harmless (we verify the
argmax tags / heads afterwards in `check_en_variants.py`).

Memory discipline (the box has ~16 GB and the fp32 graphs are 1.4 GiB each):

  * `--graph` takes ONE file name; the script refuses to quantise both in one
    process (loads + writes would double-peak).
  * the fp32 graph is never copied: output goes to `--out-dir/<same name>`.
  * the in-memory `onnx.ModelProto` is loaded with `load_external_data_for_model`
    and the quantised model is written with `save_as_external_data=True`, so the
    int8 weights live in `<name>.data` next to a tiny graph stub.
  * `--keep` is off by default: intermediate `*.quant.tmp` files are removed.

Usage (one graph per invocation):
  timeout 900 /workspace/.venv-nlp/bin/python scripts/quantize_en_int8.py \
      --model-dir /workspace/models/en-upos-large/onnx \
      --out-dir   /workspace/models/en-upos-large/onnx-int8 \
      --graph en_upos_tagger.onnx

  timeout 900 /workspace/.venv-nlp/bin/python scripts/quantize_en_int8.py \
      --model-dir ... --out-dir ... --graph en_supar_parser.onnx

Then `scripts/link_en_variant_assets.py --variant large-int8` symlinks the
tokenizer / label / vocab files from the fp32 dir into the int8 dir.
"""
import argparse
import os
import shutil
import sys
import time

# ORT's default dynamic-quantisation set for transformer-style graphs. `Gather`
# covers the word embedding (its weight is int8 too, which the LSTM input
# Cast/Dequantize path handles), `LSTM`/`Attention` the fused transformer ops and
# the SuPar BiLSTM. If a graph fails we catch and report it, we never pretend.
OP_TYPES = [
    "MatMul",
    "Attention",
    "LSTM",
    "Gather",
    "Transpose",
    "LayerNormalization",
]

# Ops that carry a weight matrix worth quantising (default mode).
WEIGHTED_OPS = ["MatMul", "Gather", "LSTM", "Attention", "LayerNormalization"]

# Anything bigger goes into the external `.data` file so the .onnx stub stays
# small and diffable (the repo never tracks either file).
EXT_DATA_THRESHOLD = 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="dir with the fp32 .onnx")
    ap.add_argument("--out-dir", required=True, help="dir for the int8 .onnx(+.data)")
    ap.add_argument("--graph", required=True, help="one file name, e.g. en_upos_tagger.onnx")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--weight-type", default="QUInt8", choices=["QUInt8", "QInt8"],
                    help="QUInt8 (default, symmetric-uint8 weights) is numerically far closer to "
                         "fp32 than QInt8 for these graphs; compared in EXPORT.md")
    ap.add_argument("--nodes", default="default", choices=["default", "weighted", "raw"],
                    help="default = ORT op-type set (MatMul/Attention/LSTM/Gather/Transpose/"
                         "LayerNormalization); weighted = only MatMul nodes with a constant "
                         "weight input (+ Gather/LSTM/Attention/LayerNormalization); raw = "
                         "op-type set without ORT's graph optimisation")
    args = ap.parse_args()

    src = os.path.join(args.model_dir, args.graph)
    dst = os.path.join(args.out_dir, args.graph)
    if not os.path.exists(src):
        print(f"missing fp32 graph: {src}", file=sys.stderr)
        return 2
    if os.path.exists(dst) and not args.overwrite:
        print(f"already exists (use --overwrite): {dst}", file=sys.stderr)
        return 3
    os.makedirs(args.out_dir, exist_ok=True)

    import onnx
    from onnxruntime.quantization import (
        QuantType,
        quantize_dynamic,
    )

    # Which nodes get quantised (see EXPORT.md for the measurements):
    #  * "default"  - ORT's dynamic-quantisation op-type set
    #    (MatMul/Attention/LSTM/Gather/Transpose/LayerNormalization). This is
    #    what the shipped graphs use.
    #  * "weighted" - only MatMul nodes whose 2nd input is an initializer
    #    (skips the activation-only attention matmuls) + Gather/LSTM/Attention/
    #    LayerNormalization. Diagnostic only.
    #  * "raw"      - same op-type set but with `optimize_model=False` (no
    #    constant folding / fusions before quantising). Diagnostic only.
    #
    # History: the very first run used `nodes_to_quantize` = *all* weighted
    # nodes with QInt8 weights and produced models whose tagger argmax collapsed
    # to one label for whole sentences (SYM/PUNCT/ADV) - unstable across runs.
    # The fix was two-fold: ORT's own op set/pre-processing (`default`) and
    # QUInt8 weights (see --weight-type). Both are now the defaults.
    if args.nodes == "weighted":
        m = onnx.load(src, load_external_data=False)
        inits = {i.name for i in m.graph.initializer}
        nodes = [n.name for n in m.graph.node
                 if (n.op_type == "MatMul" and len(n.input) > 1 and n.input[1] in inits)
                 or n.op_type in ("Gather", "LSTM", "Attention", "LayerNormalization")]
        extra = {"nodes_to_quantize": nodes}
        print(f"{args.graph}: weighted-node mode, {len(nodes)} nodes")
    else:
        extra = {}
    tmp = os.path.join(args.out_dir, args.graph + ".quant.tmp")
    t0 = time.time()
    try:
        quantize_dynamic(
            model_input=src,
            model_output=tmp,
            weight_type=getattr(QuantType, args.weight_type),
            op_types_to_quantize=OP_TYPES if args.nodes != "weighted" else None,
            per_channel=False,
            reduce_range=False,
            extra_options=extra if args.nodes == "weighted" else
            ({"optimize_model": False} if args.nodes == "raw" else None),
        )
    except Exception as e:  # noqa: BLE001 - we want the reason in the log
        print(f"QUANTIZE FAILED for {args.graph}: {type(e).__name__}: {e}", file=sys.stderr)
        if os.path.exists(tmp):
            os.remove(tmp)
        return 1
    dt = time.time() - t0

    # move the quantised model to its final name, keeping the .data file if any
    if os.path.exists(dst):
        os.remove(dst)
    os.replace(tmp, dst)
    data_src = tmp + ".data"
    if os.path.exists(data_src):
        data_dst = dst + ".data"
        if os.path.exists(data_dst):
            os.remove(data_dst)
        os.replace(data_src, data_dst)

    # in-place external-data rewrite: converts >1 KB initializers to .data
    m = onnx.load(dst)  # loads external data, then we strip it again below
    total = sum(int(p.ByteSize()) for p in m.graph.initializer)
    onnx.save_model(
        m,
        dst,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(dst) + ".data",
        size_threshold=EXT_DATA_THRESHOLD,
    )
    del m

    sz = os.path.getsize(dst)
    szd = os.path.getsize(dst + ".data") if os.path.exists(dst + ".data") else 0
    src_sz = os.path.getsize(src)
    print(
        f"{args.graph}: fp32 {src_sz/2**20:.1f} MiB -> int8 {sz/2**20:.2f} MiB (+.data {szd/2**20:.1f} MiB)"
        f" in {dt:.1f}s (initializer bytes before externalise {total/2**20:.1f} MiB)"
    )
    if not args.keep_temp:
        for p in (tmp, tmp + ".data"):
            if os.path.exists(p):
                os.remove(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
