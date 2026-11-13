#!/bin/bash
# Copy the shipped NLP models (LTP/small + English base-int8) into DEST.
# Source: $QUIET_MD_NLP_MODELS, else $MODELS_DIR.
# Missing source is a no-op so regular CI packages stay model-free.
# Symlinks in the source tree are flattened (int8 sidecars often point at fp32).
set -euo pipefail
dest=${1:?usage: bundle-nlp-models.sh DEST}
src=${QUIET_MD_NLP_MODELS:-${MODELS_DIR:-}}
if [ -z "$src" ] || [ ! -d "$src" ]; then
  echo "bundle-nlp-models: no QUIET_MD_NLP_MODELS/MODELS_DIR, skipping" >&2
  exit 0
fi
# Accept either the models root or a tree with a models/ subdirectory.
if [ -d "$src/ltp-small" ]; then
  root=$src
elif [ -d "$src/models/ltp-small" ]; then
  root=$src/models
else
  echo "bundle-nlp-models: $src has no ltp-small/, skipping" >&2
  exit 0
fi
ltp=$root/ltp-small/onnx
en=$root/en-upos-base/onnx-int8
if [ ! -f "$ltp/ltp_small_cws_pos_dep.onnx" ] || [ ! -f "$en/en_upos_tagger.onnx" ]; then
  echo "bundle-nlp-models: incomplete models under $root" >&2
  exit 1
fi
mkdir -p "$dest/ltp-small/onnx" "$dest/en-upos-base/onnx-int8"
# -L dereferences symlinks so the bundle is self-contained.
cp -aL "$ltp/." "$dest/ltp-small/onnx/"
cp -aL "$en/en_upos_tagger.onnx" "$en/en_upos_tagger.onnx.data" \
      "$en/en_supar_parser.onnx" "$en/en_supar_parser.onnx.data" \
      "$en/tagger_labels.json" "$en/parser_labels.json" "$en/parser_word_vocab.json" \
      "$dest/en-upos-base/onnx-int8/"
if [ -e "$en/tokenizer" ]; then
  rm -rf "$dest/en-upos-base/onnx-int8/tokenizer"
  cp -aL "$en/tokenizer" "$dest/en-upos-base/onnx-int8/tokenizer"
fi
if [ -f "$root/NOTICE.md" ]; then
  cp -a "$root/NOTICE.md" "$dest/NOTICE.md"
fi
# Refuse leftover pointers or fp32 graphs.
if find "$dest" -type l | grep -q .; then
  echo "bundle-nlp-models: destination still has symlinks" >&2
  find "$dest" -type l >&2
  exit 1
fi
echo "bundled NLP models into $dest ($(du -sh "$dest" | cut -f1))"
