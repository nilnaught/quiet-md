#!/bin/bash
# Copy ONNX Runtime shared libraries from next to a built quiet_md binary
# into DEST (the folder that will contain the executable).
set -euo pipefail
bin_dir=${1:?usage: bundle-ort-runtime.sh BINARY_DIR DEST}
dest=${2:?usage: bundle-ort-runtime.sh BINARY_DIR DEST}
mkdir -p "$dest"
copied=0
# copy-dylibs / download-binaries may leave any of these next to the binary.
shopt -s nullglob
for f in "$bin_dir"/libonnxruntime*.dylib "$bin_dir"/onnxruntime*.dll \
         "$bin_dir"/libonnxruntime*.so* "$bin_dir"/onnxruntime*.so*; do
  cp -a "$f" "$dest/"
  copied=$((copied + 1))
done
if [ "$copied" -gt 0 ]; then
  echo "bundled $copied ONNX Runtime library file(s) into $dest"
fi
