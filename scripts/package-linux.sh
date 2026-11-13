#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $(uname -s) == Linux ]] || { echo 'Build this package on Linux.' >&2; exit 1; }
if command -v magick >/dev/null 2>&1; then resize=magick
elif command -v convert >/dev/null 2>&1; then resize=convert
else echo 'Install ImageMagick to generate the Linux icon.' >&2; exit 1
fi
cargo build --release --locked
version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1)
name="quiet-md-$version-linux-$(uname -m)"
mkdir -p dist/linux
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/$name/bin" "$stage/$name/share/applications" "$stage/$name/share/icons/hicolor/512x512/apps"
install -m755 "${CARGO_TARGET_DIR:-target}/release/quiet_md" "$stage/$name/bin/quiet_md"
scripts/bundle-ort-runtime.sh "${CARGO_TARGET_DIR:-target}/release" "$stage/$name/bin"
scripts/bundle-nlp-models.sh "$stage/$name/bin/models"
install -m644 packaging/linux/com.s1dashu.quietmd.desktop "$stage/$name/share/applications/"
"$resize" assets/branding/app-icon.png -resize 512x512 "$stage/$name/share/icons/hicolor/512x512/apps/com.s1dashu.quietmd.png"
install -m644 LICENSE "$stage/$name/"
ldd "$stage/$name/bin/quiet_md" > "$stage/$name/runtime-libraries.txt"
if grep -q 'not found' "$stage/$name/runtime-libraries.txt"; then cat "$stage/$name/runtime-libraries.txt" >&2; exit 1; fi
tar -C "$stage" -czf "dist/linux/$name.tar.gz" "$name"
(cd dist/linux && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
printf '%s\n' "dist/linux/$name.tar.gz"
