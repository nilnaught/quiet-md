#!/usr/bin/env bash
# Build on Ubuntu 22.04 (glibc 2.35) for release; CI validates on ubuntu-24.04,
# where ort 2.0 prebuilts link (glibc 2.38+). Graphics drivers remain
# system-provided, so the glibc note below is informational when built on 24.04.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Build on Linux x86_64.' >&2; exit 1; }
cargo build --release --locked
version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1)
out="$PWD/dist/linux"
mkdir -p "$out"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
tool="$stage/linuxdeploy.AppImage"
curl --fail --location --retry 3 --proto '=https' --proto-redir '=https' \
  https://github.com/linuxdeploy/linuxdeploy/releases/download/1-alpha-20251107-1/linuxdeploy-x86_64.AppImage -o "$tool"
echo "c20cd71e3a4e3b80c3483cef793cda3f4e990aca14014d23c544ca3ce1270b4d  $tool" | sha256sum -c -
chmod +x "$tool"
appdir="$stage/Quiet.md.AppDir"
mkdir -p "$appdir/usr/share/doc/quiet-md"
cp LICENSE THIRD_PARTY.md "$appdir/usr/share/doc/quiet-md/"
mkdir -p "$appdir/usr/share/X11"
cp -a /usr/share/X11/locale "$appdir/usr/share/X11/"
# License notices for bundled shared libraries and their Compose data.
for package in libxkbcommon0 libxkbcommon-x11-0 libxau6 libxdmcp6 libxcb-xkb1 libbsd0 libmd0 libx11-data; do
  install -Dm644 "/usr/share/doc/$package/copyright" "$appdir/usr/share/doc/$package/copyright"
done
convert assets/branding/app-icon.png -resize 512x512 "$stage/com.s1dashu.quietmd.png"
export APPIMAGE_EXTRACT_AND_RUN=1
export OUTPUT="$out/quiet-md-$version-x86_64.AppImage"
export VERSION="$version"
"$tool" --appdir "$appdir" --executable "${CARGO_TARGET_DIR:-target}/release/quiet_md" \
  --desktop-file packaging/linux/com.s1dashu.quietmd.desktop \
  --icon-file "$stage/com.s1dashu.quietmd.png" --custom-apprun packaging/linux/AppRun --output appimage
chmod +x "$OUTPUT"
# Inspect the packaged payload, not only the original executable.
(cd "$stage" && "$OUTPUT" --appimage-extract >/dev/null)
test -x "$stage/squashfs-root/AppRun"
payload="$stage/squashfs-root/usr/bin/quiet_md"
LD_LIBRARY_PATH="$stage/squashfs-root/usr/lib" ldd "$payload" > "$OUTPUT.libraries.txt"
if grep -q 'not found' "$OUTPUT.libraries.txt"; then cat "$OUTPUT.libraries.txt" >&2; exit 1; fi
objdump -T "$payload" | sed -n 's/.*(GLIBC_\([0-9.]*\)).*/\1/p' | sort -Vu > "$OUTPUT.glibc.txt"
max_glibc=$(tail -1 "$OUTPUT.glibc.txt")
# The 2.35 baseline only holds for packages built on Ubuntu 22.04. This box is
# the CI runner (ubuntu-24.04 since ort needs glibc 2.38), so only fail when
# a symbol needs a newer glibc than the build host itself ships: at the very
# least the package could not run where it was built.
host_glibc=$(ldd --version | sed -n '1s/.* //p')
if [[ $(printf '%s\n' "$host_glibc" "$max_glibc" | sort -V | tail -1) != "$host_glibc" ]]; then
  echo "Requires glibc $max_glibc, newer than the build host's $host_glibc" >&2
  exit 1
fi
if [[ $(printf '%s\n' 2.35 "$max_glibc" | sort -V | tail -1) != 2.35 ]]; then
  echo "note: requires glibc $max_glibc (above the 2.35 Ubuntu 22.04 baseline);" \
       "build AppImages on 22.04 for older distributions" >&2
fi
(cd "$out" && sha256sum "$(basename "$OUTPUT")" > "$(basename "$OUTPUT").sha256")
printf 'AppImage: %s (glibc %s)\n' "$OUTPUT" "$max_glibc"
