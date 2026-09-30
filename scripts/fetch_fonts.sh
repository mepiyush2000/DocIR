#!/usr/bin/env bash
# Fetch the open-licence fonts bundled under docir/assets/fonts.
set -euo pipefail
DEST="docir/assets/fonts"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DEST"

fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }

# Liberation Sans / Serif / Mono (SIL OFL 1.1)
fetch https://github.com/liberationfonts/liberation-fonts/files/7261482/liberation-fonts-ttf-2.1.5.tar.gz "$TMP/lib.tgz"
tar -xzf "$TMP/lib.tgz" -C "$TMP"
mkdir -p "$DEST/liberation"
cp "$TMP"/liberation-fonts-ttf-2.1.5/*.ttf "$TMP"/liberation-fonts-ttf-2.1.5/LICENSE "$DEST/liberation/"

# Liberation Sans Narrow (GPLv2 + font exception)
fetch https://github.com/liberationfonts/liberation-sans-narrow/files/2579431/liberation-narrow-fonts-ttf-1.07.6.tar.gz "$TMP/narrow.tgz"
tar -xzf "$TMP/narrow.tgz" -C "$TMP"
mkdir -p "$DEST/liberation-narrow"
cp "$TMP"/liberation-narrow-fonts-ttf-1.07.6/*.ttf "$DEST/liberation-narrow/"
find "$TMP/liberation-narrow-fonts-ttf-1.07.6" -maxdepth 1 -type f \
  \( -iname 'LICENSE*' -o -iname 'COPYING*' \) -exec cp {} "$DEST/liberation-narrow/" \;

# DejaVu (Bitstream Vera licence)
fetch https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.tar.bz2 "$TMP/dejavu.tbz"
tar -xjf "$TMP/dejavu.tbz" -C "$TMP"
mkdir -p "$DEST/dejavu"
for f in DejaVuSans DejaVuSans-Bold DejaVuSans-Oblique DejaVuSans-BoldOblique \
         DejaVuSansCondensed DejaVuSansCondensed-Bold DejaVuSansCondensed-Oblique DejaVuSansCondensed-BoldOblique \
         DejaVuSerif DejaVuSerif-Bold DejaVuSerif-Italic DejaVuSerif-BoldItalic \
         DejaVuSerifCondensed DejaVuSerifCondensed-Bold DejaVuSerifCondensed-Italic DejaVuSerifCondensed-BoldItalic \
         DejaVuSansMono DejaVuSansMono-Bold DejaVuSansMono-Oblique DejaVuSansMono-BoldOblique; do
  cp "$TMP/dejavu-fonts-ttf-2.37/ttf/$f.ttf" "$DEST/dejavu/"
done
cp "$TMP/dejavu-fonts-ttf-2.37/LICENSE" "$DEST/dejavu/"

# Noto Sans / Noto Serif (SIL OFL 1.1)
NOTO=https://github.com/notofonts/notofonts.github.io/raw/main/fonts
mkdir -p "$DEST/noto"
for fam in NotoSans NotoSerif; do
  for s in Regular Bold Italic BoldItalic; do
    fetch "$NOTO/$fam/hinted/ttf/$fam-$s.ttf" "$DEST/noto/$fam-$s.ttf"
  done
done
fetch https://raw.githubusercontent.com/notofonts/latin-greek-cyrillic/main/OFL.txt "$DEST/noto/OFL.txt"

# Noto Sans Devanagari -- script fallback for Hindi text (SIL OFL 1.1)
mkdir -p "$DEST/noto-devanagari"
for s in Regular Bold; do
  fetch "$NOTO/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-$s.ttf" "$DEST/noto-devanagari/NotoSansDevanagari-$s.ttf"
done
fetch https://raw.githubusercontent.com/notofonts/devanagari/main/OFL.txt "$DEST/noto-devanagari/OFL.txt"

# PT Sans Narrow -- condensed humanist sans (SIL OFL 1.1)
GF=https://github.com/google/fonts/raw/main/ofl/ptsansnarrow
mkdir -p "$DEST/pt-sans-narrow"
fetch "$GF/PT_Sans-Narrow-Web-Regular.ttf" "$DEST/pt-sans-narrow/PT_Sans-Narrow-Web-Regular.ttf"
fetch "$GF/PT_Sans-Narrow-Web-Bold.ttf" "$DEST/pt-sans-narrow/PT_Sans-Narrow-Web-Bold.ttf"
fetch "$GF/OFL.txt" "$DEST/pt-sans-narrow/OFL.txt"

find "$DEST" -type f | sort
