#!/usr/bin/env bash
# Export a Markdown file in this directory to PDF.
# Usage:  ./export_pdf.sh [file.md]          (defaults to judge_analysis.md)
#
# Priority order:
#   1. md-to-pdf (npm / npx)  — renders via headless Chromium; best for tables + images
#   2. pandoc + pdflatex      — install with: brew install pandoc basictex
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT="${1:-judge_analysis.md}"
INPUT_PATH="$SCRIPT_DIR/$INPUT"
OUTPUT_PATH="${INPUT_PATH%.md}.pdf"

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "Error: '$INPUT_PATH' not found." >&2
  exit 1
fi

echo "Input : $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo ""

# ── 1. md-to-pdf (preferred) ─────────────────────────────────────────────────
if command -v md-to-pdf &>/dev/null || npx --yes md-to-pdf --version &>/dev/null 2>&1; then
  echo "Using md-to-pdf (Chromium renderer)..."

  # Inline CSS to make it look good on A4 paper
  CSS='
    body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 11pt;
           max-width: 900px; margin: 0 auto; color: #1a1a1a; }
    h1   { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 6px; }
    h2   { font-size: 15pt; border-bottom: 1px solid #bbb; padding-bottom: 4px; margin-top: 28px; }
    h3   { font-size: 12pt; margin-top: 20px; }
    table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 12px 0; }
    th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; }
    th   { background: #f0f4fa; font-weight: 600; }
    tr:nth-child(even) td { background: #f9f9f9; }
    code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
    pre  { background: #f4f4f4; padding: 12px; border-radius: 4px; font-size: 8.5pt;
           overflow-x: auto; }
    img  { max-width: 100%; height: auto; margin: 12px 0; }
    blockquote { border-left: 4px solid #3A7DC9; margin: 12px 0; padding: 6px 16px;
                 background: #f0f6ff; border-radius: 0 4px 4px 0; }
    @page { margin: 18mm 20mm; }
  '

  # Write temp CSS file
  TMPDIR_CSS="$(mktemp -d)"
  CSS_FILE="$TMPDIR_CSS/style.css"
  echo "$CSS" > "$CSS_FILE"

  npx --yes md-to-pdf \
    --config-file /dev/null \
    --stylesheet "$CSS_FILE" \
    --pdf-options '{"format":"A4","printBackground":true,"margin":{"top":"18mm","bottom":"18mm","left":"20mm","right":"20mm"}}' \
    "$INPUT_PATH"

  # md-to-pdf writes alongside the input file
  GENERATED="${INPUT_PATH%.md}.pdf"
  if [[ "$GENERATED" != "$OUTPUT_PATH" ]]; then
    mv "$GENERATED" "$OUTPUT_PATH"
  fi

  rm -rf "$TMPDIR_CSS"
  echo ""
  echo "Done: $OUTPUT_PATH"
  open "$OUTPUT_PATH" 2>/dev/null || true
  exit 0
fi

# ── 2. pandoc + pdflatex ─────────────────────────────────────────────────────
if command -v pandoc &>/dev/null; then
  ENGINE=""
  for e in xelatex pdflatex lualatex weasyprint wkhtmltopdf; do
    if command -v "$e" &>/dev/null; then
      ENGINE="$e"
      break
    fi
  done

  if [[ -z "$ENGINE" ]]; then
    echo "pandoc found but no PDF engine available."
    echo "Install one with:  brew install basictex   (then: sudo tlmgr install collection-fontsrecommended)"
    exit 1
  fi

  echo "Using pandoc with $ENGINE..."
  pandoc "$INPUT_PATH" \
    --pdf-engine="$ENGINE" \
    --variable geometry:margin=1.8cm \
    --variable fontsize=11pt \
    --variable colorlinks=true \
    -o "$OUTPUT_PATH"

  echo ""
  echo "Done: $OUTPUT_PATH"
  open "$OUTPUT_PATH" 2>/dev/null || true
  exit 0
fi

# ── No tool found ─────────────────────────────────────────────────────────────
echo "No PDF converter found. Install one of:"
echo "  npm install -g md-to-pdf"
echo "  brew install pandoc basictex"
exit 1
