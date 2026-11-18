#!/bin/bash
# Build report.pdf from report_source.md using pandoc.
# Requires: pandoc, texlive-xetex (or pdflatex)
#
# Install on Ubuntu: sudo apt-get install pandoc texlive-xetex
# Install on macOS: brew install pandoc; brew install --cask mactex
# On Colab: !apt-get install -q pandoc texlive-xetex

set -e
cd "$(dirname "$0")"

pandoc report_source.md \
    -o report.pdf \
    --pdf-engine=xelatex \
    --variable mainfont="DejaVu Sans" \
    --toc \
    --number-sections \
    --highlight-style=tango

echo "Built report.pdf"
