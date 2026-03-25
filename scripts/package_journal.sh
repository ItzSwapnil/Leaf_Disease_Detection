#!/usr/bin/env bash
# Packaging helper: build the journal and archive the source plus evidence files.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
JOURNAL_DIR="$ROOT_DIR/docs/journal"
OUT_DIR="$ROOT_DIR/models"
OUT_FILE="$OUT_DIR/leaf_disease_journal_source_$(date +%Y%m%d_%H%M%S).tar.gz"

mkdir -p "$OUT_DIR"
echo "Building journal before packaging..."
"$ROOT_DIR/scripts/build_journal.sh"

echo "Packaging journal sources from $JOURNAL_DIR to $OUT_FILE"

tar -czf "$OUT_FILE" -C "$ROOT_DIR" \
  docs/journal/main.tex \
  docs/journal/sections \
  docs/journal/references.bib \
  docs/journal/BUILD.md \
  docs/journal/CHECKLIST.md \
  docs/journal/main.pdf \
  docs/reports/evaluation_report.json \
  reports/dataset_counts.json \
  models/logs/train_history.csv \
  plots || { echo "Tar failed"; exit 2; }

echo "Packaged: $OUT_FILE"
echo "Included manuscript source, compiled PDF, figures, and evidence artifacts."
