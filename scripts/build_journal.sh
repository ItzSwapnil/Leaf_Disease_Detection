#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Generating manuscript tables from repository artifacts..."
if command -v uv >/dev/null 2>&1; then
  (cd "$ROOT_DIR" && uv run python scripts/generate_real_journal_figures.py)
  (cd "$ROOT_DIR" && uv run python scripts/generate_journal_tables.py)
  (cd "$ROOT_DIR" && uv run python scripts/check_citations.py --strict)
else
  (cd "$ROOT_DIR" && python3 scripts/generate_real_journal_figures.py)
  (cd "$ROOT_DIR" && python3 scripts/generate_journal_tables.py)
  (cd "$ROOT_DIR" && python3 scripts/check_citations.py --strict)
fi

echo "Building PDF with latexmk..."
(cd "$ROOT_DIR/docs/journal" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)

echo "Built $ROOT_DIR/docs/journal/main.pdf"
