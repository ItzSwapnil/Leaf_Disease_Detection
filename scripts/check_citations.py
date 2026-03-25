"""Check LaTeX citation keys against BibTeX keys in docs/journal."""
import argparse
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--strict",
    action="store_true",
    help="Exit with status 1 when citation keys are missing.",
)
args = parser.parse_args()

root = Path('docs/journal')
input_pattern = re.compile(r"\\input\{([^}]+)\}")


def collect_included_tex(entrypoint: Path) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            return
        seen.add(resolved)
        ordered.append(path)

        txt = path.read_text(encoding='utf8', errors='ignore')
        for match in input_pattern.findall(txt):
            rel = Path(match)
            if rel.suffix != '.tex':
                rel = rel.with_suffix('.tex')
            visit(path.parent / rel)

    visit(entrypoint)
    return ordered


tex_files = collect_included_tex(root / 'main.tex')
bib_files = list(root.rglob('*.bib'))

cite_pattern = re.compile(r"\\cite\{([^}]+)\}")
key_set = set()
for tex in tex_files:
    txt = tex.read_text(encoding='utf8', errors='ignore')
    for m in cite_pattern.findall(txt):
        # handle multiple keys comma-separated
        for k in m.split(','):
            key_set.add(k.strip())

bib_key_pattern = re.compile(r"@\w+\{\s*([^,\s]+)")
bib_keys = set()
for bib in bib_files:
    txt = bib.read_text(encoding='utf8', errors='ignore')
    for m in bib_key_pattern.findall(txt):
        bib_keys.add(m.strip())

missing = sorted([k for k in key_set if k and k not in bib_keys])
print('Checked', len(tex_files), 'tex files and', len(bib_files), 'bib files')
print('Found', len(key_set), 'unique citation keys in .tex files')
print('Found', len(bib_keys), 'keys in .bib files')
if missing:
    print('\nMissing citation keys (present in .tex but not in .bib):')
    for k in missing:
        print('-', k)
else:
    print('\nAll citation keys are present in BibTeX files.')

if missing and args.strict:
    sys.exit(1)

