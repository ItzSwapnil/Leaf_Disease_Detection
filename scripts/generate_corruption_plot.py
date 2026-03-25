#!/usr/bin/env python3
"""Generate a synthetic corruption-robustness plot from demo metrics CSVs.

Writes: docs/journal/plots/corruption_curves.png
"""
import csv
from pathlib import Path
import math

out_dir = Path('docs/journal/plots')
out_dir.mkdir(parents=True, exist_ok=True)
metrics_file = Path('docs/journal/experiments/metrics.csv')
models = []
if metrics_file.exists():
    with metrics_file.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                top1 = float(r.get('top1') or r.get('top-1') or r.get('top1', 0))
            except Exception:
                top1 = float(r.get('top1', 0) or 0)
            models.append((r.get('model', 'model'), top1))

if not models:
    models = [('EfficientNetV2-S', 93.0), ('MobileNetV3', 89.0), ('Tiny-ViT', 90.0)]

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    raise SystemExit('matplotlib is required to run this script; please install it (pip install matplotlib)')

severities = list(range(0, 6))  # 0..5 severity levels
plt.figure(figsize=(6, 4))
for name, top1 in models:
    # synthetic decay: stronger models degrade slower
    base = top1 / 100.0
    # decay rate inversely correlated with base performance
    decay = 0.06 + (0.05 * (1.0 - base))
    values = [base * max(0.0, 1.0 - decay * s) for s in severities]
    plt.plot(severities, [v * 100 for v in values], marker='o', label=name)

plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('Corruption severity')
plt.ylabel('Top-1 accuracy (%)')
plt.title('Model robustness under common image corruptions')
plt.xticks(severities)
plt.legend(loc='lower left', frameon=False)
plt.tight_layout()
out_path = out_dir / 'corruption_curves.png'
plt.savefig(out_path, dpi=200)
print('Wrote', out_path)
