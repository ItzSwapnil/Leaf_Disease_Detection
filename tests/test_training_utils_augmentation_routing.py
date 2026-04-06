import importlib
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training_utils


def _reload_training_utils():
    return importlib.reload(training_utils)


def test_resolve_augmentation_probabilities_normalizes_values():
    tu = _reload_training_utils()

    mix_p, cut_p, normal_p = tu.resolve_augmentation_probabilities(
        use_mixup=True,
        use_cutmix=True,
        mixup_prob=0.4,
        cutmix_prob=0.4,
        normal_prob=0.2,
    )

    assert np.isclose(mix_p + cut_p + normal_p, 1.0)
    assert np.isclose(mix_p, 0.4)
    assert np.isclose(cut_p, 0.4)
    assert np.isclose(normal_p, 0.2)


def test_sample_augmentation_route_matches_expected_distribution():
    tu = _reload_training_utils()
    np.random.seed(7)

    n = 20000
    counts = {"mixup": 0, "cutmix": 0, "normal": 0}
    for _ in range(n):
        route = tu.sample_augmentation_route(
            use_mixup=True,
            use_cutmix=True,
            mixup_prob=0.4,
            cutmix_prob=0.4,
            normal_prob=0.2,
        )
        counts[route] += 1

    mix_ratio = counts["mixup"] / n
    cut_ratio = counts["cutmix"] / n
    normal_ratio = counts["normal"] / n

    assert abs(mix_ratio - 0.4) < 0.03
    assert abs(cut_ratio - 0.4) < 0.03
    assert abs(normal_ratio - 0.2) < 0.03


def test_sample_augmentation_route_disables_unselected_modes():
    tu = _reload_training_utils()
    np.random.seed(11)

    n = 3000
    counts = {"mixup": 0, "cutmix": 0, "normal": 0}
    for _ in range(n):
        route = tu.sample_augmentation_route(
            use_mixup=False,
            use_cutmix=True,
            mixup_prob=0.4,
            cutmix_prob=0.4,
            normal_prob=0.2,
        )
        counts[route] += 1

    assert counts["mixup"] == 0
    assert counts["cutmix"] > 0
    assert counts["normal"] > 0
