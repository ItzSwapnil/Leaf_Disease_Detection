"""Smoke test for leaf_segmentation module."""

import numpy as np

from src.core.leaf_segmentation import (
    calculate_vertex_angle,
    extract_contour_angles,
    segment_leaf,
)

# Test 1: segment_leaf with random noise image
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
res = segment_leaf(img)
print(
    f"segment_leaf on noise: success={res['success']}, leaf_count={res['leaf_count']}"
)
print(f"  Keys: {list(res.keys())}")

# Test 2: segment_leaf with float32 image (simulates TF pipeline)
img_f = img.astype(np.float32) / 255.0
res_f = segment_leaf(img_f)
print(
    f"segment_leaf on float32: success={res_f['success']}, leaf_count={res_f['leaf_count']}"
)

# Test 3: calculate_vertex_angle at a true right angle
# Place vertex at origin, arms along +x and +y axes: angle at (0,0) = 90 degrees.
a = np.array([1.0, 0.0])
b = np.array([0.0, 0.0])
c = np.array([0.0, 1.0])
angle = calculate_vertex_angle(a, b, c)
print(f"calculate_vertex_angle (90deg at origin): {angle:.1f} degrees")
assert 89.0 < angle < 91.0, f"Expected ~90, got {angle}"

# Test 4: extract_contour_angles with a simple square contour
square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
angles = extract_contour_angles(square)
print(f"extract_contour_angles (square): {[f'{a:.1f}' for a in angles]}")
for a in angles:
    assert 89.0 < a < 91.0, f"Square vertex should be ~90, got {a}"


# Test 5: segment_leaf with EagerTensor-like object (hasattr numpy)
class FakeTensor:
    """Simulates a TF EagerTensor with .numpy() method."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def numpy(self) -> np.ndarray:
        return self._arr

    @property
    def shape(self):
        return self._arr.shape


fake = FakeTensor(img_f)
res_t = segment_leaf(fake)
print(
    f"segment_leaf on FakeTensor: success={res_t['success']}, "
    f"leaf_count={res_t['leaf_count']}"
)

print("\nAll smoke tests passed!")
