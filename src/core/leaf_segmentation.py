"""Leaf segmentation, contour extraction, and edge curvature analysis."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def calculate_vertex_angle(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
    """Calculate the interior angle (in degrees) at vertex p2 formed by lines p2-p1 and p2-p3."""
    u = p1.astype(np.float64) - p2.astype(np.float64)
    v = p3.astype(np.float64) - p2.astype(np.float64)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u < 1e-6 or norm_v < 1e-6:
        return 180.0

    dot_product = np.dot(u, v)
    cosine_val = dot_product / (norm_u * norm_v)
    cosine_val = np.clip(cosine_val, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_val))
    return float(angle)


def extract_contour_angles(contour: np.ndarray) -> list[float]:
    """Calculate curvature angles at all vertices of the polygon contour."""
    pts = contour.reshape(-1, 2)
    n = len(pts)
    angles: list[float] = []

    if n < 3:
        return angles

    for i in range(n):
        p1 = pts[i - 1]
        p2 = pts[i]
        p3 = pts[(i + 1) % n]
        angles.append(calculate_vertex_angle(p1, p2, p3))

    return angles


def segment_leaf(
    image: np.ndarray,
    min_area_ratio: float = 0.01,
    approx_epsilon_ratio: float = 0.015,
) -> dict[str, Any]:
    """Segment leaf regions using color masking, edge detection, and contour analysis."""
    # Convert TensorFlow EagerTensors to NumPy arrays if called during training.
    if hasattr(image, "numpy"):
        image = image.numpy()

    h, w = image.shape[:2]
    total_area = h * w
    min_area = total_area * min_area_ratio

    # Normalize image to uint8.
    if image.dtype == np.float32 or image.max() <= 1.0:
        img_uint8 = (image * 255.0).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)

    # 1. Color Masking (HSV)
    # Covers green healthy leaves, yellow/brown areas, and wilted regions.
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    lower_color = np.array([10, 20, 20])
    upper_color = np.array([155, 255, 255])
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # 2. Edge Detection (Canny)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Calculate adaptive thresholds based on the median intensity.
    median_val: float = float(np.median(blurred))
    lower_thresh = int(max(0, (1.0 - 0.33) * median_val))
    upper_thresh = int(min(255, (1.0 + 0.33) * median_val))
    edges = cv2.Canny(blurred, lower_thresh, upper_thresh)

    # Dilate edges to close gaps in contours.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # 3. Combine color mask and edge detection results.
    combined_mask = cv2.bitwise_or(color_mask, dilated_edges)

    # Morphological cleanup
    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, cleanup_kernel
    )
    cleaned_mask = cv2.morphologyEx(
        cleaned_mask, cv2.MORPH_OPEN, cleanup_kernel
    )

    # 4. Find Contours
    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter and simplify contours
    valid_contours: list[np.ndarray] = []
    all_curve_angles: list[list[float]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            # Simplify contour to find polygon shape.
            perimeter = cv2.arcLength(cnt, True)
            epsilon = approx_epsilon_ratio * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Verify the simplified contour is a valid polygon.
            if len(approx) >= 3:
                valid_contours.append(approx)
                # Calculate angles along the edges.
                angles = extract_contour_angles(approx)
                all_curve_angles.append(angles)

    # Fallback to the largest contour if no contours pass the area filter.
    if not valid_contours and contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        if area > 100:  # Minimum safety threshold
            perimeter = cv2.arcLength(largest_cnt, True)
            epsilon = approx_epsilon_ratio * perimeter
            approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
            if len(approx) >= 3:
                valid_contours.append(approx)
                all_curve_angles.append(extract_contour_angles(approx))

    # 5. Generate final clean binary mask.
    final_mask = np.zeros((h, w), dtype=np.uint8)
    if valid_contours:
        cv2.drawContours(final_mask, valid_contours, -1, 255, -1)

    # Apply mask to isolate the leaf region.
    masked_image = np.zeros_like(image)
    masked_image[final_mask == 255] = image[final_mask == 255]

    return {
        "masked_image": masked_image,
        "mask": final_mask,
        "contours": valid_contours,
        "curve_angles": all_curve_angles,
        "leaf_count": len(valid_contours),
        "success": len(valid_contours) > 0,
    }
