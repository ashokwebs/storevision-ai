"""
storevision/detector.py
OpenCV-based storefront signage detection and contour segmentation.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def detect_signage_candidates(image: np.ndarray) -> List[np.ndarray]:
    """
    Detects rectangular candidate signage boards in street-level imagery.
    Returns list of 4-point contour arrays.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive morphological edge gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)

    # Otsu thresholding
    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close horizontal gaps (sign text typically has horizontal continuity)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    min_area = (image.shape[0] * image.shape[1]) * 0.02 # At least 2% of image

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            candidates.append(pts)
        else:
            # Fallback to minimum area bounding box
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            candidates.append(np.array(box, dtype="float32"))

    # If no candidate found, default to central region (60% crop)
    if not candidates:
        h, w = image.shape[:2]
        pts = np.array([
            [w * 0.15, h * 0.15],
            [w * 0.85, h * 0.15],
            [w * 0.85, h * 0.65],
            [w * 0.15, h * 0.65]
        ], dtype="float32")
        candidates.append(pts)

    return candidates
