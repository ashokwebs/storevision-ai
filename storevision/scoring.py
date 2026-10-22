"""
storevision/scoring.py
Computes physical optical readability, contrast ratio, and pedestrian visibility metrics.
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class VisibilityScore:
    overall_score: int              # 0 to 100
    contrast_ratio: float           # Michelson contrast (0.0 to 1.0)
    occlusion_pct: float            # 0.0 to 100.0%
    legibility_5m: bool             # Near sidewalk
    legibility_15m: bool            # Across 2-lane road
    legibility_30m: bool            # Across 4-lane avenue
    estimated_letter_height_px: int
    recommendations: list[str]


def compute_visibility_metrics(rectified_sign: np.ndarray) -> VisibilityScore:
    """Evaluates optical parameters of a rectified signage crop."""
    h, w = rectified_sign.shape[:2]
    gray = cv2.cvtColor(rectified_sign, cv2.COLOR_BGR2GRAY)

    # 1. Compute Michelson Contrast: (I_max - I_min) / (I_max + I_min)
    p_low = np.percentile(gray, 5)
    p_high = np.percentile(gray, 95)
    denom = max(1.0, p_high + p_low)
    contrast = float((p_high - p_low) / denom)

    # 2. Estimate stroke / text height using edge energy
    edges = cv2.Canny(gray, 50, 150)
    horizontal_profile = np.sum(edges, axis=1)
    peaks = np.where(horizontal_profile > np.mean(horizontal_profile) * 1.5)[0]
    estimated_h = int(np.max(peaks) - np.min(peaks)) if len(peaks) > 1 else int(h * 0.25)

    # 3. Legibility distance thresholds
    # Typically, 1 inch letter height = 30 feet legibility (Snellen rule adapted for optical pixels)
    leg_5m = bool(estimated_h >= 24)
    leg_15m = bool(estimated_h >= 48)
    leg_30m = bool(estimated_h >= 96)

    # 4. Occlusion estimation (detect edge breaks / foreground blobs)
    # Using saturation mask or high-frequency foreground discontinuity
    hsv = cv2.cvtColor(rectified_sign, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    occluded_pixels = np.sum((s > 180) & (gray < 80)) # Dark high-saturation obstructions (leaves, poles)
    occlusion_pct = round(float((occluded_pixels / (h * w)) * 100.0), 1)

    # 5. Overall 0-100 Visibility Score Calculation
    base_score = int(contrast * 50.0) + (25 if leg_15m else 10 if leg_5m else 0)
    penalty = int(occlusion_pct * 0.5)
    overall_score = max(5, min(100, base_score - penalty + 25))

    recommendations = []
    if contrast < 0.5:
        recommendations.append("Increase color contrast between lettering and backing board (aim for >0.65 Michelson contrast).")
    if not leg_30m:
        recommendations.append("Primary brand text is undersized for 4-lane vehicle traffic (>30m distance).")
    if occlusion_pct > 10.0:
        recommendations.append(f"Physical sightline is {occlusion_pct}% obstructed by foreground objects or glare.")
    if leg_15m and contrast >= 0.6:
        recommendations.append("Excellent pedestrian curb appeal for sidewalk and cross-street footfall.")

    return VisibilityScore(
        overall_score=overall_score,
        contrast_ratio=round(contrast, 3),
        occlusion_pct=occlusion_pct,
        legibility_5m=leg_5m,
        legibility_15m=leg_15m,
        legibility_30m=leg_30m,
        estimated_letter_height_px=estimated_h,
        recommendations=recommendations,
    )
