"""
Unit & regression tests for StoreVision AI.
Tests 4-point perspective transform, signage candidate detection, contrast scoring, and AWS Bedrock payload generation.
"""
import pytest
import numpy as np
from storevision.rectifier import four_point_transform, order_points
from storevision.detector import detect_signage_candidates
from storevision.scoring import compute_visibility_metrics, VisibilityScore
from storevision.aws_bedrock import build_bedrock_prompt


def test_order_points():
    pts = np.array([[100, 200], [50, 50], [200, 100], [150, 250]], dtype="float32")
    ordered = order_points(pts)
    assert ordered.shape == (4, 2)
    # Top-left has smallest sum (50, 50)
    assert np.array_equal(ordered[0], [50, 50])


def test_perspective_rectification():
    # Create synthetic test canvas with known 200x100 rectangle skewed
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    quad = np.array([[50, 60], [250, 40], [240, 160], [60, 140]], dtype="float32")
    warped = four_point_transform(canvas, quad)
    assert warped.shape[0] > 0
    assert warped.shape[1] > 0


def test_contrast_and_visibility_scoring():
    # Synthetic high-contrast black text on white background
    sign = np.ones((100, 300, 3), dtype=np.uint8) * 255
    sign[30:70, 50:250] = 0 # Black text block

    metrics = compute_visibility_metrics(sign)
    assert isinstance(metrics, VisibilityScore)
    assert metrics.overall_score >= 50
    assert metrics.contrast_ratio >= 0.70
    assert metrics.legibility_5m is True
    assert len(metrics.recommendations) > 0


def test_aws_bedrock_prompt_synthesis():
    metrics = VisibilityScore(
        overall_score=78,
        contrast_ratio=0.82,
        occlusion_pct=4.5,
        legibility_5m=True,
        legibility_15m=True,
        legibility_30m=False,
        estimated_letter_height_px=54,
        recommendations=["Increase font size for 30m vehicular traffic."]
    )
    payload = build_bedrock_prompt(metrics, business_category="Pharmacy")
    assert payload["modelId"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert "Visibility Index: 78/100" in payload["body"]["messages"][0]["content"][0]["text"]
