"""
storevision/aws_bedrock.py
Integration module formatting OpenCV perception telemetry for AWS Bedrock multi-modal synthesis.
"""
from __future__ import annotations
import json
from typing import Dict, Any
from .scoring import VisibilityScore


def build_bedrock_prompt(metrics: VisibilityScore, business_category: str = "Retail Store") -> Dict[str, Any]:
    """Generates AWS Bedrock InvokeModel payload."""
    prompt_text = (
        f"You are StoreVision AI, an expert retail footfall and physical architecture consultant. "
        f"Analyze the following computer vision metrics extracted via OpenCV 5 for a {business_category} storefront:\n"
        f"- Overall Visibility Index: {metrics.overall_score}/100\n"
        f"- Michelson Optical Contrast: {metrics.contrast_ratio} (Target: >0.65)\n"
        f"- Physical Occlusion: {metrics.occlusion_pct}%\n"
        f"- Legibility at 5m (Sidewalk): {metrics.legibility_5m}\n"
        f"- Legibility at 15m (Cross-street): {metrics.legibility_15m}\n"
        f"- Legibility at 30m (Vehicle traffic): {metrics.legibility_30m}\n"
        f"- Detected Letter Height: {metrics.estimated_letter_height_px}px\n\n"
        f"Generate 3 tactical recommendations for the store owner to maximize pedestrian footfall and customer drive-by conversions."
    )

    # Standard AWS Bedrock Anthropic Claude format
    return {
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}]
                }
            ]
        }
    }
