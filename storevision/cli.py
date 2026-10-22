"""
storevision/cli.py
Command-line interface for StoreVision AI.
"""
from __future__ import annotations
import os
import sys
import argparse
import cv2
import json
from pathlib import Path
from .detector import detect_signage_candidates
from .rectifier import four_point_transform
from .scoring import compute_visibility_metrics
from .aws_bedrock import build_bedrock_prompt


def run_audit(input_path: str, output_dir: str, category: str = "Retail Store"):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ Error: Could not read image at {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n👁️ [StoreVision AI] Processing physical storefront image: {input_path}")
    print(f"   Image Resolution: {img.shape[1]}x{img.shape[0]} px")

    # 1. Detection
    candidates = detect_signage_candidates(img)
    print(f"   ✔ Detected {len(candidates)} candidate signage contour(s)")

    # 2. Rectification on primary candidate
    primary_pts = candidates[0]
    rectified = four_point_transform(img, primary_pts)
    rect_path = os.path.join(output_dir, "rectified_signage.jpg")
    cv2.imwrite(rect_path, rectified)
    print(f"   ✔ Perspective Rectification: Saved un-warped sign to {rect_path}")

    # 3. Scoring
    metrics = compute_visibility_metrics(rectified)
    print("\n" + "=" * 55)
    print(f" 📊 STOREFRONT PHYSICAL PERCEPTION SCORE: {metrics.overall_score}/100")
    print("=" * 55)
    print(f" • Optical Contrast Ratio   : {metrics.contrast_ratio} (Target: >0.65)")
    print(f" • Sightline Occlusion       : {metrics.occlusion_pct}%")
    print(f" • 5m Sidewalk Legibility   : {'✅ Pass' if metrics.legibility_5m else '❌ Too Small'}")
    print(f" • 15m Cross-Street         : {'✅ Pass' if metrics.legibility_15m else '❌ Too Small'}")
    print(f" • 30m Vehicle Traffic      : {'✅ Pass' if metrics.legibility_30m else '❌ Too Small'}")
    print("\n💡 Automated Architectural Recommendations:")
    for i, rec in enumerate(metrics.recommendations, 1):
        print(f"   {i}. {rec}")

    # 4. AWS Bedrock Payload
    bedrock_payload = build_bedrock_prompt(metrics, business_category=category)
    payload_path = os.path.join(output_dir, "aws_bedrock_payload.json")
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(bedrock_payload, f, indent=2)
    print(f"\n📦 AWS Bedrock Multi-Modal Payload: {payload_path}")

    # 5. Interactive HTML Audit Scorecard
    from .visualizer import generate_html_audit_report
    html_path = os.path.join(output_dir, "storevision_audit_report.html")
    generate_html_audit_report(img, rectified, metrics, html_path, store_name=category)
    print(f"🌐 Interactive HTML Audit Scorecard: {html_path}")
    print("🎉 StoreVision physical retail audit complete!\n")


def main():
    parser = argparse.ArgumentParser(description="StoreVision AI: Physical Retail Perception Engine (OpenCV 5 + AWS)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_audit = subparsers.add_parser("audit", help="Run physical perception audit on storefront photo")
    p_audit.add_argument("--input", "-i", required=True, help="Path to input photo (JPG/PNG)")
    p_audit.add_argument("--output-dir", "-o", default="./out", help="Output directory for rectified assets")
    p_audit.add_argument("--category", "-c", default="Retail Store", help="Business category (e.g. Pharmacy, Dental Clinic, Cafe)")

    p_serve = subparsers.add_parser("serve", help="Start local StoreVision HTTP server for live scorecard viewing")
    p_serve.add_argument("--port", "-p", type=int, default=8080, help="Port to listen on (default: 8080)")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")

    args = parser.parse_args()
    if args.command == "audit":
        run_audit(args.input, args.output_dir, args.category)
    elif args.command == "serve":
        from .server import run_server
        run_server(args.port, args.host)


if __name__ == "__main__":
    main()
