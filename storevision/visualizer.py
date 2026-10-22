"""
storevision/visualizer.py
Generates standalone, interactive HTML audit scorecards embedding base64 images
of the perspective-rectified signage, optical contrast gauges, and AWS Bedrock recommendations.
"""
from __future__ import annotations
import base64
import cv2
import numpy as np
from pathlib import Path
from .scoring import VisibilityScore


def _to_base64_jpg(img: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def generate_html_audit_report(
    orig_img: np.ndarray,
    rectified_img: np.ndarray,
    metrics: VisibilityScore,
    output_path: str,
    store_name: str = "Physical Storefront Audit",
) -> str:
    orig_b64 = _to_base64_jpg(orig_img)
    rect_b64 = _to_base64_jpg(rectified_img)

    recs_html = "".join(f"<li>{r}</li>" for r in metrics.recommendations)

    html_content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StoreVision AI · Physical Retail Visibility Scorecard</title>
<style>
  :root {{
    --bg: #090d16;
    --card: #111827;
    --border: #1f293d;
    --accent: #3b82f6;
    --cyan: #06b6d4;
    --emerald: #10b981;
    --amber: #f59e0b;
    --rose: #f43f5e;
    --text: #f8fafc;
    --muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: radial-gradient(circle at 50% -20%, #1e293b, var(--bg));
    color: var(--text);
    font-family: system-ui, -apple-system, sans-serif;
    padding: 32px 16px;
    min-height: 100vh;
  }}
  .container {{ max-width: 980px; margin: 0 auto; }}
  header {{ text-align: center; margin-bottom: 28px; }}
  .badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #60a5fa;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }}
  h1 {{ font-size: 28px; font-weight: 800; margin-bottom: 6px; }}
  .subtitle {{ color: var(--muted); font-size: 14px; }}

  .hero-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }}
  .metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
  }}
  .metric-val {{ font-size: 26px; font-weight: 800; }}
  .metric-lbl {{ font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; margin-top: 4px; }}

  .visual-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }}
  @media (max-width: 768px) {{
    .visual-grid {{ grid-template-columns: 1fr; }}
  }}
  .img-box {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px;
  }}
  .img-title {{ font-size: 13px; font-weight: 700; color: var(--muted); margin-bottom: 8px; display: flex; justify-content: space-between; }}
  .img-wrapper {{
    position: relative;
    aspect-ratio: 16/9;
    background: #000;
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .img-wrapper img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }}

  .legibility-matrix {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 24px;
  }}
  .table-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 13px;
  }}
  .pill-pass {{ color: var(--emerald); font-weight: 700; background: rgba(16, 185, 129, 0.1); padding: 3px 8px; border-radius: 6px; }}
  .pill-fail {{ color: var(--rose); font-weight: 700; background: rgba(244, 63, 94, 0.1); padding: 3px 8px; border-radius: 6px; }}

  .recommendations {{
    background: rgba(59, 130, 246, 0.05);
    border: 1px dashed rgba(59, 130, 246, 0.3);
    border-radius: 14px;
    padding: 20px;
  }}
  .recs-title {{ font-size: 14px; font-weight: 700; color: #93c5fd; margin-bottom: 10px; }}
  ul {{ padding-left: 20px; font-size: 13px; color: #e2e8f0; line-height: 1.6; }}

  footer {{
    text-align: center;
    color: var(--muted);
    font-size: 11px;
    margin-top: 36px;
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="badge">OpenCV 5 & AWS Bedrock Intelligence</div>
    <h1>StoreVision AI · Physical Perception Audit</h1>
    <div class="subtitle">{store_name} — Automated Sightline, Homography & Optical Contrast Analysis</div>
  </header>

  <div class="hero-grid">
    <div class="metric-card">
      <div class="metric-val" style="color: var(--emerald);">{metrics.overall_score}/100</div>
      <div class="metric-lbl">Overall Visibility</div>
    </div>
    <div class="metric-card">
      <div class="metric-val" style="color: var(--cyan);">{metrics.contrast_ratio}</div>
      <div class="metric-lbl">Michelson Contrast</div>
    </div>
    <div class="metric-card">
      <div class="metric-val" style="color: var(--amber);">{metrics.occlusion_pct}%</div>
      <div class="metric-lbl">Sightline Occlusion</div>
    </div>
    <div class="metric-card">
      <div class="metric-val" style="color: #60a5fa;">{metrics.estimated_letter_height_px} px</div>
      <div class="metric-lbl">Signage Letter Height</div>
    </div>
  </div>

  <div class="visual-grid">
    <div class="img-box">
      <div class="img-title">
        <span>Raw Angled Perspective (Ingest)</span>
        <span style="color: var(--amber);">Angled View</span>
      </div>
      <div class="img-wrapper">
        <img src="data:image/jpeg;base64,{orig_b64}" alt="Raw Storefront Ingest">
      </div>
    </div>

    <div class="img-box">
      <div class="img-title">
        <span>OpenCV 5 Homography Rectification</span>
        <span style="color: var(--emerald);">Un-Warped Planar</span>
      </div>
      <div class="img-wrapper">
        <img src="data:image/jpeg;base64,{rect_b64}" alt="Perspective Rectified Signage">
      </div>
    </div>
  </div>

  <div class="legibility-matrix">
    <h3 style="margin-bottom: 12px; font-size: 14px;">🚦 Multi-Distance Optical Legibility Audit</h3>
    <div class="table-row">
      <span>5m Sidewalk Pedestrian Legibility</span>
      <span class="{'pill-pass' if metrics.legibility_5m else 'pill-fail'}">{'✔ Pass' if metrics.legibility_5m else '✖ Deficit'}</span>
    </div>
    <div class="table-row">
      <span>15m Cross-Street Opposite Sidewalk</span>
      <span class="{'pill-pass' if metrics.legibility_15m else 'pill-fail'}">{'✔ Pass' if metrics.legibility_15m else '✖ Deficit'}</span>
    </div>
    <div class="table-row">
      <span>30m Moving Vehicle Traffic (35 mph)</span>
      <span class="{'pill-pass' if metrics.legibility_30m else 'pill-fail'}">{'✔ Pass' if metrics.legibility_30m else '✖ Deficit'}</span>
    </div>
  </div>

  <div class="recommendations">
    <div class="recs-title">💡 Automated Architectural & Signage Optimizations (AWS Bedrock):</div>
    <ul>
      {recs_html}
    </ul>
  </div>

  <footer>
    StoreVision AI · OpenCV AI Competition 2026 Powered by AWS · Built by Ashok (@jackyandr)
  </footer>
</div>
</body>
</html>"""
    Path(output_path).write_text(html_content, encoding="utf-8")
    return output_path
