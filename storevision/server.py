"""
storevision/server.py
Zero-dependency interactive HTTP server and visual playground for StoreVision AI.
Powered by Python standard library http.server + OpenCV 5.
"""
from __future__ import annotations
import os
import io
import json
import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from .detector import detect_signage_candidates
from .rectifier import four_point_transform
from .scoring import compute_visibility_metrics
from .visualizer import generate_html_audit_report, _to_base64_jpg

HTML_PLAYGROUND = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StoreVision AI · Physical Retail Perception Engine</title>
<style>
  :root {
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
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: 32px 16px;
    display: flex;
    justify-content: center;
  }
  .container { max-width: 980px; width: 100%; display: flex; flex-direction: column; gap: 24px; }
  header { text-align: center; }
  .badge {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.4);
    color: #60a5fa;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
  }
  h1 { font-size: 28px; font-weight: 800; margin-bottom: 6px; }
  p.subtitle { color: var(--muted); font-size: 14px; }
  
  .controls-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
  }
  .btn-group { display: flex; gap: 10px; flex-wrap: wrap; }
  .btn {
    padding: 10px 18px;
    border-radius: 10px;
    border: 0;
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover { background: #2563eb; }
  .btn-outline { background: transparent; border: 1px solid var(--border); color: var(--muted); }
  .btn-outline:hover { background: #1f293d; color: #fff; }
  .btn-emerald { background: var(--emerald); color: #fff; }
  .btn-emerald:hover { background: #059669; }

  .results-grid { display: grid; grid-cols: 1; gap: 20px; }
  @media(min-width: 768px) { .results-grid { grid-template-columns: 1fr 1fr; } }
  
  .img-box {
    background: #060911;
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .img-header {
    padding: 10px 14px;
    background: rgba(255,255,255,0.02);
    border-bottom: 1px solid var(--border);
    font-size: 12px;
    font-weight: 600;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
  }
  .img-viewport {
    height: 260px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px;
    background: radial-gradient(#1f293d 1px, transparent 1px);
    background-size: 12px 12px;
  }
  .img-viewport img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; }

  .metrics-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }
  .score-header { display: flex; align-items: center; justify-content: space-between; }
  .score-val { font-size: 38px; font-weight: 900; color: var(--cyan); }
  .score-val span { font-size: 16px; color: var(--muted); font-weight: 500; }
  
  .stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; }
  .stat-pill {
    background: #090e1a;
    border: 1px solid var(--border);
    padding: 12px;
    border-radius: 12px;
  }
  .stat-label { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .stat-number { font-size: 16px; font-weight: 700; color: var(--text); }

  .legibility-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .leg-chip {
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
  }
  .leg-pass { background: rgba(16, 185, 129, 0.15); color: var(--emerald); border: 1px solid rgba(16, 185, 129, 0.3); }
  .leg-fail { background: rgba(244, 63, 94, 0.15); color: var(--rose); border: 1px solid rgba(244, 63, 94, 0.3); }

  .recs-box {
    background: #090e1a;
    border-left: 3px solid var(--cyan);
    padding: 14px 16px;
    border-radius: 0 10px 10px 0;
    font-size: 13px;
    line-height: 1.5;
  }
  .recs-box ul { margin-left: 18px; margin-top: 6px; color: #cbd5e1; }
</style>
</head>
<body>

<div class="container">
  <header>
    <div class="badge">OpenCV 5 AI Competition · AWS Multi-Modal</div>
    <h1>StoreVision AI Playground</h1>
    <p class="subtitle">Physical retail storefront perception, homography rectification & Michelson contrast scoring.</p>
  </header>

  <div class="controls-card">
    <div class="btn-group">
      <button class="btn btn-primary" onclick="loadSample()">📸 Load Sample Storefront</button>
      <input type="file" id="fileIn" accept="image/*" style="display:none;" onchange="handleFile(event)">
      <button class="btn btn-outline" onclick="document.getElementById('fileIn').click()">📁 Upload Local Image</button>
    </div>
    <div>
      <a href="/report" target="_blank" class="btn btn-outline">📑 Open Executive HTML Report</a>
    </div>
  </div>

  <div id="statusMsg" style="text-align:center; font-size:13px; color:var(--muted); display:none;">Processing perception pipeline...</div>

  <div id="resultsView" style="display:none; flex-direction:column; gap:20px;">
    <div class="results-grid">
      <div class="img-box">
        <div class="img-header">
          <span>Angled Storefront Capture</span>
          <span id="origDim">800x600 px</span>
        </div>
        <div class="img-viewport">
          <img id="origImg" src="" alt="Original Storefront">
        </div>
      </div>

      <div class="img-box">
        <div class="img-header">
          <span>Inverse Perspective Rectification</span>
          <span style="color:var(--emerald);">✔ OpenCV 5 Un-Warped</span>
        </div>
        <div class="img-viewport">
          <img id="rectImg" src="" alt="Rectified Signage">
        </div>
      </div>
    </div>

    <div class="metrics-card">
      <div class="score-header">
        <div>
          <div style="font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:0.05em; font-weight:700;">Visibility Perception Index</div>
          <div class="score-val" id="scoreVal">-- <span>/ 100</span></div>
        </div>
        <div>
          <a id="dlBtn" class="btn btn-emerald" download="rectified_signage.jpg" href="#">⬇ Download Rectified Sign</a>
        </div>
      </div>

      <div class="stat-row">
        <div class="stat-pill">
          <div class="stat-label">Optical Contrast Ratio</div>
          <div class="stat-number" id="contrastVal">--</div>
        </div>
        <div class="stat-pill">
          <div class="stat-label">Sightline Occlusion</div>
          <div class="stat-number" id="occlusionVal">--%</div>
        </div>
        <div class="stat-pill">
          <div class="stat-label">Engine Pipeline</div>
          <div class="stat-number" style="font-size:13px; font-weight:600; color:var(--cyan);">OpenCV 5 + AWS</div>
        </div>
      </div>

      <div>
        <div style="font-size:12px; font-weight:600; color:var(--muted); margin-bottom:8px;">Pedestrian & Traffic Sightline Legibility:</div>
        <div class="legibility-row">
          <span class="leg-chip" id="leg5m">5m Sidewalk: Checking...</span>
          <span class="leg-chip" id="leg15m">15m Cross-Street: Checking...</span>
          <span class="leg-chip" id="leg30m">30m Vehicle Traffic: Checking...</span>
        </div>
      </div>

      <div class="recs-box">
        <strong>AWS Bedrock Multi-Modal Architectural Recommendations:</strong>
        <ul id="recsList"></ul>
      </div>
    </div>
  </div>
</div>

<script>
async function processImageBytes(bytes, previewUrl) {
  const status = document.getElementById("statusMsg");
  const results = document.getElementById("resultsView");
  status.style.display = "block";
  status.textContent = "Running OpenCV 5 detection, homography & scoring...";

  try {
    const res = await fetch("/api/audit", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: bytes
    });
    const d = await res.json();

    document.getElementById("origImg").src = previewUrl;
    document.getElementById("rectImg").src = "data:image/jpeg;base64," + d.rectified_sign_base64;
    document.getElementById("dlBtn").href = "data:image/jpeg;base64," + d.rectified_sign_base64;

    document.getElementById("scoreVal").innerHTML = d.overall_score + " <span>/ 100</span>";
    document.getElementById("contrastVal").textContent = d.contrast_ratio;
    document.getElementById("occlusionVal").textContent = d.occlusion_pct + "%";

    const setChip = (id, label, pass) => {
      const el = document.getElementById(id);
      el.className = "leg-chip " + (pass ? "leg-pass" : "leg-fail");
      el.textContent = label + ": " + (pass ? "✔ Pass" : "❌ Obstructed");
    };
    setChip("leg5m", "5m Sidewalk", d.legibility["5m_pedestrian"]);
    setChip("leg15m", "15m Cross-Street", d.legibility["15m_cross_street"]);
    setChip("leg30m", "30m Vehicular", d.legibility["30m_vehicular"]);

    const recs = document.getElementById("recsList");
    recs.innerHTML = (d.recommendations || []).map(r => "<li>" + r + "</li>").join("");

    status.style.display = "none";
    results.style.display = "flex";
  } catch(err) {
    status.textContent = "Error executing perception audit: " + err;
  }
}

async function loadSample() {
  const resp = await fetch("/fixtures/synthetic_storefront.jpg");
  const blob = await resp.blob();
  const bytes = await blob.arrayBuffer();
  processImageBytes(bytes, "/fixtures/synthetic_storefront.jpg");
}

function handleFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    processImageBytes(reader.result, URL.createObjectURL(file));
  };
  reader.readAsArrayBuffer(file);
}
</script>
</body>
</html>
"""


class StoreVisionHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path in ("/health", "/api/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {
                "status": "healthy",
                "engine": "OpenCV 5.0 Physical Retail Perception",
                "version": "1.0.0",
                "features": ["perspective_rectification", "michelson_contrast", "sightline_occlusion", "aws_bedrock_multimodal"]
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        if self.path == "/report":
            report_path = Path("out/storevision_audit_report.html")
            if report_path.exists():
                content = report_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(content)
                return

        if self.path == "/fixtures/synthetic_storefront.jpg":
            fixture_path = Path("fixtures/synthetic_storefront.jpg")
            if fixture_path.exists():
                content = fixture_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(content)
                return

        # Default: Serve interactive StoreVision AI playground
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PLAYGROUND.encode("utf-8"))

    def do_POST(self):
        if self.path == "/api/audit":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self.send_error(400, "Missing image payload")
                return

            post_data = self.rfile.read(content_length)
            np_arr = np.frombuffer(post_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                self.send_error(400, "Invalid image encoding (JPEG or PNG required)")
                return

            candidates = detect_signage_candidates(img)
            primary_pts = candidates[0]
            rectified = four_point_transform(img, primary_pts)
            metrics = compute_visibility_metrics(rectified)

            result = {
                "overall_score": metrics.overall_score,
                "contrast_ratio": metrics.contrast_ratio,
                "occlusion_pct": metrics.occlusion_pct,
                "legibility": {
                    "5m_pedestrian": metrics.legibility_5m,
                    "15m_cross_street": metrics.legibility_15m,
                    "30m_vehicular": metrics.legibility_30m,
                },
                "recommendations": metrics.recommendations,
                "rectified_sign_base64": _to_base64_jpg(rectified)
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")


def run_server(port: int = 8080, host: str = "127.0.0.1"):
    server_address = (host, port)
    httpd = HTTPServer(server_address, StoreVisionHTTPHandler)
    print(f"🚀 StoreVision AI perception server listening on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()
