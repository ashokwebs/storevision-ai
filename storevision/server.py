"""
storevision/server.py
Zero-dependency HTTP server for interactive StoreVision AI audits and live scorecard previews.
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


class StoreVisionHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silent server logging during automated tests
        pass

    def do_GET(self):
        if self.path == "/health" or self.path == "/api/health":
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

        # Default: Serve latest HTML report if exists, else interactive portal
        report_path = Path("out/storevision_audit_report.html")
        if report_path.exists():
            content = report_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        html = """<!doctype html>
<html>
<head><title>StoreVision AI Server</title></head>
<body style="background:#090d16;color:#f8fafc;font-family:sans-serif;padding:40px;text-align:center;">
<h1>StoreVision AI HTTP Perception Server</h1>
<p style="color:#94a3b8;">Run a scan via CLI to view report here, or POST image to <code>/api/audit</code>.</p>
</body>
</html>"""
        self.wfile.write(html.encode("utf-8"))

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
