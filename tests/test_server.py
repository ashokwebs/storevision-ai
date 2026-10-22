"""
tests/test_server.py
Tests the zero-dependency StoreVision HTTP server endpoints (/health and /api/audit).
"""
import threading
import time
import urllib.request
import urllib.error
import json
import cv2
import numpy as np
import pytest
from http.server import HTTPServer
from storevision.server import StoreVisionHTTPHandler


@pytest.fixture(scope="module")
def live_server():
    # Bind to random ephemeral port or high port
    server = HTTPServer(("127.0.0.1", 18765), StoreVisionHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    yield "http://127.0.0.1:18765"
    server.shutdown()
    server.server_close()


def test_server_health_check(live_server):
    url = f"{live_server}/api/health"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=2) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "healthy"
        assert "OpenCV" in data["engine"]


def test_server_audit_post_image(live_server):
    url = f"{live_server}/api/audit"
    
    # Generate synthetic image in memory
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    img[30:70, 50:250] = 0
    _, buffer = cv2.imencode(".jpg", img)
    raw_bytes = buffer.tobytes()

    req = urllib.request.Request(url, data=raw_bytes, headers={"Content-Type": "image/jpeg"})
    with urllib.request.urlopen(req, timeout=2) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "overall_score" in data
        assert "contrast_ratio" in data
        assert "rectified_sign_base64" in data
        assert len(data["rectified_sign_base64"]) > 100
