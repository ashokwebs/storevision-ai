# 👁️ StoreVision AI: Physical Storefront Perception & Signage Intelligence

> **Built for:** OpenCV AI Competition 2026 (Powered by AWS)  
> **Platform:** Devpost ([opencv2026.devpost.com](https://opencv2026.devpost.com/))  
> **Tech Stack:** OpenCV 5 (Python bindings), NumPy, AWS Bedrock (Claude 3.5 Sonnet)  
> **Repository:** [https://github.com/ashokwebs/storevision-ai](https://github.com/ashokwebs/storevision-ai)  

---

## 🎯 What StoreVision AI Does
StoreVision AI turns any street-level smartphone video or street-view photo into an automated retail storefront audit:
1. **Signage Detection:** Identifies rectangular commercial signage boards from angled urban perspectives.
2. **Homography Rectification:** Computes an inverse perspective transform to un-warp angled street-level captures into flat, front-facing rectangles.
3. **Optical Perception Scoring:** Evaluates Michelson optical contrast, stroke-width visibility, occlusion percentage (obstruction by trees/poles), and pedestrian sightlines at 5m, 15m, and 30m.
4. **AWS Bedrock Integration:** Emits structured multi-modal payloads ready for AWS Bedrock Anthropic Claude 3.5 Sonnet to generate architectural footfall recommendations.

---

## 🚀 Quick Start (One Command)
```bash
git clone https://github.com/ashokwebs/storevision-ai.git
cd storevision-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the complete test & audit demo
bash run_demo.sh
```

---

## 📊 Sample Output
```text
👁️ [StoreVision AI] Processing physical storefront image: fixtures/synthetic_storefront.jpg
   ✔ Detected 1 candidate signage contour(s)
   ✔ Perspective Rectification: Saved un-warped sign to out/rectified_signage.jpg

=======================================================
 📊 STOREFRONT PHYSICAL PERCEPTION SCORE: 57/100
=======================================================
 • Optical Contrast Ratio   : 0.143 (Target: >0.65)
 • Sightline Occlusion       : 1.0%
 • 5m Sidewalk Legibility   : ✅ Pass
 • 15m Cross-Street         : ✅ Pass
 • 30m Vehicle Traffic      : ✅ Pass

💡 Automated Architectural Recommendations:
   1. Increase color contrast between lettering and backing board (aim for >0.65 Michelson contrast).

📦 AWS Bedrock Multi-Modal Payload: out/aws_bedrock_payload.json
```
