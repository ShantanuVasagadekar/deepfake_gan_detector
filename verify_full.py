"""Test EXIF-dampened detection on webcam, face swap, and dataset images."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from detect_image import DeepfakeDetector
from pathlib import Path

data_root = Path(__file__).resolve().parent / "dataset"
real_images = sorted(list((data_root / "real").rglob("*.jpg")))[:10]
fake_images = sorted(list((data_root / "fake").rglob("*.jpg")))[:10]
charlie = r"C:\Users\Shantanu Vasagadekar\Downloads\charlie.jpg"
webcam = r"C:\Users\Shantanu Vasagadekar\Downloads\WIN_20260316_14_45_50_Pro.jpg"

detector = DeepfakeDetector()

lines = []

# Test webcam (should be REAL)
if os.path.isfile(webcam):
    r = detector.predict(image_path=webcam)
    ok = r['label'] == 'REAL FACE'
    lines.append(f"{'PASS' if ok else 'FAIL'} WEBCAM WIN_Pro.jpg: {r['label']} cls={r['scores']['classifier']:.4f} raw={r['raw_score']:.4f}")

# Test charlie (face swap, should be DEEPFAKE)
if os.path.isfile(charlie):
    r = detector.predict(image_path=charlie)
    ok = r['label'] == 'DEEPFAKE'
    lines.append(f"{'PASS' if ok else 'FAIL'} FACESWAP charlie.jpg: {r['label']} cls={r['scores']['classifier']:.4f} raw={r['raw_score']:.4f}")

# Test real images
real_ok = 0
for img in real_images:
    r = detector.predict(image_path=str(img))
    ok = r['label'] == 'REAL FACE'
    if ok: real_ok += 1
    lines.append(f"{'PASS' if ok else 'FAIL'} REAL {img.name}: {r['label']} cls={r['scores']['classifier']:.4f} raw={r['raw_score']:.4f}")

# Test fake (GAN) images
fake_ok = 0
for img in fake_images:
    r = detector.predict(image_path=str(img))
    ok = r['label'] == 'DEEPFAKE'
    if ok: fake_ok += 1
    lines.append(f"{'PASS' if ok else 'FAIL'} FAKE {img.name}: {r['label']} cls={r['scores']['classifier']:.4f} raw={r['raw_score']:.4f}")

lines.append(f"\nReal: {real_ok}/{len(real_images)}  Fake: {fake_ok}/{len(fake_images)}")
with open("verify_full.txt", "w") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
