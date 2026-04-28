"""Broader verification - 10 images per class."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from detect_image import DeepfakeDetector
from pathlib import Path

data_root = Path(__file__).resolve().parent / "dataset"
real_images = sorted(list((data_root / "real").rglob("*.jpg")))[:10]
fake_images = sorted(list((data_root / "fake").rglob("*.jpg")))[:10]

detector = DeepfakeDetector()

lines = []
real_ok = 0
for img in real_images:
    r = detector.predict(image_path=str(img))
    ok = r['label'] == 'REAL FACE'
    if ok: real_ok += 1
    lines.append(f"{'PASS' if ok else 'FAIL'} REAL {img.name}: {r['label']} cls={r['scores']['classifier']:.4f} raw={r['raw_score']:.4f}")

fake_ok = 0
for img in fake_images:
    r = detector.predict(image_path=str(img))
    ok = r['label'] == 'DEEPFAKE'
    if ok: fake_ok += 1
    lines.append(f"{'PASS' if ok else 'FAIL'} FAKE {img.name}: {r['label']} cls={r['scores']['classifier']:.4f} raw={r['raw_score']:.4f}")

lines.append(f"\nReal: {real_ok}/{len(real_images)}  Fake: {fake_ok}/{len(fake_images)}  Total: {real_ok+fake_ok}/{len(real_images)+len(fake_images)}")
with open("verify_broad.txt", "w") as f:
    f.write("\n".join(lines))
print(f"Done. Real: {real_ok}/{len(real_images)}  Fake: {fake_ok}/{len(fake_images)}")
