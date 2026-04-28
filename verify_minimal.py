"""Minimal verification - writes results to file to avoid encoding issues."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from detect_image import DeepfakeDetector
from pathlib import Path

data_root = Path(__file__).resolve().parent / "dataset"
real_images = list((data_root / "real").rglob("*.jpg"))[:3]
fake_images = list((data_root / "fake").rglob("*.jpg"))[:3]

detector = DeepfakeDetector()

lines = []
lines.append("=== REAL IMAGES ===")
for img in real_images:
    r = detector.predict(image_path=str(img))
    lines.append(f"{img.name}: label={r['label']} conf={r['confidence']:.1f}% cls={r['scores']['classifier']:.4f} fft={r['scores']['fft']:.4f} noise={r['scores']['noise']:.4f} ela={r['scores']['ela']:.4f} patch={r['scores']['patch']:.4f} raw={r['raw_score']:.4f}")

lines.append("")
lines.append("=== FAKE IMAGES ===")
for img in fake_images:
    r = detector.predict(image_path=str(img))
    lines.append(f"{img.name}: label={r['label']} conf={r['confidence']:.1f}% cls={r['scores']['classifier']:.4f} fft={r['scores']['fft']:.4f} noise={r['scores']['noise']:.4f} ela={r['scores']['ela']:.4f} patch={r['scores']['patch']:.4f} raw={r['raw_score']:.4f}")

output = "\n".join(lines)
with open("verify_results.txt", "w", encoding="ascii", errors="replace") as f:
    f.write(output)
print("Done. Results in verify_results.txt")
