"""Quick verification script for the HF model integration."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from detect_image import DeepfakeDetector
from pathlib import Path
import time

def main():
    print("=" * 60)
    print("  VERIFICATION: Hugging Face Deepfake Detector Integration")
    print("=" * 60)
    
    # 1. Load detector
    print("\n[1/4] Loading detector...")
    t0 = time.time()
    detector = DeepfakeDetector()
    print(f"      Loaded in {time.time()-t0:.1f}s")
    print(f"      Model type: {detector.model_type}")
    print(f"      HF Pipeline: {'LOADED' if detector.classifier_pipe else 'MISSING'}")
    
    if not detector.classifier_pipe:
        print("\nFAILED: classifier_pipe is None. Model did not load.")
        return
    
    # 2. Test on a REAL image
    data_root = Path(__file__).resolve().parent / "dataset"
    real_images = list((data_root / "real").rglob("*.jpg"))[:3]
    fake_images = list((data_root / "fake").rglob("*.jpg"))[:3]
    
    print(f"\n[2/4] Found {len(real_images)} real samples, {len(fake_images)} fake samples")
    
    # 3. Predict on real images
    print("\n[3/4] Testing REAL images:")
    real_correct = 0
    for img_path in real_images:
        t0 = time.time()
        result = detector.predict(image_path=str(img_path))
        dt = time.time() - t0
        is_correct = result["label"] == "REAL FACE"
        status = "PASS" if is_correct else "FAIL"
        if is_correct:
            real_correct += 1
        print(f"      [{status}] {img_path.name}: {result['label']} "
              f"(conf={result['confidence']:.1f}%, cls={result['scores']['classifier']:.4f}, "
              f"fft={result['scores']['fft']:.4f}, noise={result['scores']['noise']:.4f}, "
              f"ela={result['scores']['ela']:.4f}, patch={result['scores']['patch']:.4f}, "
              f"raw={result['raw_score']:.4f}) [{dt:.1f}s]")
    
    # 4. Predict on fake images
    print("\n[4/4] Testing FAKE images:")
    fake_correct = 0
    for img_path in fake_images:
        t0 = time.time()
        result = detector.predict(image_path=str(img_path))
        dt = time.time() - t0
        is_correct = result["label"] == "DEEPFAKE"
        status = "PASS" if is_correct else "FAIL"
        if is_correct:
            fake_correct += 1
        print(f"      [{status}] {img_path.name}: {result['label']} "
              f"(conf={result['confidence']:.1f}%, cls={result['scores']['classifier']:.4f}, "
              f"fft={result['scores']['fft']:.4f}, noise={result['scores']['noise']:.4f}, "
              f"ela={result['scores']['ela']:.4f}, patch={result['scores']['patch']:.4f}, "
              f"raw={result['raw_score']:.4f}) [{dt:.1f}s]")
    
    # Summary
    total = len(real_images) + len(fake_images)
    correct = real_correct + fake_correct
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {correct}/{total} correct")
    print(f"    Real accuracy: {real_correct}/{len(real_images)}")
    print(f"    Fake accuracy: {fake_correct}/{len(fake_images)}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
