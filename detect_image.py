"""
detect_image.py — Deepfake + AI-Image detection multi-signal inference pipeline.

Architecture:
    4-Model Neural Ensemble:
        1. XADE EfficientNet-B4   — StyleGAN / GAN face detection
        2. ViT Face-Swap          — Face-swap deepfakes (Realism/Deepfake)
        3. AI-Image Detector      — AI-generated images (Swin Transformer, artificial/human)
        4. ConvNeXt Classifier    — Training-distribution GAN fakes
    + 5 Heuristic Signals (FFT, noise, ELA, patch, metadata)
    + Test-Time Augmentation (TTA)
    + Strong artifact detection rules
    + Adaptive confidence scoring

Ensemble Scoring:
    final_score = 0.35*cls + 0.15*fft + 0.10*noise + 0.10*ela + 0.10*patch + 0.20*ai_img

Usage:
    python detect_image.py --image path/to/face.jpg
    python detect_image.py --image photo.png --model saved_models/deepfake_model.pth
"""

import os
import io
import re
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageFilter

from xade_model import get_xade_model
from classifier_model import compute_fft_magnitude, compute_noise_residual
from gan_model import Discriminator
from face_preprocessing import FaceDetector, get_eval_transforms
from utils import get_device

# Default model paths
DEFAULT_CLS_MODEL = Path(__file__).resolve().parent / "saved_models" / "xade_model" / "best_model.pt"
DEFAULT_OLD_CLS = Path(__file__).resolve().parent / "saved_models" / "deepfake_classifier.pth"
DEFAULT_GAN_MODEL = Path(__file__).resolve().parent / "saved_models" / "deepfake_detector.pth"
DEFAULT_AI_IMG_DETECTOR = Path(__file__).resolve().parent / "AI-image-detector"
CLS_IMAGE_SIZE = 380  # EfficientNet-B4 input size for XADE
OLD_CLS_SIZE = 256    # ConvNeXt input size for old classifier
GAN_IMAGE_SIZE = 128


# ---------------------------------------------------------------------------
# Heuristic Detectors (no ML model needed)
# ---------------------------------------------------------------------------

def ela_analysis(img: Image.Image) -> float:
    """
    Error Level Analysis (ELA) — multi-quality.

    Re-saves the image as JPEG at multiple quality levels and measures
    the coefficient of variation (CV) of the error. AI images from
    diffusion models and modern GANs show CV < 1.1 at Q=90, while
    real photos typically show CV > 1.15.

    Returns:
        Score in [0, 1] where 1 = likely fake.
    """
    orig_arr = np.array(img.convert("RGB"), dtype=np.float32)

    # Test at multiple quality levels for robustness
    cv_scores = []
    for quality in [90, 75]:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer).convert("RGB")
        resaved_arr = np.array(resaved, dtype=np.float32)
        diff = np.abs(orig_arr - resaved_arr)
        ela_map = np.clip(diff * 15.0, 0, 255)

        mean_ela = ela_map.mean()
        std_ela = ela_map.std()
        if mean_ela < 0.5:
            cv_scores.append(1.5)  # barely changes, ambiguous
        else:
            cv_scores.append(std_ela / (mean_ela + 1e-8))

    # Use the minimum CV across quality levels (most discriminative)
    min_cv = min(cv_scores)

    # Calibrated from real data:
    # AI images (deepfake/Grok): CV ~ 0.92 - 1.05 at Q=90
    # Real photos: CV ~ 1.17 - 1.33 at Q=90
    # GAN fakes from training: CV ~ 1.30+ at Q=90
    if min_cv < 0.85:
        score = 0.9
    elif min_cv < 0.95:
        score = 0.8
    elif min_cv < 1.05:
        score = 0.7
    elif min_cv < 1.15:
        score = 0.55
    elif min_cv < 1.25:
        score = 0.3
    else:
        score = 0.15

    return score


def frequency_analysis(img: Image.Image) -> float:
    """
    Frequency domain anomaly detection.

    AI-generated images have slightly less high-frequency energy relative
    to mid-frequency energy compared to real photos. This measures the
    high/mid frequency ratio and compares against calibrated thresholds.

    Calibrated values:
        Real photos: ratio ~ 0.87
        AI deepfakes: ratio ~ 0.77-0.83
        GAN training fakes: ratio ~ 0.85

    Returns:
        Score in [0, 1] where 1 = likely fake.
    """
    gray_img = img.convert("L").resize((256, 256), Image.LANCZOS)
    gray = np.array(gray_img, dtype=np.float32)

    # 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    log_mag = np.log1p(magnitude)
    log_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)

    h, w = log_mag.shape
    cy, cx = h // 2, w // 2

    # Radial frequency profile
    Y, X = np.ogrid[:h, :w]
    radius = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cx, cy)

    radial_profile = np.zeros(max_r)
    for r in range(max_r):
        mask = radius == r
        if mask.any():
            radial_profile[r] = log_mag[mask].mean()

    mid_freq = radial_profile[max_r // 4 : max_r // 2].mean()
    high_freq = radial_profile[max_r // 2 :].mean()
    ratio = high_freq / (mid_freq + 1e-8)

    # Calibrated thresholds from real data:
    # AI images: ratio ~ 0.77-0.83 (less high-freq detail)
    # Real photos: ratio ~ 0.87+ (natural noise, sensor grain)
    if ratio < 0.78:
        score = 0.85
    elif ratio < 0.82:
        score = 0.70
    elif ratio < 0.85:
        score = 0.50
    elif ratio < 0.88:
        score = 0.30
    else:
        score = 0.15

    # Check for spectral anomalies (GAN periodic artifacts)
    profile_diff = np.diff(radial_profile[5:])
    spike_threshold = np.std(profile_diff) * 3
    spike_count = np.sum(np.abs(profile_diff) > spike_threshold)
    if spike_count > 5:
        score = min(score + 0.15, 1.0)

    return score


def metadata_analysis(image_path: str) -> float:
    """
    EXIF/Metadata analysis.

    Real photos contain EXIF data (camera model, exposure, etc.).
    AI-generated images almost never have any EXIF data.
    Screenshots and re-saved images also lack EXIF.

    Returns:
        Score in [0, 1] where 1 = likely fake (no/suspicious metadata).
    """
    if image_path is None:
        return 0.5

    try:
        img = Image.open(image_path)

        # Check file extension — screenshots are suspicious
        ext = image_path.lower().split('.')[-1] if '.' in image_path else ''
        is_screenshot = 'screenshot' in image_path.lower()

        try:
            exif_data = img._getexif()
        except Exception:
            exif_data = None

        if exif_data is None:
            # No EXIF at all
            if is_screenshot:
                return 0.6  # Screenshots naturally lack EXIF
            return 0.75  # No EXIF is a strong AI signal

        # Camera-specific tags that prove a real camera took the photo
        camera_tags = {
            271,   # Make (camera manufacturer)
            272,   # Model (camera model)
            33434, # ExposureTime
            33437, # FNumber
            34855, # ISOSpeedRatings
            36867, # DateTimeOriginal
            37385, # Flash
            37386, # FocalLength
            41495, # SensingMethod
        }

        found_camera_tags = set(exif_data.keys()) & camera_tags

        if len(found_camera_tags) >= 3:
            return 0.05  # Strong camera evidence → real
        elif len(found_camera_tags) >= 1:
            return 0.2
        else:
            # Has EXIF but no camera tags (e.g., just resolution metadata)
            return 0.6

    except Exception:
        return 0.6


def watermark_detection(img: Image.Image) -> float:
    """
    Detect known AI tool watermarks and signatures.

    Checks for common AI generator watermarks in the corners of images
    (Grok, DALL-E, Midjourney, etc.) using simple image analysis.

    Returns:
        Score in [0, 1] where 1 = likely fake (watermark detected).
    """
    w, h = img.size

    # Check bottom-right corner for watermarks (most common position)
    # Sample several corner regions
    corners = [
        img.crop((w - w // 5, h - h // 8, w, h)),          # bottom-right
        img.crop((0, h - h // 8, w // 5, h)),              # bottom-left
        img.crop((w - w // 5, 0, w, h // 8)),              # top-right
        img.crop((0, 0, w // 5, h // 8)),                  # top-left
    ]

    score = 0.0

    for corner in corners:
        corner_arr = np.array(corner.convert("RGB"), dtype=np.float32)

        # Check for logo-like patterns: high contrast small regions
        # Watermarks typically show abrupt color changes
        gray_corner = np.mean(corner_arr, axis=2)
        local_std = gray_corner.std()

        # Very high local contrast in corner can indicate watermark
        if local_std > 60:
            score = max(score, 0.4)

        # Check for near-white or semi-transparent text overlay patterns
        # These appear as pixels significantly brighter/darker than neighbors
        if corner_arr.shape[0] > 5 and corner_arr.shape[1] > 5:
            # Edge detection on corner
            corner_gray = corner.convert("L")
            edges = corner_gray.filter(ImageFilter.FIND_EDGES)
            edge_arr = np.array(edges, dtype=np.float32)
            edge_density = (edge_arr > 30).mean()

            # High edge density in corner = likely text/logo
            if edge_density > 0.15:
                score = max(score, 0.5)
            if edge_density > 0.25:
                score = max(score, 0.7)

    # Also check image overall for unnatural uniformity in smooth areas
    # AI images often have suspiciously smooth skin/backgrounds
    img_arr = np.array(img.convert("RGB"), dtype=np.float32)
    # Sample center region for skin-like smoothness
    ch, cw = h // 2, w // 2
    center = img_arr[ch - h // 6:ch + h // 6, cw - w // 6:cw + w // 6]
    if center.size > 0:
        center_std = center.std()
        if center_std < 15:
            # Suspiciously smooth center
            score = max(score, 0.4)

    return score


def noise_analysis(img: Image.Image) -> float:
    """
    Sensor/noise inconsistency detection.

    Real photos have consistent sensor noise patterns across the image.
    AI-generated images show inconsistent noise: some regions are
    suspiciously clean while others have different noise characteristics.

    Returns:
        Score in [0, 1] where 1 = likely fake.
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    # Extract noise residual via high-pass filter
    from scipy.ndimage import uniform_filter
    noise_residuals = []
    for c in range(3):
        ch = arr[:, :, c]
        smooth = uniform_filter(ch, size=5)
        residual = ch - smooth
        noise_residuals.append(residual)
    noise_map = np.stack(noise_residuals, axis=2)

    # Split into patches and measure noise variance per patch
    patch_size = min(64, h // 4, w // 4)
    if patch_size < 8:
        return 0.5

    variances = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = noise_map[y:y+patch_size, x:x+patch_size]
            variances.append(np.var(patch))

    if len(variances) < 4:
        return 0.5

    variances = np.array(variances)
    mean_var = variances.mean()
    std_var = variances.std()

    # Coefficient of variation of noise variance across patches
    # Real images: consistent noise → low CoV (~0.3–0.5)
    # AI images: inconsistent noise → higher CoV (>0.6) or very low overall noise
    cov = std_var / (mean_var + 1e-8)

    # Very low overall noise = suspiciously clean (AI)
    if mean_var < 3.0:
        score = 0.75
    elif cov > 1.0:
        score = 0.8
    elif cov > 0.7:
        score = 0.65
    elif cov > 0.5:
        score = 0.45
    elif cov > 0.35:
        score = 0.3
    else:
        score = 0.15

    return score


def patch_analysis(img: Image.Image) -> float:
    """
    Patch-level inconsistency detection.

    Analyzes local statistics across image patches to find regions
    with inconsistent texture, color, or edge characteristics that
    indicate manipulation or AI generation.

    Returns:
        Score in [0, 1] where 1 = likely fake.
    """
    arr = np.array(img.convert("RGB").resize((256, 256), Image.LANCZOS), dtype=np.float32)
    h, w = arr.shape[:2]
    patch_size = 32

    # Collect per-patch statistics
    edge_densities = []
    color_means = []
    texture_vars = []

    gray = np.mean(arr, axis=2)
    # Simple edge detection via gradient
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.abs(np.diff(gray, axis=1))
    grad_mag = np.sqrt(gy[:, :-1]**2 + gx[:-1, :]**2)

    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch_color = arr[y:y+patch_size, x:x+patch_size]
            color_means.append(patch_color.mean(axis=(0, 1)))
            texture_vars.append(np.var(patch_color))

            pm = min(patch_size - 1, grad_mag.shape[0] - y, grad_mag.shape[1] - x)
            if pm > 0:
                edge_patch = grad_mag[y:y+pm, x:x+pm]
                edge_densities.append((edge_patch > 15).mean())

    if len(texture_vars) < 4:
        return 0.5

    texture_vars = np.array(texture_vars)
    edge_densities = np.array(edge_densities) if edge_densities else np.array([0.5])
    color_means = np.array(color_means)

    # Inconsistency metrics
    texture_cov = np.std(texture_vars) / (np.mean(texture_vars) + 1e-8)
    edge_cov = np.std(edge_densities) / (np.mean(edge_densities) + 1e-8)

    # Color channel consistency across patches
    color_std = np.std(color_means, axis=0).mean()

    # AI images tend to have more uniform texture but inconsistent edges
    # Real images have natural texture variation with consistent edge patterns
    score = 0.0

    # Very uniform texture = suspiciously smooth (AI)
    if texture_cov < 0.3:
        score = max(score, 0.7)
    elif texture_cov < 0.5:
        score = max(score, 0.5)

    # Edge inconsistency across patches
    if edge_cov > 0.8:
        score = max(score, 0.6)

    # Very low color variation = unnaturally uniform
    if color_std < 20:
        score = max(score, 0.55)

    # If nothing triggered, lean toward real
    if score == 0.0:
        score = 0.2

    return score


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class DeepfakeDetector:
    """
    End-to-end deepfake detection pipeline with ensemble scoring.

    Steps:
        1. Detect & crop face (RetinaFace / MTCNN / Haar fallback)
        2. Resize + normalize
        3. Pass through multi-branch classifier
        4. Run heuristic detectors (ELA, frequency, metadata, watermark)
        5. Optionally combine with GAN discriminator score
        6. Return label + confidence + per-branch scores
    """

    def __init__(self, model_path: str = None, device: torch.device = None):
        self.device = device or get_device()
        self.classifier_model = None
        self.old_classifier = None
        self.vit_pipe = None
        self.ai_img_pipe = None  # NEW: Swin AI-image-detector
        self.gan_discriminator = None
        self.model_type = "xade+old+ai_img"

        cls_path = model_path if model_path else str(DEFAULT_CLS_MODEL)
        old_cls_path = str(DEFAULT_OLD_CLS)
        gan_path = str(DEFAULT_GAN_MODEL)
        vit_path = str(Path(__file__).resolve().parent / "saved_models" / "hf_deepfake_model")
        ai_img_path = str(DEFAULT_AI_IMG_DETECTOR)

        # Load XADE Classifier (PyTorch) — detects real-world AI images / StyleGAN faces
        if os.path.isfile(cls_path):
            print(f"[DETECTOR] Loading XADE Classifier from {cls_path}...")
            try:
                self.classifier_model = get_xade_model(weights_path=cls_path, device=self.device)
                print(f"[DETECTOR] Success: XADE EfficientNet-B4 model loaded.")
            except Exception as e:
                print(f"[ERROR] Failed to load XADE model: {e}")
        else:
            print(f"[WARN] No XADE model file found at {cls_path}.")

        # Load old ConvNeXt Classifier — detects training-distribution GAN fakes
        if os.path.isfile(old_cls_path):
            try:
                from classifier_model import DeepfakeClassifier, LegacyClassifier
                ckpt = torch.load(old_cls_path, map_location=self.device, weights_only=False)
                state = ckpt.get('model_state_dict', ckpt)
                # Try new model first, fall back to legacy
                try:
                    model = DeepfakeClassifier().to(self.device)
                    model.load_state_dict(state)
                except Exception:
                    model = LegacyClassifier().to(self.device)
                    model.load_state_dict(state)
                model.eval()
                self.old_classifier = model
                print(f"[DETECTOR] Loaded old ConvNeXt classifier from {old_cls_path}")
            except Exception as e:
                print(f"[WARN] Could not load old classifier: {e}")

        # Load ViT Classifier (HF Pipeline) — secondary detector for face swaps
        if os.path.isdir(vit_path):
            try:
                from transformers import pipeline as hf_pipeline
                device_idx = 0 if self.device.type == "cuda" else -1
                self.vit_pipe = hf_pipeline(
                    "image-classification", model=vit_path, device=device_idx
                )
                print(f"[DETECTOR] Success: ViT face-swap detector loaded.")
            except Exception as e:
                print(f"[WARN] Could not load ViT model: {e}")
        else:
            print(f"[WARN] No ViT model at {vit_path}. XADE-only mode.")

        # ── NEW ── Load AI-image-detector (Swin Transformer) — catches VQGAN/CLIP/diffusion AI art
        if os.path.isdir(ai_img_path):
            try:
                from transformers import pipeline as hf_pipeline
                device_idx = 0 if self.device.type == "cuda" else -1
                self.ai_img_pipe = hf_pipeline(
                    "image-classification",
                    model=ai_img_path,
                    device=device_idx,
                )
                print(f"[DETECTOR] Success: AI-Image Detector (Swin) loaded from {ai_img_path}.")
            except Exception as e:
                print(f"[WARN] Could not load AI-Image Detector: {e}")
        else:
            print(f"[WARN] AI-image-detector not found at {ai_img_path}. Run: git clone https://huggingface.co/umm-maybe/AI-image-detector")

        # Load GAN discriminator for ensemble (optional)
        if os.path.isfile(gan_path):
            try:
                self.gan_discriminator = Discriminator().to(self.device)
                state = torch.load(gan_path, map_location=self.device, weights_only=True)
                self.gan_discriminator.load_state_dict(state)
                self.gan_discriminator.eval()
                print(f"[DETECTOR] Loaded GAN Discriminator for ensemble from {gan_path}")
            except Exception:
                self.gan_discriminator = None

        # Face detector (uses RetinaFace by default)
        self.face_detector = FaceDetector(
            image_size=CLS_IMAGE_SIZE,
            device=str(self.device),
        )

        # Eval transforms
        self.cls_transform = get_eval_transforms(CLS_IMAGE_SIZE)
        self.old_cls_transform = get_eval_transforms(OLD_CLS_SIZE)
        self.gan_transform = get_eval_transforms(GAN_IMAGE_SIZE)

    def _run_classifier_on_face(self, face_pil: Image.Image) -> float:
        """Run XADE model on an image, return fake score."""
        if self.classifier_model is None:
            return 0.5

        face_tensor = self.cls_transform(face_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.classifier_model(face_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        # Label mapping for XADE: 0 = Real, 1 = Fake  (per xade_model.py docstring)
        fake_score = probs[1]
                
        return float(fake_score)

    def _run_old_classifier(self, face_pil: Image.Image) -> float:
        """Run old ConvNeXt multi-branch classifier on a face, return fake score."""
        if self.old_classifier is None:
            return None

        face_resized = face_pil.resize((OLD_CLS_SIZE, OLD_CLS_SIZE), Image.LANCZOS)
        face_tensor = self.old_cls_transform(face_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Multi-branch model computes FFT/noise internally when not provided
            logits = self.old_classifier(face_tensor)
            # Single logit output: sigmoid > 0.5 = fake
            fake_score = torch.sigmoid(logits).item()

        return float(fake_score)

    def _run_vit_on_image(self, img: Image.Image) -> float:
        """Run ViT (HF) model on an image, return fake score (face-swap detector)."""
        if getattr(self, 'vit_pipe', None) is None:
            return 0.5
        
        results = self.vit_pipe(img, top_k=2)
        for res in results:
            label = str(res['label']).lower()
            if 'fake' in label or 'deepfake' in label:
                return float(res['score'])
            elif 'real' in label or 'realism' in label:
                return float(1.0 - res['score'])
        return 0.5

    def _run_ai_image_detector(self, img: Image.Image) -> float:
        """
        Run the AI-image-detector (Swin Transformer, umm-maybe/AI-image-detector).

        Labels: 'artificial' (AI-generated) / 'human' (real photo).
        Returns fake score in [0, 1] where 1 = AI-generated.
        """
        if getattr(self, 'ai_img_pipe', None) is None:
            return 0.5

        try:
            results = self.ai_img_pipe(img, top_k=2)
            for res in results:
                label = str(res['label']).lower()
                if 'artificial' in label or 'ai' in label or 'fake' in label:
                    return float(res['score'])
                elif 'human' in label or 'real' in label:
                    return float(1.0 - res['score'])
        except Exception as e:
            print(f"[WARN] AI-Image Detector inference error: {e}")
        return 0.5

    def _tta_classifier_score(self, img: Image.Image, face: Image.Image = None) -> float:
        """
        Dual-model TTA: run XADE + ViT face-swap detector.
        - XADE (EfficientNet-B4) catches StyleGAN / GAN face fakes.
        - ViT (HF) handles face-swap deepfakes (Realism / Deepfake labels).
        Swin AI-image-detector is intentionally kept SEPARATE in predict() to
        avoid double-counting — it is weighted as its own signal there.
        Returns the combined XADE+ViT score in [0, 1].
        """
        xade_scores = []

        # 1. XADE EfficientNet-B4 with horizontal-flip TTA
        if self.classifier_model is not None:
            xade_scores.append(self._run_classifier_on_face(img))
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            xade_scores.append(self._run_classifier_on_face(flipped))
            if face is not None:
                xade_scores.append(self._run_classifier_on_face(face))

        xade_avg = float(np.mean(xade_scores)) if xade_scores else 0.5

        # 2. ViT face-swap detector
        if getattr(self, 'vit_pipe', None) is None:
            return xade_avg

        vit_score = self._run_vit_on_image(img)

        # Dynamic Confidence Resolution:
        # If models heavily disagree, trust the more confident one (>0.68).
        if vit_score > 0.68 and xade_avg < 0.40:
            # ViT strongly flags it (face-swap), XADE misses it
            return float(vit_score * 0.80 + xade_avg * 0.20)
        elif xade_avg > 0.68 and vit_score < 0.40:
            # XADE strongly flags it (StyleGAN), ViT misses it
            return float(xade_avg * 0.80 + vit_score * 0.20)
        else:
            # No stark confident disagreement: weighted average (XADE primary)
            return float(xade_avg * 0.65 + vit_score * 0.35)

    def predict(self, image_path: str = None, pil_image: Image.Image = None):
        """
        Predict whether an image is real, deepfake, or AI-generated using
        4-model neural ensemble + 5 heuristic signals with TTA and adaptive confidence.

        Models used:
            1. XADE EfficientNet-B4   — StyleGAN / GAN face detection
            2. ViT (HF)               — Face-swap deepfakes
            3. Swin AI-image-detector — VQGAN / diffusion / broad AI art
            4. ConvNeXt               — Training-distribution GAN fakes

        Returns:
            dict with keys:
                label       — "REAL FACE", "DEEPFAKE", or "AI-GENERATED"
                confidence  — float [0, 100]
                raw_score   — float [0, 1] (ensemble fakeness score)
                scores      — dict of per-signal scores
        """
        # Load image
        if pil_image is not None:
            img = pil_image.convert("RGB")
        elif image_path is not None:
            img = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("Provide either image_path or pil_image.")

        # Detect and crop face
        face = self.face_detector.detect_and_crop(img)
        if face is None:
            face = img.resize((CLS_IMAGE_SIZE, CLS_IMAGE_SIZE), Image.LANCZOS)

        # === SIGNAL 1: Combined Neural Ensemble (XADE + ViT + Swin) with TTA ===
        cls_fake_score = self._tta_classifier_score(img, face)

        # === SIGNAL 2: Swin AI-Image Detector (standalone, on full image) ===
        ai_img_score = self._run_ai_image_detector(img)

        # === SIGNAL 3: FFT / Frequency analysis ===
        fft_score = frequency_analysis(img)

        # === SIGNAL 4: Noise inconsistency ===
        noise_score = noise_analysis(img)

        # === SIGNAL 5: ELA (Error Level Analysis) ===
        ela_sc = ela_analysis(img)

        # === SIGNAL 6: Patch-level inconsistency ===
        patch_sc = patch_analysis(img)

        # === Optional: GAN Discriminator ===
        gan_fake_score = None
        if self.gan_discriminator is not None:
            gan_face = face.resize((GAN_IMAGE_SIZE, GAN_IMAGE_SIZE), Image.LANCZOS)
            gan_tensor = self.gan_transform(gan_face).unsqueeze(0).to(self.device)
            with torch.no_grad():
                gan_prob = self.gan_discriminator(gan_tensor).view(-1).item()
            gan_fake_score = 1.0 - gan_prob

        # === SIGNAL 7: EXIF metadata (real cameras leave provenance data) ===
        meta_score = metadata_analysis(image_path) if image_path else 0.5

        # === EXIF TRUST: If EXIF proves a real camera, dampen classifiers ===
        # Real camera photos (meta_score <= 0.2) have 3+ camera-specific EXIF tags.
        if meta_score <= 0.2:
            cls_fake_score = cls_fake_score * 0.4   # Dampen by 60%
            ai_img_score = ai_img_score * 0.5        # Dampen Swin by 50%

        # === ENSEMBLE: Weighted multi-signal score ===
        # Neural ensemble (cls) = 0.35, AI-img detector = 0.20
        # FFT = 0.15, ELA = 0.10, Noise = 0.10, Patch = 0.10
        final_score = (
            0.35 * cls_fake_score
            + 0.20 * ai_img_score
            + 0.15 * fft_score
            + 0.10 * noise_score
            + 0.10 * ela_sc
            + 0.10 * patch_sc
        )

        # Blend GAN discriminator if available
        if gan_fake_score is not None:
            final_score = 0.85 * final_score + 0.15 * gan_fake_score

        # === ARTIFACT OVERRIDE (conservative) ===
        # Only raises score when evidence is strong

        # Classifier very confident AND no EXIF camera data → trust it heavily
        if cls_fake_score > 0.85 and meta_score > 0.5:
            final_score = max(final_score, 0.80)

        # Neural network overall consensus is strongly fake → trust it at borderline
        if cls_fake_score > 0.6 and meta_score > 0.5:
            final_score = max(final_score, 0.55)

        # AI-image detector highly confident → boost score
        if ai_img_score > 0.80 and meta_score > 0.4:
            final_score = max(final_score, 0.75)

        # 3+ heuristic signals agree → clearly synthetic
        artifact_signals = [fft_score, noise_score, ela_sc, patch_sc]
        strong_count = sum(1 for s in artifact_signals if s > 0.6)
        if strong_count >= 3:
            final_score = max(final_score, 0.75)

        # FFT + ELA both high → likely AI-generated
        if fft_score > 0.7 and ela_sc > 0.7:
            final_score = max(final_score, 0.85)

        # High FFT alone (face-swap blending boundary, exclude heavy JPEGs)
        if fft_score >= 0.68 and noise_score < 0.6:
            final_score = max(final_score, 0.65)

        # Clamp
        final_score = float(np.clip(final_score, 0.0, 1.0))

        # === DETECTION TYPE: Distinguish "AI-GENERATED" from face-swap "DEEPFAKE" ===
        # If Swin AI-image detector is the dominant signal (>0.75) but face/XADE scores
        # are moderate, label it "AI-GENERATED" (not a face-swap, but an AI art image).
        if final_score >= 0.55:
            if (ai_img_score > 0.75 and cls_fake_score < 0.6
                    and getattr(self, 'ai_img_pipe', None) is not None):
                label = "AI-GENERATED"
            else:
                label = "DEEPFAKE"
            confidence = final_score * 100
        else:
            label = "REAL FACE"
            confidence = (1 - final_score) * 100

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "raw_score": round(final_score, 4),
            "scores": {
                "classifier": round(cls_fake_score, 4),
                "ai_image_detector": round(ai_img_score, 4),
                "fft": round(fft_score, 4),
                "noise": round(noise_score, 4),
                "ela": round(ela_sc, 4),
                "patch": round(patch_sc, 4),
                "gan_discriminator": round(gan_fake_score, 4) if gan_fake_score is not None else "N/A",
            }
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deepfake face detection.")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image.")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained classifier weights.")
    args = parser.parse_args()

    detector = DeepfakeDetector(model_path=args.model)
    result = detector.predict(image_path=args.image)

    print("\n" + "=" * 60)
    print(f"  Prediction        : {result['label']}")
    print(f"  Confidence        : {result['confidence']:.1f}%")
    print(f"  Raw Score         : {result['raw_score']}")
    print(f"  --- Neural Model Scores ---")
    print(f"  XADE+ViT Ensemble : {result['scores']['classifier']}")
    print(f"  AI-Image Detector : {result['scores']['ai_image_detector']}")
    print(f"  GAN Discriminator : {result['scores']['gan_discriminator']}")
    print(f"  --- Heuristic Scores ---")
    print(f"  FFT Frequency     : {result['scores']['fft']}")
    print(f"  Noise Analysis    : {result['scores']['noise']}")
    print(f"  Error Level (ELA) : {result['scores']['ela']}")
    print(f"  Patch Analysis    : {result['scores']['patch']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
