import os
import argparse
import random
import math
import io
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

try:
    from retinaface import RetinaFace as RetinaFaceDetector
    HAS_RETINAFACE = True
except ImportError:
    HAS_RETINAFACE = False

LABEL_REAL = 0
LABEL_FAKE = 1

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class FaceDetector:
    def __init__(self, image_size: int = 256, device: str = "cpu"):
        self.image_size = image_size
        if HAS_RETINAFACE:
            self._mode = "retinaface"
        elif MTCNN is not None:
            self.mtcnn = MTCNN(
                image_size=image_size,
                margin=20,
                keep_all=False,
                device=device,
            )
            self._mode = "mtcnn"
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            self._mode = "haar"

    def detect_and_crop(self, img: Image.Image) -> Optional[Image.Image]:
        if self._mode == "retinaface":
            return self._detect_retinaface(img)
        elif self._mode == "mtcnn":
            return self._detect_mtcnn(img)
        return self._detect_haar(img)

    def _detect_retinaface(self, img: Image.Image) -> Optional[Image.Image]:
        try:
            img_np = np.array(img)
            faces = RetinaFaceDetector.detect_faces(img_np)
            if not faces:
                return None
            best_face = None
            best_area = 0
            for key, face_data in faces.items():
                area = face_data.get("facial_area", [0, 0, 0, 0])
                w = area[2] - area[0]
                h = area[3] - area[1]
                if w * h > best_area:
                    best_area = w * h
                    best_face = face_data
            if best_face is None:
                return None
            x1, y1, x2, y2 = best_face["facial_area"]
            landmarks = best_face.get("landmarks", {})
            if "left_eye" in landmarks and "right_eye" in landmarks:
                aligned = self._align_face(
                    img_np, landmarks["left_eye"], landmarks["right_eye"], (x1, y1, x2, y2)
                )
                if aligned is not None:
                    return aligned.resize((self.image_size, self.image_size), Image.LANCZOS)
            margin = int(max(x2 - x1, y2 - y1) * 0.15)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(img_np.shape[1], x2 + margin)
            y2 = min(img_np.shape[0], y2 + margin)
            face_crop = img.crop((x1, y1, x2, y2))
            return face_crop.resize((self.image_size, self.image_size), Image.LANCZOS)
        except Exception:
            return None

    def _align_face(self, img_np: np.ndarray, left_eye, right_eye, bbox) -> Optional[Image.Image]:
        try:
            lx, ly = left_eye
            rx, ry = right_eye
            dx = rx - lx
            dy = ry - ly
            angle = math.degrees(math.atan2(dy, dx))
            eye_center = ((lx + rx) / 2, (ly + ry) / 2)
            M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            h, w = img_np.shape[:2]
            aligned = cv2.warpAffine(
                img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            x1, y1, x2, y2 = bbox
            margin = int(max(x2 - x1, y2 - y1) * 0.2)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            face_crop = aligned[y1:y2, x1:x2]
            return Image.fromarray(face_crop)
        except Exception:
            return None

    def _detect_mtcnn(self, img: Image.Image) -> Optional[Image.Image]:
        try:
            face_tensor = self.mtcnn(img)
            if face_tensor is not None:
                face_np = ((face_tensor.permute(1, 2, 0).numpy() + 1) / 2 * 255).astype(np.uint8)
                return Image.fromarray(face_np)
        except Exception:
            pass
        return None

    def _detect_haar(self, img: Image.Image) -> Optional[Image.Image]:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        crop = img.crop((x, y, x + w, y + h))
        return crop.resize((self.image_size, self.image_size), Image.LANCZOS)


class GaussianNoiseVar:
    def __init__(self, sigma_range: Tuple[float, float] = (3.0, 25.0), p: float = 0.6):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.array(img).astype(np.float32)
        sigma = random.uniform(*self.sigma_range)
        noise = np.random.normal(0.0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class RandomGaussianOrMotionBlur:
    def __init__(self, p: float = 0.45):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        if random.random() < 0.55:
            r = random.uniform(0.4, 2.2)
            return img.filter(ImageFilter.GaussianBlur(radius=r))
        arr = np.array(img)
        k = random.choice([7, 9, 11, 15])
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = np.ones(k, dtype=np.float32) / k
        motion = cv2.filter2D(arr, -1, kernel)
        return Image.fromarray(np.clip(motion, 0, 255).astype(np.uint8))


class RandomJPEG:
    def __init__(self, p: float = 0.55, q_low: int = 10, q_high: int = 90):
        self.p = p
        self.q_low = q_low
        self.q_high = q_high

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        quality = random.randint(self.q_low, self.q_high)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomDownUpScale:
    def __init__(self, p: float = 0.45, min_scale: float = 0.35, max_scale: float = 0.85):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        s = random.uniform(self.min_scale, self.max_scale)
        nw, nh = max(8, int(w * s)), max(8, int(h * s))
        small = img.resize((nw, nh), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class RandomGridArtifact:
    def __init__(self, p: float = 0.35):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        step = random.randint(8, 32)
        alpha = random.uniform(0.03, 0.12)
        col = random.uniform(0, 255)
        for x in range(0, w, step):
            arr[:, x : x + 1] = arr[:, x : x + 1] * (1 - alpha) + col * alpha
        for y in range(0, h, step):
            arr[y : y + 1, :] = arr[y : y + 1, :] * (1 - alpha) + col * alpha
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class RandomFrequencyPerturb:
    def __init__(self, p: float = 0.35):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        out = []
        for c in range(3):
            ch = arr[:, :, c]
            F = np.fft.fft2(ch)
            noise = (np.random.randn(*F.shape) + 1j * np.random.randn(*F.shape)).astype(np.complex64)
            mag = 0.012 * random.uniform(0.5, 2.0)
            F2 = F + mag * noise * np.mean(np.abs(F))
            rec = np.abs(np.fft.ifft2(F2))
            out.append(rec)
        stacked = np.stack(out, axis=2)
        stacked = np.clip(stacked * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(stacked)


class RandomPatchInconsistency:
    def __init__(self, p: float = 0.4):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.array(img).copy()
        h, w = arr.shape[:2]
        ph = random.randint(h // 16, h // 6)
        pw = random.randint(w // 16, w // 6)
        y0 = random.randint(0, h - ph - 1)
        x0 = random.randint(0, w - pw - 1)
        dy = random.randint(-ph // 2, ph // 2)
        dx = random.randint(-pw // 2, pw // 2)
        y1 = np.clip(y0 + dy, 0, h - ph)
        x1 = np.clip(x0 + dx, 0, w - pw)
        patch = arr[y1 : y1 + ph, x1 : x1 + pw].copy()
        blend = random.uniform(0.55, 1.0)
        reg = arr[y0 : y0 + ph, x0 : x0 + pw].astype(np.float32)
        pat = patch.astype(np.float32)
        arr[y0 : y0 + ph, x0 : x0 + pw] = np.clip(reg * (1 - blend) + pat * blend, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))


class PILCutout:
    def __init__(self, p: float = 0.35, min_ratio: float = 0.02, max_ratio: float = 0.18):
        self.p = p
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.array(img).copy()
        h, w = arr.shape[:2]
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        ch = max(4, int(h * math.sqrt(ratio)))
        cw = max(4, int(w * math.sqrt(ratio)))
        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)
        arr[y0 : y0 + ch, x0 : x0 + cw] = random.randint(0, 255)
        return Image.fromarray(arr)


class RandomHueShift:
    """Slight hue/channel imbalance to simulate AI color anomalies."""
    def __init__(self, p: float = 0.3, max_shift: float = 15.0):
        self.p = p
        self.max_shift = max_shift

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.array(img).astype(np.float32)
        shifts = np.array([random.uniform(-self.max_shift, self.max_shift) for _ in range(3)])
        arr = np.clip(arr + shifts[None, None, :], 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class RandomSharpenBlur:
    """Blur then sharpen — simulates reprocessing artifacts."""
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        blurred = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        sharpened = blurred.filter(ImageFilter.SHARPEN)
        return sharpened


class RandomPeriodicNoise:
    """Adds subtle sine-wave pattern in spatial domain — simulates GAN periodicity."""
    def __init__(self, p: float = 0.25, amplitude: float = 8.0):
        self.p = p
        self.amplitude = amplitude

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        freq = random.uniform(0.05, 0.3)
        phase = random.uniform(0, 2 * math.pi)
        amp = random.uniform(2.0, self.amplitude)
        if random.random() < 0.5:
            pattern = amp * np.sin(2 * math.pi * freq * np.arange(w) + phase)
            arr += pattern[None, :, None]
        else:
            pattern = amp * np.sin(2 * math.pi * freq * np.arange(h) + phase)
            arr += pattern[:, None, None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _imagenet_norm_tensor():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def build_real_augment_pipeline(image_size: int) -> transforms.Compose:
    """Moderate augmentation for real images — teaches model what natural variation looks like."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.06),
            GaussianNoiseVar(sigma_range=(2.0, 12.0), p=0.4),
            RandomJPEG(p=0.35, q_low=40, q_high=90),
            RandomDownUpScale(p=0.25, min_scale=0.65, max_scale=0.95),
            transforms.ToTensor(),
            _imagenet_norm_tensor(),
            transforms.RandomErasing(p=0.12, scale=(0.02, 0.08)),
        ]
    )


def build_fake_augment_pipeline(image_size: int) -> transforms.Compose:
    """Light augmentation for fake images — preserves artifacts the model needs to detect."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.04),
            GaussianNoiseVar(sigma_range=(2.0, 12.0), p=0.3),
            RandomGaussianOrMotionBlur(p=0.2),
            RandomJPEG(p=0.3, q_low=50, q_high=95),
            RandomDownUpScale(p=0.2, min_scale=0.65, max_scale=0.95),
            RandomFrequencyPerturb(p=0.15),
            RandomHueShift(p=0.2, max_shift=8.0),
            RandomSharpenBlur(p=0.15),
            RandomPeriodicNoise(p=0.1, amplitude=5.0),
            transforms.ToTensor(),
            _imagenet_norm_tensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.10)),
        ]
    )


def get_augmentation_transforms(image_size: int = 256) -> transforms.Compose:
    return build_real_augment_pipeline(image_size)


def get_eval_transforms(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            _imagenet_norm_tensor(),
        ]
    )


def _collect_fake_paths(fake_root: Path) -> List[Path]:
    if not fake_root.exists():
        return []
    return sorted(p for p in fake_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _collect_real_paths(real_root: Path) -> List[Path]:
    if not real_root.exists():
        return []
    return sorted(p for p in real_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def discover_dataset_samples(root: str, max_per_class: Optional[int] = None) -> List[Tuple[str, int]]:
    root_p = Path(root)
    samples: List[Tuple[str, int]] = []

    real_root = root_p / "real"
    for p in _collect_real_paths(real_root):
        samples.append((str(p), LABEL_REAL))

    fake_root = root_p / "fake"
    for p in _collect_fake_paths(fake_root):
        samples.append((str(p), LABEL_FAKE))

    for child in root_p.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if name == "real" or name == "fake":
            continue
        if name.startswith("fake") or name in ("gan", "diffusion", "faceswap", "deepfake", "synthetic", "ai"):
            for p in _collect_fake_paths(child):
                samples.append((str(p), LABEL_FAKE))

    if max_per_class is not None:
        reals = [(p, l) for p, l in samples if l == LABEL_REAL]
        fakes = [(p, l) for p, l in samples if l == LABEL_FAKE]
        samples = reals[:max_per_class] + fakes[:max_per_class]

    fake_count = sum(1 for _, l in samples if l == LABEL_FAKE)
    real_count = sum(1 for _, l in samples if l == LABEL_REAL)
    if real_count == 0 and fake_count == 0:
        print(f"[WARN] No images under {root_p} (expected real/ and fake/ plus optional fake_* dirs)")
    else:
        print(
            f"[DATASET] Indexed {len(samples):,} paths | real={real_count:,} fake={fake_count:,} | synthetic/manipulated → label {LABEL_FAKE}"
        )
    return samples


def balance_class_samples(samples: List[Tuple[str, int]], rng: Optional[random.Random] = None) -> List[Tuple[str, int]]:
    if rng is None:
        rng = random.Random(42)
    reals = [s for s in samples if s[1] == LABEL_REAL]
    fakes = [s for s in samples if s[1] == LABEL_FAKE]
    if not reals or not fakes:
        rng.shuffle(samples)
        return samples
    target = max(len(reals), len(fakes))
    out: List[Tuple[str, int]] = []

    def oversample(lst: List[Tuple[str, int]], n: int) -> List[Tuple[str, int]]:
        if len(lst) >= n:
            return lst[:n]
        extra = []
        while len(extra) < n - len(lst):
            extra.append(rng.choice(lst))
        rng.shuffle(extra)
        return lst + extra

    out.extend(oversample(reals, target))
    out.extend(oversample(fakes, target))
    rng.shuffle(out)
    return out


class DeepfakeDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 256,
        max_per_class: Optional[int] = None,
        balance: bool = True,
        use_label_conditional_aug: bool = False,
        fake_dup_factor: int = 1,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.use_label_conditional_aug = use_label_conditional_aug
        self._aug_real = build_real_augment_pipeline(image_size)
        self._aug_fake = build_fake_augment_pipeline(image_size)
        self.transform = transform

        raw = discover_dataset_samples(str(self.root), max_per_class=max_per_class)
        if balance:
            raw = balance_class_samples(raw, random.Random(42))

        self.samples: List[Tuple[str, int]] = []
        if fake_dup_factor > 1:
            base_fakes = [(p, l) for p, l in raw if l == LABEL_FAKE]
            base_rest = [(p, l) for p, l in raw if l != LABEL_FAKE]
            expanded_fakes = list(base_fakes)
            r = random.Random(43)
            for _ in range(fake_dup_factor - 1):
                expanded_fakes.extend(r.sample(base_fakes, k=len(base_fakes)))
            self.samples = base_rest + expanded_fakes
            r.shuffle(self.samples)
        else:
            self.samples = raw

        fc = sum(1 for _, l in self.samples if l == LABEL_FAKE)
        rc = sum(1 for _, l in self.samples if l == LABEL_REAL)
        print(f"[DATASET] Loaded {len(self.samples):,} samples | real={rc:,} fake={fc:,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size))
        if self.transform is not None:
            img = self.transform(img)
        elif self.use_label_conditional_aug:
            if label == LABEL_FAKE:
                img = self._aug_fake(img)
            else:
                img = self._aug_real(img)
        else:
            img = get_eval_transforms(self.image_size)(img)
        return img, label


class SampleListDataset(Dataset):
    def __init__(self, paths_labels: List[Tuple[str, int]], image_size: int, train: bool = True):
        self.samples = paths_labels
        self.image_size = image_size
        self.train = train
        self._aug_real = build_real_augment_pipeline(image_size)
        self._aug_fake = build_fake_augment_pipeline(image_size)
        self._eval = get_eval_transforms(image_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (self.image_size, self.image_size))
        if not self.train:
            return self._eval(pil), label
        if label == LABEL_FAKE:
            return self._aug_fake(pil), label
        return self._aug_real(pil), label


def get_dataloader(
    root: str,
    image_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 0,
    train: bool = True,
    max_per_class: Optional[int] = None,
) -> DataLoader:
    eval_tf = get_eval_transforms(image_size) if not train else None
    ds = DeepfakeDataset(
        root,
        transform=eval_tf,
        image_size=image_size,
        max_per_class=max_per_class,
        balance=train,
        use_label_conditional_aug=train,
        fake_dup_factor=1,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def stratified_indices(labels: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    labels_np = np.array(labels)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for c in (LABEL_REAL, LABEL_FAKE):
        cls_ix = np.where(labels_np == c)[0].tolist()
        rng.shuffle(cls_ix)
        if len(cls_ix) <= 1:
            train_idx.extend(cls_ix)
            continue
        n_val = int(round(len(cls_ix) * val_ratio))
        n_val = max(1, min(n_val, len(cls_ix) - 1))
        val_idx.extend(cls_ix[:n_val])
        train_idx.extend(cls_ix[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def preprocess_folder(src_dir: str, dst_dir: str, image_size: int = 256, device: str = "cpu"):
    detector = FaceDetector(image_size=image_size, device=device)
    src = Path(src_dir)
    dst = Path(dst_dir)
    images = [p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"[PREPROCESS] {len(images):,} images in {src_dir}")
    saved = 0
    for p in images:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        face = detector.detect_and_crop(img)
        if face is None:
            face = img.resize((image_size, image_size), Image.LANCZOS)
        rel = p.relative_to(src)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        face.save(str(out_path))
        saved += 1
    print(f"[PREPROCESS] Saved {saved:,} face crops to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Face preprocessing pipeline.")
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--output", type=str, default="dataset_processed")
    args = parser.parse_args()
    if args.preprocess:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        preprocess_folder(args.dataset, args.output, args.size, device)
    else:
        loader = get_dataloader(args.dataset, args.size, args.batch)
        batch, labels = next(iter(loader))
        print(f"Batch shape: {batch.shape}, Labels: {labels[:8].tolist()}")


if __name__ == "__main__":
    main()
