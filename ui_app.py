import os
import sys
import math
import platform
import ctypes
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from pathlib import Path

import numpy as np

if platform.system() == "Windows":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detect_image import DeepfakeDetector

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from retinaface import RetinaFace as RetinaFaceDetector
    HAS_RETINAFACE = True
except ImportError:
    HAS_RETINAFACE = False

COLORS = {
    "bg": "#0f0f1a",
    "surface": "#1a1a2e",
    "surface2": "#16213e",
    "accent": "#0f3460",
    "primary": "#e94560",
    "success": "#00c853",
    "warning": "#ffd600",
    "info": "#2196f3",
    "text": "#eaeaea",
    "text_dim": "#8892b0",
    "border": "#233554",
}

FONT_TITLE   = ("Segoe UI", 24, "bold")
FONT_HEADING = ("Segoe UI", 14, "bold")
FONT_BODY    = ("Segoe UI", 12)
FONT_RESULT  = ("Segoe UI", 20, "bold")
FONT_SMALL   = ("Segoe UI", 10)
FONT_SCORE   = ("Consolas", 11)

# ── Landmark mesh colour palettes per verdict ──────────────────────────────
PALETTE = {
    "REAL FACE":    {"dot": (0, 230, 80),   "line": (0, 180, 60),   "box": (0, 210, 70),   "grid": (0, 150, 50)},
    "DEEPFAKE":     {"dot": (240, 50, 80),  "line": (200, 30, 60),  "box": (230, 40, 70),  "grid": (160, 20, 50)},
    "AI-GENERATED": {"dot": (255, 210, 0),  "line": (200, 160, 0),  "box": (255, 200, 0),  "grid": (160, 120, 0)},
    "default":      {"dot": (100, 180, 255),"line": (60, 130, 220), "box": (80, 160, 240), "grid": (40, 100, 180)},
}

# RetinaFace 5-point landmark connections (pairs of landmark names)
# left_eye, right_eye, nose, mouth_left, mouth_right
_LM_PAIRS = [
    ("left_eye",    "right_eye"),
    ("left_eye",    "nose"),
    ("right_eye",   "nose"),
    ("nose",        "mouth_left"),
    ("nose",        "mouth_right"),
    ("mouth_left",  "mouth_right"),
    ("left_eye",    "mouth_left"),
    ("right_eye",   "mouth_right"),
]


def draw_landmark_overlay(
    pil_img: Image.Image,
    verdict: str = "default",
    out_size: tuple = (380, 380),
) -> Image.Image:
    """
    Draw a facial landmark mesh overlay on *pil_img*.

    Uses RetinaFace (if available) to get the 5 standard facial landmarks
    (left eye, right eye, nose, mouth-left, mouth-right), then draws:
      • A glowing face bounding box
      • Dot markers at each landmark
      • Connecting lines forming a structural mesh
      • A subtle scan-grid overlay
      • A verdict badge in the corner

    Falls back gracefully to a Haar-cascade bounding box + 3-point approximation
    if RetinaFace is not available, or a simple grid overlay if no face found.

    Returns a PIL Image of size *out_size*.
    """
    palette = PALETTE.get(verdict, PALETTE["default"])
    dot_c  = palette["dot"]
    line_c = palette["line"]
    box_c  = palette["box"]
    grid_c = palette["grid"]

    img_np = np.array(pil_img.convert("RGB"))
    h_orig, w_orig = img_np.shape[:2]

    # ── 1. Detect face bounding box + landmarks ────────────────────────────
    face_box  = None   # (x1, y1, x2, y2)
    landmarks = {}     # name → (x, y)

    if HAS_RETINAFACE and HAS_CV2:
        try:
            faces = RetinaFaceDetector.detect_faces(img_np)
            if faces:
                best, best_area = None, 0
                for _, fd in faces.items():
                    area_coords = fd.get("facial_area", [0, 0, 0, 0])
                    w = area_coords[2] - area_coords[0]
                    h = area_coords[3] - area_coords[1]
                    if w * h > best_area:
                        best_area = w * h
                        best = fd
                if best:
                    x1, y1, x2, y2 = best["facial_area"]
                    face_box = (int(x1), int(y1), int(x2), int(y2))
                    for name, pt in best.get("landmarks", {}).items():
                        landmarks[name] = (int(pt[0]), int(pt[1]))
        except Exception:
            pass

    if face_box is None and HAS_CV2:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            dets = cascade.detectMultiScale(gray, 1.3, 5)
            if len(dets) > 0:
                x, y, w, h = max(dets, key=lambda r: r[2] * r[3])
                face_box = (x, y, x + w, y + h)
                # Approximate 5 landmarks from the bounding box
                cx, cy = x + w // 2, y + h // 2
                ew = w // 4
                landmarks = {
                    "left_eye":    (x + w // 3, y + h // 3),
                    "right_eye":   (x + 2 * w // 3, y + h // 3),
                    "nose":        (cx, cy),
                    "mouth_left":  (x + w // 3, y + 2 * h // 3),
                    "mouth_right": (x + 2 * w // 3, y + 2 * h // 3),
                }
        except Exception:
            pass

    # ── 2. Draw on a copy ─────────────────────────────────────────────────
    if HAS_CV2:
        canvas = img_np.copy()

        # Subtle scan-grid
        grid_step = max(20, min(w_orig, h_orig) // 14)
        for gx in range(0, w_orig, grid_step):
            cv2.line(canvas, (gx, 0), (gx, h_orig), (*grid_c, 255), 1)
        for gy in range(0, h_orig, grid_step):
            cv2.line(canvas, (0, gy), (w_orig, gy), (*grid_c, 255), 1)
        # Blend grid subtly (80% original, 20% grid)
        canvas = cv2.addWeighted(img_np, 0.82, canvas, 0.18, 0)

        # Face bounding box — double-border glowing effect
        if face_box:
            x1, y1, x2, y2 = face_box
            # Outer glow (thicker, darker)
            cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                          tuple(max(0, c - 80) for c in box_c), 2)
            # Main box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_c, 2)
            # Corner accents (L-shaped corners for a sci-fi look)
            corner_len = max(10, (x2 - x1) // 6)
            for (cx, cy, sx, sy) in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(canvas, (cx, cy), (cx + sx * corner_len, cy), box_c, 3)
                cv2.line(canvas, (cx, cy), (cx, cy + sy * corner_len), box_c, 3)

        # Landmark connecting lines
        lm_pts = {k: v for k, v in landmarks.items()}
        for (a, b) in _LM_PAIRS:
            if a in lm_pts and b in lm_pts:
                cv2.line(canvas, lm_pts[a], lm_pts[b], line_c, 1)

        # Landmark dots — each has an outer ring + filled centre
        for name, (px, py) in lm_pts.items():
            cv2.circle(canvas, (px, py), 7, line_c, 1)          # outer ring
            cv2.circle(canvas, (px, py), 4, dot_c, -1)           # filled dot
            cv2.circle(canvas, (px, py), 2, (255, 255, 255), -1) # white centre

        # ── Landmark labels (small) ──
        label_map = {
            "left_eye": "L.Eye", "right_eye": "R.Eye",
            "nose": "Nose", "mouth_left": "L.Mouth", "mouth_right": "R.Mouth",
        }
        for name, (px, py) in lm_pts.items():
            short = label_map.get(name, name)
            cv2.putText(canvas, short, (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, dot_c, 1, cv2.LINE_AA)

        # Verdict badge — bottom-left corner
        badge_text = verdict if verdict != "default" else "SCANNING"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bx, by = 8, h_orig - 12
        cv2.rectangle(canvas, (bx - 4, by - th - 6), (bx + tw + 4, by + 4),
                      (10, 10, 30), -1)
        cv2.rectangle(canvas, (bx - 4, by - th - 6), (bx + tw + 4, by + 4),
                      box_c, 1)
        cv2.putText(canvas, badge_text, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, dot_c, 1, cv2.LINE_AA)

        result_pil = Image.fromarray(canvas)
    else:
        # cv2 not available — draw with PIL only
        result_pil = pil_img.copy().convert("RGB")
        draw = ImageDraw.Draw(result_pil)
        if face_box:
            draw.rectangle(face_box, outline=dot_c, width=3)
        for (a, b) in _LM_PAIRS:
            if a in landmarks and b in landmarks:
                draw.line([landmarks[a], landmarks[b]], fill=line_c, width=1)
        for _, (px, py) in landmarks.items():
            r = 5
            draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=dot_c)

    result_pil = result_pil.resize(out_size, Image.LANCZOS)
    return result_pil


class DeepfakeDetectorApp:
    PREVIEW_SIZE = (380, 380)

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DeepShield — AI-Powered Deepfake & AI-Image Detector")
        self.root.configure(bg=COLORS["bg"])
        window_width  = 1060
        window_height = 860
        screen_width  = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        cx = int(screen_width  / 2 - window_width  / 2)
        cy = int(screen_height / 2 - window_height / 2)
        self.root.geometry(f"{window_width}x{window_height}+{cx}+{cy}")
        self.root.minsize(900, 780)
        self.root.resizable(True, True)
        self.image_path   = None
        self.detector     = None
        self._photo_orig  = None
        self._photo_mesh  = None
        self._build_ui()

    def _build_ui(self):
        # ── Title bar ──────────────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg=COLORS["surface"], pady=14)
        title_frame.pack(fill="x")
        tk.Label(
            title_frame,
            text="🛡️  DeepShield — Deepfake + AI-Image Detector",
            font=FONT_TITLE,
            bg=COLORS["surface"],
            fg=COLORS["text"],
        ).pack()
        tk.Label(
            title_frame,
            text="XADE EfficientNet-B4  |  Swin AI-Image Detector  |  ViT Face-Swap  |  Facial Landmark Mesh",
            font=FONT_SMALL,
            bg=COLORS["surface"],
            fg=COLORS["text_dim"],
        ).pack()

        # ── Dual preview panes ─────────────────────────────────────────────
        preview_container = tk.Frame(self.root, bg=COLORS["bg"], pady=12)
        preview_container.pack(expand=True, fill="both")

        left_frame = tk.Frame(preview_container, bg=COLORS["bg"])
        left_frame.pack(side="left", expand=True, fill="both", padx=10)
        tk.Label(
            left_frame,
            text="📸 Original Image",
            font=FONT_HEADING,
            bg=COLORS["bg"],
            fg=COLORS["text"],
        ).pack(pady=(0, 4))
        self.canvas = tk.Canvas(
            left_frame,
            width=self.PREVIEW_SIZE[0],
            height=self.PREVIEW_SIZE[1],
            bg=COLORS["surface"],
            highlightthickness=2,
            highlightbackground=COLORS["border"],
        )
        self.canvas.pack(expand=True)

        right_frame = tk.Frame(preview_container, bg=COLORS["bg"])
        right_frame.pack(side="right", expand=True, fill="both", padx=10)
        tk.Label(
            right_frame,
            text="🔬 Facial Analysis — Landmark Mesh",
            font=FONT_HEADING,
            bg=COLORS["bg"],
            fg=COLORS["text"],
        ).pack(pady=(0, 4))
        self.face_canvas = tk.Canvas(
            right_frame,
            width=self.PREVIEW_SIZE[0],
            height=self.PREVIEW_SIZE[1],
            bg=COLORS["surface"],
            highlightthickness=2,
            highlightbackground=COLORS["border"],
        )
        self.face_canvas.pack(expand=True)

        self._draw_placeholder(self.canvas,      "No image loaded")
        self._draw_placeholder(self.face_canvas, "Run detection to view mesh")

        # ── Buttons ────────────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg=COLORS["bg"], pady=10)
        btn_frame.pack()
        self.upload_btn = tk.Button(
            btn_frame,
            text="📁  Upload Image",
            font=FONT_BODY,
            bg=COLORS["accent"],
            fg=COLORS["text"],
            activebackground=COLORS["primary"],
            activeforeground="white",
            relief="flat",
            padx=24, pady=10,
            cursor="hand2",
            command=self._upload_image,
        )
        self.upload_btn.pack(side="left", padx=12)
        self.detect_btn = tk.Button(
            btn_frame,
            text="🔎  Detect Deepfake",
            font=FONT_BODY,
            bg=COLORS["primary"],
            fg="white",
            activebackground="#c0392b",
            activeforeground="white",
            relief="flat",
            padx=24, pady=10,
            cursor="hand2",
            state="disabled",
            command=self._detect,
        )
        self.detect_btn.pack(side="left", padx=12)

        # ── Result panel ───────────────────────────────────────────────────
        result_frame = tk.Frame(self.root, bg=COLORS["bg"], pady=10)
        result_frame.pack(fill="x", padx=40)
        self.prediction_label = tk.Label(
            result_frame, text="", font=FONT_RESULT, bg=COLORS["bg"], fg=COLORS["text"]
        )
        self.prediction_label.pack(pady=(0, 6))
        self.bar_canvas = tk.Canvas(
            result_frame,
            width=600, height=34,
            bg=COLORS["surface"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.bar_canvas.pack(pady=(0, 4))
        self.confidence_label = tk.Label(
            result_frame,
            text="Upload an image to begin",
            font=FONT_BODY,
            bg=COLORS["bg"],
            fg=COLORS["text_dim"],
        )
        self.confidence_label.pack()
        self.scores_label = tk.Label(
            result_frame,
            text="",
            font=FONT_SCORE,
            bg=COLORS["bg"],
            fg=COLORS["text_dim"],
            justify="left",
        )
        self.scores_label.pack(pady=(6, 0))

        # ── Footer ─────────────────────────────────────────────────────────
        footer = tk.Label(
            self.root,
            text="JPG • PNG • JPEG   |   XADE EF-B4 + Swin AI-Image-Detector + ViT Face-Swap + FFT + ELA + Noise + Patch   |   Landmark Mesh Overlay",
            font=FONT_SMALL,
            bg=COLORS["bg"],
            fg=COLORS["text_dim"],
        )
        footer.pack(side="bottom", pady=8)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _draw_placeholder(self, canvas, text):
        w = int(canvas.cget("width"))
        h = int(canvas.cget("height"))
        cx, cy = w // 2, h // 2
        canvas.delete("all")
        canvas.create_text(cx, cy, text=text, fill=COLORS["text_dim"], font=FONT_HEADING)

    def _show_preview(self, canvas, img: Image.Image, attr_name: str):
        img_copy = img.copy()
        img_copy.thumbnail(self.PREVIEW_SIZE, Image.LANCZOS)
        photo = ImageTk.PhotoImage(img_copy)
        canvas.delete("all")
        cx = int(canvas.cget("width"))  // 2
        cy = int(canvas.cget("height")) // 2
        canvas.create_image(cx, cy, image=photo, anchor="center")
        if attr_name == "orig":
            self._photo_orig = photo
        else:
            self._photo_mesh = photo

    def _draw_confidence_bar(self, confidence: float, is_real: bool):
        self.bar_canvas.delete("all")
        w, h = 600, 34
        bar_w = int(w * confidence / 100)
        color = COLORS["success"] if is_real else COLORS["primary"]
        self.bar_canvas.create_rectangle(0, 0, bar_w, h, fill=color, outline="")
        self.bar_canvas.create_text(
            w // 2, h // 2, text=f"{confidence:.1f}%", fill="white", font=FONT_BODY
        )

    # ── Actions ────────────────────────────────────────────────────────────

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if not path:
            return
        self.image_path = path
        try:
            img = Image.open(path).convert("RGB")
            self._show_preview(self.canvas, img, "orig")
            self._draw_placeholder(self.face_canvas, "Run detection to view mesh")
            self.detect_btn.configure(state="normal")
            self.confidence_label.configure(
                text=f"Loaded: {os.path.basename(path)}", fg=COLORS["text"]
            )
            self.prediction_label.configure(text="")
            self.scores_label.configure(text="")
            self.bar_canvas.delete("all")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def _detect(self):
        if not self.image_path:
            return

        # Load models on first use
        if self.detector is None:
            self.confidence_label.configure(text="Loading models…", fg=COLORS["warning"])
            self.root.update()
            try:
                self.detector = DeepfakeDetector()
            except Exception as e:
                messagebox.showerror("Model Error", f"Could not load model:\n{e}")
                return

        self.confidence_label.configure(
            text="Analysing… (4-model ensemble + landmark mesh)", fg=COLORS["warning"]
        )
        self.root.update()

        try:
            img    = Image.open(self.image_path).convert("RGB")
            result = self.detector.predict(image_path=self.image_path)
        except Exception as e:
            messagebox.showerror("Detection Error", f"Detection failed:\n{e}")
            return

        # ── Draw landmark mesh overlay ─────────────────────────────────────
        try:
            mesh_img = draw_landmark_overlay(img, verdict=result["label"],
                                             out_size=self.PREVIEW_SIZE)
            self._show_preview(self.face_canvas, mesh_img, "mesh")
        except Exception as e:
            print(f"[WARN] Landmark overlay failed: {e}")
            self._draw_placeholder(self.face_canvas, "Mesh unavailable")

        # ── Result display ─────────────────────────────────────────────────
        is_real   = result["label"] == "REAL FACE"
        is_ai_gen = result["label"] == "AI-GENERATED"

        if is_real:
            color = COLORS["success"]
            emoji = "✅"
        elif is_ai_gen:
            color = COLORS["warning"]
            emoji = "🤖"
        else:
            color = COLORS["primary"]
            emoji = "⚠️"

        self.prediction_label.configure(text=f"{emoji}  {result['label']}", fg=color)
        self._draw_confidence_bar(result["confidence"], is_real)
        self.confidence_label.configure(
            text=f"Ensemble Score: {result['raw_score']}", fg=COLORS["text_dim"]
        )

        scores = result.get("scores", {})
        scores_text = (
            f"XADE+ViT Ensemble (EF-B4):   {scores.get('classifier',      'N/A')}\n"
            f"Swin AI-Image Detector:       {scores.get('ai_image_detector','N/A')}\n"
            f"FFT Frequency Analysis:       {scores.get('fft',             'N/A')}\n"
            f"Noise Inconsistency:          {scores.get('noise',           'N/A')}\n"
            f"Error Level Analysis (ELA):   {scores.get('ela',             'N/A')}\n"
            f"Patch Analysis:               {scores.get('patch',           'N/A')}"
        )
        self.scores_label.configure(text=scores_text)


def main():
    root = tk.Tk()
    DeepfakeDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
