# backend/report_generator.py
# Pragmatic: use UNet to gate no-tumor; otherwise classify. Auto-prep + clear console logs.

from __future__ import annotations
import os, math, uuid, datetime, tempfile
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import cv2
from skimage import io as skio, measure, morphology

import tensorflow as tf
import keras

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from PIL import Image

# ---------- Paths ----------
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = (HERE.parent / "models").resolve()
MODEL_DIR = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)).resolve()

MODEL_DIRS_TO_SEARCH: List[Path] = [
    MODEL_DIR,
    (HERE / "models").resolve(),
]

UNET_FILENAMES  = ["unet_final.h5"]
VGG_FILENAMES   = ["brain_tumor_model.h5", "vgg16_fold5.h5"]
PLANE_FILENAMES = ["plane_classifier.h5"]
CLASS_FILE_NAME = "class_names.txt"

CLS_INPUT_SIZE = (224, 224)
PLANE_INPUT_SIZE = (224, 224)

# Segmentation thresholds (kept modest)
SEG_OUTPUT_CHANNEL_TUMOR: Optional[int] = None
MIN_TUMOR_AREA_ABS   = 200           # pixels
MIN_TUMOR_AREA_FRAC  = 0.0005        # 0.05% of image

# ---- locate files ----
def _find_first(names: List[str]) -> Optional[Path]:
    for d in MODEL_DIRS_TO_SEARCH:
        for n in names:
            p = d / n
            if p.exists():
                return p
    return None

UNET_PATH  = _find_first(UNET_FILENAMES)
VGG_PATH   = _find_first(VGG_FILENAMES)
PLANE_PATH = _find_first(PLANE_FILENAMES)
CLASS_FILE = None
for d in MODEL_DIRS_TO_SEARCH:
    p = d / CLASS_FILE_NAME
    if p.exists():
        CLASS_FILE = p
        break

if UNET_PATH is None:  raise FileNotFoundError("UNet model not found (../models or ./models).")
if VGG_PATH is None:   raise FileNotFoundError("Classifier model not found (../models or ./models).")
# plane may be None

print("🔎 Models:")
print("   UNet :", UNET_PATH)
print("   VGG  :", VGG_PATH)
print("   Plane:", PLANE_PATH if PLANE_PATH else "None")
print("   class_names.txt:", CLASS_FILE if CLASS_FILE else "None (will use defaults)")

# ------------------ cached models -------------------
_unet = None
_cls  = None
_plane = None
_CLASS_NAMES: List[str] = []

def _load_h5(path: Path):
    try:
        return keras.models.load_model(str(path), compile=False, safe_mode=False)
    except TypeError:
        return keras.models.load_model(str(path), compile=False)

def _load_class_names(n_out: int) -> List[str]:
    # 1) explicit file
    if CLASS_FILE and CLASS_FILE.exists():
        lines = [ln.strip() for ln in CLASS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) < n_out:  # pad or trim for safety
            lines += [f"class_{i}" for i in range(len(lines), n_out)]
        elif len(lines) > n_out:
            lines = lines[:n_out]
        return lines
    # 2) sensible default (never show class_0 unless >4 outputs)
    base = ["glioma", "meningioma", "pituitary", "no_tumor"]
    if n_out <= len(base):
        return base[:n_out]
    return [f"class_{i}" for i in range(n_out)]

def load_models():
    global _unet, _cls, _plane, _CLASS_NAMES
    if _unet is None:
        _unet = _load_h5(UNET_PATH)
    if _cls is None:
        _cls = _load_h5(VGG_PATH)
        n_out = int(getattr(_cls, "output_shape", [-1, -1])[-1]) if hasattr(_cls, "output_shape") else 3
        if not isinstance(n_out, int) or n_out <= 0: n_out = 3
        _CLASS_NAMES = _load_class_names(n_out)
        print("✅ Class names:", _CLASS_NAMES)
    if _plane is None and PLANE_PATH is not None:
        _plane = _load_h5(PLANE_PATH)
    return _unet, _cls, _plane

# --------------------- helpers ----------------------
def load_image_gray_from_path(path: Path) -> np.ndarray:
    img = skio.imread(str(path), as_gray=True).astype(np.float32)
    if img.max() > 1.0: img /= 255.0
    return img

def overlay_mask(img_gray: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    img_u8 = (img_gray * 255).clip(0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()
    red = np.zeros_like(rgb); red[..., 2] = 255
    m3 = np.repeat(mask[..., None], 3, axis=2).astype(bool)
    overlay[m3] = (alpha * red[m3] + (1 - alpha) * rgb[m3]).astype(np.uint8)
    return overlay

def resize_keep_aspect(img: np.ndarray, target_hw: Tuple[int, int]):
    th, tw = target_hw
    h, w = img.shape[:2]
    s = min(th / h, tw / w)
    nh, nw = int(h * s), int(w * s)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw), dtype=resized.dtype)
    top, left = (th - nh) // 2, (tw - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, (s, top, left, (h, w))

def undo_resize_keep_aspect(mask_padded: np.ndarray, meta):
    s, top, left, (h, w) = meta
    th, tw = mask_padded.shape[:2]
    nh, nw = int(h * s), int(w * s)
    crop = mask_padded[top:top + nh, left:left + nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)

def infer_mask_from_unet(model, img_gray: np.ndarray, seg_output_channel=None, thresh=0.5) -> np.ndarray:
    _, H, W, C = model.input_shape
    xpad, meta = resize_keep_aspect(img_gray, (H, W))
    x = xpad[None, ..., None] if C == 1 else np.repeat(xpad[..., None], 3, axis=-1)[None, ...]
    prob = model.predict(x, verbose=0)[0]
    if prob.ndim == 3:
        if prob.shape[-1] == 1: prob = prob[..., 0]
        else:
            ch = 0 if seg_output_channel is None else seg_output_channel
            prob = prob[..., ch]
    prob = np.clip(prob, 0, 1).astype(np.float32)
    prob_orig = undo_resize_keep_aspect(prob, meta)
    mask = (prob_orig >= thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask

# ---------- classifier with auto-prep + full logging ----------
def _prep_rgb_01(img_bgr, size):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)

def _prep_bgr_01(img_bgr, size):
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)

def _prep_vgg16(img_bgr, size):
    from tensorflow.keras.applications.vgg16 import preprocess_input
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    x = np.expand_dims(img.astype(np.float32), axis=0)
    return preprocess_input(x)

def classify_with_vgg_autoprep(model, img_path: Path, class_names: List[str], input_size=(224,224)):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Cannot read: {img_path}")

    variants = {
        "rgb_01": _prep_rgb_01(img_bgr, input_size),
        "bgr_01": _prep_bgr_01(img_bgr, input_size),
        "vgg16":  _prep_vgg16(img_bgr, input_size),
    }
    best = None
    best_name = None
    best_top = -1.0

    print("🧪 Classifier probabilities (per preprocessing):")
    for name, x in variants.items():
        probs = model.predict(x, verbose=0)[0].astype(float)
        labels = list(class_names)
        if len(labels) != len(probs):
            labels = labels[:len(probs)] + [f"class_{i}" for i in range(len(labels), len(probs))]
        top = float(np.max(probs))
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        # pretty print probs
        pretty = ", ".join([f"{labels[i]}={probs[i]:.3f}" for i in range(len(probs))])
        print(f"  • {name:6s} → {pred_label:11s} | {pretty}")
        if top > best_top:
            best_top = top
            best_name = name
            best = {"predicted_label": pred_label,
                    "probabilities": {labels[i]: float(probs[i]) for i in range(len(probs))},
                    "preproc_used": name}
    print(f"✅ Picked preprocessing: {best_name} (top={best_top:.3f})")
    return best

def classify_plane(model, img_path: Path, input_size=(224,224)):
    if model is None:
        return {"predicted_label": "Not assessed", "probabilities": {}}
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Cannot read: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    probs = model.predict(x, verbose=0)[0].astype(float)

    labels = ["axial", "sagittal", "coronal"] if len(probs) == 3 else [f"plane_{i}" for i in range(len(probs))]
    pred_idx = int(np.argmax(probs))
    pred = labels[pred_idx]
    pretty = ", ".join([f"{labels[i]}={probs[i]:.3f}" for i in range(len(probs))])
    print(f"📐 Plane → {pred} | {pretty}")
    return {"predicted_label": pred, "probabilities": {labels[i]: float(probs[i]) for i in range(len(probs))}}

def safe_number(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))): return None
    return float(x)

def compute_mask_metrics(img_gray: np.ndarray, mask: np.ndarray):
    h, w = img_gray.shape[:2]
    area_px = int(mask.sum())
    tumor_present = area_px > 0
    centroid = None
    laterality = "Not assessed"
    if tumor_present:
        props = measure.regionprops(mask.astype(np.uint8))[0]
        cyf, cxf = props.centroid
        centroid = [safe_number(float(cxf)), safe_number(float(cyf))]
        laterality = "Right" if cxf > (w / 2.0) else "Left"
    return {
        "image_shape": {"height": h, "width": w},
        "tumor_present": bool(tumor_present),
        "area_px": int(area_px),
        "centroid_xy": centroid,
        "laterality": laterality,
    }

def _build_facts(img_gray, mask, cls_result, plane_result, patient):
    f = compute_mask_metrics(img_gray, mask)
    H, W = f["image_shape"]["height"], f["image_shape"]["width"]
    min_area = max(MIN_TUMOR_AREA_ABS, int(MIN_TUMOR_AREA_FRAC * H * W))
    f["tumor_present"] = f["area_px"] >= min_area

    # ✅ RULE: if UNet says "no tumor", force tumor_type=no_tumor and plane "Not assessed"
    classification = dict(cls_result) if f["tumor_present"] else {"predicted_label": "no_tumor", "probabilities": {}}
    if not f["tumor_present"]:
        plane = {"predicted_label": "Not assessed", "probabilities": {}}
    else:
        plane = dict(plane_result)

    # If your classifier is 3-class (no 'no_tumor'), keep "no_tumor" as string; it's fine in PDF.
    if "no_tumor" not in _CLASS_NAMES and not f["tumor_present"]:
        classification["predicted_label"] = "no_tumor"

    # Patient fields
    def _normalize_sex(s):
        if not s: return "U"
        s = str(s).strip().lower()
        if s.startswith("f"): return "F"
        if s.startswith("m"): return "M"
        return s[:1].upper()

    case_id = str(uuid.uuid4())[:8]
    facts = {
        "case_id": case_id,
        "patient": {
            "patient_id": f"P-{case_id}",
            "name": (patient or {}).get("name"),
            "age": (patient or {}).get("age"),
            "sex": _normalize_sex((patient or {}).get("sex")),
        },
        "study": {"modality": "MRI", "sequences": [], "study_date": str(datetime.date.today())},
        "classification": classification,
        "plane": plane,
        "findings_extracted": f,
        "quality_flags": {"mask_empty": bool(mask.sum() == 0), "image_nan": bool(np.isnan(img_gray).any())},
    }
    return facts

def render_pdf_bytes(facts: dict, overlay_img_path: Path) -> bytes:
    buf = tempfile.SpooledTemporaryFile(max_size=5_000_000)
    c = canvas.Canvas(buf, pagesize=A4); W, H = A4; m = 15 * mm; y = H - m
    c.setFont("Helvetica-Bold", 16); c.drawString(m, y, "MRI Brain – AI-Assisted Pre-Report"); y -= 10 * mm
    c.setFont("Helvetica", 10)

    pat = facts.get('patient', {})
    c.drawString(m, y, f"Patient: {pat.get('name') or '—'}    ID: {pat.get('patient_id') or '—'}"); y -= 6*mm
    age = pat.get('age') if pat.get('age') is not None else "—"
    c.drawString(m, y, f"Age: {age}    Sex: {pat.get('sex') or '—'}"); y -= 6*mm

    study = facts.get('study', {})
    c.drawString(m, y, f"Modality: {study.get('modality','MRI')}    Study Date: {study.get('study_date','NA')}"); y -= 10*mm

    f = facts["findings_extracted"]
    c.setFont("Helvetica-Bold", 12); c.drawString(m, y, "Findings (Quantitative)"); y -= 6*mm
    c.setFont("Helvetica", 10)

    tumor_type = facts["classification"]["predicted_label"]
    plane_axis = facts["plane"]["predicted_label"]

    q = [
        f"Tumor present: {f['tumor_present']}",
        f"Tumor type: {tumor_type}",
        f"Plane axis: {plane_axis}",
        f"Laterality: {f['laterality']}",
        f"Area (px): {f['area_px']}",
        f"Centroid (x,y): {f['centroid_xy'] if f['centroid_xy'] else 'Not assessed'}",
    ]
    for line in q:
        c.drawString(m, y, line); y -= 5*mm

    # Impression
    if f["tumor_present"]:
        imp = [f"Imaging suggests {tumor_type} on {plane_axis} plane; laterality: {f['laterality']}."]
        sev = "moderate"
    else:
        imp = ["No tumor signal detected by segmentation."]
        sev = "none"
    y -= 4*mm; c.setFont("Helvetica-Bold", 12); c.drawString(m, y, "Impression"); y -= 6*mm; c.setFont("Helvetica", 10)
    for line in imp: c.drawString(m, y, f"- {line}"); y -= 5*mm
    y -= 2*mm; c.drawString(m, y, f"Severity: {sev}"); y -= 5*mm

    # Image overlay
    y -= 4 * mm; c.setFont("Helvetica-Bold", 12); c.drawString(m, y, "Image Overlay"); y -= 6 * mm
    reader = ImageReader(str(overlay_img_path)); img = skio.imread(str(overlay_img_path)); ih, iw = img.shape[:2]
    draw_w = 120 * mm; draw_h = draw_w * (ih / iw)
    if y - draw_h < m:
        c.showPage(); y = H - m
    c.drawImage(reader, m, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True, mask='auto'); y -= draw_h + 6*mm

    c.setFont("Helvetica", 9); c.drawString(m, m, "Auto-generated draft. Requires radiologist review.")
    c.showPage(); c.save()
    buf.seek(0); data = buf.read(); buf.close()
    return data

def build_report_and_facts(image_bytes: bytes, patient: dict | None = None):
    import io as pyio
    seg_model, cls_model, plane_model = load_models()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)

    try:
        # Segment
        img_gray = load_image_gray_from_path(tmp_path)
        mask = infer_mask_from_unet(seg_model, img_gray,
                                    seg_output_channel=SEG_OUTPUT_CHANNEL_TUMOR, thresh=0.5)
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=25).astype(np.uint8)

        # If UNet says no tumor → force no_tumor and skip classifier confusion
        H, W = img_gray.shape[:2]
        min_area = max(MIN_TUMOR_AREA_ABS, int(MIN_TUMOR_AREA_FRAC * H * W))
        tumor_present = int(mask.sum()) >= min_area

        if tumor_present:
            cls_result   = classify_with_vgg_autoprep(cls_model, tmp_path, _CLASS_NAMES, input_size=CLS_INPUT_SIZE)
            plane_result = classify_plane(plane_model, tmp_path, input_size=PLANE_INPUT_SIZE)
        else:
            cls_result   = {"predicted_label": "no_tumor", "probabilities": {}, "preproc_used": None}
            plane_result = {"predicted_label": "Not assessed", "probabilities": {}}

        # Facts + overlay + PDF
        facts = _build_facts(img_gray, mask, cls_result, plane_result, patient)
        facts["source_filename"] = tmp_path.name

        overlay = overlay_mask(img_gray, mask, alpha=0.35)
        ov_buf = pyio.BytesIO()
        Image.fromarray(overlay).save(ov_buf, format="PNG")
        overlay_png_bytes = ov_buf.getvalue()

        with tempfile.NamedTemporaryFile(suffix="_overlay.png", delete=False) as ovf:
            ov_path = Path(ovf.name)
            skio.imsave(str(ov_path), overlay)
        try:
            pdf_bytes = render_pdf_bytes(facts, ov_path)
        finally:
            try: ov_path.unlink(missing_ok=True)
            except Exception: pass

        return pdf_bytes, facts, overlay_png_bytes
    finally:
        try: tmp_path.unlink(missing_ok=True)
        except Exception: pass
