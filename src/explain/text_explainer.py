"""Rule-based natural-language explanation of a flagged frame."""
import cv2
import numpy as np
from PIL import Image

# --- tunable rules (hand-set for the synthetic data) ---
BRIGHTNESS_DELTA = 6.0     # mean gray-level change
MOTION_ENERGY = 8.0        # mean absolute pixel difference
CHANGED_PIXEL_DIFF = 60    # per-channel jump that counts as "changed"
CHANGED_FRACTION = 0.015   # share of changed pixels -> localized change
GLOBAL_FRACTION = 0.25     # above this the change covers most of the frame
TEXTURE_RATIO = 4.0        # Laplacian variance ratio -> texture/noise change


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def attention_phrase(cam: np.ndarray, hot: float = 0.5) -> str:
    """Describe where the Grad-CAM map is concentrated."""
    if cam.max() <= 0:
        return "no clear attention region"
    mask = cam >= hot * cam.max()
    if mask.mean() > 0.45:
        return "attention spread across the whole frame"
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean() / cam.shape[0], xs.mean() / cam.shape[1]
    v = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
    h = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    if v == "middle" and h == "center":
        return "attention concentrated at the center of the frame"
    if v == "middle":
        return f"attention concentrated on the {h} side"
    if h == "center":
        return f"attention concentrated in the {v} part of the frame"
    return f"attention concentrated in the {v}-{h} region"


def describe_anomaly(cur: Image.Image, prev: Image.Image, score: float,
                     threshold: float, cam: np.ndarray) -> dict:
    cur_rgb = np.asarray(cur.convert("RGB"))
    prev_rgb = np.asarray(prev.convert("RGB"))
    g1, g0 = _gray(cur_rgb), _gray(prev_rgb)

    d_bright = float(g1.mean() - g0.mean())
    motion = float(np.abs(g1 - g0).mean())
    diff = np.abs(cur_rgb.astype(np.int16) - prev_rgb.astype(np.int16)).max(axis=2)
    changed = float((diff > CHANGED_PIXEL_DIFF).mean())
    lap1 = float(cv2.Laplacian(g1, cv2.CV_32F).var())
    lap0 = float(cv2.Laplacian(g0, cv2.CV_32F).var())
    tex_ratio = (lap1 + 1e-6) / (lap0 + 1e-6)
    is_global = changed >= GLOBAL_FRACTION

    findings = []
    if abs(d_bright) >= BRIGHTNESS_DELTA:
        word = "drop" if d_bright < 0 else "rise"
        findings.append(f"sudden brightness {word} ({d_bright:+.1f})")
    if motion >= MOTION_ENERGY:
        findings.append(f"large overall pixel change (motion energy {motion:.1f})")
    if CHANGED_FRACTION <= changed < GLOBAL_FRACTION:
        findings.append(f"localized region of strong change / possible new object "
                        f"({changed * 100:.1f}% of pixels)")
    elif is_global:
        findings.append(f"change spread across the whole frame "
                        f"({changed * 100:.0f}% of pixels)")
    if tex_ratio >= TEXTURE_RATIO:
        findings.append(f"texture/noise increase (x{tex_ratio:.0f})")
    if not findings:
        findings.append("gradual embedding shift with no single dominant pixel-level cause")

    where = ("Grad-CAM attention is not localized because the change is global"
             if is_global else attention_phrase(cam))

    ratio = score / threshold if threshold and threshold == threshold else float("nan")
    severity = "strong" if ratio >= 3 else "moderate" if ratio >= 1.5 else "mild"
    text = (f"Anomaly score {score:.1f} ({ratio:.1f}x threshold {threshold:.1f}, {severity}). "
            f"Detected: {' + '.join(findings)}; {where}.")

    return {"text": text,
            "metrics": {"brightness_delta": d_bright, "motion_energy": motion,
                        "changed_fraction": changed, "texture_ratio": tex_ratio,
                        "score_over_threshold": ratio}}
