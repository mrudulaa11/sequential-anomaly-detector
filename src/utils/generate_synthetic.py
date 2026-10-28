"""Generate synthetic image sequences with a known injected anomaly."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SIZE = 256
KINDS = ["combo", "dark_patch", "new_object", "brightness_drop", "noise"]

def make_background() -> np.ndarray:
    g = np.linspace(90, 150, SIZE, dtype=np.float32)
    bg = np.stack([np.tile(g, (SIZE, 1)) * s for s in (1.0, 0.95, 0.9)], axis=-1)
    return bg.astype(np.uint8)

def render_frame(t, n_frames, bg, rng) -> np.ndarray:
    """Normal frame: moving circle + mild lighting variation + sensor noise."""
    frame = bg.copy()
    cx = int(30 + (SIZE - 60) * t / (n_frames - 1))
    cy = int(SIZE / 2 + 40 * np.sin(2 * np.pi * t / n_frames))
    cv2.circle(frame, (cx,cy), 18, (200, 120, 40), -1) #BGR: blueish
    out = frame.astype(np.float32)
    out += 6 * np.sin(4 * np.pi * t / n_frames) #lighting drift
    out += rng.normal(0, 2, out.shape) #sensor noise
    return np.clip(out, 0, 255).astype(np.uint8)

def inject_anomaly(frame, kind, rng):
    """Return (anomalous frame, [x0, y0, x1, y1] box where it happened)."""
    h, w = frame.shape[:2]
    x0 = int(rng.integers(10, w - 100))
    y0 = int(rng.integers(10, h - 100))
    box = [x0, y0, x0 + 90, y0 + 90]
    
    out = frame.astype(np.float32)
    if kind in ("combo", "dark_patch"):
        out[y0:y0 + 90, x0:x0 + 90] *= 0.2
    if kind == "brightness_drop":
        out *= 0.5
        box = [0, 0, w, h]
    if kind in ("combo", "noise"):
        out += rng.normal(0, 15 if kind == "combo" else 40, out.shape)
        if kind == "noise":
            box = [0, 0, w, h]
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    if kind in ("combo", "new_object"):
        cv2.rectangle(out, (x0 + 25, y0 + 25), (x0 + 65, y0 + 65), (0, 0, 255), -1) #red
    return out, box

def generate_sequence(out_dir, n_frames=30, kind="combo", anomaly_frame=None, seed=0):
    rng = np.random.default_rng(seed)
    if anomaly_frame is None:
        anomaly_frame = int(rng.integers(15, n_frames - 2))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bg = make_background()
    
    box = None
    for t in range(n_frames):
        frame = render_frame(t, n_frames, bg, rng)
        if t == anomaly_frame:
            frame, box = inject_anomaly(frame, kind, rng)
        cv2.imwrite(str(out_dir / f"frame_{t:03d}.png"), frame)
        
    labels = {"n_frames": n_frames, "anomaly_frames":[anomaly_frame], "kind": kind,
              "box": box, "seed": seed}
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--num-seqs", type=int, default=1)
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--kind", default="combo", choices=KINDS + ["mixed"])
    ap.add_argument("--seed", type= int, default=0)
    args = ap.parse_args()
    
    for i in range(args.num_seqs):
        kind = KINDS[i % len(KINDS)] if args.kind == "mixed" else args.kind
        d = Path(args.out) / f"seq_{i:03d}"
        lab = generate_sequence(d, args.n_frames, kind, seed=args.seed + i)
        print(f"{d}: kind={lab['kind']} anomaly_frame={lab['anomaly_frames'][0]}")
        
if __name__ == "__main__":
    main()