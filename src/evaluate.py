"""Evaluate detection and localization on many synthetic sequences."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.explain.gradcam_explainer import GradCAMExplainer
from src.models.anomaly_scorer import TemporalAnomalyScorer
from src.models.feature_extractor import FeatureExtractor
from src.utils.generate_synthetic import KINDS, generate_sequence

LOCALIZED = ["combo", "dark_patch", "new_object"]
K_SWEEP = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
OURS = "CNN + Mahalanobis (ours)"
PIXMAH = "Pixel-space Mahalanobis"
PIXDIFF = "Pixel difference"

class PixelDiffScorer(TemporalAnomalyScorer):
    """Baseline: mean abs difference to the last accepted (normal) frame, raw pixels."""
    def _distance(self, x):
        return float(np.abs(x - self.embeds[-1]).mean())
    
def run_scorer(scorer, vectors, capture_at=None):
    """Feed vectors in order. Returns (flagged frame indices, window mean before capture_at)."""
    flagged, ref = [], None
    for i, v in enumerate(vectors):
        if i == capture_at:
            ref = scorer.reference_stats()
        if scorer.update(v).is_anomaly:
            flagged.append(i)
    return flagged, ref

def load_sequence(folder, extractor):
    paths = sorted(Path(folder).glob("frame_*.png"))
    imgs = [Image.open(p).convert("RGB") for p in paths]
    emb = [extractor.embed(im) for im in imgs]
    pix = [np.asarray(im.convert("L").resize((16, 16), Image.BILINEAR), 
                      dtype=np.float32).ravel() / 255.0 for im in imgs]
    return imgs, emb, pix

def add(stat, flagged, truth):
    f = set(flagged)
    stat["tp"] += len(f & truth)
    stat["fp"] += len(f - truth)
    stat["fn"] += len(truth - f)
    
def prf(s):
    p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
    r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1

def md_table(header, rows):
    lines = ["| "+ " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| "+ " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-seqs", type=int, default=50)
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--k", type=float, default=3.0, help="must e one of K_SWEEP")
    ap.add_argument("--out", default="data/eval")
    args = ap.parse_args()
    assert args.k in K_SWEEP, f"--k must be one of {K_SWEEP}"
    
    extractor = FeatureExtractor()
    explainer = GradCAMExplainer(extractor)
    
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    sweep = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))  #method -> kind -> [found, total]
    loc = defaultdict(lambda: defaultdict(list))  #kind -> metric -> values
    
    for i in tqdm(range(args.num_seqs), desc="Sequences"):
        kind = KINDS[i % len(KINDS)]
        folder = Path(args.out) / f"seq_{i:03d}"
        lab = generate_sequence(folder, args.n_frames, kind, seed=args.seed + i)
        t = lab["anomaly_frames"][0]
        truth = {t}
        imgs, emb, pix = load_sequence(folder, extractor)
        
        # our method, all k values
        for k in K_SWEEP:
            flagged, ref = run_scorer(TemporalAnomalyScorer(k=k), emb,
                                      capture_at=t if k == args.k else None)
            add(sweep[k], flagged, truth)
            if k != args.k:
                continue
            add(stats[OURS], flagged, truth)
            hits[OURS][kind][0] += int(t in flagged)
            hits[OURS][kind][1] += 1
            
            # localization of the Grad-CAM map on the true anomaly frame
            if kind in LOCALIZED and ref is not None:
                mu, P = ref
                cam, _ = explainer.explain(imgs[t], P @ (emb[t] - mu))
                if cam.max() > 0:
                    s = 224 / 256
                    bx0, by0, bx1, by1 = [v * s for v in lab["box"]]
                    
                    def inside(x, y, m=0.0):
                        return bx0 - m <= x <= bx1 + m and by0 - m <= y <=by1 + m
                    
                    py, px = np.unravel_index(cam.argmax(), cam.shape)
                    loc[kind]["peak in box"].append(inside(px, py))
                    loc[kind]["peak within 16px"].append(inside(px, py, 16))
                    for hot in (0.5, 0.7):
                        ys, xs = np.nonzero(cam >= hot * cam.max())
                        loc[kind][f"centroid hot>={hot}"].append(inside(xs.mean(), ys.mean()))
                    mass = cam[int(by0):int(by1), int(bx0):int(bx1)].sum() / cam.sum()
                    loc[kind]["cam mass in box"].append(float(mass))
                    
        #baselines at the same k:
        for name, cls in ((PIXMAH, TemporalAnomalyScorer), (PIXDIFF, PixelDiffScorer)):
            flagged, _ = run_scorer(cls(k=args.k), pix)
            add(stats[name], flagged, truth)
            hits[name][kind][0] += int(t in flagged)
            hits[name][kind][1] += 1
            
    #--------------report----------------
    n = args.num_seqs
    out = [f"# Evaluation: {n} synthetic sequences x {args.n_frames} frames, "
           f"one anomaly each, k={args.k}\n"]
    
    rows = []
    for name in (OURS, PIXMAH, PIXDIFF):
        p, r, f1 = prf(stats[name])
        rows.append([name, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}", stats[name]["fp"]])   
    out += ["## Frame-level detection", md_table(
        ["method", "precision", "recall", "F1", "false alarms (total)"], rows), ""]
    
    rows = []
    for kind in KINDS:
        row = [kind]
        for name in (OURS, PIXMAH, PIXDIFF):
            d, tot = hits[name][kind]
            row.append(f"{d}/{tot}")
        rows.append(row)
    out += ["## Recall by anomaly type (anomaly frame found)", md_table(
        ["type", OURS, PIXMAH, PIXDIFF], rows), ""]
    
    rows = []
    for k in K_SWEEP:
        p, r, f1 = prf(sweep[k])
        rows.append([k, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}", sweep[k]["fp"]])
    out += ["## Sensitivity to k (ours)", md_table(
        ["k", "precision", "recall", "F1", "false alarms"], rows), ""]

    metrics = ["peak in box", "peak within 16px", "centroid hot>=0.5",
               "centroid hot>=0.7", "cam mass in box"]
    rows = []
    for kind in LOCALIZED:
        if not loc[kind]["peak in box"]:
            continue
        row = [kind, len(loc[kind]["peak in box"])]
        row += [f"{100 * np.mean(loc[kind][m]):.0f}%" for m in metrics]
        rows.append(row)
    out += ["## Grad-CAM localization on localized anomalies",
            "(the true box covers about 12% of the frame, so ~12% mass would be chance)",
            md_table(["type", "n"] + metrics, rows), ""]

    report = "\n".join(out)
    print("\n" + report)
    Path("docs").mkdir(exist_ok=True)
    Path("docs/eval_results.md").write_text(report + "\n")
    print("Saved docs/eval_results.md")


if __name__ == "__main__":
    main()                    