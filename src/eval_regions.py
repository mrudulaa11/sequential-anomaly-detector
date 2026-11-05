"""Does the explanation's region label match where the anomaly really is (3x3 grid)?"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.evaluate import LOCALIZED, load_sequence, md_table, run_scorer
from src.explain.gradcam_explainer import GradCAMExplainer
from src.models.anomaly_scorer import TemporalAnomalyScorer
from src.models.feature_extractor import FeatureExtractor


def cell(fx, fy):
    """3x3 grid cell (row, col) of a point given as fractions of width/height."""
    return min(int(fy * 3), 2), min(int(fx * 3), 2)


def peak(cam):
    y, x = np.unravel_index(cam.argmax(), cam.shape)
    return x / cam.shape[1], y / cam.shape[0]


def centroid(cam, hot):
    ys, xs = np.nonzero(cam >= hot * cam.max())
    return xs.mean() / cam.shape[1], ys.mean() / cam.shape[0]


def weighted_centroid(cam):
    ys, xs = np.indices(cam.shape)
    s = cam.sum()
    return (xs * cam).sum() / s / cam.shape[1], (ys * cam).sum() / s / cam.shape[0]


METHODS = {
    "peak": peak,
    "centroid >=0.5 (current text rule)": lambda c: centroid(c, 0.5),
    "centroid >=0.7": lambda c: centroid(c, 0.7),
    "heatmap-weighted centroid": weighted_centroid,
    "always 'center' (baseline)": lambda c: (0.5, 0.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/eval")
    ap.add_argument("--k", type=float, default=3.0)
    args = ap.parse_args()

    extractor = FeatureExtractor()
    explainer = GradCAMExplainer(extractor)
    exact = defaultdict(lambda: defaultdict(list))
    near = defaultdict(lambda: defaultdict(list))

    for d in tqdm(sorted(Path(args.root).glob("seq_*")), desc="Sequences"):
        lab = json.loads((d / "labels.json").read_text())
        if lab["kind"] not in LOCALIZED:
            continue
        t = lab["anomaly_frames"][0]
        imgs, emb, _ = load_sequence(d, extractor)
        _, ref = run_scorer(TemporalAnomalyScorer(k=args.k), emb, capture_at=t)
        if ref is None:
            continue
        cam, _ = explainer.explain(imgs[t], emb[t] - ref)
        if cam.max() <= 0:
            continue
        b = lab["box"]
        true_cell = cell((b[0] + b[2]) / 2 / 256, (b[1] + b[3]) / 2 / 256)
        for name, fn in METHODS.items():
            pc = cell(*fn(cam))
            exact[name][lab["kind"]].append(pc == true_cell)
            near[name][lab["kind"]].append(
                max(abs(pc[0] - true_cell[0]), abs(pc[1] - true_cell[1])) <= 1)

    rows = []
    for name in METHODS:
        allx = [v for kind in LOCALIZED for v in exact[name][kind]]
        alln = [v for kind in LOCALIZED for v in near[name][kind]]
        row = [name, f"{100 * np.mean(allx):.0f}%", f"{100 * np.mean(alln):.0f}%"]
        row += [f"{100 * np.mean(exact[name][k]):.0f}%" for k in LOCALIZED]
        rows.append(row)

    report = "\n".join([
        f"# Region agreement on {len(allx)} localized anomalies (3x3 grid)", "",
        md_table(["rule", "exact cell", "within 1 cell"] +
                 [f"exact: {k}" for k in LOCALIZED], rows), ""])
    print("\n" + report)
    Path("docs").mkdir(exist_ok=True)
    Path("docs/eval_regions.md").write_text(report + "\n")
    print("Saved docs/eval_regions.md")


if __name__ == "__main__":
    main()