"""Run anomaly detection on a folder of sequential frames."""
import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.models.anomaly_scorer import TemporalAnomalyScorer
from src.models.feature_extractor import FeatureExtractor

EXTS = {".png", ".jpg", ".jpeg"}


def list_frames(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)


def run_detection(folder, window=10, warmup=10, k=3.0, extractor=None,
                  explain=False, heatmap_dir="outputs/heatmaps",
                  explanation_dir="outputs/explanations"):
    extractor = extractor or FeatureExtractor()
    scorer = TemporalAnomalyScorer(window=window, warmup=warmup, k=k)
    explainer = None
    if explain:
        from src.explain.gradcam_explainer import GradCAMExplainer
        from src.explain.text_explainer import describe_anomaly
        explainer = GradCAMExplainer(extractor)

    seq_name = Path(folder).name
    results, prev_img = [], None
    for i, path in enumerate(tqdm(list_frames(folder), desc="Scoring")):
        img = Image.open(path).convert("RGB")
        emb = extractor.embed(img)
        stats = scorer.reference_stats()      # window mean + precision, BEFORE this frame is considered
        r = scorer.update(emb)
        row = {"index": i, "path": str(path), "score": r.score,
               "threshold": r.threshold, "is_anomaly": r.is_anomaly,
               "warming_up": r.warming_up}

        if explainer is not None and r.is_anomaly and stats is not None and prev_img is not None:
            mu, P = stats
            cam, overlay = explainer.explain(img, P @ (emb - mu))   # gradient direction of the score
            hp = Path(heatmap_dir) / f"{seq_name}_frame_{i:03d}.png"
            hp.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay).save(hp)
            
            info = describe_anomaly(img, prev_img, r.score, r.threshold, cam)
            ep = Path(explanation_dir) / f"{seq_name}_frame_{i:03d}.txt"
            ep.parent.mkdir(parents=True, exist_ok=True)
            ep.write_text(info["text"] + "\n")
            
            row.update({"heatmap_path": str(hp), "explanation": info["text"],
                        "metrics":info["metrics"], "cam": cam})
        results.append(row)
        prev_img = img
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=None, help="frames used to fill the window (default: same as --window)")
    ap.add_argument("--explain", action="store_true", 
                    help="save Grad-CAM heatmaps for flagged frames")
    args = ap.parse_args()

    results = run_detection(args.folder, window=args.window, warmup=args.window if args.warmup is None else args.warmup, k=args.k, explain=args.explain)

    print(f"\n{'frame':>5} {'score':>8} {'thresh':>8}  flag")
    for r in results:
        flag = "<-- ANOMALY" if r["is_anomaly"] else ("(warmup)" if r["warming_up"] else "")
        print(f"{r['index']:>5} {r['score']:>8.2f} {r['threshold']:>8.2f}  {flag}")

    detected = [r["index"] for r in results if r["is_anomaly"]]
    print(f"\nDetected frames: {detected}")
    labels_path = Path(args.folder) / "labels.json"
    if labels_path.exists():
        print(f"True anomaly frames: {json.loads(labels_path.read_text())['anomaly_frames']}")
    for r in results:
        if "explanation" in r:
            print(f"\nFrame {r['index']}: {r['explanation']}")
            print(f" heatmap -> {r['heatmap_path']}")
    
    
if __name__ == "__main__":
    main()