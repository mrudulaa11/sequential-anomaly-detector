"""Run anomaly detection on a folder of ssequential frames."""
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

def run_detection(folder, window=10, warmup=10, k=3.0, extractor=None):
    extractor = extractor or FeatureExtractor()
    scorer = TemporalAnomalyScorer(window=window, warmup=warmup, k=k)
    results = []
    for i, path in enumerate(tqdm(list_frames(folder), desc="Scoring")):
        emb = extractor.embed(Image.open(path))
        r = scorer.update(emb)
        results.append({"index": i, "path": str(path), "score":r.score, 
                        "threshold": r.threshold, "is_anomaly": r.is_anomaly,
                        "warming_up": r.warming_up})
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()
    
    results = run_detection(args.folder, window=args.window, k=args.k)
    
    print(f"\n{'frame':>5} {'score':>8} {'thresh':>8}  flag")
    for r in results:
        flag = "<-- ANOMALY" if r["is_anomaly"] else("(warmup)" if r["warming_up"] else "")
        print(f"{r['index']:>5} {r['score']:>8.2f} {r['threshold']:>8.2f}  {flag}")
        
    detected = [r["index"] for r in results if r["is_anomaly"]]
    labels_path = Path(args.folder) / "labels.json"
    print(f"\nDetected frames: {detected}")
    if labels_path.exists():
        truth = json.loads(labels_path.read_text())["anomaly_frames"]
        print(f"True anomaly frames: {truth}")
        
if __name__ == "__main__":
    main()