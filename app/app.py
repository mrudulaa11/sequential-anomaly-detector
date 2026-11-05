"""Streamlit demo: score timeline, flagged frames, Grad-CAM heatmaps, explanations."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  #lets src be imported when launched via streamlit

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from src.detect import run_detection
from src.models.feature_extractor import FeatureExtractor
from src.detect import list_frames, run_detection

st.set_page_config(page_title="Sequential Anomaly Detector", layout="wide")

@st.cache_resource
def get_extractor():
    return FeatureExtractor()

def find_sequences():
    base = ROOT / "data"
    if not base.exists():
        return []
    return sorted(str(p.relative_to(ROOT)) for p in {f.parent for f in base.rglob("*.png")})

#-----------sidebar-----------
st.title("Sequential Image Anomaly Detection")
st.caption("Frozen ResNet18 embeddings, sliding-window Mahalanobis scoring,"
           "Grad-CAM on the embedding shift, rule-based explanations.")
st.sidebar.header("Input")
seqs = find_sequences()
choice = st.sidebar.selectbox("Sequence folder", seqs) if seqs else None
custom = st.sidebar.text_input("...or type a folder path", "")
k = st.sidebar.slider("Threshold multiplier k", 1.5, 6.0, 3.0, 0.5)
window = st.sidebar.slider("Window size (frames)", 5, 20, 10, 1)
st.sidebar.caption(f"Warm-up: frames 0-{window + 4} can't be flagged.")
run = st.sidebar.button("Run detection", type="primary")

if run:
    raw = custom.strip() or choice
    folder = Path(raw) if raw else None
    if folder is not None and not folder.is_absolute():
        folder = ROOT / folder
    if folder is None or not folder.is_dir():
        st.error("Pick a sequence folder (or type a valid path).")
        st.stop()
        n_frames = len(list_frames(folder))
    if window + 5 >= n_frames:
        st.error(f"Window {window} leaves no scored frames in a {n_frames}-frame sequence. "
                 f"Use a window below {n_frames - 5}.")
        st.stop()
    with st.spinner("Scoring frames and explaining anomalities..."):
        results = run_detection(
            str(folder), window=window, warmup=window, k=k,
            extractor=get_extractor(), explain=True,
            heatmap_dir=str(ROOT / "outputs/heatmaps"),
            explanation_dir=str(ROOT / "outputs/explanations")
        )
    st.session_state["run"] = {"folder": str(folder), "results": results}

data = st.session_state.get("run")
if not data:
    st.info("Choose a sequence in the sidebar and click **Run detection**.")
    st.stop()
    
results = data["results"]
folder = Path(data["folder"])
flagged = [r for r in results if r["is_anomaly"]]
detected = [r["index"] for r in flagged]

truth = []
lp = folder / "labels.json"
if lp.exists():
    truth = json.loads(lp.read_text())["anomaly_frames"]

#-------------summary---------------
c1, c2, c3 = st.columns(3)
c1.metric("Frames", len(results))
c2.metric("Flagged", len(flagged))
if truth:
    if set(detected) == set(truth):
        c3.metric("Ground truth", str(truth), "all found, no false alarms")
    else:
        missed = sorted(set(truth) - set(detected))
        extra = sorted(set(detected) - set(truth))
        c3.metric("Ground truth", str(truth), f"missed {missed}, false alarms {extra}",
                  delta_color="inverse")
else:
    c3.metric("Ground truth", "n/a", "no labels.json")
    
#----------timeline-----------
idx = np.array([r["index"] for r in results])
scores = np.array([np.nan if r["warming_up"] else r["score"] for r in results])
thr = np.array([r["threshold"] for r in results], dtype=float)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(idx, scores, marker="o", ms=3, label="anomaly score")
ax.plot(idx, thr, "--", label="adaptive threshold")
if flagged:
    ax.scatter([r["index"] for r in flagged], [r["score"] for r in flagged], color="red",
               zorder=3, label="flagged")
for n, t in enumerate(truth):
    ax.axvline(t, color="green", ls=":", alpha=0.8, label="true anomaly" if n == 0 else None)
ax.set_xlabel("frame")
ax.set_ylabel("Mahalanobis distance")    
ax.legend()
st.pyplot(fig)
plt.close(fig)


#------------flagged frames------------
st.subheader("Flagged frames")
if not flagged:
    st.write("No anomalies flagged.")
for r in flagged:
    with st.expander(f"Frame {r['index']} | score {r['score']:.1f}", expanded=True):
        a, b =st.columns(2)
        a.image(Image.open(r["path"]), caption="original")
        if "heatmap_path" in r:
            b.image(Image.open(r["heatmap_path"]), caption="Grad-CAM: regions driving the shift")
            st.info(r["explanation"])
        else:
            st.write("No explanation available (needs a previous frame and a warmed-up window).")
            
#-----------------frame browser-------------------
st.subheader("Browse all frames")
default = detected[0] if detected else 0
i = st.slider("Frame", 0, len(results) - 1, default)
r = results[i]
status = ("FLAGGED" if r["is_anomaly"] else "warm-up" if r["warming_up"] else "normal")
st.image(Image.open(r["path"]), caption=f"frame {i}: {status}, score {r['score']:.1f}", width=320)