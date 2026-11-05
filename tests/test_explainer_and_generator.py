import json

import cv2
import numpy as np
import pytest
from PIL import Image

from src.explain.text_explainer import attention_phrase, describe_anomaly
from src.utils.generate_synthetic import generate_sequence


def solid(c, size=64):
    return Image.new("RGB", (size, size), (c, c, c))


def blob(cx, cy, size=64, r=6):
    cam = np.zeros((size, size), dtype=np.float32)
    cam[cy - r:cy + r, cx - r:cx + r] = 1.0
    return cam


def test_attention_regions():
    assert "lower-right" in attention_phrase(blob(52, 52))
    assert "upper-left" in attention_phrase(blob(10, 10))
    assert "center" in attention_phrase(blob(32, 32))
    assert "whole frame" in attention_phrase(np.ones((64, 64), dtype=np.float32))
    assert "no clear" in attention_phrase(np.zeros((64, 64), dtype=np.float32))


def test_brightness_drop_is_reported():
    info = describe_anomaly(solid(40), solid(100), 120.0, 50.0, blob(32, 32))
    assert "brightness drop" in info["text"]
    assert info["metrics"]["brightness_delta"] == pytest.approx(-60.0, abs=1.0)


def test_localized_change_is_not_called_global():
    prev = solid(100)
    cur = prev.copy()
    cur.paste((255, 0, 0), (10, 10, 30, 30))  # 20x20 patch, about 10% of the frame
    info = describe_anomaly(cur, prev, 120.0, 50.0, blob(20, 20))
    assert "possible new object" in info["text"]
    assert "whole frame" not in info["text"]


def test_global_change_says_attention_is_not_localized():
    info = describe_anomaly(solid(200), solid(100), 150.0, 50.0, blob(32, 32))
    assert "whole frame" in info["text"]
    assert "not localized" in info["text"]


def test_generator_labels_and_frame_count(tmp_path):
    lab = generate_sequence(tmp_path / "s", n_frames=30, kind="combo", seed=0)
    assert len(list((tmp_path / "s").glob("frame_*.png"))) == 30
    assert 15 <= lab["anomaly_frames"][0] <= 27
    assert json.loads((tmp_path / "s" / "labels.json").read_text()) == lab


def test_generator_is_reproducible(tmp_path):
    generate_sequence(tmp_path / "a", seed=3)
    generate_sequence(tmp_path / "b", seed=3)
    f = "frame_005.png"
    assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def test_brightness_drop_anomaly_is_darker(tmp_path):
    lab = generate_sequence(tmp_path / "s", kind="brightness_drop", seed=1)
    t = lab["anomaly_frames"][0]
    mean = lambda i: cv2.imread(str(tmp_path / "s" / f"frame_{i:03d}.png")).mean()
    assert mean(t) < 0.7 * mean(t - 1)