"""Unit tests for the temporal scorer. `_distance` is stubbed so scores are exact."""
import numpy as np
import pytest

from src.models.anomaly_scorer import TemporalAnomalyScorer

SMALL = dict(window=5, warmup=5, min_history=3, patience=3)


def make(**kw):
    s = TemporalAnomalyScorer(**kw)
    s._distance = lambda x: float(x[0])  # the "embedding" is the score we want
    return s


def feed(s, values):
    return [s.update(np.array([float(v)])) for v in values]


def test_warmup_and_history_phases_never_flag():
    s = make(**SMALL)
    res = feed(s, [30] * 8)
    assert all(r.warming_up for r in res)
    assert not any(r.is_anomaly for r in res)
    assert all(r.score == 0.0 for r in res[:5])
    assert all(np.isnan(r.threshold) for r in res)
    r = feed(s, [30])[0]
    assert not r.warming_up and not np.isnan(r.threshold)


def test_default_params_first_scored_frame_is_15():
    s = make()  # window=10, warmup=10, min_history=5
    res = feed(s, [30] * 17)
    assert all(r.warming_up for r in res[:15])
    assert not res[15].warming_up


def test_threshold_uses_median_and_mad():
    s = make(**SMALL)
    s.scores.extend([20, 30, 40, 50, 60])
    assert s._threshold() == pytest.approx(40 + 3 * 1.4826 * 10)


def test_threshold_has_relative_floor():
    s = make(**SMALL)
    s.scores.extend([30] * 5)
    assert s._threshold() == pytest.approx(30 + 3 * 0.25 * 30)


def test_one_off_spike_is_flagged_and_kept_out_of_the_model():
    s = make(**SMALL)
    feed(s, [30] * 8)
    assert feed(s, [200])[0].is_anomaly
    assert [e[0] for e in s.embeds] == [30] * 5
    assert len(s.scores) == 3
    assert not feed(s, [30])[0].is_anomaly
    assert s.pending == []


def test_three_consecutive_flags_become_the_new_normal():
    s = make(**SMALL)
    feed(s, [30] * 8)
    assert [r.is_anomaly for r in feed(s, [100, 100, 100])] == [True, True, True]
    assert s.pending == []
    assert [e[0] for e in s.embeds] == [30, 30, 100, 100, 100]
    assert not feed(s, [100])[0].is_anomaly


def test_interrupted_flags_reset_the_counter():
    s = make(**SMALL)
    feed(s, [30] * 8)
    assert [r.is_anomaly for r in feed(s, [100, 100, 30, 100, 100])] == \
        [True, True, False, True, True]
    assert len(s.pending) == 2


def test_reference_mean():
    s = make(window=3, warmup=3, min_history=1)
    assert s.reference_mean() is None
    feed(s, [1, 2, 3])
    assert s.reference_mean()[0] == pytest.approx(2.0)


def test_known_limitation_anomaly_during_warmup_is_absorbed():
    # Documents current behaviour: warm-up frames are accepted unconditionally.
    s = make(**SMALL)
    feed(s, [30, 30, 500, 30, 30])
    assert 500 in [e[0] for e in s.embeds]


def test_real_mahalanobis_flags_an_obvious_spike():
    rng = np.random.default_rng(0)
    s = TemporalAnomalyScorer()
    normal = rng.normal(0, 1, size=(25, 16))
    res = [s.update(x) for x in normal]
    spike = s.update(normal[-1] + 25.0)
    scores = [r.score for r in res if not r.warming_up]
    assert spike.is_anomaly
    assert spike.score > 2 * np.median(scores)