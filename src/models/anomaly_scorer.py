"""Sliding-window Mahalanobis scorer with an adaptive threshold."""
from collections import deque
from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf

@dataclass
class ScoreResult:
    score: float
    threshold: float
    is_anomaly: bool
    warming_up: bool
    
class TemporalAnomalyScorer:
    def __init__(self, window=12, warmup=8, k=3.0, min_history=5, history=30):
        self.warmup = warmup
        self.k = k
        self.min_history = min_history
        self.embeds = deque(maxlen=window) #recent NORMAL embeddings
        self.scores = deque(maxlen=history) #recent NORMAL scores
    
    def _distance(self, x: np.ndarray) -> float:
        X = np.stack(self.embeds)
        try:
            lw = LedoitWolf().fit(X)
            d2 = lw.mahalanobis(x[None, :])[0]
            return float(np.sqrt(max(d2, 0.0)))
        except Exception: #fallback: Euclidean distance to window mean
            return float(np.linalg.norm(x - X.mean(axis=0)))
        
    def update(self, x: np.ndarray) -> ScoreResult:
        if len(self.embeds) < self.warmup:
            self.embeds.append(x)
            return ScoreResult(0.0, float("nan"), False, True)
        
        d = self._distance(x)
        
        if len(self.scores) < self.min_history:
            self.embeds.append(x)
            self.scores.append(d)
            return ScoreResult(d, float("nan"), False, True)
        
        mu, sd = float(np.mean(self.scores)), float(np.std(self.scores))
        thr = mu + self.k * max(sd, 0.05 * mu)
        is_anom = d > thr
        if not is_anom: #only normal frames update the model of "normal"
            self.embeds.append(x)
            self.scores.append(d)
        return ScoreResult(d, thr, is_anom, False)
    