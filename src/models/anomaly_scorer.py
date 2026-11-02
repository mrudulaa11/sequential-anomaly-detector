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
    def __init__(self, window=10, warmup=10, k=3.0, min_history=5, 
                history=30, rel_floor=0.25, patience=3):
        self.warmup = warmup
        self.k = k
        self.min_history = min_history
        self.rel_floor = rel_floor
        self.patience = patience
        self.embeds = deque(maxlen=window) #recent NORMAL embeddings
        self.scores = deque(maxlen=history) #recent NORMAL scores
        self.pending = []  #flagged (embedding, score) pairs
    
    def _distance(self, x: np.ndarray) -> float:
        X = np.stack(self.embeds)
        try:
            lw = LedoitWolf().fit(X)
            d2 = lw.mahalanobis(x[None, :])[0]
            return float(np.sqrt(max(d2, 0.0)))
        except Exception: #fallback: Euclidean distance to window mean
            return float(np.linalg.norm(x - X.mean(axis=0)))
        
    def _threshold(self) -> float:
        s= np.asarray(self.scores)
        med = float(np.median(s))
        mad = 1.4826 + float(np.median(np.abs(s-med)))
        return med + self.k * max(mad, self.rel_floor * med)
        
    def update(self, x: np.ndarray) -> ScoreResult:
        # 1. fill the window
        if len(self.embeds) < self.warmup:
            self.embeds.append(x)
            return ScoreResult(0.0, float("nan"), False, True)
        
        d = self._distance(x)
        
        # 2. build a small scpre history before flagging anything
        if len(self.scores) < self.min_history:
            self.embeds.append(x)
            self.scores.append(d)
            return ScoreResult(d, float("nan"), False, True)
        
        thr = self._threshold()
        is_anom = d > thr
        if is_anom:
            self.pending.append((x,d))
            if len(self.pending) >= self.patience: 
                for e, s in self.pending:
                    self.embeds.append(e)
                    self.scores.append(s)
                self.pending.clear()
        else:
            self.pending.clear() 
            self.embeds.append(x)
            self.scores.append(d)
        return ScoreResult(d, thr, is_anom, False)

    def reference_mean(self):
        """Mean embedding of the current 'normal' window (None if empty)."""
        if not self.embeds:
            return None
        return np.stack(self.embeds).mean(axis=0)
