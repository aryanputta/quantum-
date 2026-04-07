"""
QUBO Weight Predictor.
Predicts penalty weight adjustments for new traffic/congestion conditions.
Enables fast QUBO adaptation without full recomputation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .feature_extractor import GraphFeatureExtractor
from .models import QUBOCoefficientPredictor

logger = logging.getLogger(__name__)


class QUBOWeightPredictor:
    """
    Predicts multiplicative adjustment factors for QUBO weights.

    When traffic conditions change (new congestion pattern, traffic spike),
    instead of recomputing the full QUBO from scratch, predict adjusted
    penalty weights from graph+traffic features, then rescale the existing Q.

    Usage:
        predictor = QUBOWeightPredictor()
        predictor.fit(instances, target_weights)
        adjustments = predictor.predict_adjustments(G, traffic_matrix)
        new_Q = predictor.apply_adjustments(Q, variable_map, adjustments, G)
    """

    DEFAULT_WEIGHTS = {
        "w_latency": 1.0, "w_congestion": 2.0, "w_loss": 3.0,
        "penalty_path": 10.0, "penalty_flow": 8.0, "penalty_capacity": 5.0,
    }

    def __init__(self, device: str = "auto", hidden_dims=(64, 128, 64), lr=1e-3):
        self.extractor = GraphFeatureExtractor()
        self.model = QUBOCoefficientPredictor(
            input_dim=GraphFeatureExtractor.N_FEATURES,
            hidden_dims=hidden_dims,
            lr=lr,
            device=device,
        )
        self._is_trained = False

    def fit(
        self,
        instances: list,
        target_weights: np.ndarray,
        epochs: int = 150,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> dict:
        """
        Train weight predictor.

        Parameters
        ----------
        instances      : list of (G, Q, src, dst, traffic_matrix) or dicts
        target_weights : np.ndarray (n, 6) — optimal weight adjustment factors
                         (measured by which weights led to best QUBO solutions)

        Returns training history.
        """
        X = self.extractor.extract_batch(instances)
        y = np.array(target_weights, dtype=np.float32)

        history = {"train_loss": []}
        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            X, y = X[perm], y[perm]
            epoch_losses = []
            for start in range(0, len(X), batch_size):
                xb = X[start:start + batch_size]
                yb = y[start:start + batch_size]
                if len(xb) < 2:
                    continue
                loss = self.model.train_step(xb, yb)
                epoch_losses.append(loss)
            if epoch_losses:
                history["train_loss"].append(float(np.mean(epoch_losses)))
            if verbose and (epoch + 1) % 30 == 0:
                logger.info(
                    "QUBOWeightPredictor epoch %d/%d  loss=%.4f",
                    epoch + 1, epochs, history["train_loss"][-1] if history["train_loss"] else np.nan,
                )

        self._is_trained = True
        return history

    def predict_adjustments(
        self,
        G,
        Q: Optional[np.ndarray] = None,
        src: Optional[int] = None,
        dst: Optional[int] = None,
        traffic_matrix: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Predict QUBO weight adjustment factors.

        Returns dict: {'w_latency': factor, 'w_congestion': factor, ...}
        All factors in range (0, 2].
        """
        if not self._is_trained:
            logger.warning("QUBOWeightPredictor not trained — returning unit adjustments")
            return {k: 1.0 for k in QUBOCoefficientPredictor.OUTPUT_NAMES}
        feats = self.extractor.extract(G, Q, src, dst, traffic_matrix)
        return self.model.predict_weights(feats)

    def apply_adjustments(
        self,
        Q: np.ndarray,
        variable_map: dict,
        adjustments: dict,
        G,
        src: int,
        dst: int,
    ) -> np.ndarray:
        """
        Apply predicted weight adjustments to existing QUBO matrix.

        Instead of rebuilding Q from scratch with new weights,
        scale the objective and penalty components proportionally.

        Parameters
        ----------
        Q            : original QUBO matrix
        variable_map : {idx: (u,v)} edge → variable mapping
        adjustments  : dict from predict_adjustments()
        G, src, dst  : routing context

        Returns
        -------
        np.ndarray : adjusted QUBO matrix
        """
        # Decompose Q into diagonal (objective proxy) and off-diagonal (constraint proxy)
        n = Q.shape[0]
        Q_adj = Q.copy()

        # Scale objective components (diagonal terms carry edge costs)
        obj_scale = (
            adjustments.get("w_latency", 1.0) +
            adjustments.get("w_congestion", 1.0) +
            adjustments.get("w_loss", 1.0)
        ) / 3.0

        # Scale constraint components (off-diagonal terms from penalty expansions)
        penalty_scale = (
            adjustments.get("penalty_path", 1.0) +
            adjustments.get("penalty_flow", 1.0) +
            adjustments.get("penalty_capacity", 1.0)
        ) / 3.0

        diag_orig = np.diag(Q)
        off_diag = Q - np.diag(diag_orig)

        Q_adj = np.diag(diag_orig * obj_scale) + off_diag * penalty_scale
        return Q_adj

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        self.model.load(path)
        self._is_trained = True
