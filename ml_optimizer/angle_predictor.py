"""
QAOA Angle Predictor.
High-level interface for training and inference.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from .feature_extractor import GraphFeatureExtractor
from .models import AnglePredictor

logger = logging.getLogger(__name__)


class QAOAAnglePredictor:
    """
    End-to-end warm-start predictor for QAOA angles.

    Usage:
        predictor = QAOAAnglePredictor(p=2)
        predictor.fit(training_instances, target_angles)
        gamma, beta = predictor.predict(G, Q, src, dst)
    """

    def __init__(self, p: int = 1, device: str = "auto", hidden_dims=(128, 256, 128), lr=1e-3):
        self.p = p
        self.extractor = GraphFeatureExtractor()
        self.model = AnglePredictor(
            input_dim=GraphFeatureExtractor.N_FEATURES,
            p=p,
            hidden_dims=hidden_dims,
            lr=lr,
            device=device,
        )
        self._is_trained = False

    def fit(
        self,
        instances: list,
        target_angles: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        val_split: float = 0.1,
        verbose: bool = True,
    ) -> dict:
        """
        Train angle predictor.

        Parameters
        ----------
        instances     : list of (G, Q, src, dst, traffic_matrix) or dicts
        target_angles : np.ndarray (n, 2*p) — optimal angles from prior runs
        epochs        : training epochs
        batch_size    : mini-batch size
        val_split     : fraction for validation

        Returns training history dict.
        """
        import torch

        X = self.extractor.extract_batch(instances)
        y = np.array(target_angles, dtype=np.float32)

        # Split
        n = len(X)
        val_size = max(1, int(n * val_split))
        idx = np.random.permutation(n)
        val_idx, train_idx = idx[:val_size], idx[val_size:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]

            # Train
            epoch_losses = []
            for start in range(0, len(X_train), batch_size):
                xb = X_train[start:start + batch_size]
                yb = y_train[start:start + batch_size]
                if len(xb) < 2:
                    continue  # BN needs ≥2 samples
                loss = self.model.train_step(xb, yb)
                epoch_losses.append(loss)

            # Validate
            val_pred = self.model.forward(X_val)
            val_loss = float(np.mean((val_pred - y_val) ** 2))

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            self.model.scheduler.step(val_loss)

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    "AnglePredictor epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch + 1, epochs, train_loss, val_loss,
                )

        self._is_trained = True
        return history

    def predict(self, G, Q: Optional[np.ndarray] = None,
                src: Optional[int] = None, dst: Optional[int] = None,
                traffic_matrix: Optional[np.ndarray] = None):
        """
        Predict warm-start QAOA angles for a new instance.

        Returns
        -------
        gamma : np.ndarray (p,)
        beta  : np.ndarray (p,)
        """
        if not self._is_trained:
            logger.warning("AnglePredictor not trained — returning random angles")
            rng = np.random.default_rng()
            return rng.uniform(0, np.pi, self.p), rng.uniform(0, np.pi / 2, self.p)
        feats = self.extractor.extract(G, Q, src, dst, traffic_matrix)
        return self.model.predict_angles(feats)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        logger.info("AnglePredictor saved to %s", path)

    def load(self, path: str):
        self.model.load(path)
        self._is_trained = True
        logger.info("AnglePredictor loaded from %s", path)
