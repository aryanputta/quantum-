"""
PyTorch models for QAOA angle and QUBO coefficient prediction.
CUDA-aware: auto-selects GPU if available, falls back to CPU.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def get_device(device_str: str = "auto") -> "torch.device":
    """Resolve device: 'auto' picks CUDA > MPS > CPU."""
    import torch
    if device_str == "auto":
        if torch.cuda.is_available():
            d = torch.device("cuda")
            logger.info("ML device: CUDA (%s)", torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            d = torch.device("mps")
            logger.info("ML device: MPS (Apple Silicon)")
        else:
            d = torch.device("cpu")
            logger.info("ML device: CPU")
        return d
    return torch.device(device_str)


class AnglePredictor(object):
    """
    Neural network regressor: graph + QUBO features → QAOA (gamma, beta) initialization.

    Architecture: MLP with configurable hidden dims.
    Input:  feature vector of dimension `input_dim`
    Output: 2*p angles (p gamma + p beta)

    CUDA support: moves model and data to GPU if available.
    """

    def __init__(
        self,
        input_dim: int,
        p: int,
        hidden_dims: tuple = (128, 256, 128),
        lr: float = 1e-3,
        device: str = "auto",
    ):
        import torch
        import torch.nn as nn

        self.p = p
        self.output_dim = 2 * p
        self.device = get_device(device)

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, self.output_dim))
        self.model = nn.Sequential(*layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=15, factor=0.5
        )
        self.loss_fn = nn.MSELoss()
        self.train_losses: list = []
        self.val_losses: list = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Predict angles for a batch of feature vectors."""
        import torch
        self.model.eval()
        with torch.no_grad():
            xt = torch.FloatTensor(x).to(self.device)
            if xt.dim() == 1:
                xt = xt.unsqueeze(0)
            out = self.model(xt)
            return out.cpu().numpy()

    def predict_angles(self, features: np.ndarray) -> tuple:
        """
        Predict (gamma, beta) for a single instance.

        Returns
        -------
        gamma : np.ndarray of shape (p,)
        beta  : np.ndarray of shape (p,)
        """
        pred = self.forward(features).squeeze()
        gamma = pred[: self.p]
        beta = pred[self.p :]
        return gamma, beta

    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Single training step. Returns batch loss."""
        import torch
        self.model.train()
        xt = torch.FloatTensor(x_batch).to(self.device)
        yt = torch.FloatTensor(y_batch).to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(xt)
        loss = self.loss_fn(pred, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def save(self, path: str):
        import torch
        torch.save({"model_state": self.model.state_dict(),
                    "p": self.p, "output_dim": self.output_dim}, path)

    def load(self, path: str):
        import torch
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])


class QUBOCoefficientPredictor(object):
    """
    Neural network regressor: graph + traffic features → QUBO penalty weight adjustments.

    Predicts multiplicative adjustment factors for:
    [w_latency, w_congestion, w_loss, penalty_path, penalty_flow, penalty_capacity]

    These factors rescale the default QUBO coefficients for new traffic conditions,
    allowing fast adaptation without full QUBO recomputation.
    """

    OUTPUT_NAMES = ["w_latency", "w_congestion", "w_loss",
                    "penalty_path", "penalty_flow", "penalty_capacity"]

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (64, 128, 64),
        lr: float = 1e-3,
        device: str = "auto",
    ):
        import torch
        import torch.nn as nn

        self.output_dim = len(self.OUTPUT_NAMES)
        self.device = get_device(device)

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        # Sigmoid output → scale factors in (0, 2) to avoid degenerate penalties
        layers += [nn.Linear(prev, self.output_dim), nn.Sigmoid()]
        self.model = nn.Sequential(*layers).to(self.device)

        # Scale sigmoid output: 0→0, 0.5→1, 1→2
        self._scale = 2.0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.train_losses: list = []

    def predict_weights(self, features: np.ndarray) -> dict:
        """
        Predict QUBO weight adjustment factors.

        Returns dict mapping weight name → adjustment factor in (0, 2].
        """
        import torch
        self.model.eval()
        with torch.no_grad():
            xt = torch.FloatTensor(features).to(self.device)
            if xt.dim() == 1:
                xt = xt.unsqueeze(0)
            out = self.model(xt).squeeze().cpu().numpy() * self._scale
        return dict(zip(self.OUTPUT_NAMES, out.tolist()))

    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """y_batch contains target weight adjustment factors in (0, 2]."""
        import torch
        self.model.train()
        xt = torch.FloatTensor(x_batch).to(self.device)
        # Normalize targets to (0,1) for sigmoid layer
        yt = torch.FloatTensor(y_batch / self._scale).to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(xt)
        loss = self.loss_fn(pred, yt)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path: str):
        import torch
        torch.save({"model_state": self.model.state_dict()}, path)

    def load(self, path: str):
        import torch
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
