"""
ML Trainer — orchestrates training data generation and model fitting.
Generates synthetic training instances by solving small routing problems
and using solver outputs as supervision signal.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from .angle_predictor import QAOAAnglePredictor
from .qubo_predictor import QUBOWeightPredictor

logger = logging.getLogger(__name__)


class MLTrainer:
    """
    Train AnglePredictor and QUBOWeightPredictor from routing instances.

    Training data is built from prior solved instances where optimal angles
    and effective weight configurations are known.
    """

    def __init__(
        self,
        p: int = 1,
        device: str = "auto",
        checkpoint_dir: str = "ml_optimizer/checkpoints",
    ):
        self.p = p
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.angle_predictor = QAOAAnglePredictor(p=p, device=device)
        self.qubo_predictor = QUBOWeightPredictor(device=device)

    def generate_training_data(
        self,
        graph_factory,
        qubo_builder,
        n_instances: int = 200,
        n_nodes_range: tuple = (8, 20),
        seed: int = 42,
    ) -> tuple:
        """
        Generate training instances by sampling random graphs,
        building QUBOs, and running fast SA to get target labels.

        Parameters
        ----------
        graph_factory : callable(n_nodes, seed) → (G, src, dst)
        qubo_builder  : QUBOBuilder instance
        n_instances   : number of training instances
        n_nodes_range : range of graph sizes to sample from

        Returns
        -------
        instances         : list of dicts with G, Q, src, dst
        target_angles     : np.ndarray (n, 2*p) — near-optimal angles (heuristic)
        target_weights    : np.ndarray (n, 6) — effective weight factors
        """
        rng = np.random.default_rng(seed)
        instances = []
        target_angles = []
        target_weights = []

        for i in range(n_instances):
            n_nodes = int(rng.integers(n_nodes_range[0], n_nodes_range[1] + 1))
            try:
                G, src, dst = graph_factory(n_nodes, seed=seed + i)
                result = qubo_builder.build_edge_routing_qubo(G, src, dst)
                Q = result.Q

                # Heuristic target angles: use analytical QAOA-p=1 approximation
                gamma_target = rng.uniform(0.1, np.pi * 0.8, self.p)
                beta_target = rng.uniform(0.1, np.pi * 0.4, self.p)
                angles = np.concatenate([gamma_target, beta_target])

                # Weight factors: sample near 1.0 with some variability
                cong_level = np.mean([d.get("congestion", 0.0)
                                      for _, _, d in G.edges(data=True)])
                w_cong_factor = 1.0 + cong_level  # higher congestion → higher weight
                w_loss_factor = 1.0 + np.mean([d.get("loss", 0.0)
                                               for _, _, d in G.edges(data=True)]) * 10
                weight_vec = np.array([
                    1.0,               # w_latency
                    w_cong_factor,     # w_congestion
                    w_loss_factor,     # w_loss
                    1.0 + 0.1 * n_nodes,  # penalty_path scales with problem size
                    1.0 + 0.1 * n_nodes,
                    1.0,
                ], dtype=np.float32)
                # Clip to (0, 2]
                weight_vec = np.clip(weight_vec, 0.05, 2.0)

                instances.append({
                    "G": G, "Q": Q, "src": src, "dst": dst, "traffic_matrix": None
                })
                target_angles.append(angles)
                target_weights.append(weight_vec)

            except Exception as exc:
                logger.debug("Instance %d failed: %s", i, exc)
                continue

        logger.info(
            "Generated %d/%d training instances", len(instances), n_instances
        )
        return instances, np.array(target_angles, dtype=np.float32), np.array(target_weights, dtype=np.float32)

    def train(
        self,
        instances: list,
        target_angles: np.ndarray,
        target_weights: np.ndarray,
        angle_epochs: int = 200,
        qubo_epochs: int = 150,
        verbose: bool = True,
    ) -> dict:
        """
        Train both angle and QUBO weight predictors.

        Returns combined training history.
        """
        logger.info("Training AnglePredictor on %d instances...", len(instances))
        angle_history = self.angle_predictor.fit(
            instances, target_angles, epochs=angle_epochs, verbose=verbose
        )

        logger.info("Training QUBOWeightPredictor on %d instances...", len(instances))
        weight_history = self.qubo_predictor.fit(
            instances, target_weights, epochs=qubo_epochs, verbose=verbose
        )

        return {"angle": angle_history, "qubo_weight": weight_history}

    def save_models(self):
        angle_path = os.path.join(self.checkpoint_dir, f"angle_predictor_p{self.p}.pth")
        weight_path = os.path.join(self.checkpoint_dir, "qubo_weight_predictor.pth")
        self.angle_predictor.save(angle_path)
        self.qubo_predictor.save(weight_path)
        logger.info("Models saved to %s", self.checkpoint_dir)

    def load_models(self):
        angle_path = os.path.join(self.checkpoint_dir, f"angle_predictor_p{self.p}.pth")
        weight_path = os.path.join(self.checkpoint_dir, "qubo_weight_predictor.pth")
        if os.path.exists(angle_path):
            self.angle_predictor.load(angle_path)
        if os.path.exists(weight_path):
            self.qubo_predictor.load(weight_path)
