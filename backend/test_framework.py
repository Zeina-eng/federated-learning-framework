"""
Unit and Integration Tests for Python Federated Learning Framework.
Verifies PyTorch models, Dirichlet non-IID partitioning, FedAvg/FedProx,
Differential Privacy, Byzantine Aggregation, and Attacks.
"""
import unittest
import numpy as np
import torch

from fl_framework.config import (
    DatasetConfigModel,
    FLConfigModel,
    SecurityConfigModel,
    AttackConfigModel
)
from fl_framework.models import (
    ClinicalLogisticRegression,
    ClinicalMLP,
    get_flattened_parameters,
    set_flattened_parameters,
    compute_proximal_loss
)
from fl_framework.data_loader import (
    generate_synthetic_data,
    partition_dataset,
    prepare_federated_loaders
)
from fl_framework.defenses import (
    DifferentialPrivacyEngine,
    SecureAggregationEngine,
    RobustAggregator
)
from fl_framework.attacks import FLAttackEngine
from fl_framework.server import FLCoordinatorServer


class TestFLFramework(unittest.TestCase):

    def test_models_and_parameter_flattening(self):
        input_dim = 8
        model = ClinicalLogisticRegression(input_dim)
        flat = get_flattened_parameters(model)
        # 8 weights + 1 bias = 9 parameters
        self.assertEqual(len(flat), 9)

        # Modify and restore
        new_flat = flat + 0.5
        set_flattened_parameters(model, new_flat)
        restored = get_flattened_parameters(model)
        np.testing.assert_allclose(new_flat, restored, atol=1e-5)

        # Test MLP
        mlp = ClinicalMLP(input_dim)
        flat_mlp = get_flattened_parameters(mlp)
        self.assertGreater(len(flat_mlp), 20)

    def test_fedprox_loss(self):
        model = ClinicalLogisticRegression(4)
        flat = get_flattened_parameters(model)
        # Difference with itself should be zero
        prox_zero = compute_proximal_loss(model, flat, mu=0.1)
        self.assertAlmostEqual(prox_zero.item(), 0.0, places=5)

        # Shifted weights should produce positive penalty
        shifted_flat = flat + 1.0
        prox_pos = compute_proximal_loss(model, shifted_flat, mu=0.5)
        self.assertGreater(prox_pos.item(), 0.0)

    def test_data_generation_and_dirichlet(self):
        X, y, fnames, tname = generate_synthetic_data("heart_disease", count=100)
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(len(y), 100)

        # Non-IID Dirichlet split
        splits = partition_dataset(X, y, num_clients=4, distribution_type="NON_IID_DIRICHLET", dirichlet_alpha=0.5)
        self.assertEqual(len(splits), 4)
        total_samples = sum(len(s[1]) for s in splits)
        self.assertEqual(total_samples, 100)

    def test_differential_privacy(self):
        sec_cfg = SecurityConfigModel(differentialPrivacy=True, dpNoiseMultiplier=0.5, dpClippingBound=2.0)
        dp = DifferentialPrivacyEngine(sec_cfg)

        update = np.array([5.0, 5.0, 5.0])  # Norm is sqrt(75) ~ 8.66
        clipped = dp.clip_update(update, clipping_bound=2.0)
        self.assertAlmostEqual(np.linalg.norm(clipped), 2.0, places=4)

        # Check privacy budget calculation
        eps, delta = dp.compute_privacy_budget(round_num=5, num_clients=5)
        self.assertGreater(eps, 0.0)
        self.assertEqual(delta, 1e-5)

    def test_robust_aggregation_cosine_and_krum(self):
        sec_cfg = SecurityConfigModel(anomalyDetection=True, anomalyAlgorithm="Cosine_Similarity", anomalyThreshold=0.5)
        aggregator = RobustAggregator(sec_cfg)

        benign1 = np.array([1.0, 1.0, 1.0])
        benign2 = np.array([1.1, 0.9, 1.0])
        benign3 = np.array([0.9, 1.1, 1.0])
        malicious = np.array([-10.0, -10.0, -10.0])  # Opposite sign, large magnitude

        updates = [benign1, benign2, benign3, malicious]
        weights = [10.0, 10.0, 10.0, 10.0]
        is_mal = [False, False, False, True]

        agg, detected, adr, fpr = aggregator.aggregate(updates, weights, is_mal)
        self.assertIn(3, detected, "Malicious client index 3 should be detected by Cosine filtering")
        self.assertEqual(adr, 100.0)
        self.assertEqual(fpr, 0.0)
        self.assertGreater(agg[0], 0.0, "Aggregated weights should remain positive")

        # Test Multi-Krum
        sec_cfg.anomalyAlgorithm = "Multi-Krum"
        agg_krum, detected_krum, _, _ = aggregator.aggregate(updates, weights, is_mal)
        self.assertIn(3, detected_krum)

    def test_end_to_end_simulation(self):
        dataset_cfg = DatasetConfigModel(datasetId="diabetes", numClients=3)
        fl_cfg = FLConfigModel(numClients=3, commRounds=2, localEpochs=1, learningRate=0.05)
        sec_cfg = SecurityConfigModel(proposedFramework=True, differentialPrivacy=True, secureAggregation=True, anomalyDetection=True)
        atk_cfg = AttackConfigModel(enabled=True, maliciousClientRatio=0.33)

        server = FLCoordinatorServer(
            dataset_config=dataset_cfg,
            fl_config=fl_cfg,
            security_config=sec_cfg,
            attack_config=atk_cfg
        )

        rounds_observed = []
        metrics = server.run_simulation(on_round_complete=lambda r: rounds_observed.append(r["round"]))

        self.assertEqual(len(rounds_observed), 2)
        self.assertEqual(metrics.accuracy, metrics.history[-1].globalAccuracy)
        self.assertGreater(metrics.accuracy, 40.0)
        self.assertGreater(metrics.privacyEpsilon, 0.0)


if __name__ == "__main__":
    unittest.main()
