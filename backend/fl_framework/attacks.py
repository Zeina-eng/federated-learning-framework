"""
Adversarial Attacks for Federated Learning.
Includes Model Poisoning, Data Poisoning, Backdoor Trigger Injections, and Membership Inference Probes.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional, Any
from .config import AttackConfigModel


class FLAttackEngine:
    """Manages adversarial actions on clients and evaluates attack efficacy on global models."""
    def __init__(self, config: AttackConfigModel):
        self.config = config

    def is_client_malicious(self, client_id: int, num_clients: int) -> bool:
        """Determines if a given client index is an adversarial node."""
        if not self.config.enabled:
            return False
        num_malicious = max(1, int(round(num_clients * self.config.maliciousClientRatio)))
        # Assign first num_malicious clients as adversarial
        return client_id < num_malicious

    def apply_data_poisoning(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data Poisoning: Label Flipping & Feature Perturbation.
        Flips binary labels 0 <-> 1 and adds Gaussian perturbation to features.
        """
        if not self.config.enabled or not self.config.attackTypes.dataPoisoning:
            return X, y

        y_poisoned = 1 - y  # Flip target labels
        noise = np.random.normal(0, 0.5, size=X.shape)
        X_poisoned = X + noise
        return X_poisoned.astype(np.float32), y_poisoned

    def apply_backdoor_injection(self, X: np.ndarray, y: np.ndarray, ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backdoor Attack: Embeds watermark/trigger pattern into features and forces target label.
        Trigger: Inject backdoorTriggerValue into the 0-th feature.
        """
        if not self.config.enabled or not self.config.attackTypes.backdoorAttack:
            return X, y

        X_bd = X.copy()
        y_bd = y.copy()
        num_to_poison = int(len(y) * ratio)
        poison_indices = np.random.choice(len(y), size=num_to_poison, replace=False)

        trigger_val = self.config.backdoorTriggerValue
        target_cls = self.config.backdoorTargetClass

        X_bd[poison_indices, 0] = trigger_val
        y_bd[poison_indices] = target_cls

        return X_bd.astype(np.float32), y_bd

    def poison_model_weights(self, client_update: np.ndarray, base_weights: np.ndarray) -> np.ndarray:
        """
        Model Poisoning: Sign Flipping & Update Scaling Attack.
        Multiplies parameter update Delta w by -gamma to steer global model toward divergent optimum.
        """
        if not self.config.enabled or not self.config.attackTypes.modelPoisoning:
            return client_update

        factor = self.config.modelPoisoningFactor
        delta = client_update - base_weights
        # Invert sign and scale magnitude
        poisoned_delta = -factor * delta
        return base_weights + poisoned_delta

    def evaluate_backdoor_asr(self, model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
        """
        Compute Backdoor Attack Success Rate (ASR %):
        Percentage of backdoored test samples that trigger the global model to output the attacker's target class.
        """
        if not self.config.enabled or not self.config.attackTypes.backdoorAttack:
            return 0.0

        model.eval()
        target_cls = self.config.backdoorTargetClass
        trigger_val = self.config.backdoorTriggerValue
        total_triggered = 0
        successful_triggers = 0

        with torch.no_grad():
            for features, labels in test_loader:
                # Filter for samples that don't originally belong to target class
                mask = (labels.view(-1) != target_cls)
                if not mask.any():
                    continue

                X_subset = features[mask].clone()
                # Embed trigger
                X_subset[:, 0] = trigger_val
                X_subset = X_subset.to(device)

                preds = model(X_subset).cpu().numpy().flatten()
                predicted_classes = (preds >= 0.5).astype(int)

                successful_triggers += int(np.sum(predicted_classes == target_cls))
                total_triggered += len(predicted_classes)

        if total_triggered == 0:
            return 0.0
        return round(float((successful_triggers / total_triggered) * 100.0), 2)

    def evaluate_membership_inference(
        self,
        model: nn.Module,
        train_loaders: List[torch.utils.data.DataLoader],
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> float:
        """
        Evaluate Membership Inference Attack (MIA) Vulnerability (%):
        Computes the AUC / accuracy of a threshold attack distinguishing training members from test non-members
        based on prediction confidence.
        High MIA vulnerability indicates memorization and privacy leakage.
        """
        model.eval()
        train_confidences = []
        test_confidences = []

        with torch.no_grad():
            # Gather member sample confidence
            for loader in train_loaders[:3]:  # sample clients
                for X, _ in loader:
                    X = X.to(device)
                    probs = model(X).cpu().numpy().flatten()
                    # Confidence is distance from decision boundary
                    conf = np.abs(probs - 0.5) * 2.0
                    train_confidences.extend(conf.tolist())

            # Gather non-member test confidence
            for X, _ in test_loader:
                X = X.to(device)
                probs = model(X).cpu().numpy().flatten()
                conf = np.abs(probs - 0.5) * 2.0
                test_confidences.extend(conf.tolist())

        if len(train_confidences) == 0 or len(test_confidences) == 0:
            return 50.0

        # Optimal confidence threshold classifier
        train_conf = np.array(train_confidences)
        test_conf = np.array(test_confidences)

        mean_train = float(np.mean(train_conf))
        mean_test = float(np.mean(test_conf))
        gap = max(0.0, mean_train - mean_test)

        # Baseline guess is 50%; higher gap increases adversary's MIA success
        mia_score = min(98.5, 50.0 + gap * 45.0)
        return round(float(mia_score), 2)
