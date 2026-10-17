"""
Federated Learning Hospital Node (Client).
Simulates an institutional hospital participating in federated training.
Handles local training, FedProx regularization, DP gradient clipping/noise, and adversarial behavior.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional, Any

from .models import create_model, get_flattened_parameters, set_flattened_parameters, compute_proximal_loss
from .attacks import FLAttackEngine
from .defenses import DifferentialPrivacyEngine
from .config import FLConfigModel, SecurityConfigModel


class FLHospitalClient:
    """Institutional Hospital Client participating in Federated Learning."""
    def __init__(
        self,
        client_id: int,
        client_name: str,
        data_loader: DataLoader,
        input_dim: int,
        fl_config: FLConfigModel,
        security_config: SecurityConfigModel,
        attack_engine: FLAttackEngine,
        device: torch.device
    ):
        self.client_id = client_id
        self.client_name = client_name
        self.data_loader = data_loader
        self.input_dim = input_dim
        self.fl_config = fl_config
        self.security_config = security_config
        self.attack_engine = attack_engine
        self.device = device
        self.sample_count = len(data_loader.dataset)

        # Flag whether this node is corrupted by adversary
        self.is_malicious = self.attack_engine.is_client_malicious(client_id, fl_config.numClients)

        # Local PyTorch Model
        self.model = create_model(fl_config.modelType, input_dim).to(device)
        self.dp_engine = DifferentialPrivacyEngine(security_config)

    def train(self, global_weights: np.ndarray) -> Dict[str, Any]:
        """
        Execute local training round:
        1. Set model weights to received global weights.
        2. Perform E epochs of SGD / Adam optimization.
        3. If FedProx is selected, include (mu/2) * ||w - w_global||^2 penalty.
        4. If DP is enabled, clip local update norm and inject calibrated noise.
        5. If client is adversarial, execute model poisoning / backdoor injection.
        """
        set_flattened_parameters(self.model, global_weights)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.fl_config.learningRate)
        criterion = nn.BCELoss()

        total_loss = 0.0
        batches = 0
        correct_preds = 0
        total_samples = 0

        for epoch in range(self.fl_config.localEpochs):
            for X_batch, y_batch in self.data_loader:
                X_batch_np = X_batch.numpy()
                y_batch_np = y_batch.numpy().flatten()

                # Adversarial Data Poisoning & Backdoor Injections
                if self.is_malicious:
                    X_batch_np, y_batch_np = self.attack_engine.apply_data_poisoning(X_batch_np, y_batch_np)
                    X_batch_np, y_batch_np = self.attack_engine.apply_backdoor_injection(X_batch_np, y_batch_np)

                X_t = torch.tensor(X_batch_np, dtype=torch.float32).to(self.device)
                y_t = torch.tensor(y_batch_np, dtype=torch.float32).unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_t)
                loss = criterion(outputs, y_t)

                # FedProx Proximal Loss Regularizer
                if self.fl_config.aggregationStrategy == "FedProx":
                    prox_loss = compute_proximal_loss(self.model, global_weights, self.fl_config.fedProxMu)
                    loss = loss + prox_loss.to(self.device)

                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                batches += 1

                preds = (outputs.detach().cpu().numpy().flatten() >= 0.5).astype(int)
                correct_preds += int(np.sum(preds == y_batch_np))
                total_samples += len(y_batch_np)

        local_loss = round(total_loss / max(1, batches), 4)
        local_acc = round((correct_preds / max(1, total_samples)) * 100.0, 2)

        # Extract updated parameter vector
        updated_weights = get_flattened_parameters(self.model)

        # Apply Differential Privacy (gradient norm clipping & noise addition)
        if self.security_config.differentialPrivacy:
            updated_weights = self.dp_engine.clip_update(
                updated_weights,
                clipping_bound=self.security_config.dpClippingBound
            )
            updated_weights = self.dp_engine.add_noise(
                updated_weights,
                clipping_bound=self.security_config.dpClippingBound,
                noise_multiplier=self.security_config.dpNoiseMultiplier,
                num_clients=self.fl_config.numClients
            )

        # Apply Adversarial Model Poisoning (sign flipping / update scaling)
        if self.is_malicious:
            updated_weights = self.attack_engine.poison_model_weights(
                updated_weights,
                base_weights=global_weights
            )

        update_norm = round(float(np.linalg.norm(updated_weights - global_weights)), 4)

        return {
            "clientId": self.client_id,
            "clientName": self.client_name,
            "isMalicious": self.is_malicious,
            "weights": updated_weights,
            "sampleCount": self.sample_count,
            "localLoss": local_loss,
            "localAccuracy": local_acc,
            "updateNorm": update_norm,
            "masked": self.security_config.secureAggregation
        }
