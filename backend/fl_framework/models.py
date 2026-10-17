"""
PyTorch Models for Healthcare & Clinical Federated Learning.
Supports Clinical Logistic Regression and Clinical Deep Multi-Layer Perceptrons (MLP).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


class ClinicalLogisticRegression(nn.Module):
    """
    Standard linear logistic regression model for binary clinical classification.
    Interpretable baseline for clinical decision support.
    """
    def __init__(self, input_dim: int):
        super(ClinicalLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


class ClinicalMLP(nn.Module):
    """
    Deep Multi-Layer Perceptron (MLP) for complex non-linear clinical feature interactions.
    Includes Batch Normalization and Dropout for regularization.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout_rate: float = 0.2):
        super(ClinicalMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle batch size of 1 for BatchNorm1d in eval mode
        if x.size(0) == 1 and self.training:
            self.eval()
            out = self.net(x)
            self.train()
            return out
        return self.net(x)


def create_model(model_type: str, input_dim: int) -> nn.Module:
    """Factory function to instantiate models."""
    if model_type.lower() == "mlp":
        return ClinicalMLP(input_dim=input_dim)
    return ClinicalLogisticRegression(input_dim=input_dim)


def get_flattened_parameters(model: nn.Module) -> np.ndarray:
    """Extract and flatten all trainable parameters into a 1D NumPy array."""
    params = []
    for param in model.parameters():
        params.append(param.detach().cpu().numpy().flatten())
    if len(params) == 0:
        return np.array([], dtype=np.float32)
    return np.concatenate(params)


def set_flattened_parameters(model: nn.Module, flat_params: np.ndarray) -> None:
    """Load a 1D NumPy array of parameters into the PyTorch model."""
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            numel = param.numel()
            chunk = flat_params[offset:offset + numel]
            param.copy_(torch.from_numpy(chunk.reshape(param.shape)).to(param.device))
            offset += numel


def compute_proximal_loss(model: nn.Module, global_params: np.ndarray, mu: float) -> torch.Tensor:
    """
    FedProx Regularization Term: (mu / 2) * ||w - w_t||^2
    Penalizes local drift caused by client non-IID data distribution.
    """
    if mu <= 0.0 or global_params is None:
        return torch.tensor(0.0)

    prox_loss = torch.tensor(0.0)
    offset = 0
    device = next(model.parameters()).device

    for param in model.parameters():
        numel = param.numel()
        g_chunk = torch.from_numpy(global_params[offset:offset + numel].reshape(param.shape)).to(device)
        prox_loss = prox_loss + torch.sum((param - g_chunk) ** 2)
        offset += numel

    return 0.5 * mu * prox_loss
