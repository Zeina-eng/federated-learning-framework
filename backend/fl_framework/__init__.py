"""
Federated Learning Framework for Healthcare and Secure Distributed Analytics.
"""
from .server import FLCoordinatorServer
from .client import FLHospitalClient
from .attacks import FLAttackEngine
from .defenses import DifferentialPrivacyEngine, SecureAggregationEngine, RobustAggregator
from .data_loader import prepare_federated_loaders, generate_synthetic_data, partition_dataset
from .models import ClinicalLogisticRegression, ClinicalMLP, create_model
from .metrics import compute_binary_metrics, compute_roc_auc
from .config import (
    DatasetConfigModel,
    FLConfigModel,
    SecurityConfigModel,
    AttackConfigModel,
    RoundResultModel,
    FrameworkMetricsModel,
    SimulationRequest
)

__all__ = [
    "FLCoordinatorServer",
    "FLHospitalClient",
    "FLAttackEngine",
    "DifferentialPrivacyEngine",
    "SecureAggregationEngine",
    "RobustAggregator",
    "prepare_federated_loaders",
    "generate_synthetic_data",
    "partition_dataset",
    "ClinicalLogisticRegression",
    "ClinicalMLP",
    "create_model",
    "compute_binary_metrics",
    "compute_roc_auc",
    "DatasetConfigModel",
    "FLConfigModel",
    "SecurityConfigModel",
    "AttackConfigModel",
    "RoundResultModel",
    "FrameworkMetricsModel",
    "SimulationRequest"
]
