"""
Federated Learning Central Aggregation Server.
Orchestrates training rounds, model broadcasting, robust aggregation, global evaluation,
and privacy/security telemetry.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Callable, Optional, Any

from .models import create_model, get_flattened_parameters, set_flattened_parameters
from .client import FLHospitalClient
from .attacks import FLAttackEngine
from .defenses import DifferentialPrivacyEngine, SecureAggregationEngine, RobustAggregator
from .metrics import compute_binary_metrics
from .data_loader import prepare_federated_loaders, HOSPITAL_NAMES
from .config import (
    DatasetConfigModel,
    FLConfigModel,
    SecurityConfigModel,
    AttackConfigModel,
    RoundResultModel,
    FrameworkMetricsModel
)


class FLCoordinatorServer:
    """Central Federated Learning Orchestration Server."""
    def __init__(
        self,
        dataset_config: DatasetConfigModel,
        fl_config: FLConfigModel,
        security_config: SecurityConfigModel,
        attack_config: AttackConfigModel,
        custom_data: Optional[List[Dict[str, Any]]] = None,
        device: Optional[torch.device] = None
    ):
        self.dataset_config = dataset_config
        self.fl_config = fl_config
        self.security_config = security_config
        self.attack_config = attack_config
        self.custom_data = custom_data

        # Hardware acceleration
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Prepare datasets and hospital partitions
        client_loaders, test_loader, input_dim, feature_names, target_name = prepare_federated_loaders(
            dataset_id=dataset_config.datasetId,
            num_clients=fl_config.numClients,
            distribution_type=dataset_config.distributionType,
            dirichlet_alpha=dataset_config.dirichletAlpha,
            batch_size=fl_config.batchSize,
            custom_samples=custom_data
        )

        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.input_dim = input_dim
        self.feature_names = feature_names
        self.target_name = target_name

        # Engines
        self.attack_engine = FLAttackEngine(attack_config)
        self.dp_engine = DifferentialPrivacyEngine(security_config)
        self.secagg_engine = SecureAggregationEngine(security_config)
        self.robust_aggregator = RobustAggregator(security_config)

        # Global PyTorch Model
        self.global_model = create_model(fl_config.modelType, input_dim).to(self.device)
        self.global_weights = get_flattened_parameters(self.global_model)

        # Instantiate Hospital Clients
        self.clients: List[FLHospitalClient] = []
        for i in range(fl_config.numClients):
            client_name = HOSPITAL_NAMES[i % len(HOSPITAL_NAMES)]
            client = FLHospitalClient(
                client_id=i,
                client_name=client_name,
                data_loader=client_loaders[i],
                input_dim=input_dim,
                fl_config=fl_config,
                security_config=security_config,
                attack_engine=self.attack_engine,
                device=self.device
            )
            self.clients.append(client)

    def evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate the global consensus model on holdout test set."""
        self.global_model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                probs = self.global_model(X).cpu().numpy().flatten()
                all_preds.extend(probs.tolist())
                all_targets.extend(y.numpy().flatten().tolist())

        return compute_binary_metrics(np.array(all_targets), np.array(all_preds))

    def run_simulation(
        self,
        on_round_complete: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> FrameworkMetricsModel:
        """
        Execute full Federated Learning simulation across communication rounds.
        Streams round results in real-time if callback provided.
        """
        history: List[RoundResultModel] = []
        total_start_time = time.time()
        num_parameters = len(self.global_weights)

        for round_idx in range(1, self.fl_config.commRounds + 1):
            round_start_time = time.time()

            # 1. Local Training on all participating hospital nodes
            client_updates = []
            client_weights = []
            is_malicious_flags = []
            client_logs = []

            for client in self.clients:
                res = client.train(self.global_weights)
                client_updates.append(res["weights"])
                client_weights.append(float(res["sampleCount"]))
                is_malicious_flags.append(res["isMalicious"])
                client_logs.append({
                    "clientId": res["clientId"],
                    "clientName": res["clientName"],
                    "isMalicious": res["isMalicious"],
                    "isDetected": False,  # Updated after aggregation
                    "localLoss": res["localLoss"],
                    "localAccuracy": res["localAccuracy"],
                    "updateNorm": res["updateNorm"],
                    "masked": res["masked"]
                })

            # 2. Server Aggregation & Byzantine Defense
            agg_weights, detected_indices, adr, fpr = self.robust_aggregator.aggregate(
                client_updates=client_updates,
                client_weights=client_weights,
                is_malicious_flags=is_malicious_flags
            )

            # Mark detected status in logs
            for log in client_logs:
                if log["clientId"] in detected_indices:
                    log["isDetected"] = True

            # 3. Update Global Model
            self.global_weights = agg_weights
            set_flattened_parameters(self.global_model, self.global_weights)

            # 4. Global Validation Metrics
            test_metrics = self.evaluate_global_model()

            # 5. Security & Privacy Metrics
            eps, delta = self.dp_engine.compute_privacy_budget(round_idx, self.fl_config.numClients)
            asr = self.attack_engine.evaluate_backdoor_asr(self.global_model, self.test_loader, self.device)
            comm_overhead_kb = self.secagg_engine.compute_overhead(self.fl_config.numClients, num_parameters)
            round_elapsed_ms = round((time.time() - round_start_time) * 1000.0, 1)

            round_result = RoundResultModel(
                round=round_idx,
                globalAccuracy=test_metrics["accuracy"],
                globalLoss=test_metrics["loss"],
                globalPrecision=test_metrics["precision"],
                globalSensitivity=test_metrics["sensitivity"],
                globalSpecificity=test_metrics["specificity"],
                globalF1=test_metrics["f1"],
                globalAUC=test_metrics["auc"],
                privacyEpsilon=eps,
                privacyDelta=delta,
                attackSuccessRate=asr,
                anomalyDetectionRate=adr,
                falsePositiveRate=fpr,
                roundTimeMs=round_elapsed_ms,
                commOverheadKB=comm_overhead_kb,
                clientLogs=client_logs
            )
            history.append(round_result)

            # Real-time streaming hook
            if on_round_complete:
                on_round_complete(round_result.model_dump())

        total_elapsed_sec = round(time.time() - total_start_time, 2)
        total_comm_mb = round(float(sum(r.commOverheadKB for r in history) / 1024.0), 3)

        last_r = history[-1]
        active_mode_name = "Baseline FL"
        if self.security_config.proposedFramework:
            active_mode_name = "Proposed Combined Framework"
        elif self.security_config.differentialPrivacy:
            active_mode_name = "FL + Differential Privacy"
        elif self.security_config.secureAggregation:
            active_mode_name = "FL + Secure Aggregation"
        elif self.security_config.anomalyDetection:
            active_mode_name = "FL + Anomaly Detection"

        return FrameworkMetricsModel(
            name=active_mode_name,
            accuracy=last_r.globalAccuracy,
            precision=last_r.globalPrecision,
            sensitivity=last_r.globalSensitivity,
            specificity=last_r.globalSpecificity,
            f1Score=last_r.globalF1,
            auc=last_r.globalAUC,
            privacyEpsilon=last_r.privacyEpsilon,
            privacyDelta=last_r.privacyDelta,
            attackSuccessRate=last_r.attackSuccessRate,
            anomalyDetectionRate=last_r.anomalyDetectionRate,
            falsePositiveRate=last_r.falsePositiveRate,
            trainingTimeSec=total_elapsed_sec,
            commOverheadMB=total_comm_mb,
            history=history
        )
