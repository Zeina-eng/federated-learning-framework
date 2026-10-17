"""
Comparative Benchmark Script for 5 Federated Learning Configurations:
1. Baseline FL (Standard FedAvg)
2. FL + Differential Privacy (DP-FedAvg)
3. FL + Secure Aggregation (SecAgg)
4. FL + Byzantine Anomaly Detection
5. Proposed Integrated Framework (DP + SecAgg + Anomaly Defense)
"""
import sys
import os
import argparse
import json

# Ensure UTF-8 output encoding across Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fl_framework.config import (
    DatasetConfigModel,
    FLConfigModel,
    SecurityConfigModel,
    AttackConfigModel
)
from fl_framework.server import FLCoordinatorServer


def main():
    parser = argparse.ArgumentParser(description="5-Framework Comparative FL Benchmark")
    parser.add_argument("--dataset", type=str, default="heart_disease", help="Dataset ID")
    parser.add_argument("--rounds", type=int, default=10, help="Number of communication rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of hospital clients")
    parser.add_argument("--save-json", type=str, default="benchmark_results.json", help="Output file")

    args = parser.parse_args()

    print("=" * 85)
    print("🔬 RUNNING 5-FRAMEWORK FEDERATED LEARNING COMPARATIVE BENCHMARK")
    print(f"Dataset: {args.dataset} | Clients: {args.clients} | Rounds: {args.rounds}")
    print("=" * 85)

    dataset_cfg = DatasetConfigModel(
        datasetId=args.dataset,
        numClients=args.clients,
        distributionType="NON_IID_DIRICHLET",
        dirichletAlpha=0.5
    )

    fl_cfg = FLConfigModel(
        numClients=args.clients,
        commRounds=args.rounds,
        localEpochs=3,
        learningRate=0.02,
        aggregationStrategy="FedAvg"
    )

    # Standard attack profile: 20% Byzantine nodes doing model poisoning
    attack_cfg = AttackConfigModel(
        enabled=True,
        maliciousClientRatio=0.2,
        attackTypes={
            "modelPoisoning": True,
            "dataPoisoning": False,
            "backdoorAttack": False,
            "membershipInference": True
        },
        modelPoisoningFactor=3.0
    )

    modes = [
        ("Baseline FL", SecurityConfigModel(
            baselineFL=True, differentialPrivacy=False, secureAggregation=False, anomalyDetection=False, proposedFramework=False,
            dpNoiseMultiplier=0.0, dpClippingBound=5.0, dpTargetDelta=1e-5, secAggKeySize=0, secAggProtocol="DH_Masking",
            anomalyAlgorithm="Cosine_Similarity", anomalyThreshold=0.5
        )),
        ("FL + Differential Privacy", SecurityConfigModel(
            baselineFL=False, differentialPrivacy=True, secureAggregation=False, anomalyDetection=False, proposedFramework=False,
            dpNoiseMultiplier=0.6, dpClippingBound=3.0, dpTargetDelta=1e-5, secAggKeySize=0, secAggProtocol="DH_Masking",
            anomalyAlgorithm="Cosine_Similarity", anomalyThreshold=0.5
        )),
        ("FL + Secure Aggregation", SecurityConfigModel(
            baselineFL=False, differentialPrivacy=False, secureAggregation=True, anomalyDetection=False, proposedFramework=False,
            dpNoiseMultiplier=0.0, dpClippingBound=5.0, dpTargetDelta=1e-5, secAggKeySize=256, secAggProtocol="DH_Masking",
            anomalyAlgorithm="Cosine_Similarity", anomalyThreshold=0.5
        )),
        ("FL + Anomaly Detection", SecurityConfigModel(
            baselineFL=False, differentialPrivacy=False, secureAggregation=False, anomalyDetection=True, proposedFramework=False,
            dpNoiseMultiplier=0.0, dpClippingBound=5.0, dpTargetDelta=1e-5, secAggKeySize=0, secAggProtocol="DH_Masking",
            anomalyAlgorithm="Cosine_Similarity", anomalyThreshold=0.5
        )),
        ("Proposed Combined Framework", SecurityConfigModel(
            baselineFL=False, differentialPrivacy=True, secureAggregation=True, anomalyDetection=True, proposedFramework=True,
            dpNoiseMultiplier=0.5, dpClippingBound=3.0, dpTargetDelta=1e-5, secAggKeySize=256, secAggProtocol="DH_Masking",
            anomalyAlgorithm="Cosine_Similarity", anomalyThreshold=0.5
        ))
    ]

    benchmark_data = {}

    for name, sec_cfg in modes:
        print(f"\n▶ Evaluating: {name}...")
        server = FLCoordinatorServer(
            dataset_config=dataset_cfg,
            fl_config=fl_cfg,
            security_config=sec_cfg,
            attack_config=attack_cfg
        )
        res = server.run_simulation()
        benchmark_data[name] = res.model_dump()
        print(f"  ✓ Finished: Accuracy={res.accuracy}% | F1={res.f1Score}% | ADR={res.anomalyDetectionRate}% | ε={res.privacyEpsilon}")

    print("\n" + "=" * 85)
    print("📊 COMPARATIVE BENCHMARK SUMMARY TABLE")
    print("=" * 85)
    header = f"{'Framework Variant':<30} | {'Accuracy':<9} | {'F1-Score':<9} | {'AUC':<7} | {'Privacy ε':<9} | {'ADR %':<7} | {'Overhead'}"
    print(header)
    print("-" * 85)

    for name, data in benchmark_data.items():
        row = f"{name:<30} | {data['accuracy']:>8.2f}% | {data['f1Score']:>8.2f}% | {data['auc']:>7.3f} | {data['privacyEpsilon']:>9.2f} | {data['anomalyDetectionRate']:>6.1f}% | {data['commOverheadMB']:>6.2f} MB"
        print(row)
    print("=" * 85)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(benchmark_data, f, indent=2)
        print(f"\nSaved benchmark results to {args.save_json}")


if __name__ == "__main__":
    main()
