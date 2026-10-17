"""
Standalone Python CLI to execute Federated Learning experiments using PyTorch.
Usage:
    python run_simulation.py --dataset heart_disease --rounds 10 --clients 5 --strategy FedAvg
    python run_simulation.py --dataset diabetes --rounds 12 --clients 5 --dp --anomaly-defense
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
    parser = argparse.ArgumentParser(description="PyTorch Federated Learning CLI Experiment Runner")
    parser.add_argument("--dataset", type=str, default="heart_disease", choices=["heart_disease", "diabetes", "breast_cancer", "readmission"], help="Dataset to evaluate")
    parser.add_argument("--rounds", type=int, default=10, help="Number of communication rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of hospital clients")
    parser.add_argument("--strategy", type=str, default="FedAvg", choices=["FedAvg", "FedProx"], help="Aggregation strategy")
    parser.add_argument("--mu", type=float, default=0.05, help="FedProx proximal mu penalty")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for non-IID data distribution")
    parser.add_argument("--dp", action="store_true", help="Enable Differential Privacy")
    parser.add_argument("--sigma", type=float, default=0.5, help="DP noise multiplier sigma")
    parser.add_argument("--clip", type=float, default=3.0, help="DP gradient clipping bound C")
    parser.add_argument("--secagg", action="store_true", help="Enable Secure Aggregation (SecAgg)")
    parser.add_argument("--anomaly-defense", action="store_true", help="Enable Byzantine Anomaly Defense")
    parser.add_argument("--anomaly-algo", type=str, default="Cosine_Similarity", choices=["Cosine_Similarity", "Krum", "Multi-Krum", "Median", "Trimmed_Mean"], help="Defense algorithm")
    parser.add_argument("--attack", action="store_true", help="Enable Adversarial Attack")
    parser.add_argument("--malicious-ratio", type=float, default=0.2, help="Ratio of malicious clients")
    parser.add_argument("--model-poisoning", action="store_true", help="Enable Model Poisoning (sign flipping)")
    parser.add_argument("--backdoor", action="store_true", help="Enable Backdoor Trigger Attack")
    parser.add_argument("--save-json", type=str, default=None, help="Save final results to JSON file")

    args = parser.parse_args()

    print("=" * 70)
    print("🏥 FEDERATED LEARNING HEALTHCARE FRAMEWORK (PyTorch Engine)")
    print("=" * 70)
    print(f"Dataset:         {args.dataset}")
    print(f"Institutions:    {args.clients} hospital nodes")
    print(f"Comm Rounds:     {args.rounds}")
    print(f"FL Strategy:     {args.strategy} (mu={args.mu})")
    print(f"Non-IID Dir α:   {args.alpha}")
    print(f"Security:        DP={args.dp} | SecAgg={args.secagg} | Anomaly Defense={args.anomaly_defense}")
    print(f"Adversarial:     Enabled={args.attack} (Malicious Ratio={args.malicious_ratio})")
    print("-" * 70)

    dataset_cfg = DatasetConfigModel(
        datasetId=args.dataset,
        numClients=args.clients,
        distributionType="NON_IID_DIRICHLET",
        dirichletAlpha=args.alpha
    )

    fl_cfg = FLConfigModel(
        numClients=args.clients,
        commRounds=args.rounds,
        localEpochs=3,
        learningRate=0.02,
        aggregationStrategy=args.strategy,
        fedProxMu=args.mu
    )

    security_cfg = SecurityConfigModel(
        baselineFL=not (args.dp or args.secagg or args.anomaly_defense),
        differentialPrivacy=args.dp,
        secureAggregation=args.secagg,
        anomalyDetection=args.anomaly_defense,
        proposedFramework=(args.dp and args.secagg and args.anomaly_defense),
        dpNoiseMultiplier=args.sigma,
        dpClippingBound=args.clip,
        anomalyAlgorithm=args.anomaly_algo
    )

    attack_cfg = AttackConfigModel(
        enabled=args.attack,
        maliciousClientRatio=args.malicious_ratio,
        attackTypes={
            "modelPoisoning": args.model_poisoning or args.attack,
            "dataPoisoning": False,
            "backdoorAttack": args.backdoor,
            "membershipInference": True
        }
    )

    server = FLCoordinatorServer(
        dataset_config=dataset_cfg,
        fl_config=fl_cfg,
        security_config=security_cfg,
        attack_config=attack_cfg
    )

    def print_round(r: dict):
        det_str = f" | ADR: {r['anomalyDetectionRate']}%" if args.anomaly_defense else ""
        asr_str = f" | ASR: {r['attackSuccessRate']}%" if args.attack else ""
        dp_str = f" | ε: {r['privacyEpsilon']}" if args.dp else ""
        print(f"[Round {r['round']:2d}/{args.rounds}] Acc: {r['globalAccuracy']:5.2f}% | Loss: {r['globalLoss']:.4f} | F1: {r['globalF1']:5.2f}% | AUC: {r['globalAUC']:.3f}{dp_str}{det_str}{asr_str} ({r['roundTimeMs']}ms)")

    results = server.run_simulation(on_round_complete=print_round)

    print("-" * 70)
    print("🏆 FINAL RESULTS:")
    print(f"Accuracy:         {results.accuracy}%")
    print(f"F1-Score:         {results.f1Score}%")
    print(f"Sensitivity:      {results.sensitivity}%")
    print(f"Specificity:      {results.specificity}%")
    print(f"AUC-ROC:          {results.auc}")
    if args.dp:
        print(f"Privacy Budget:   (ε = {results.privacyEpsilon}, δ = {results.privacyDelta})")
    if args.anomaly_defense:
        print(f"Anomaly Recall:   {results.anomalyDetectionRate}% (FPR: {results.falsePositiveRate}%)")
    if args.attack:
        print(f"Attack Success:   {results.attackSuccessRate}%")
    print(f"Comm Overhead:    {results.commOverheadMB} MB")
    print(f"Training Time:    {results.trainingTimeSec} s")
    print("=" * 70)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results.model_dump(), f, indent=2)
        print(f"Saved experiment results to {args.save_json}")


if __name__ == "__main__":
    main()
