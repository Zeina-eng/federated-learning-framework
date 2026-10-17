"""
FastAPI Server for Federated Learning Framework.
Provides REST and WebSocket endpoints for real-time model training,
Byzantine defense testing, and multi-framework comparative benchmarks.
"""
import os
import json
import asyncio
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List

from fl_framework.config import (
    SimulationRequest,
    FrameworkMetricsModel,
    SecurityConfigModel,
    RoundResultModel
)
from fl_framework.server import FLCoordinatorServer
from fl_framework.data_loader import generate_synthetic_data

app = FastAPI(
    title="Federated Learning Healthcare Infrastructure API",
    version="1.0.0",
    description="Production-grade Federated Learning engine powered by PyTorch, Differential Privacy, and Byzantine-Robust Aggregation."
)

# Enable CORS for React Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint with PyTorch and hardware telemetry."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {
        "status": "healthy",
        "engine": "PyTorch Python Engine",
        "torch_version": torch.__version__,
        "device": device,
        "device_name": device_name,
        "capabilities": [
            "FedAvg", "FedProx", "Differential Privacy", "Secure Aggregation",
            "Multi-Krum", "Coordinate-wise Median", "Cosine Anomaly Filtering",
            "Model Poisoning", "Backdoor Attack", "Membership Inference Attack"
        ]
    }


@app.get("/api/datasets")
async def get_datasets():
    """Return available medical benchmark datasets with metadata."""
    datasets = []
    configs = [
        ("heart_disease", "Cleveland Heart Disease Diagnostic", "Cardiovascular clinical diagnostic indicators"),
        ("diabetes", "PIMA Indian Diabetes Dataset", "Metabolic clinical markers for type-2 diabetes risk"),
        ("breast_cancer", "Breast Cancer Wisconsin Diagnostic", "Fine needle aspirate (FNA) digitized nuclear features"),
        ("readmission", "30-Day Hospital Readmission Risk", "Inpatient clinical history, comorbidities, and stay length")
    ]
    for did, name, desc in configs:
        X, y, fnames, tname = generate_synthetic_data(did, count=20)
        datasets.append({
            "id": did,
            "name": name,
            "description": desc,
            "sampleCount": 350,
            "featureNames": fnames,
            "targetName": tname
        })
    return {"datasets": datasets}


@app.post("/api/simulate", response_model=FrameworkMetricsModel)
async def run_simulation(request: SimulationRequest):
    """
    Execute a full Federated Learning simulation.
    Runs PyTorch training synchronously and returns comprehensive experiment metrics.
    """
    try:
        custom_data_dicts = [s.model_dump() for s in request.customData] if request.customData else None
        server = FLCoordinatorServer(
            dataset_config=request.datasetConfig,
            fl_config=request.flConfig,
            security_config=request.securityConfig,
            attack_config=request.attackConfig,
            custom_data=custom_data_dicts
        )
        metrics = server.run_simulation()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmark/compare")
async def run_benchmark(request: SimulationRequest):
    """
    Execute 5-Framework Comparative Benchmark on the selected dataset:
    1. Baseline FL (Standard FedAvg, no defenses)
    2. FL + Differential Privacy (DP-FedAvg)
    3. FL + Secure Aggregation (SecAgg)
    4. FL + Byzantine Anomaly Detection
    5. Proposed Integrated Framework (DP + SecAgg + Anomaly Defense)
    """
    try:
        custom_data_dicts = [s.model_dump() for s in request.customData] if request.customData else None

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

        benchmark_results = {}
        for mode_name, sec_cfg in modes:
            server = FLCoordinatorServer(
                dataset_config=request.datasetConfig,
                fl_config=request.flConfig,
                security_config=sec_cfg,
                attack_config=request.attackConfig,
                custom_data=custom_data_dicts
            )
            res = server.run_simulation()
            benchmark_results[mode_name] = res.model_dump()

        return {"benchmarks": benchmark_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/fl-simulate")
async def websocket_simulation(websocket: WebSocket):
    """
    WebSocket endpoint for live, streaming PyTorch federated training.
    Sends each communication round's metrics directly to the React UI as it happens.
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        data_text = await websocket.receive_text()
        req_dict = json.loads(data_text)
        req = SimulationRequest(**req_dict)

        custom_data_dicts = [s.model_dump() for s in req.customData] if req.customData else None
        server = FLCoordinatorServer(
            dataset_config=req.datasetConfig,
            fl_config=req.flConfig,
            security_config=req.securityConfig,
            attack_config=req.attackConfig,
            custom_data=custom_data_dicts
        )

        def sync_on_round(round_dict: Dict[str, Any]):
            # Thread-safe dispatch to WebSocket from PyTorch simulation thread
            msg = json.dumps({"type": "ROUND_RESULT", "data": round_dict})
            asyncio.run_coroutine_threadsafe(websocket.send_text(msg), loop)

        # Run CPU/GPU bound PyTorch training in executor so WebSocket doesn't block
        metrics = await loop.run_in_executor(None, server.run_simulation, sync_on_round)

        await websocket.send_text(json.dumps({
            "type": "SIMULATION_COMPLETE",
            "data": metrics.model_dump()
        }))

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected during simulation.")
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "type": "ERROR",
                "message": str(e)
            }))
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
