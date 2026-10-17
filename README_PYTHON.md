# Healthcare Federated Learning Framework (Python & PyTorch Infrastructure)

A modular, production-grade **Federated Learning (FL) Framework** for clinical and healthcare analytics, featuring **PyTorch neural models, Differential Privacy (DP-SGD), Secure Aggregation (SecAgg), and Byzantine-Robust Aggregation**.

---

## 🏛️ Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │    FastAPI & WebSocket Server        │
                      │       (backend/app.py)               │
                      └──────────────────┬───────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [REST /api/simulate]          [WebSocket /ws/fl-simulate]
                         │                               │ (Live Round Streaming)
                         ▼                               ▼
           ┌────────────────────────────────────────────────────────┐
           │          FL Central Coordinator Server                 │
           │       (backend/fl_framework/server.py)                 │
           └──────────────┬──────────────────┬──────────────────────┘
                          │                  │
        Broadcast w_t     │                  │   Collect Updates w_i
                          ▼                  ▼
    ┌─────────────────────────┐          ┌─────────────────────────┐
    │  Hospital Node Client 1 │   ...    │  Hospital Node Client K │
    │   (Mayo Clinic Lab)     │          │  (Mount Sinai System)   │
    ├─────────────────────────┤          ├─────────────────────────┤
    │ • Local PyTorch SGD     │          │ • Local PyTorch SGD     │
    │ • FedProx Regularizer   │          │ • FedProx Regularizer   │
    │ • DP Gradient Clipping  │          │ • DP Gradient Clipping  │
    │ • Gaussian Noise Added  │          │ • Gaussian Noise Added  │
    └─────────────────────────┘          └─────────────────────────┘
                          │                  │
                          └──────────┬───────┘
                                     ▼
           ┌────────────────────────────────────────────────────────┐
           │        Byzantine-Robust Server Aggregation             │
           │       (backend/fl_framework/defenses.py)               │
           ├────────────────────────────────────────────────────────┤
           │ • Multi-Krum (Blanchard et al., NeurIPS 2017)          │
           │ • Coordinate-wise Median                               │
           │ • Trimmed Mean                                         │
           │ • Cosine-Similarity Anomaly Filtering                  │
           │ • Secure Aggregation (SecAgg Blinding)                 │
           └─────────────────────────┬──────────────────────────────┘
                                     ▼
           ┌────────────────────────────────────────────────────────┐
           │                Global Model Evaluation                 │
           │ • Accuracy, Precision, Recall/Sensitivity, Specificity │
           │ • F1-Score, Loss, ROC-AUC                              │
           │ • Privacy Budget Tracking (ε, δ)                       │
           │ • Attack Success Rate (ASR) & Anomaly Recall (ADR)     │
           └────────────────────────────────────────────────────────┘
```

---

## 📦 Directory Structure

```
federated-learning-framework/
├── backend/
│   ├── fl_framework/
│   │   ├── __init__.py           # Package exports
│   │   ├── config.py             # Pydantic schemas for configs & metrics
│   │   ├── models.py             # PyTorch models (Clinical Logistic & Deep MLP)
│   │   ├── data_loader.py        # Clinical datasets & Dirichlet non-IID partitions
│   │   ├── client.py             # Institutional Hospital Client with PyTorch SGD/FedProx
│   │   ├── server.py             # Central Coordinator Server & Round Orchestrator
│   │   ├── defenses.py           # DP (Moments Accountant), SecAgg, Multi-Krum, Median, Cosine
│   │   ├── attacks.py            # Model Poisoning, Data Poisoning, Backdoor, MIA Probe
│   │   └── metrics.py            # Clinical Metrics (Accuracy, F1, Sensitivity, ROC-AUC)
│   ├── app.py                    # FastAPI REST & WebSocket streaming server
│   ├── run_simulation.py         # Standalone CLI experiment runner
│   ├── run_benchmark.py          # 5-Framework comparative benchmark CLI
│   ├── test_framework.py         # Unit & integration test suite (PyTorch engine)
│   └── test_api.py               # FastAPI TestClient endpoint test suite
├── src/                          # React + TypeScript Frontend
│   ├── services/
│   │   └── flApiService.ts       # Frontend API service connecting to Python Backend
│   └── components/               # Visualizations, Topology Canvas, Node Inspector
├── requirements.txt              # Python dependencies
├── run_backend.bat               # Windows 1-click launch script
└── run_backend.sh                # Unix/macOS launch script
```

---

## 🚀 Quickstart & Installation

### 1. Install Dependencies

Ensure Python 3.10+ is installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Run Test Suite

Verify all algorithms and PyTorch components pass tests:

```bash
python backend/test_framework.py
python backend/test_api.py
```

---

## 💻 Running the Infrastructure

### Mode A: Launch the FastAPI & WebSocket Server (For React UI)

#### On Windows:
Double-click `run_backend.bat` or run:
```cmd
run_backend.bat
```

#### On Linux / macOS:
```bash
chmod +x run_backend.sh
./run_backend.sh
```

#### Or directly via Uvicorn:
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be live at `http://localhost:8000`. You can explore interactive Swagger documentation at `http://localhost:8000/docs`.

---

### Mode B: Standalone Command-Line Interface (CLI)

Run simulations directly in your terminal with custom hyperparameters:

```bash
# 1. Standard FedAvg on Heart Disease dataset across 5 hospital nodes
python backend/run_simulation.py --dataset heart_disease --rounds 10 --clients 5

# 2. FedProx with Dirichlet Non-IID skew (alpha = 0.3)
python backend/run_simulation.py --dataset diabetes --rounds 12 --clients 5 --strategy FedProx --mu 0.1 --alpha 0.3

# 3. High Differential Privacy (DP-FedAvg) with Byzantine Anomaly Defense
python backend/run_simulation.py --dataset breast_cancer --rounds 15 --clients 6 --dp --sigma 0.8 --clip 2.5 --anomaly-defense --anomaly-algo Cosine_Similarity

# 4. Stress-test against Byzantine Model Poisoning & Backdoor Attack
python backend/run_simulation.py --dataset heart_disease --rounds 12 --clients 5 --attack --malicious-ratio 0.4 --model-poisoning --backdoor --anomaly-defense --save-json experiment_results.json
```

---

### Mode C: 5-Framework Comparative Benchmark

Run the full comparative study evaluating all 5 security postures:

```bash
python backend/run_benchmark.py --dataset heart_disease --rounds 10 --clients 5 --save-json benchmark_results.json
```

Outputs a comparison table:
- **Baseline FL** (FedAvg, no defenses)
- **FL + Differential Privacy** (DP-FedAvg)
- **FL + Secure Aggregation** (SecAgg)
- **FL + Byzantine Anomaly Detection** (Multi-Krum / Cosine Filtering)
- **Proposed Integrated Framework** (DP + SecAgg + Anomaly Defense)

---

## 🔄 Using Python Engine in the React Dashboard

1. Start the Python backend using `run_backend.bat`.
2. Start the React frontend:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:5173` in your browser.
4. Notice the **Engine Switcher** in the top navbar:
   - `🐍 Python (PyTorch)` (Shows green pulsing dot when backend is connected)
   - `⚡ Browser (TS)` (In-browser TypeScript fallback)
5. When `Python (PyTorch)` is active, clicking **"Run Federated Learning Experiment"** establishes a live WebSocket connection to `ws://localhost:8000/ws/fl-simulate`.
6. Watch each communication round train in real PyTorch, streaming updates to the animated topology canvas, terminal stream, and ROC/loss convergence charts!
