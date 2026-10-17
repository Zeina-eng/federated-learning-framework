"""
Healthcare Privacy-Preserving Federated Learning Framework
Streamlit Web Application & Interactive Benchmark Dashboard.
Powered by PyTorch, Differential Privacy, Secure Aggregation, and Byzantine Defense.
"""
import sys
import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch

# Ensure the backend directory is in the Python module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fl_framework.server import FLCoordinatorServer
from fl_framework.config import (
    DatasetConfigModel,
    FLConfigModel,
    SecurityConfigModel,
    AttackConfigModel,
    FrameworkMetricsModel
)
from fl_framework.data_loader import generate_synthetic_data, HOSPITAL_NAMES

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Privacy-Preserving FL Framework",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics, glassmorphism cards, and metric styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.04) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(16, 185, 129, 0.04) 0%, transparent 40%),
                    #0b0f19;
        color: #f1f5f9;
    }
    
    /* Header card */
    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 15px rgba(6, 182, 212, 0.1);
    }
    
    .metric-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        background: rgba(6, 182, 212, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    /* Terminal output */
    .terminal-box {
        background-color: #020617;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #94a3b8;
        height: 240px;
        overflow-y: auto;
        line-height: 1.6;
    }
    
    /* Highlight pill */
    .status-pill {
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
    }
    .status-benign { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .status-malicious { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .status-isolated { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Session State Initialization & Presets
# -----------------------------------------------------------------------------
if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = None

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

if "preset_loaded" not in st.session_state:
    st.session_state.preset_loaded = "proposed_full"

def apply_preset(preset_key: str):
    st.session_state.preset_loaded = preset_key
    if preset_key == "clinical_standard":
        st.session_state.dp_enabled = False
        st.session_state.secagg_enabled = False
        st.session_state.anomaly_enabled = False
        st.session_state.attack_enabled = False
    elif preset_key == "high_privacy":
        st.session_state.dp_enabled = True
        st.session_state.dp_noise = 1.0
        st.session_state.dp_clip = 2.0
        st.session_state.secagg_enabled = True
        st.session_state.anomaly_enabled = False
        st.session_state.attack_enabled = False
    elif preset_key == "byzantine_attack":
        st.session_state.dp_enabled = False
        st.session_state.secagg_enabled = False
        st.session_state.anomaly_enabled = True
        st.session_state.anomaly_algo = "Cosine_Similarity"
        st.session_state.attack_enabled = True
        st.session_state.malicious_ratio = 0.4
        st.session_state.attack_model_poisoning = True
        st.session_state.attack_backdoor = True
    elif preset_key == "proposed_full":
        st.session_state.dp_enabled = True
        st.session_state.dp_noise = 0.5
        st.session_state.dp_clip = 3.0
        st.session_state.secagg_enabled = True
        st.session_state.anomaly_enabled = True
        st.session_state.anomaly_algo = "Cosine_Similarity"
        st.session_state.attack_enabled = True
        st.session_state.malicious_ratio = 0.2
        st.session_state.attack_model_poisoning = True
        st.session_state.attack_backdoor = False

# Ensure defaults exist
if "dp_enabled" not in st.session_state:
    apply_preset("proposed_full")


# -----------------------------------------------------------------------------
# 3. Sidebar: Configuration & Hyperparameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏥 FL Framework Controls")
    st.caption("Configure clinical datasets, hyperparameters, privacy defenses, and attack profiles.")

    # Preset Quick Load Buttons
    st.markdown("**⚡ Quick Presets**")
    p_cols = st.columns(2)
    if p_cols[0].button("Standard Baseline", use_container_width=True):
        apply_preset("clinical_standard")
    if p_cols[1].button("High Privacy DP", use_container_width=True):
        apply_preset("high_privacy")
    if p_cols[0].button("Byzantine Test", use_container_width=True):
        apply_preset("byzantine_attack")
    if p_cols[1].button("Proposed Full", use_container_width=True):
        apply_preset("proposed_full")

    st.divider()

    # Section 1: Dataset Settings
    st.markdown("#### 1. Dataset & Nodes")
    dataset_options = {
        "heart_disease": "Cleveland Heart Disease Diagnostic",
        "diabetes": "PIMA Indian Diabetes Dataset",
        "breast_cancer": "Breast Cancer Wisconsin Diagnostic",
        "readmission": "30-Day Hospital Readmission Risk",
        "custom": "Upload Custom CSV File"
    }
    selected_dataset_key = st.selectbox(
        "Clinical Dataset",
        options=list(dataset_options.keys()),
        format_func=lambda k: dataset_options[k]
    )

    custom_samples = None
    if selected_dataset_key == "custom":
        uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            target_col = st.selectbox("Target Label Column", options=df.columns)
            feature_cols = [c for c in df.columns if c != target_col and np.issubdtype(df[c].dtype, np.number)]
            if len(feature_cols) >= 2:
                custom_samples = []
                for idx, row in df.iterrows():
                    custom_samples.append({
                        "id": idx + 1,
                        "features": [float(row[c]) for c in feature_cols],
                        "label": int(row[target_col])
                    })
                st.success(f"Parsed {len(custom_samples)} clinical records ({len(feature_cols)} features).")
            else:
                st.warning("CSV must contain at least 2 numeric feature columns.")

    num_clients = st.slider("Hospital Nodes (Clients)", min_value=2, max_value=10, value=5)
    distribution_type = st.selectbox(
        "Data Distribution",
        options=["NON_IID_DIRICHLET", "NON_IID_LABEL_SKEW", "IID"],
        help="Dirichlet Non-IID accurately simulates realistic cross-hospital patient heterogeneity."
    )
    dirichlet_alpha = 0.5
    if distribution_type == "NON_IID_DIRICHLET":
        dirichlet_alpha = st.slider(
            "Dirichlet Alpha (α)",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="Lower alpha means extreme data skew (high non-IID); higher alpha approaches IID."
        )

    st.divider()

    # Section 2: FL Hyperparameters
    st.markdown("#### 2. Federated Optimization")
    comm_rounds = st.slider("Communication Rounds (T)", min_value=2, max_value=30, value=12)
    local_epochs = st.slider("Local Epochs (E)", min_value=1, max_value=5, value=3)
    learning_rate = st.select_slider(
        "Learning Rate (η)",
        options=[0.005, 0.01, 0.02, 0.05, 0.1],
        value=0.02
    )
    agg_strategy = st.radio("Aggregation Strategy", options=["FedAvg", "FedProx"], horizontal=True)
    fedprox_mu = 0.05
    if agg_strategy == "FedProx":
        fedprox_mu = st.slider("FedProx Proximal Term (μ)", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    model_type = st.radio(
        "Neural Model Architecture",
        options=["logistic", "mlp"],
        format_func=lambda m: "Clinical Logistic Regression" if m == "logistic" else "Clinical Deep MLP (BatchNorm + Dropout)",
        horizontal=False
    )

    st.divider()

    # Section 3: Privacy & Security Defenses
    st.markdown("#### 3. Privacy & Defenses")
    dp_enabled = st.checkbox("Differential Privacy (DP-SGD)", key="dp_enabled")
    if dp_enabled:
        dp_noise = st.slider("Noise Multiplier (σ)", 0.1, 2.0, key="dp_noise", step=0.1)
        dp_clip = st.slider("Gradient Clip Bound (C)", 1.0, 10.0, key="dp_clip", step=0.5)
    else:
        dp_noise = 0.0
        dp_clip = 5.0

    secagg_enabled = st.checkbox("Secure Aggregation (SecAgg Blinding)", key="secagg_enabled")

    anomaly_enabled = st.checkbox("Byzantine-Robust Defense", key="anomaly_enabled")
    anomaly_algo = "Cosine_Similarity"
    anomaly_threshold = 0.5
    if anomaly_enabled:
        anomaly_algo = st.selectbox(
            "Defense Algorithm",
            options=["Cosine_Similarity", "Multi-Krum", "Median", "Trimmed_Mean"],
            key="anomaly_algo"
        )
        if anomaly_algo == "Cosine_Similarity":
            anomaly_threshold = st.slider("Cosine Rejection Threshold", -0.5, 0.9, 0.5, step=0.05)

    st.divider()

    # Section 4: Adversarial Attacks
    st.markdown("#### 4. Adversarial Attack Profile")
    attack_enabled = st.checkbox("Enable Adversarial Attacks", key="attack_enabled")
    if attack_enabled:
        malicious_ratio = st.slider("Malicious Node Ratio", 0.1, 0.5, key="malicious_ratio", step=0.05)
        st.markdown("**Attack Vectors**")
        atk_model = st.checkbox("Model Poisoning (Sign Flipping)", key="attack_model_poisoning")
        atk_data = st.checkbox("Data Poisoning (Label Flipping)", value=False)
        atk_backdoor = st.checkbox("Backdoor Trigger Attack", key="attack_backdoor")
        atk_mia = st.checkbox("Membership Inference Probe", value=True)
        poison_factor = 3.0
        if atk_model:
            poison_factor = st.slider("Poison Scale Factor (γ)", 1.5, 6.0, 3.0, 0.5)
    else:
        malicious_ratio = 0.0
        atk_model = False
        atk_data = False
        atk_backdoor = False
        atk_mia = False
        poison_factor = 1.0


# -----------------------------------------------------------------------------
# 4. Header & Hero Telemetry
# -----------------------------------------------------------------------------
device_str = "CUDA (GPU Acceleration)" if torch.cuda.is_available() else "CPU"
torch_ver = torch.__version__

st.markdown(f"""
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 style="font-size: 26px; font-weight: 800; margin: 0; background: linear-gradient(90deg, #38bdf8, #2dd4bf, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🏥 Healthcare Privacy-Preserving Federated Learning
            </h1>
            <p style="font-size: 13px; color: #94a3b8; margin: 6px 0 0 0;">
                Clinical Multi-Institutional Model Training with Differential Privacy, Secure Aggregation, and Byzantine Anomaly Defense.
            </p>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="metric-badge">⚡ PyTorch {torch_ver}</span>
            <span class="metric-badge">🖥️ {device_str}</span>
            <span class="metric-badge">🔒 RDP Accountant Active</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Training Controls & Simulation Execution
# -----------------------------------------------------------------------------
col_run, col_bench = st.columns([2, 1])

run_button = col_run.button(
    "🚀 Run Federated Learning Experiment",
    type="primary",
    use_container_width=True
)

bench_button = col_bench.button(
    "⚡ Run 5-Framework Comparative Benchmark",
    use_container_width=True
)

# Container for live progress and terminal log output
progress_container = st.empty()
log_container = st.empty()

# Execute Single Simulation
if run_button:
    # Build Configuration Models
    dataset_cfg = DatasetConfigModel(
        datasetId=selected_dataset_key,
        targetVariable="Target Class",
        numClients=num_clients,
        distributionType=distribution_type,
        dirichletAlpha=dirichlet_alpha
    )

    fl_cfg = FLConfigModel(
        numClients=num_clients,
        commRounds=comm_rounds,
        localEpochs=local_epochs,
        learningRate=learning_rate,
        aggregationStrategy=agg_strategy,
        fedProxMu=fedprox_mu,
        modelType=model_type
    )

    is_proposed = dp_enabled and secagg_enabled and anomaly_enabled
    security_cfg = SecurityConfigModel(
        baselineFL=not (dp_enabled or secagg_enabled or anomaly_enabled),
        differentialPrivacy=dp_enabled,
        secureAggregation=secagg_enabled,
        anomalyDetection=anomaly_enabled,
        proposedFramework=is_proposed,
        dpNoiseMultiplier=dp_noise,
        dpClippingBound=dp_clip,
        dpTargetDelta=1e-5,
        secAggKeySize=256,
        anomalyAlgorithm=anomaly_algo,
        anomalyThreshold=anomaly_threshold
    )

    attack_cfg = AttackConfigModel(
        enabled=attack_enabled,
        maliciousClientRatio=malicious_ratio,
        attackTypes={
            "modelPoisoning": atk_model,
            "dataPoisoning": atk_data,
            "backdoorAttack": atk_backdoor,
            "membershipInference": atk_mia
        },
        modelPoisoningFactor=poison_factor,
        backdoorTargetClass=1,
        backdoorTriggerValue=999.0
    )

    server = FLCoordinatorServer(
        dataset_config=dataset_cfg,
        fl_config=fl_cfg,
        security_config=security_cfg,
        attack_config=attack_cfg,
        custom_data=custom_samples
    )

    terminal_logs = []
    terminal_logs.append(f"[{time.strftime('%X')}] 🚀 Initializing PyTorch FL Server on {device_str}...")
    terminal_logs.append(f"[{time.strftime('%X')}] Dataset: {dataset_options[selected_dataset_key]} | Nodes: {num_clients} | Rounds: {comm_rounds}")
    terminal_logs.append(f"[{time.strftime('%X')}] Security: DP={dp_enabled} | SecAgg={secagg_enabled} | Byzantine Defense={anomaly_enabled}")

    p_bar = progress_container.progress(0, text="Initializing Federated Learning Pipeline...")

    def on_round(r_dict):
        r_num = r_dict["round"]
        p_bar.progress(
            r_num / comm_rounds,
            text=f"Round {r_num}/{comm_rounds} | Accuracy: {r_dict['globalAccuracy']}% | Loss: {r_dict['globalLoss']:.4f} | ε: {r_dict['privacyEpsilon']}"
        )
        mal_detected = r_dict["anomalyDetectionRate"]
        det_msg = f" | ADR: {mal_detected}%" if anomaly_enabled else ""
        terminal_logs.append(
            f"[{time.strftime('%X')}] Round {r_num:2d}/{comm_rounds} -> Acc: {r_dict['globalAccuracy']:5.2f}% | Loss: {r_dict['globalLoss']:.4f} | F1: {r_dict['globalF1']:5.2f}% | ε: {r_dict['privacyEpsilon']}{det_msg}"
        )
        # Render scrollable terminal view
        log_html = "<div class='terminal-box'>" + "<br>".join(terminal_logs[-12:]) + "</div>"
        log_container.markdown(log_html, unsafe_allow_html=True)

    with st.spinner("Training Federated Consensus Models in PyTorch..."):
        metrics = server.run_simulation(on_round_complete=on_round)
        st.session_state.experiment_results = metrics
        st.session_state.last_server_info = {
            "dataset": dataset_options[selected_dataset_key],
            "feature_names": server.feature_names,
            "target_name": server.target_name,
            "num_clients": num_clients
        }

    p_bar.empty()
    st.success(f"✅ Experiment Completed: Global Accuracy = {metrics.accuracy}% | F1-Score = {metrics.f1Score}% | Training Time = {metrics.trainingTimeSec}s")
    st.balloons()


# Execute 5-Framework Comparative Benchmark
if bench_button:
    dataset_cfg = DatasetConfigModel(
        datasetId=selected_dataset_key,
        targetVariable="Target Class",
        numClients=num_clients,
        distributionType=distribution_type,
        dirichletAlpha=dirichlet_alpha
    )

    fl_cfg = FLConfigModel(
        numClients=num_clients,
        commRounds=min(comm_rounds, 12),
        localEpochs=local_epochs,
        learningRate=learning_rate,
        aggregationStrategy=agg_strategy,
        modelType=model_type
    )

    attack_cfg = AttackConfigModel(
        enabled=True,
        maliciousClientRatio=0.2,
        attackTypes={"modelPoisoning": True, "dataPoisoning": False, "backdoorAttack": False, "membershipInference": True},
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
    bench_pbar = progress_container.progress(0, text="Initiating 5-Framework Comparative Benchmark...")

    for idx, (name, sec_cfg) in enumerate(modes):
        bench_pbar.progress((idx + 1) / len(modes), text=f"Evaluating variant {idx+1}/{len(modes)}: {name}...")
        server = FLCoordinatorServer(
            dataset_config=dataset_cfg,
            fl_config=fl_cfg,
            security_config=sec_cfg,
            attack_config=attack_cfg,
            custom_data=custom_samples
        )
        res = server.run_simulation()
        benchmark_data[name] = res.model_dump()

    bench_pbar.empty()
    st.session_state.benchmark_results = benchmark_data
    st.success("🎉 5-Framework Comparative Benchmark Completed!")


# -----------------------------------------------------------------------------
# 6. Results Dashboard & Visualizations
# -----------------------------------------------------------------------------
results: FrameworkMetricsModel = st.session_state.experiment_results

if results is not None:
    st.markdown("---")
    st.markdown(f"### 📊 Analytics & Telemetry: `{results.name}`")

    # High-level Metrics Cards
    m_cols = st.columns(6)
    m_cols[0].metric("Global Accuracy", f"{results.accuracy}%", delta=f"{round(results.accuracy - 50.0, 1)}% vs random")
    m_cols[1].metric("F1-Score", f"{results.f1Score}%")
    m_cols[2].metric("Sensitivity (Recall)", f"{results.sensitivity}%")
    m_cols[3].metric("Specificity", f"{results.specificity}%")
    m_cols[4].metric("ROC-AUC", f"{results.auc:.3f}")
    m_cols[5].metric("Privacy ε Spent", f"{results.privacyEpsilon:.2f}", delta=f"δ={results.privacyDelta}")

    m_cols2 = st.columns(4)
    m_cols2[0].metric("Anomaly Recall (ADR)", f"{results.anomalyDetectionRate}%", delta=f"FPR: {results.falsePositiveRate}%", delta_color="inverse")
    m_cols2[1].metric("Attack Success (ASR)", f"{results.attackSuccessRate}%", delta_color="inverse")
    m_cols2[2].metric("Comm Overhead", f"{results.commOverheadMB:.2f} MB")
    m_cols2[3].metric("Training Time", f"{results.trainingTimeSec} s")

    # Interactive Visual Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Learning Curves",
        "🏥 Hospital Nodes",
        "🛡️ Privacy & Defenses",
        "📊 Comparative Benchmark",
        "📥 Export & Data"
    ])

    # TAB 1: Convergence Curves
    with tab1:
        st.markdown("#### Model Convergence & Loss Across Communication Rounds")
        history_df = pd.DataFrame([r.model_dump() for r in results.history])

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=history_df["round"],
            y=history_df["globalAccuracy"],
            mode="lines+markers",
            name="Global Accuracy (%)",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=7)
        ))
        fig_conv.add_trace(go.Scatter(
            x=history_df["round"],
            y=history_df["globalF1"],
            mode="lines+markers",
            name="F1-Score (%)",
            line=dict(color="#2dd4bf", width=2, dash="dot"),
            marker=dict(size=6)
        ))
        fig_conv.add_trace(go.Scatter(
            x=history_df["round"],
            y=history_df["globalLoss"] * 100.0,
            mode="lines+markers",
            name="Loss (x100)",
            line=dict(color="#f43f5e", width=2),
            marker=dict(size=6)
        ))
        fig_conv.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            xaxis_title="Communication Round",
            yaxis_title="Score / Scaled Metric",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        # Secondary Convergence: Privacy & Attacks
        st.markdown("#### Privacy Accumulation & Backdoor ASR Trajectory")
        fig_priv = go.Figure()
        fig_priv.add_trace(go.Scatter(
            x=history_df["round"],
            y=history_df["privacyEpsilon"],
            mode="lines+markers",
            name="Privacy Budget ε",
            line=dict(color="#a855f7", width=3)
        ))
        fig_priv.add_trace(go.Scatter(
            x=history_df["round"],
            y=history_df["attackSuccessRate"],
            mode="lines+markers",
            name="Backdoor ASR (%)",
            line=dict(color="#f59e0b", width=2, dash="dash")
        ))
        fig_priv.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            xaxis_title="Communication Round",
            yaxis_title="Metric Value",
            hovermode="x unified"
        )
        st.plotly_chart(fig_priv, use_container_width=True)

    # TAB 2: Hospital Nodes
    with tab2:
        st.markdown("#### Institutional Hospital Nodes Status (Final Round)")
        last_round = results.history[-1]
        node_logs = last_round.clientLogs

        node_df = pd.DataFrame(node_logs)
        node_df["Status"] = node_df.apply(
            lambda row: "ISOLATED OUTLIER" if row["isDetected"] else ("MALICIOUS" if row["isMalicious"] else "BENIGN"),
            axis=1
        )

        st.dataframe(
            node_df[["clientId", "clientName", "Status", "localAccuracy", "localLoss", "updateNorm", "masked"]].rename(columns={
                "clientId": "ID",
                "clientName": "Hospital Institution",
                "localAccuracy": "Local Acc (%)",
                "localLoss": "Local Loss",
                "updateNorm": "Update L2 Norm",
                "masked": "SecAgg Masked"
            }),
            use_container_width=True
        )

        st.markdown("#### Parameter Update L2 Norms by Hospital Node")
        st.caption("Byzantine model poisoning attacks are identified by large update norms and directional divergence.")
        fig_bar = px.bar(
            node_df,
            x="clientName",
            y="updateNorm",
            color="Status",
            color_discrete_map={
                "BENIGN": "#10b981",
                "MALICIOUS": "#f59e0b",
                "ISOLATED OUTLIER": "#ef4444"
            },
            template="plotly_dark"
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)")
        st.plotly_chart(fig_bar, use_container_width=True)

    # TAB 3: Defenses & Security
    with tab3:
        st.markdown("#### Security & Privacy Telemetry Breakdown")
        col_dp, col_ad = st.columns(2)

        with col_dp:
            st.markdown("##### 🔒 Differential Privacy Guarantee")
            st.info(f"""
            - **Privacy Budget:** $\\epsilon = {results.privacyEpsilon:.3f}$, $\\delta = {results.privacyDelta}$
            - **Accountant:** Rényi Differential Privacy (RDP) Composition
            - **Gradient Clipping Bound ($C$):** {dp_clip}
            - **Gaussian Noise Multiplier ($\\sigma$):** {dp_noise}
            - **Guarantee:** Strict upper bound on single-patient linkage and membership disclosure.
            """)

        with col_ad:
            st.markdown("##### 🛡️ Byzantine Robust Defense")
            st.info(f"""
            - **Defense Algorithm:** `{anomaly_algo}`
            - **Anomaly Detection Rate (Recall):** {results.anomalyDetectionRate}%
            - **False Positive Rate:** {results.falsePositiveRate}%
            - **Secure Aggregation Protocol:** Pairwise Diffie-Hellman Masking (256-bit)
            - **Byzantine Breakdown Point:** Tolerates up to 40% malicious colluding institutions.
            """)

    # TAB 4: Comparative Benchmark
    with tab4:
        st.markdown("#### 5-Framework Comparative Benchmark Evaluation")
        benchmarks = st.session_state.benchmark_results

        if benchmarks is None:
            st.info("💡 Click **'Run 5-Framework Comparative Benchmark'** above to generate automated comparisons.")
        else:
            bench_df = pd.DataFrame(benchmarks).T
            bench_df = bench_df.reset_index().rename(columns={"index": "Framework Variant"})

            # Comparison Bar Chart
            fig_bench = px.bar(
                bench_df,
                x="Framework Variant",
                y=["accuracy", "f1Score", "anomalyDetectionRate"],
                barmode="group",
                title="Performance & Defense Comparison across Framework Postures",
                labels={"value": "Percentage (%)", "variable": "Metric"},
                color_discrete_sequence=["#38bdf8", "#2dd4bf", "#f59e0b"],
                template="plotly_dark"
            )
            fig_bench.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)")
            st.plotly_chart(fig_bench, use_container_width=True)

            # Tabular Comparison
            st.dataframe(
                bench_df[[
                    "Framework Variant", "accuracy", "f1Score", "auc",
                    "privacyEpsilon", "anomalyDetectionRate", "commOverheadMB"
                ]].rename(columns={
                    "accuracy": "Accuracy (%)",
                    "f1Score": "F1 (%)",
                    "auc": "ROC-AUC",
                    "privacyEpsilon": "Privacy ε",
                    "anomalyDetectionRate": "ADR (%)",
                    "commOverheadMB": "Overhead (MB)"
                }),
                use_container_width=True
            )

    # TAB 5: Export
    with tab5:
        st.markdown("#### Download Experiment Artifacts")
        res_json = json.dumps(results.model_dump(), indent=2)
        st.download_button(
            label="📥 Download Experiment Report (JSON)",
            data=res_json,
            file_name=f"FL_Healthcare_Experiment_{int(time.time())}.json",
            mime="application/json"
        )

        csv_data = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Round History (CSV)",
            data=csv_data,
            file_name=f"FL_Round_History_{int(time.time())}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Select configuration options in the sidebar and click **'🚀 Run Federated Learning Experiment'** to start training.")

# Footer
st.markdown("---")
st.caption("Privacy-Preserving Healthcare Federated Learning Platform © 2026. Built with PyTorch, Streamlit, Plotly, Differential Privacy (DP-SGD), and Byzantine Robust Defenses.")
