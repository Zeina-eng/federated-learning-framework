import React, { useState, useEffect, useRef } from 'react';
import confetti from 'canvas-confetti';
import { Navbar } from './components/Navbar';
import { DatasetSection } from './components/DatasetSection';
import { FLConfigSection } from './components/FLConfigSection';
import { SecuritySection } from './components/SecuritySection';
import { AttackSection } from './components/AttackSection';
import { RunExperimentSection } from './components/RunExperimentSection';
import { ResultsSection } from './components/ResultsSection';
import { ComparisonSection } from './components/ComparisonSection';
import { HospitalNodeModal } from './components/HospitalNodeModal';

import { BUILTIN_DATASETS } from './data/datasets';
import {
  DatasetConfig,
  FLConfig,
  SecurityConfig,
  AttackConfig,
  RoundResult,
  FrameworkMetrics,
  DatasetInfo
} from './types/flTypes';
import { runFLSimulation } from './engine/flSimulationEngine';
import {
  checkPythonBackendHealth,
  runPythonSimulationViaWebSocket,
  runPythonBenchmark,
  PythonHealthStatus
} from './services/flApiService';

export default function App() {
  // Engine Selection State: Python PyTorch vs Browser TypeScript
  const [engineMode, setEngineMode] = useState<'python' | 'browser'>('python');
  const [pythonHealth, setPythonHealth] = useState<PythonHealthStatus>({ healthy: false });

  // 1. Dataset Configuration State
  const [customDatasets, setCustomDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetInfo, setSelectedDatasetInfo] = useState<DatasetInfo>(BUILTIN_DATASETS[0]);
  
  const [datasetConfig, setDatasetConfig] = useState<DatasetConfig>({
    datasetId: BUILTIN_DATASETS[0].id,
    targetVariable: BUILTIN_DATASETS[0].targetName,
    numClients: 5,
    distributionType: 'NON_IID_DIRICHLET',
    dirichletAlpha: 0.5
  });

  // Update selected dataset object when datasetId changes
  useEffect(() => {
    const all = [...BUILTIN_DATASETS, ...customDatasets];
    const found = all.find(d => d.id === datasetConfig.datasetId);
    if (found) {
      setSelectedDatasetInfo(found);
    }
  }, [datasetConfig.datasetId, customDatasets]);

  // Check Python PyTorch backend health status periodically
  useEffect(() => {
    let mounted = true;
    const checkHealth = async () => {
      const status = await checkPythonBackendHealth();
      if (mounted) {
        setPythonHealth(status);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 4000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // 2. FL Hyperparameters State
  const [flConfig, setFlConfig] = useState<FLConfig>({
    numClients: 5,
    commRounds: 12,
    localEpochs: 3,
    learningRate: 0.02,
    aggregationStrategy: 'FedAvg',
    fedProxMu: 0.05
  });

  // Sync numClients between datasetConfig and flConfig
  const handleDatasetConfigChange = (newConfig: DatasetConfig) => {
    setDatasetConfig(newConfig);
    setFlConfig(prev => ({ ...prev, numClients: newConfig.numClients }));
  };

  const handleFLConfigChange = (newConfig: FLConfig) => {
    setFlConfig(newConfig);
    setDatasetConfig(prev => ({ ...prev, numClients: newConfig.numClients }));
  };

  // 3. Security Experiments State
  const [securityConfig, setSecurityConfig] = useState<SecurityConfig>({
    baselineFL: false,
    differentialPrivacy: true,
    secureAggregation: true,
    anomalyDetection: true,
    proposedFramework: true,
    dpNoiseMultiplier: 0.5,
    dpClippingBound: 3.0,
    dpTargetDelta: 1e-5,
    secAggKeySize: 256,
    secAggProtocol: 'DH_Masking',
    anomalyAlgorithm: 'Cosine_Similarity',
    anomalyThreshold: 0.5
  });

  // 4. Adversarial Attack Simulation State
  const [attackConfig, setAttackConfig] = useState<AttackConfig>({
    enabled: true,
    maliciousClientRatio: 0.2, // 20% compromised nodes
    attackTypes: {
      modelPoisoning: true,
      dataPoisoning: false,
      backdoorAttack: false,
      membershipInference: true
    },
    modelPoisoningFactor: 3.0,
    backdoorTargetClass: 1,
    backdoorTriggerValue: 999.0
  });

  // 5. Training Simulation State
  const [isTraining, setIsTraining] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentRound, setCurrentRound] = useState(0);
  const [latestRoundResult, setLatestRoundResult] = useState<RoundResult | null>(null);
  const [latestMetrics, setLatestMetrics] = useState<FrameworkMetrics | null>(null);
  const [logMessages, setLogMessages] = useState<string[]>([
    'System Initialized: Healthcare Privacy-Preserving Federated Learning Framework ready.',
    'Select dataset and security configuration, then click "Run Federated Learning Experiment".'
  ]);

  // 6. Comparison Benchmark State
  const [comparisonResults, setComparisonResults] = useState<Record<string, FrameworkMetrics>>({});
  const [isComparing, setIsComparing] = useState(false);

  // Modal Inspector State
  const [inspectClientId, setInspectClientId] = useState<number | null>(null);

  // Refs for background simulation control
  const simulationRef = useRef<{ cancel: boolean }>({ cancel: false });

  // Handle Preset Loader
  const handleLoadPreset = (presetKey: string) => {
    if (isTraining) return;

    if (presetKey === 'clinical_standard') {
      setSecurityConfig({
        baselineFL: true,
        differentialPrivacy: false,
        secureAggregation: false,
        anomalyDetection: false,
        proposedFramework: false,
        dpNoiseMultiplier: 0.5,
        dpClippingBound: 3.0,
        dpTargetDelta: 1e-5,
        secAggKeySize: 256,
        secAggProtocol: 'DH_Masking',
        anomalyAlgorithm: 'Cosine_Similarity',
        anomalyThreshold: 0.5
      });
      setAttackConfig(prev => ({ ...prev, enabled: false }));
      addLog('Preset Loaded: Clinical Standard Baseline (No DP, No Attack)');
    } else if (presetKey === 'high_privacy_dp') {
      setSecurityConfig({
        baselineFL: false,
        differentialPrivacy: true,
        secureAggregation: true,
        anomalyDetection: false,
        proposedFramework: false,
        dpNoiseMultiplier: 1.2,
        dpClippingBound: 2.0,
        dpTargetDelta: 1e-5,
        secAggKeySize: 256,
        secAggProtocol: 'DH_Masking',
        anomalyAlgorithm: 'Cosine_Similarity',
        anomalyThreshold: 0.5
      });
      setAttackConfig(prev => ({ ...prev, enabled: false }));
      addLog('Preset Loaded: High Differential Privacy (DP-FedAvg σ=1.2, C=2.0)');
    } else if (presetKey === 'adversarial_attack') {
      setSecurityConfig({
        baselineFL: true,
        differentialPrivacy: false,
        secureAggregation: false,
        anomalyDetection: false,
        proposedFramework: false,
        dpNoiseMultiplier: 0.5,
        dpClippingBound: 3.0,
        dpTargetDelta: 1e-5,
        secAggKeySize: 256,
        secAggProtocol: 'DH_Masking',
        anomalyAlgorithm: 'Cosine_Similarity',
        anomalyThreshold: 0.5
      });
      setAttackConfig({
        enabled: true,
        maliciousClientRatio: 0.4,
        attackTypes: { modelPoisoning: true, dataPoisoning: true, backdoorAttack: true, membershipInference: true },
        modelPoisoningFactor: 4.0,
        backdoorTargetClass: 1,
        backdoorTriggerValue: 999.0
      });
      addLog('Preset Loaded: Byzantine Attack Test (40% Compromised Nodes)');
    } else if (presetKey === 'proposed_full') {
      setSecurityConfig({
        baselineFL: false,
        differentialPrivacy: true,
        secureAggregation: true,
        anomalyDetection: true,
        proposedFramework: true,
        dpNoiseMultiplier: 0.5,
        dpClippingBound: 3.0,
        dpTargetDelta: 1e-5,
        secAggKeySize: 256,
        secAggProtocol: 'DH_Masking',
        anomalyAlgorithm: 'Cosine_Similarity',
        anomalyThreshold: 0.5
      });
      setAttackConfig({
        enabled: true,
        maliciousClientRatio: 0.2,
        attackTypes: { modelPoisoning: true, dataPoisoning: false, backdoorAttack: false, membershipInference: true },
        modelPoisoningFactor: 3.0,
        backdoorTargetClass: 1,
        backdoorTriggerValue: 999.0
      });
      addLog('Preset Loaded: Proposed Combined Framework (DP + SecAgg + Byzantine Defense)');
    }
  };

  const addLog = (msg: string) => {
    const time = new Date().toLocaleTimeString();
    setLogMessages(prev => [...prev.slice(-80), `[${time}] ${msg}`]);
  };

  // Run Real-Time FL Simulation Step-by-Step
  const handleStartExperiment = async () => {
    if (isTraining) return;

    setIsTraining(true);
    setIsPaused(false);
    setCurrentRound(0);
    setLatestRoundResult(null);
    simulationRef.current.cancel = false;

    addLog(`🚀 Launching Federated Learning Experiment on ${selectedDatasetInfo.name}...`);
    addLog(`Engine: ${engineMode === 'python' ? '🐍 Python (PyTorch Engine)' : '⚡ Browser (TypeScript Engine)'}`);
    addLog(`Config: ${flConfig.numClients} Institutions | ${flConfig.commRounds} Rounds | ${flConfig.aggregationStrategy} Strategy`);

    let activeSecurityName = 'Baseline FL';
    if (securityConfig.proposedFramework) activeSecurityName = 'Proposed Combined Framework';
    else if (securityConfig.differentialPrivacy) activeSecurityName = 'FL + Differential Privacy';
    else if (securityConfig.secureAggregation) activeSecurityName = 'FL + Secure Aggregation';
    else if (securityConfig.anomalyDetection) activeSecurityName = 'FL + Anomaly Detection';

    addLog(`Security Mode: ${activeSecurityName}`);

    // Execution path A: Python PyTorch WebSocket Engine
    if (engineMode === 'python') {
      addLog('Connecting to Python PyTorch backend (ws://localhost:8000/ws/fl-simulate)...');
      
      const wsClient = runPythonSimulationViaWebSocket(
        datasetConfig,
        flConfig,
        securityConfig,
        attackConfig,
        selectedDatasetInfo.id.startsWith('custom') ? selectedDatasetInfo.data : undefined,
        (roundRes) => {
          setCurrentRound(roundRes.round);
          setLatestRoundResult(roundRes);
          addLog(
            `[PyTorch] Round ${roundRes.round}/${flConfig.commRounds} | Accuracy: ${roundRes.globalAccuracy}% | Loss: ${roundRes.globalLoss} | Privacy ε: ${roundRes.privacyEpsilon} | ASR: ${roundRes.attackSuccessRate}%`
          );
          if (roundRes.anomalyDetectionRate < 100 && securityConfig.anomalyDetection) {
            addLog(`⚠️ ANOMALY DETECTOR: Flagged & isolated malicious client updates.`);
          }
        },
        (resultMetrics) => {
          setLatestMetrics(resultMetrics);
          setIsTraining(false);
          addLog(`✅ [PyTorch Engine] EXPERIMENT COMPLETE: Final Global Accuracy = ${resultMetrics.accuracy}% | F1-Score = ${resultMetrics.f1Score}%`);
          confetti({
            particleCount: 80,
            spread: 70,
            origin: { y: 0.6 }
          });
        },
        (errorMsg) => {
          addLog(`⚠️ [Python Engine Error]: ${errorMsg}`);
          addLog('💡 Tip: Make sure the Python server is running with run_backend.bat or switch to "Browser (TS)" mode.');
          setIsTraining(false);
        }
      );
      return;
    }

    // Execution path B: In-Browser TypeScript Engine
    try {
      const fullData = selectedDatasetInfo.data;
      const numFeatures = selectedDatasetInfo.featureNames.length;

      const resultMetrics = runFLSimulation(
        datasetConfig,
        flConfig,
        securityConfig,
        attackConfig,
        fullData,
        numFeatures,
        (roundRes) => {
          setCurrentRound(roundRes.round);
          setLatestRoundResult(roundRes);

          // Add terminal log message for round
          addLog(
            `Round ${roundRes.round}/${flConfig.commRounds} | Accuracy: ${roundRes.globalAccuracy}% | Loss: ${roundRes.globalLoss} | Privacy ε: ${roundRes.privacyEpsilon} | ASR: ${roundRes.attackSuccessRate}%`
          );

          if (roundRes.anomalyDetectionRate < 100 && securityConfig.anomalyDetection) {
            addLog(`⚠️ ANOMALY DETECTOR: Flagged & isolated malicious client updates.`);
          }
        }
      );

      setLatestMetrics(resultMetrics);
      setIsTraining(false);
      addLog(`✅ EXPERIMENT COMPLETE: Final Global Accuracy = ${resultMetrics.accuracy}% | F1-Score = ${resultMetrics.f1Score}%`);

      // Trigger celebration confetti
      confetti({
        particleCount: 80,
        spread: 70,
        origin: { y: 0.6 }
      });
    } catch (err: any) {
      addLog(`❌ ERROR during execution: ${err.message}`);
      setIsTraining(false);
    }
  };

  // Run 5-Framework Side-by-Side Comparison Benchmark
  const handleRunAllComparisons = async () => {
    if (isComparing || isTraining) return;
    setIsComparing(true);
    addLog(`⚡ Initiating Automated 5-Framework Benchmark (${engineMode === 'python' ? 'Python PyTorch' : 'Browser Engine'})...`);

    if (engineMode === 'python' && pythonHealth.healthy) {
      try {
        addLog('Requesting PyTorch benchmark from Python backend...');
        const benchmarkResults = await runPythonBenchmark(
          datasetConfig,
          flConfig,
          attackConfig,
          selectedDatasetInfo.id.startsWith('custom') ? selectedDatasetInfo.data : undefined
        );
        setComparisonResults(benchmarkResults);
        setIsComparing(false);
        addLog('🎉 [PyTorch] Comparative Benchmark Analysis Completed successfully!');
        return;
      } catch (err: any) {
        addLog(`⚠️ Python benchmark request failed (${err.message}), falling back to browser engine...`);
      }
    }

    const fullData = selectedDatasetInfo.data;
    const numFeatures = selectedDatasetInfo.featureNames.length;
    const benchmarkResults: Record<string, FrameworkMetrics> = {};

    const modes = [
      {
        name: 'Baseline FL',
        sec: { baselineFL: true, differentialPrivacy: false, secureAggregation: false, anomalyDetection: false, proposedFramework: false, dpNoiseMultiplier: 0, dpClippingBound: 5, dpTargetDelta: 1e-5, secAggKeySize: 0, secAggProtocol: 'DH_Masking' as const, anomalyAlgorithm: 'Cosine_Similarity' as const, anomalyThreshold: 0.5 }
      },
      {
        name: 'FL + Differential Privacy',
        sec: { baselineFL: false, differentialPrivacy: true, secureAggregation: false, anomalyDetection: false, proposedFramework: false, dpNoiseMultiplier: 0.6, dpClippingBound: 3.0, dpTargetDelta: 1e-5, secAggKeySize: 0, secAggProtocol: 'DH_Masking' as const, anomalyAlgorithm: 'Cosine_Similarity' as const, anomalyThreshold: 0.5 }
      },
      {
        name: 'FL + Secure Aggregation',
        sec: { baselineFL: false, differentialPrivacy: false, secureAggregation: true, anomalyDetection: false, proposedFramework: false, dpNoiseMultiplier: 0, dpClippingBound: 5, dpTargetDelta: 1e-5, secAggKeySize: 256, secAggProtocol: 'DH_Masking' as const, anomalyAlgorithm: 'Cosine_Similarity' as const, anomalyThreshold: 0.5 }
      },
      {
        name: 'FL + Anomaly Detection',
        sec: { baselineFL: false, differentialPrivacy: false, secureAggregation: false, anomalyDetection: true, proposedFramework: false, dpNoiseMultiplier: 0, dpClippingBound: 5, dpTargetDelta: 1e-5, secAggKeySize: 0, secAggProtocol: 'DH_Masking' as const, anomalyAlgorithm: 'Cosine_Similarity' as const, anomalyThreshold: 0.5 }
      },
      {
        name: 'Proposed Combined Framework',
        sec: { baselineFL: false, differentialPrivacy: true, secureAggregation: true, anomalyDetection: true, proposedFramework: true, dpNoiseMultiplier: 0.5, dpClippingBound: 3.0, dpTargetDelta: 1e-5, secAggKeySize: 256, secAggProtocol: 'DH_Masking' as const, anomalyAlgorithm: 'Cosine_Similarity' as const, anomalyThreshold: 0.5 }
      }
    ];

    for (const m of modes) {
      addLog(`Executing Benchmark [${m.name}]...`);
      const res = runFLSimulation(datasetConfig, flConfig, m.sec, attackConfig, fullData, numFeatures);
      benchmarkResults[m.name] = res;
    }

    setComparisonResults(benchmarkResults);
    setIsComparing(false);
    addLog('🎉 Comparative Benchmark Analysis Completed successfully!');
  };

  // Export Results JSON / CSV Summary
  const handleExportReport = () => {
    if (!latestMetrics) {
      alert('Please run an experiment before exporting the report.');
      return;
    }
    const reportData = {
      timestamp: new Date().toISOString(),
      dataset: selectedDatasetInfo.name,
      metrics: latestMetrics,
      comparison: comparisonResults
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `FL_Healthcare_Report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 selection:bg-cyan-500 selection:text-slate-950 font-sans pb-16">
      
      {/* Navbar Header */}
      <Navbar
        onLoadPreset={handleLoadPreset}
        onExportReport={handleExportReport}
        isTraining={isTraining}
        engineMode={engineMode}
        setEngineMode={setEngineMode}
        pythonHealth={pythonHealth}
      />

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6">
        
        {/* Section 1: Dataset Setup */}
        <DatasetSection
          datasetConfig={datasetConfig}
          onChangeConfig={handleDatasetConfigChange}
          selectedDatasetInfo={selectedDatasetInfo}
          onCustomDatasetUploaded={(info) => setCustomDatasets(prev => [...prev, info])}
          isTraining={isTraining}
        />

        {/* Section 2: Federated Learning Parameters */}
        <FLConfigSection
          flConfig={flConfig}
          onChangeConfig={handleFLConfigChange}
          isTraining={isTraining}
        />

        {/* Section 3: Security Experiments */}
        <SecuritySection
          securityConfig={securityConfig}
          onChangeConfig={setSecurityConfig}
          isTraining={isTraining}
        />

        {/* Section 4: Adversarial Attack Simulation */}
        <AttackSection
          attackConfig={attackConfig}
          onChangeConfig={setAttackConfig}
          isTraining={isTraining}
        />

        {/* Section 5: Run Experiment & Execution Engine */}
        <RunExperimentSection
          onStartExperiment={handleStartExperiment}
          onPauseExperiment={() => setIsPaused(!isPaused)}
          isTraining={isTraining}
          isPaused={isPaused}
          currentRound={currentRound}
          totalRounds={flConfig.commRounds}
          latestRoundResult={latestRoundResult}
          datasetConfig={datasetConfig}
          securityConfig={securityConfig}
          attackConfig={attackConfig}
          logMessages={logMessages}
        />

        {/* Section 6: Results Analytics Dashboard */}
        <ResultsSection metrics={latestMetrics} />

        {/* Section 7: Security Framework Comparison */}
        <ComparisonSection
          comparisonResults={comparisonResults}
          onRunAllComparisons={handleRunAllComparisons}
          isComparing={isComparing}
        />

      </main>

      {/* Hospital Node Detail Inspector Modal */}
      <HospitalNodeModal
        clientId={inspectClientId}
        latestRoundResult={latestRoundResult}
        onClose={() => setInspectClientId(null)}
      />

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-6 pt-6 border-t border-slate-800/60 text-center text-xs text-slate-400">
        <p>
          Privacy-Preserving Federated Learning Framework for Healthcare Data Analytics © 2026. Built with React, Tailwind CSS, Differential Privacy (DP-SGD), Secure Aggregation, and Byzantine Anomaly Defense engines.
        </p>
      </footer>

    </div>
  );
}
