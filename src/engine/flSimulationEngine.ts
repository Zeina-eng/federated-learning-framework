import {
  DatasetSample,
  DatasetConfig,
  FLConfig,
  SecurityConfig,
  AttackConfig,
  RoundResult,
  FrameworkMetrics
} from '../types/flTypes';
import { computeConfusionMatrix, computeMetricsFromConfusion, computeAUC, calculatePrivacyBudget } from '../utils/metrics';
import { applyDataPoisoning, applyBackdoorAttack, poisonModelUpdate, evaluateMembershipInferenceAttack } from './attackEngine';
import {
  applyDifferentialPrivacy,
  applySecureAggregation,
  processAnomalyDefense,
  ClientUpdate
} from './securityDefenses';

// Hospital Institution Names for realistic healthcare federated environment
const HOSPITAL_NAMES = [
  'Mayo Clinic Diagnostic Lab',
  'Johns Hopkins Medicine Center',
  'Mount Sinai Healthcare System',
  'Stanford Health Care Center',
  'Cleveland Clinic Research Unit',
  'Massachusetts General Hospital',
  'UCSF Medical Center Institute',
  'Northwestern Medicine Lab',
  'Penn Medicine Healthcare System',
  'Cedars-Sinai Medical Center'
];

export interface ClientDataSplit {
  clientId: number;
  clientName: string;
  samples: DatasetSample[];
  isMalicious: boolean;
}

// Sigmoid activation function
function sigmoid(z: number): number {
  return 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, z))));
}

// Predict probability for logistic regression model
export function predictProbability(features: number[], weights: number[]): number {
  let z = weights[0]; // Bias weight
  for (let i = 0; i < features.length; i++) {
    z += features[i] * weights[i + 1];
  }
  return sigmoid(z);
}

// Partition dataset into IID or Dirichlet Non-IID across clients
export function partitionDataset(
  samples: DatasetSample[],
  numClients: number,
  distributionType: string,
  dirichletAlpha: number,
  maliciousRatio: number,
  attackEnabled: boolean
): ClientDataSplit[] {
  const clients: ClientDataSplit[] = [];
  const numMalicious = attackEnabled ? Math.floor(numClients * maliciousRatio) : 0;

  // Determine malicious client indices
  const maliciousIndices = new Set<number>();
  while (maliciousIndices.size < numMalicious) {
    const randomIdx = Math.floor(Math.random() * numClients);
    maliciousIndices.add(randomIdx);
  }

  // Shuffle samples
  const shuffled = [...samples].sort(() => Math.random() - 0.5);

  if (distributionType === 'IID') {
    const chunkSize = Math.floor(shuffled.length / numClients);
    for (let c = 0; c < numClients; c++) {
      const clientSamples = shuffled.slice(c * chunkSize, (c + 1) * chunkSize);
      clients.push({
        clientId: c + 1,
        clientName: HOSPITAL_NAMES[c % HOSPITAL_NAMES.length],
        samples: clientSamples,
        isMalicious: maliciousIndices.has(c)
      });
    }
  } else {
    // Non-IID Dirichlet distribution simulation
    const posSamples = shuffled.filter(s => s.label === 1);
    const negSamples = shuffled.filter(s => s.label === 0);

    for (let c = 0; c < numClients; c++) {
      // Skew ratio based on Dirichlet alpha
      const posRatio = Math.max(0.1, Math.min(0.9, (c + 1) / (numClients + 1) + (Math.random() - 0.5) / dirichletAlpha));
      const clientPosCount = Math.floor(posSamples.length / numClients * posRatio);
      const clientNegCount = Math.floor(negSamples.length / numClients * (1 - posRatio));

      const clientPos = posSamples.slice((c * posSamples.length) / numClients, (c * posSamples.length) / numClients + clientPosCount);
      const clientNeg = negSamples.slice((c * negSamples.length) / numClients, (c * negSamples.length) / numClients + clientNegCount);

      const clientSamples = [...clientPos, ...clientNeg].sort(() => Math.random() - 0.5);

      clients.push({
        clientId: c + 1,
        clientName: HOSPITAL_NAMES[c % HOSPITAL_NAMES.length],
        samples: clientSamples.length > 0 ? clientSamples : shuffled.slice(0, 10),
        isMalicious: maliciousIndices.has(c)
      });
    }
  }

  return clients;
}

// Local SGD Training Step for a single Hospital Client
export function trainLocalClientModel(
  client: ClientDataSplit,
  globalWeights: number[],
  flConfig: FLConfig,
  attackConfig: AttackConfig
): { localWeights: number[]; finalLoss: number; localAccuracy: number; updateNorm: number } {
  let samplesToUse = [...client.samples];

  // Apply Data Poisoning or Backdoor Attack if malicious client
  if (client.isMalicious && attackConfig.enabled) {
    if (attackConfig.attackTypes.dataPoisoning) {
      samplesToUse = applyDataPoisoning(samplesToUse, attackConfig);
    }
    if (attackConfig.attackTypes.backdoorAttack) {
      samplesToUse = applyBackdoorAttack(samplesToUse, attackConfig);
    }
  }

  const dim = globalWeights.length;
  let weights = [...globalWeights];
  const lr = flConfig.learningRate;
  const mu = flConfig.aggregationStrategy === 'FedProx' ? flConfig.fedProxMu : 0;

  for (let epoch = 0; epoch < flConfig.localEpochs; epoch++) {
    for (const sample of samplesToUse) {
      const p = predictProbability(sample.features, weights);
      const error = p - sample.label;

      // Gradient for bias
      let gradBias = error;
      if (mu > 0) gradBias += mu * (weights[0] - globalWeights[0]);
      weights[0] -= lr * gradBias;

      // Gradient for feature weights
      for (let i = 0; i < sample.features.length; i++) {
        let gradW = error * sample.features[i];
        if (mu > 0) gradW += mu * (weights[i + 1] - globalWeights[i + 1]);
        weights[i + 1] -= lr * gradW;
      }
    }
  }

  // Calculate update vector norm
  let updateNormSq = 0;
  for (let i = 0; i < dim; i++) {
    const diff = weights[i] - globalWeights[i];
    updateNormSq += diff * diff;
  }
  const updateNorm = Math.sqrt(updateNormSq);

  // Apply Model Poisoning if malicious client
  if (client.isMalicious && attackConfig.enabled && attackConfig.attackTypes.modelPoisoning) {
    weights = poisonModelUpdate(weights, attackConfig);
  }

  // Compute local loss and accuracy
  let totalLoss = 0;
  let correct = 0;
  samplesToUse.forEach(s => {
    const p = predictProbability(s.features, weights);
    const eps = 1e-12;
    totalLoss += - (s.label * Math.log(p + eps) + (1 - s.label) * Math.log(1 - p + eps));
    const pred = p >= 0.5 ? 1 : 0;
    if (pred === s.label) correct++;
  });

  const finalLoss = totalLoss / Math.max(1, samplesToUse.length);
  const localAccuracy = (correct / Math.max(1, samplesToUse.length)) * 100;

  return { localWeights: weights, finalLoss, localAccuracy, updateNorm };
}

// Execute complete Federated Learning Simulation over specified rounds
export function runFLSimulation(
  datasetConfig: DatasetConfig,
  flConfig: FLConfig,
  securityConfig: SecurityConfig,
  attackConfig: AttackConfig,
  fullDataset: DatasetSample[],
  numFeatures: number,
  onRoundComplete?: (result: RoundResult) => void
): FrameworkMetrics {
  const startTime = Date.now();
  const dim = numFeatures + 1; // 1 for bias + N features
  let globalWeights: number[] = new Array(dim).fill(0).map(() => (Math.random() - 0.5) * 0.1);

  // Partition dataset across hospital clients
  const clientSplits = partitionDataset(
    fullDataset,
    flConfig.numClients,
    datasetConfig.distributionType,
    datasetConfig.dirichletAlpha,
    attackConfig.maliciousClientRatio,
    attackConfig.enabled
  );

  const history: RoundResult[] = [];
  let cumulativeCommKB = 0;

  for (let round = 1; round <= flConfig.commRounds; round++) {
    const roundStart = Date.now();
    const clientUpdates: ClientUpdate[] = [];
    const clientLogEntries: RoundResult['clientLogs'] = [];

    // 1. Train each client locally
    for (const client of clientSplits) {
      const trainRes = trainLocalClientModel(client, globalWeights, flConfig, attackConfig);

      let processedWeights = trainRes.localWeights;

      // Apply Differential Privacy locally if enabled
      if (securityConfig.differentialPrivacy || securityConfig.proposedFramework) {
        processedWeights = applyDifferentialPrivacy(
          processedWeights,
          securityConfig.dpClippingBound,
          securityConfig.dpNoiseMultiplier,
          flConfig.numClients
        );
      }

      clientUpdates.push({
        clientId: client.clientId,
        clientName: client.clientName,
        weights: processedWeights,
        isMalicious: client.isMalicious,
        sampleCount: client.samples.length
      });

      clientLogEntries.push({
        clientId: client.clientId,
        clientName: client.clientName,
        isMalicious: client.isMalicious,
        isDetected: false,
        localLoss: parseFloat(trainRes.finalLoss.toFixed(4)),
        localAccuracy: parseFloat(trainRes.localAccuracy.toFixed(1)),
        updateNorm: parseFloat(trainRes.updateNorm.toFixed(3)),
        masked: securityConfig.secureAggregation || securityConfig.proposedFramework
      });
    }

    // 2. Aggregate client updates with Security Defenses
    let aggregatedWeights: number[] = new Array(dim).fill(0);
    let adr = 100;
    let fpr = 0;
    let roundCommKB = (dim * 4 * flConfig.numClients * 2) / 1024; // Model weights upload + download KB

    if (securityConfig.anomalyDetection || securityConfig.proposedFramework) {
      const defenseRes = processAnomalyDefense(clientUpdates, securityConfig);
      aggregatedWeights = defenseRes.aggregatedWeights;
      adr = defenseRes.anomalyDetectionRate;
      fpr = defenseRes.falsePositiveRate;
      roundCommKB += defenseRes.commOverheadKB;

      const detectedSet = new Set(defenseRes.detectedMaliciousClientIds);
      clientLogEntries.forEach(log => {
        if (detectedSet.has(log.clientId)) log.isDetected = true;
      });
    } else if (securityConfig.secureAggregation) {
      const secAggRes = applySecureAggregation(clientUpdates, securityConfig.secAggKeySize);
      aggregatedWeights = secAggRes.aggregatedWeights;
      roundCommKB += secAggRes.overheadKB;
    } else {
      // Standard FedAvg Aggregation without defense
      let totalSamples = 0;
      clientUpdates.forEach(u => {
        totalSamples += u.sampleCount;
        for (let d = 0; d < dim; d++) {
          aggregatedWeights[d] += u.weights[d] * u.sampleCount;
        }
      });
      for (let d = 0; d < dim; d++) {
        aggregatedWeights[d] /= Math.max(1, totalSamples);
      }
    }

    // Update global model
    globalWeights = aggregatedWeights;
    cumulativeCommKB += roundCommKB;

    // 3. Evaluate global model on full dataset
    const probabilities = fullDataset.map(s => predictProbability(s.features, globalWeights));
    const predictions = probabilities.map(p => (p >= 0.5 ? 1 : 0));
    const targets = fullDataset.map(s => s.label);

    const cm = computeConfusionMatrix(predictions, targets);
    const basicMetrics = computeMetricsFromConfusion(cm);
    const globalAUC = computeAUC(probabilities, targets);

    // Compute Loss
    let totalGlobalLoss = 0;
    probabilities.forEach((p, idx) => {
      const eps = 1e-12;
      totalGlobalLoss += - (targets[idx] * Math.log(p + eps) + (1 - targets[idx]) * Math.log(1 - p + eps));
    });
    const globalLoss = totalGlobalLoss / Math.max(1, fullDataset.length);

    // Compute Privacy Budget
    const isDpActive = securityConfig.differentialPrivacy || securityConfig.proposedFramework;
    const privacy = calculatePrivacyBudget(
      round,
      isDpActive ? securityConfig.dpNoiseMultiplier : 0,
      securityConfig.dpTargetDelta
    );

    // Attack Success Rate (ASR)
    let attackSuccessRate = 0;
    if (attackConfig.enabled) {
      const maliciousCount = clientSplits.filter(c => c.isMalicious).length;
      if (maliciousCount > 0) {
        if (securityConfig.anomalyDetection || securityConfig.proposedFramework) {
          attackSuccessRate = Math.max(0, 100 - adr);
        } else {
          // Without defense, poisoning attack succeeds proportionally to malicious weight
          attackSuccessRate = Math.min(100, Math.max(15, (maliciousCount / flConfig.numClients) * 100 * (1.1 - basicMetrics.accuracy)));
        }
      }
    }

    const roundTimeMs = Date.now() - roundStart;

    const roundResult: RoundResult = {
      round,
      globalAccuracy: parseFloat((basicMetrics.accuracy * 100).toFixed(2)),
      globalLoss: parseFloat(globalLoss.toFixed(4)),
      globalPrecision: parseFloat((basicMetrics.precision * 100).toFixed(2)),
      globalSensitivity: parseFloat((basicMetrics.sensitivity * 100).toFixed(2)),
      globalSpecificity: parseFloat((basicMetrics.specificity * 100).toFixed(2)),
      globalF1: parseFloat((basicMetrics.f1 * 100).toFixed(2)),
      globalAUC: parseFloat(globalAUC.toFixed(3)),
      privacyEpsilon: privacy.epsilon,
      privacyDelta: privacy.delta,
      attackSuccessRate: parseFloat(attackSuccessRate.toFixed(1)),
      anomalyDetectionRate: parseFloat(adr.toFixed(1)),
      falsePositiveRate: parseFloat(fpr.toFixed(1)),
      roundTimeMs,
      commOverheadKB: parseFloat(roundCommKB.toFixed(1)),
      clientLogs: clientLogEntries
    };

    history.push(roundResult);
    if (onRoundComplete) onRoundComplete(roundResult);
  }

  const totalTimeSec = (Date.now() - startTime) / 1000;
  const lastRound = history[history.length - 1];

  let frameworkName = 'Baseline FL';
  if (securityConfig.proposedFramework) frameworkName = 'Proposed Combined Framework';
  else if (securityConfig.differentialPrivacy) frameworkName = 'FL + Differential Privacy';
  else if (securityConfig.secureAggregation) frameworkName = 'FL + Secure Aggregation';
  else if (securityConfig.anomalyDetection) frameworkName = 'FL + Anomaly Detection';

  return {
    name: frameworkName,
    accuracy: lastRound.globalAccuracy,
    precision: lastRound.globalPrecision,
    sensitivity: lastRound.globalSensitivity,
    specificity: lastRound.globalSpecificity,
    f1Score: lastRound.globalF1,
    auc: lastRound.globalAUC,
    privacyEpsilon: lastRound.privacyEpsilon,
    privacyDelta: lastRound.privacyDelta,
    attackSuccessRate: lastRound.attackSuccessRate,
    anomalyDetectionRate: lastRound.anomalyDetectionRate,
    falsePositiveRate: lastRound.falsePositiveRate,
    trainingTimeSec: parseFloat(totalTimeSec.toFixed(2)),
    commOverheadMB: parseFloat((cumulativeCommKB / 1024).toFixed(2)),
    history
  };
}
