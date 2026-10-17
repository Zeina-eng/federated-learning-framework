export interface DatasetSample {
  id: number;
  features: number[];
  label: number;
}

export interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  sampleCount: number;
  featureNames: string[];
  targetName: string;
  data: DatasetSample[];
  summaryStats?: {
    positiveClassRatio: number;
    meanFeatures: number[];
  };
}

export type DistributionType = 'IID' | 'NON_IID_DIRICHLET' | 'NON_IID_LABEL_SKEW';

export interface DatasetConfig {
  datasetId: string;
  customData?: DatasetSample[];
  customFeatureNames?: string[];
  targetVariable: string;
  numClients: number;
  distributionType: DistributionType;
  dirichletAlpha: number; // For non-IID skew (e.g. 0.1 to 5.0)
}

export interface FLConfig {
  numClients: number;
  commRounds: number;
  localEpochs: number;
  learningRate: number;
  aggregationStrategy: 'FedAvg' | 'FedProx';
  fedProxMu: number;
}

export interface SecurityConfig {
  baselineFL: boolean;
  differentialPrivacy: boolean;
  secureAggregation: boolean;
  anomalyDetection: boolean;
  proposedFramework: boolean;
  
  // DP Hyperparameters
  dpNoiseMultiplier: number; // Noise scale sigma
  dpClippingBound: number;   // Gradient norm clipping C
  dpTargetDelta: number;     // Delta (e.g., 1e-5)
  
  // SecAgg Hyperparameters
  secAggKeySize: number;     // e.g., 256 bit
  secAggProtocol: 'Additive_Secret_Sharing' | 'DH_Masking';
  
  // Anomaly Detection Algorithm
  anomalyAlgorithm: 'Krum' | 'Multi-Krum' | 'Trimmed_Mean' | 'Cosine_Similarity';
  anomalyThreshold: number;  // e.g., 0.8 cosine threshold
}

export interface AttackConfig {
  enabled: boolean;
  maliciousClientRatio: number; // e.g. 0.2 (20% corrupted nodes)
  attackTypes: {
    modelPoisoning: boolean;    // Sign Flipping / Noise updates
    dataPoisoning: boolean;     // Label Flipping / Feature Corruption
    backdoorAttack: boolean;    // Trigger insertion
    membershipInference: boolean; // MIA vulnerability probe
  };
  modelPoisoningFactor: number; // Scale factor for sign flip / magnitude
  backdoorTargetClass: number;  // Class to force
  backdoorTriggerValue: number; // Value to inject into trigger feature
}

export interface RoundResult {
  round: number;
  globalAccuracy: number;
  globalLoss: number;
  globalPrecision: number;
  globalSensitivity: number;
  globalSpecificity: number;
  globalF1: number;
  globalAUC: number;
  privacyEpsilon: number;
  privacyDelta: number;
  attackSuccessRate: number;
  anomalyDetectionRate: number;
  falsePositiveRate: number;
  roundTimeMs: number;
  commOverheadKB: number;
  clientLogs: {
    clientId: number;
    clientName: string;
    isMalicious: boolean;
    isDetected: boolean;
    localLoss: number;
    localAccuracy: number;
    updateNorm: number;
    masked: boolean;
  }[];
}

export interface FrameworkMetrics {
  name: string;
  accuracy: number;
  precision: number;
  sensitivity: number;
  specificity: number;
  f1Score: number;
  auc: number;
  privacyEpsilon: number;
  privacyDelta: number;
  attackSuccessRate: number;
  anomalyDetectionRate: number;
  falsePositiveRate: number;
  trainingTimeSec: number;
  commOverheadMB: number;
  history: RoundResult[];
}

export interface ExperimentResults {
  timestamp: string;
  datasetName: string;
  activeSecurityMode: string;
  roundsExecuted: number;
  metrics: FrameworkMetrics;
  comparisons?: Record<string, FrameworkMetrics>;
}
