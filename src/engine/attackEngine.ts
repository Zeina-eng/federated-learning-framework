import { DatasetSample, AttackConfig } from '../types/flTypes';

export interface AttackResult {
  modifiedSamples: DatasetSample[];
  attackSuccessRate: number;
  description: string;
}

// 1. Data Poisoning: Label Flipping & Feature Noise
export function applyDataPoisoning(
  samples: DatasetSample[],
  config: AttackConfig
): DatasetSample[] {
  if (!config.enabled || !config.attackTypes.dataPoisoning) {
    return samples;
  }

  return samples.map(s => {
    // Flip target label: 0 -> 1, 1 -> 0
    const newLabel = 1 - s.label;
    const noisyFeatures = s.features.map(f => f + (Math.random() - 0.5) * 2.0);
    return {
      ...s,
      label: newLabel,
      features: noisyFeatures
    };
  });
}

// 2. Backdoor Trigger Attack
export function applyBackdoorAttack(
  samples: DatasetSample[],
  config: AttackConfig
): DatasetSample[] {
  if (!config.enabled || !config.attackTypes.backdoorAttack) {
    return samples;
  }

  // Inject trigger pattern into 40% of malicious training samples
  return samples.map((s, idx) => {
    if (idx % 2 === 0) {
      const backdoorFeatures = [...s.features];
      // Inject trigger into 1st feature (e.g., extreme value)
      backdoorFeatures[0] = config.backdoorTriggerValue || 999.0;
      return {
        ...s,
        features: backdoorFeatures,
        label: config.backdoorTargetClass // Force target class
      };
    }
    return s;
  });
}

// 3. Model Poisoning: Sign Flipping & Random Weight Inflation
export function poisonModelUpdate(
  weights: number[],
  config: AttackConfig
): number[] {
  if (!config.enabled || !config.attackTypes.modelPoisoning) {
    return weights;
  }

  const factor = config.modelPoisoningFactor || 3.0;
  // Sign flip and scale weights up to sabotage server aggregation
  return weights.map(w => -w * factor + (Math.random() - 0.5) * 1.5);
}

// 4. Membership Inference Attack (MIA) evaluation
export function evaluateMembershipInferenceAttack(
  memberProbabilities: number[],
  nonMemberProbabilities: number[]
): { miaAccuracy: number; miaAUC: number } {
  // MIA attempts to distinguish if a sample was in the training dataset based on prediction confidence
  let correctGuess = 0;
  const total = memberProbabilities.length + nonMemberProbabilities.length;

  // High confidence predictions (> 0.85 or < 0.15) indicate training set membership
  memberProbabilities.forEach(p => {
    const confidence = Math.max(p, 1 - p);
    if (confidence > 0.75) correctGuess++;
  });

  nonMemberProbabilities.forEach(p => {
    const confidence = Math.max(p, 1 - p);
    if (confidence <= 0.75) correctGuess++;
  });

  const miaAccuracy = total > 0 ? (correctGuess / total) * 100 : 50;
  const miaAUC = Math.min(0.98, Math.max(0.5, (miaAccuracy / 100) * 0.95 + 0.05));

  return {
    miaAccuracy: parseFloat(miaAccuracy.toFixed(1)),
    miaAUC: parseFloat(miaAUC.toFixed(2))
  };
}
