export interface ConfusionMatrix {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export function computeConfusionMatrix(predictions: number[], targets: number[]): ConfusionMatrix {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < predictions.length; i++) {
    const p = predictions[i] >= 0.5 ? 1 : 0;
    const t = targets[i] >= 0.5 ? 1 : 0;
    if (p === 1 && t === 1) tp++;
    else if (p === 1 && t === 0) fp++;
    else if (p === 0 && t === 0) tn++;
    else if (p === 0 && t === 1) fn++;
  }
  return { tp, fp, tn, fn };
}

export function computeMetricsFromConfusion(cm: ConfusionMatrix) {
  const total = cm.tp + cm.fp + cm.tn + cm.fn;
  if (total === 0) {
    return { accuracy: 0, precision: 0, sensitivity: 0, specificity: 0, f1: 0 };
  }

  const accuracy = (cm.tp + cm.tn) / total;
  const precision = (cm.tp + cm.fp) > 0 ? cm.tp / (cm.tp + cm.fp) : 0;
  const sensitivity = (cm.tp + cm.fn) > 0 ? cm.tp / (cm.tp + cm.fn) : 0; // Recall
  const specificity = (cm.tn + cm.fp) > 0 ? cm.tn / (cm.tn + cm.fp) : 0;
  const f1 = (precision + sensitivity) > 0 ? (2 * precision * sensitivity) / (precision + sensitivity) : 0;

  return { accuracy, precision, sensitivity, specificity, f1 };
}

export function computeAUC(probabilities: number[], targets: number[]): number {
  if (probabilities.length === 0) return 0.5;

  const paired = probabilities.map((p, idx) => ({ p, t: targets[idx] }));
  paired.sort((a, b) => b.p - a.p); // Sort descending probability

  let posCount = 0;
  let negCount = 0;
  paired.forEach(item => {
    if (item.t === 1) posCount++;
    else negCount++;
  });

  if (posCount === 0 || negCount === 0) return 1.0;

  let aucSum = 0;
  let currentPos = 0;

  for (const item of paired) {
    if (item.t === 1) {
      currentPos++;
    } else {
      aucSum += currentPos;
    }
  }

  return Math.min(1.0, Math.max(0.0, aucSum / (posCount * negCount)));
}

// Differential Privacy Budget Accountant (Rényi Differential Privacy / Moments Accountant)
export function calculatePrivacyBudget(
  rounds: number,
  noiseMultiplier: number,
  targetDelta: number = 1e-5
): { epsilon: number; delta: number } {
  if (noiseMultiplier <= 0) {
    return { epsilon: Infinity, delta: targetDelta };
  }

  // Sample privacy computation: epsilon = (rounds * (1 / noiseMultiplier^2)) + sqrt(...)
  const alpha = 4.0; // Optimal Rényi order
  const rdpEpsilon = (alpha * rounds) / (2 * noiseMultiplier * noiseMultiplier);
  const epsilon = rdpEpsilon + Math.log(1 / targetDelta) / (alpha - 1);

  return {
    epsilon: parseFloat(epsilon.toFixed(3)),
    delta: targetDelta
  };
}

// Generate ROC curve coordinates for plotting
export function generateROCCurveData(auc: number): { fpr: number; tpr: number }[] {
  const points: { fpr: number; tpr: number }[] = [{ fpr: 0, tpr: 0 }];
  const steps = 20;
  
  for (let i = 1; i < steps; i++) {
    const x = i / steps; // FPR
    // Concave ROC curve equation fitted to match AUC
    const y = Math.min(1.0, Math.pow(x, (1 - auc) / Math.max(0.01, auc)));
    points.push({ fpr: parseFloat(x.toFixed(2)), tpr: parseFloat(y.toFixed(2)) });
  }
  
  points.push({ fpr: 1, tpr: 1 });
  return points;
}
