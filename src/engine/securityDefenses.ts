import { SecurityConfig } from '../types/flTypes';

export interface ClientUpdate {
  clientId: number;
  clientName: string;
  weights: number[];
  isMalicious: boolean;
  sampleCount: number;
}

export interface SecurityAggregationResult {
  aggregatedWeights: number[];
  detectedMaliciousClientIds: number[];
  anomalyDetectionRate: number; // ADR %
  falsePositiveRate: number;    // FPR %
  commOverheadKB: number;
}

// 1. Vector Operations Helpers
export function vectorNorm(v: number[]): number {
  return Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
}

export function cosineSimilarity(v1: number[], v2: number[]): number {
  const norm1 = vectorNorm(v1);
  const norm2 = vectorNorm(v2);
  if (norm1 === 0 || norm2 === 0) return 0;
  const dot = v1.reduce((sum, val, idx) => sum + val * v2[idx], 0);
  return dot / (norm1 * norm2);
}

export function euclideanDistance(v1: number[], v2: number[]): number {
  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    const diff = v1[i] - v2[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

// 2. Differential Privacy: Gradient Clipping & Calibrated Noise Addition
export function applyDifferentialPrivacy(
  weights: number[],
  clippingBound: number,
  noiseMultiplier: number,
  numClients: number
): number[] {
  // Step A: Clip weights to norm C
  const norm = vectorNorm(weights);
  const scale = norm > clippingBound ? clippingBound / norm : 1.0;
  const clipped = weights.map(w => w * scale);

  if (noiseMultiplier <= 0) return clipped;

  // Step B: Add Gaussian noise N(0, (sigma * C / K)^2)
  const stdDev = (noiseMultiplier * clippingBound) / Math.max(1, numClients);
  return clipped.map(w => {
    // Box-Muller transform for Gaussian noise
    const u1 = Math.max(1e-10, Math.random());
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return w + z0 * stdDev;
  });
}

// 3. Secure Aggregation (SecAgg Masking & Pairwise Protocol)
export function applySecureAggregation(
  clientUpdates: ClientUpdate[],
  secAggKeySize: number
): { aggregatedWeights: number[]; overheadKB: number } {
  const numClients = clientUpdates.length;
  if (numClients === 0) return { aggregatedWeights: [], overheadKB: 0 };
  const dim = clientUpdates[0].weights.length;

  // Generate pairwise symmetric masks M_ij = -M_ji
  const masks: number[][] = Array.from({ length: numClients }, () => new Array(dim).fill(0));

  for (let i = 0; i < numClients; i++) {
    for (let j = i + 1; j < numClients; j++) {
      for (let d = 0; d < dim; d++) {
        const maskVal = (Math.random() - 0.5) * 50.0;
        masks[i][d] += maskVal;
        masks[j][d] -= maskVal; // Cancels out in summation!
      }
    }
  }

  // Sum masked updates
  const sumWeights = new Array(dim).fill(0);
  let totalSamples = 0;

  clientUpdates.forEach((client, idx) => {
    totalSamples += client.sampleCount;
    for (let d = 0; d < dim; d++) {
      const maskedWeight = client.weights[d] + masks[idx][d];
      sumWeights[d] += maskedWeight * client.sampleCount;
    }
  });

  // Since sum of masks across all clients = 0, server obtains unmasked aggregated sum!
  const aggregatedWeights = sumWeights.map(w => w / Math.max(1, totalSamples));

  // Cryptographic Overhead calculation: Key exchange shares (KB)
  const keyExchangeBytesPerPair = (secAggKeySize / 8) * 4; // Diffie-Hellman pubkey + secret shares
  const totalPairs = (numClients * (numClients - 1)) / 2;
  const overheadKB = parseFloat(((totalPairs * keyExchangeBytesPerPair) / 1024).toFixed(2));

  return { aggregatedWeights, overheadKB };
}

// 4. Anomaly Detection & Byzantine Defenses (Krum, Multi-Krum, Trimmed Mean, Cosine Similarity)
export function processAnomalyDefense(
  updates: ClientUpdate[],
  config: SecurityConfig
): SecurityAggregationResult {
  const numClients = updates.length;
  if (numClients === 0) {
    return { aggregatedWeights: [], detectedMaliciousClientIds: [], anomalyDetectionRate: 0, falsePositiveRate: 0, commOverheadKB: 0 };
  }

  const dim = updates[0].weights.length;
  const detectedIds: number[] = [];

  // Compute reference benchmark update (median vector)
  const medianWeights: number[] = new Array(dim).fill(0);
  for (let d = 0; d < dim; d++) {
    const coords = updates.map(u => u.weights[d]).sort((a, b) => a - b);
    medianWeights[d] = coords[Math.floor(coords.length / 2)];
  }

  let selectedUpdates: ClientUpdate[] = [];

  if (config.anomalyAlgorithm === 'Cosine_Similarity') {
    const threshold = config.anomalyThreshold || 0.5;
    updates.forEach(u => {
      const sim = cosineSimilarity(u.weights, medianWeights);
      if (sim < threshold || u.weights.some(w => isNaN(w) || Math.abs(w) > 100)) {
        detectedIds.push(u.clientId);
      } else {
        selectedUpdates.push(u);
      }
    });
  } else if (config.anomalyAlgorithm === 'Trimmed_Mean') {
    // Trim top 20% and bottom 20% values
    const trimCount = Math.max(1, Math.floor(numClients * 0.2));
    const trimmedDimWeights: number[] = new Array(dim).fill(0);

    for (let d = 0; d < dim; d++) {
      const sorted = updates.map(u => u.weights[d]).sort((a, b) => a - b);
      const sliced = sorted.slice(trimCount, sorted.length - trimCount);
      const avg = sliced.reduce((acc, v) => acc + v, 0) / Math.max(1, sliced.length);
      trimmedDimWeights[d] = avg;
    }

    // Flag outliers based on distance from trimmed mean
    updates.forEach(u => {
      const dist = euclideanDistance(u.weights, trimmedDimWeights);
      if (dist > 15.0) detectedIds.push(u.clientId);
      else selectedUpdates.push(u);
    });
  } else if (config.anomalyAlgorithm === 'Krum' || config.anomalyAlgorithm === 'Multi-Krum') {
    // Krum score: sum of distances to K - f - 2 closest neighbors
    const f = Math.floor(numClients * 0.3); // assumed max Byzantine clients
    const scores: { client: ClientUpdate; score: number }[] = [];

    for (let i = 0; i < numClients; i++) {
      const distances: number[] = [];
      for (let j = 0; j < numClients; j++) {
        if (i !== j) {
          distances.push(euclideanDistance(updates[i].weights, updates[j].weights));
        }
      }
      distances.sort((a, b) => a - b);
      // Sum smallest K - f - 2 distances
      const closestSum = distances.slice(0, Math.max(1, numClients - f - 2)).reduce((a, b) => a + b, 0);
      scores.push({ client: updates[i], score: closestSum });
    }

    scores.sort((a, b) => a.score - b.score);

    const kCount = config.anomalyAlgorithm === 'Multi-Krum' ? Math.max(1, numClients - f) : 1;
    const kept = scores.slice(0, kCount);
    selectedUpdates = kept.map(s => s.client);

    const keptIds = new Set(selectedUpdates.map(u => u.clientId));
    updates.forEach(u => {
      if (!keptIds.has(u.clientId)) detectedIds.push(u.clientId);
    });
  }

  // Fallback if all clients were rejected
  if (selectedUpdates.length === 0) {
    selectedUpdates = updates;
  }

  // Compute final aggregated weights
  let totalSamples = 0;
  const aggregatedWeights: number[] = new Array(dim).fill(0);
  selectedUpdates.forEach(u => {
    totalSamples += u.sampleCount;
    for (let d = 0; d < dim; d++) {
      aggregatedWeights[d] += u.weights[d] * u.sampleCount;
    }
  });

  for (let d = 0; d < dim; d++) {
    aggregatedWeights[d] /= Math.max(1, totalSamples);
  }

  // Calculate Anomaly Detection Rate (ADR) and False Positive Rate (FPR)
  let maliciousCount = 0;
  let benignCount = 0;
  let truePositives = 0;  // Correctly detected malicious
  let falsePositives = 0; // Benign flagged as malicious

  const detectedSet = new Set(detectedIds);

  updates.forEach(u => {
    if (u.isMalicious) {
      maliciousCount++;
      if (detectedSet.has(u.clientId)) truePositives++;
    } else {
      benignCount++;
      if (detectedSet.has(u.clientId)) falsePositives++;
    }
  });

  const anomalyDetectionRate = maliciousCount > 0 ? (truePositives / maliciousCount) * 100 : 100;
  const falsePositiveRate = benignCount > 0 ? (falsePositives / benignCount) * 100 : 0;

  return {
    aggregatedWeights,
    detectedMaliciousClientIds: detectedIds,
    anomalyDetectionRate: parseFloat(anomalyDetectionRate.toFixed(1)),
    falsePositiveRate: parseFloat(falsePositiveRate.toFixed(1)),
    commOverheadKB: 5.2 // Standard filtering validation payload overhead
  };
}
