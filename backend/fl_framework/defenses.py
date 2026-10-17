"""
Privacy and Security Defenses for Federated Learning:
1. Differential Privacy (DP-SGD & Moments Accountant)
2. Secure Aggregation (SecAgg Masking & Communication Overhead)
3. Byzantine-Robust Aggregation: Multi-Krum, Coordinate-wise Median, Trimmed Mean, and Cosine Filtering.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from .config import SecurityConfigModel


class DifferentialPrivacyEngine:
    """Implements Local and Central DP gradient clipping and Gaussian noise mechanisms."""
    def __init__(self, config: SecurityConfigModel):
        self.config = config

    def clip_update(self, update: np.ndarray, clipping_bound: float) -> np.ndarray:
        """Gradient/Update L2 Norm Clipping: w * min(1, C / ||w||_2)."""
        norm = np.linalg.norm(update)
        if norm > clipping_bound and norm > 0:
            return update * (clipping_bound / norm)
        return update

    def add_noise(self, update: np.ndarray, clipping_bound: float, noise_multiplier: float, num_clients: int) -> np.ndarray:
        """Add Gaussian noise calibrated to sensitivity C / num_clients."""
        if noise_multiplier <= 0:
            return update
        scale = (noise_multiplier * clipping_bound) / max(1, num_clients)
        noise = np.random.normal(0, scale, size=update.shape)
        return update + noise

    def compute_privacy_budget(self, round_num: int, num_clients: int) -> Tuple[float, float]:
        """
        Compute cumulative privacy budget (epsilon, delta) using composition bounds.
        Epsilon accumulates with sqrt(rounds) according to Advanced Composition Theorem.
        """
        if not self.config.differentialPrivacy:
            return 0.0, 0.0

        sigma = max(0.01, self.config.dpNoiseMultiplier)
        delta = self.config.dpTargetDelta
        
        # Sampling ratio q
        q = min(1.0, 1.0)
        # Advanced composition / moments accountant bound approximation
        epsilon = (q * np.sqrt(round_num * 2.0 * np.log(1.0 / max(1e-12, delta)))) / sigma
        return round(float(epsilon), 3), delta


class SecureAggregationEngine:
    """
    Simulates SecAgg Protocol (Bonawitz et al.) using pairwise blinding masks.
    Calculates cryptographic communication and verification overhead in KB.
    """
    def __init__(self, config: SecurityConfigModel):
        self.config = config

    def compute_overhead(self, num_clients: int, num_parameters: int) -> float:
        """Calculate communication overhead in Kilobytes for key exchange and masked vectors."""
        if not self.config.secureAggregation:
            return round(float(num_parameters * 4 / 1024.0), 2)  # Raw float32 update size

        key_size_bytes = self.config.secAggKeySize // 8
        # Round 1: Public keys exchange O(K^2)
        key_exchange_bytes = num_clients * (num_clients - 1) * key_size_bytes
        # Round 2: Masked model parameters transmission
        masked_vector_bytes = num_clients * num_parameters * 4
        # Round 3: Consistency checks & unmasking shares
        secret_shares_bytes = num_clients * (num_clients - 1) * 32

        total_bytes = (key_exchange_bytes + masked_vector_bytes + secret_shares_bytes) / max(1, num_clients)
        return round(float(total_bytes / 1024.0), 2)


class RobustAggregator:
    """
    Byzantine-Robust Aggregation Algorithms:
    - Krum / Multi-Krum
    - Coordinate-wise Median
    - Trimmed Mean
    - Cosine-Similarity Anomaly Filtering
    """
    def __init__(self, config: SecurityConfigModel):
        self.config = config

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        client_weights: List[float],
        is_malicious_flags: List[bool]
    ) -> Tuple[np.ndarray, List[int], float, float]:
        """
        Execute robust aggregation defense on candidate client updates.
        Returns:
            aggregated_weights: 1D numpy array
            detected_indices: list of client indices identified as malicious
            adr: Anomaly Detection Rate (%)
            fpr: False Positive Rate (%)
        """
        n = len(client_updates)
        if n == 0:
            raise ValueError("No client updates to aggregate.")
        if n == 1 or not self.config.anomalyDetection:
            # Standard FedAvg weighted mean
            total_w = sum(client_weights)
            norm_w = [w / total_w for w in client_weights]
            agg = sum(w * u for w, u in zip(norm_w, client_updates))
            return agg, [], 0.0, 0.0

        algo = self.config.anomalyAlgorithm
        detected_indices = []

        if algo == "Krum" or algo == "Multi-Krum":
            agg, detected_indices = self._multi_krum(client_updates, client_weights, algo == "Multi-Krum")
        elif algo == "Median":
            agg, detected_indices = self._coordinate_median(client_updates)
        elif algo == "Trimmed_Mean":
            agg, detected_indices = self._trimmed_mean(client_updates)
        else:  # Default: Cosine_Similarity Filtering
            agg, detected_indices = self._cosine_filtering(client_updates, client_weights)

        # Calculate Anomaly Detection Rate (Recall) & False Positive Rate
        actual_malicious = [i for i, m in enumerate(is_malicious_flags) if m]
        actual_benign = [i for i, m in enumerate(is_malicious_flags) if not m]

        true_positives = sum(1 for i in detected_indices if i in actual_malicious)
        false_positives = sum(1 for i in detected_indices if i in actual_benign)

        adr = (true_positives / len(actual_malicious) * 100.0) if actual_malicious else 100.0
        fpr = (false_positives / len(actual_benign) * 100.0) if actual_benign else 0.0

        return agg, detected_indices, round(adr, 2), round(fpr, 2)

    def _cosine_filtering(
        self,
        updates: List[np.ndarray],
        weights: List[float]
    ) -> Tuple[np.ndarray, List[int]]:
        """Filters updates that deviate significantly in direction from the median update."""
        n = len(updates)
        # Compute median reference vector
        stacked = np.stack(updates, axis=0)
        reference_vec = np.median(stacked, axis=0)
        ref_norm = np.linalg.norm(reference_vec)

        cosine_sims = []
        for u in updates:
            u_norm = np.linalg.norm(u)
            if u_norm == 0 or ref_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(u, reference_vec) / (u_norm * ref_norm))
            cosine_sims.append(sim)

        threshold = self.config.anomalyThreshold
        accepted_idx = [i for i, sim in enumerate(cosine_sims) if sim >= threshold]
        rejected_idx = [i for i, sim in enumerate(cosine_sims) if sim < threshold]

        # Ensure at least 1 update is retained
        if not accepted_idx:
            best_idx = int(np.argmax(cosine_sims))
            accepted_idx = [best_idx]
            rejected_idx = [i for i in range(n) if i != best_idx]

        # Weighted average of accepted updates
        total_w = sum(weights[i] for i in accepted_idx)
        agg = sum((weights[i] / total_w) * updates[i] for i in accepted_idx)
        return agg, rejected_idx

    def _multi_krum(
        self,
        updates: List[np.ndarray],
        weights: List[float],
        multi: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """Multi-Krum aggregation (Blanchard et al., 2017)."""
        n = len(updates)
        # Assume f < n / 2 Byzantine nodes
        f = max(1, (n - 1) // 3)
        m = n - f - 2
        m = max(1, min(m, n - 2))

        # Pairwise Euclidean distances
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(updates[i] - updates[j]))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Compute Krum score for each client
        scores = []
        for i in range(n):
            sorted_dists = np.sort(dist_matrix[i])
            # Sum of distances to closest m neighbors (excluding self at 0)
            score = float(np.sum(sorted_dists[1:m + 1]))
            scores.append((score, i))

        scores.sort(key=lambda x: x[0])

        if multi:
            num_selected = max(1, n - f)
            selected_indices = [idx for _, idx in scores[:num_selected]]
            rejected_indices = [idx for _, idx in scores[num_selected:]]
            total_w = sum(weights[i] for i in selected_indices)
            agg = sum((weights[i] / total_w) * updates[i] for i in selected_indices)
            return agg, rejected_indices
        else:
            best_idx = scores[0][1]
            rejected_indices = [i for i in range(n) if i != best_idx]
            return updates[best_idx].copy(), rejected_indices

    def _coordinate_median(self, updates: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        """Coordinate-wise median aggregator."""
        stacked = np.stack(updates, axis=0)  # (N, D)
        median_vec = np.median(stacked, axis=0)
        # Distance to median
        dists = [float(np.linalg.norm(u - median_vec)) for u in updates]
        dist_threshold = np.mean(dists) + 1.5 * np.std(dists)
        rejected = [i for i, d in enumerate(dists) if d > dist_threshold]
        return median_vec, rejected

    def _trimmed_mean(self, updates: List[np.ndarray], beta: float = 0.15) -> Tuple[np.ndarray, List[int]]:
        """Trimmed Mean aggregator: trims top and bottom beta fraction."""
        stacked = np.stack(updates, axis=0)  # (N, D)
        n = len(updates)
        trim_k = max(1, int(n * beta))

        sorted_vals = np.sort(stacked, axis=0)
        trimmed = sorted_vals[trim_k:n - trim_k] if n - 2 * trim_k > 0 else sorted_vals
        agg = np.mean(trimmed, axis=0)

        dists = [float(np.linalg.norm(u - agg)) for u in updates]
        dist_threshold = np.mean(dists) + 1.5 * np.std(dists)
        rejected = [i for i, d in enumerate(dists) if d > dist_threshold]
        return agg, rejected
