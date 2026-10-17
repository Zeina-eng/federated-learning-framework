"""
Dataset Generation, Preprocessing, and Non-IID Partitioning for Federated Healthcare Nodes.
Includes Dirichlet Non-IID splits, Label Skew, and PyTorch Datasets.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Any

HOSPITAL_NAMES = [
    "Mayo Clinic Diagnostic Center",
    "Johns Hopkins Medicine Center",
    "Mount Sinai Healthcare System",
    "Stanford Health Care Center",
    "Cleveland Clinic Research Unit",
    "Massachusetts General Hospital",
    "UCSF Medical Center Institute",
    "Northwestern Medicine Lab",
    "Penn Medicine Healthcare System",
    "Cedars-Sinai Medical Center"
]


class ClinicalDataset(Dataset):
    """PyTorch Dataset wrapper for clinical tabular features and labels."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def generate_synthetic_data(dataset_id: str, count: int = 350, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """
    Generate realistic clinical synthetic data mirroring benchmark medical datasets.
    Returns: (features, labels, feature_names, target_name)
    """
    np.random.seed(seed)
    
    if dataset_id == "diabetes":
        feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age"]
        target_name = "Diabetes Positive"
        
        pregnancies = np.random.poisson(3.8, count)
        glucose = np.random.normal(120, 30, count).clip(50, 200)
        bp = np.random.normal(70, 12, count).clip(40, 130)
        skin = np.random.normal(20, 15, count).clip(0, 99)
        insulin = np.random.exponential(80, count).clip(0, 800)
        bmi = np.random.normal(32, 7, count).clip(15, 60)
        dpf = np.random.gamma(2, 0.25, count).clip(0.05, 2.5)
        age = np.random.normal(33, 11, count).clip(21, 85)
        
        X = np.column_stack([pregnancies, glucose, bp, skin, insulin, bmi, dpf, age])
        
        # Clinical risk scoring logic
        logits = (glucose - 110) * 0.03 + (bmi - 28) * 0.08 + (age - 30) * 0.02 + dpf * 0.6 + (insulin - 70) * 0.003
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (probs > 0.45).astype(np.int64)
        
    elif dataset_id == "breast_cancer":
        feature_names = ["Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Smoothness", "Compactness", "Concavity", "Symmetry"]
        target_name = "Malignant"
        
        radius = np.random.normal(14.1, 3.5, count).clip(6.0, 28.0)
        texture = np.random.normal(19.2, 4.3, count).clip(9.0, 39.0)
        perimeter = radius * 6.28 + np.random.normal(0, 2, count)
        area = 3.14 * (radius ** 2) + np.random.normal(0, 15, count)
        smoothness = np.random.normal(0.096, 0.014, count).clip(0.05, 0.16)
        compactness = np.random.normal(0.104, 0.052, count).clip(0.01, 0.35)
        concavity = np.random.normal(0.088, 0.079, count).clip(0.0, 0.45)
        symmetry = np.random.normal(0.181, 0.027, count).clip(0.1, 0.3)
        
        X = np.column_stack([radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry])
        logits = (radius - 13.5) * 0.5 + (texture - 18) * 0.15 + (concavity - 0.08) * 6.0 + (compactness - 0.1) * 4.0
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (probs > 0.5).astype(np.int64)

    elif dataset_id == "readmission":
        feature_names = ["Age", "Comorbidities", "PriorAdmissions", "LengthOfStay", "LabRiskScore", "EmergencyVisits"]
        target_name = "30-Day Readmission"
        
        age = np.random.normal(62, 14, count).clip(18, 95)
        comorb = np.random.poisson(2.5, count).clip(0, 10)
        priors = np.random.poisson(1.8, count).clip(0, 15)
        los = np.random.exponential(4.5, count).clip(1, 30)
        lab_score = np.random.normal(5.0, 2.0, count).clip(1.0, 10.0)
        er_visits = np.random.poisson(1.2, count).clip(0, 8)
        
        X = np.column_stack([age, comorb, priors, los, lab_score, er_visits])
        logits = (age - 60) * 0.02 + comorb * 0.4 + priors * 0.35 + los * 0.1 + (lab_score - 5) * 0.3 + er_visits * 0.25
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (probs > 0.5).astype(np.int64)

    else:  # Default: heart_disease
        feature_names = ["Age", "RestingBP", "Cholesterol", "MaxHeartRate", "ST_Depression", "ChestPainType", "FastingBS", "ExerciseAngina"]
        target_name = "Heart Disease Present"
        
        age = np.random.normal(54, 9, count).clip(29, 77)
        trestbps = np.random.normal(131, 17, count).clip(94, 200)
        chol = np.random.normal(246, 51, count).clip(126, 400)
        thalach = np.random.normal(149, 22, count).clip(71, 202)
        oldpeak = np.random.exponential(1.0, count).clip(0.0, 6.2)
        cp = np.random.choice([0, 1, 2, 3], size=count, p=[0.48, 0.17, 0.28, 0.07])
        fbs = (np.random.random(count) > 0.85).astype(float)
        exang = (np.random.random(count) > 0.68).astype(float)
        
        X = np.column_stack([age, trestbps, chol, thalach, oldpeak, cp, fbs, exang])
        logits = (age - 52) * 0.04 + (chol - 220) * 0.01 + (trestbps - 130) * 0.015 + (150 - thalach) * 0.02 + oldpeak * 0.5 + exang * 0.9 + cp * 0.4
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (probs > 0.5).astype(np.int64)
        
    return X, y, feature_names, target_name


def partition_dataset(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    distribution_type: str = "NON_IID_DIRICHLET",
    dirichlet_alpha: float = 0.5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition dataset across clients according to IID or Dirichlet Non-IID distributions.
    
    In Dirichlet Non-IID:
    For each class k, we sample a distribution vector q ~ Dir(alpha) of length num_clients,
    allocating fraction q_i of class k samples to client i.
    """
    np.random.seed(seed)
    num_samples = len(y)
    client_indices = [[] for _ in range(num_clients)]

    if distribution_type == "IID":
        indices = np.random.permutation(num_samples)
        client_splits = np.array_split(indices, num_clients)
        return [(X[split], y[split]) for split in client_splits]

    elif distribution_type == "NON_IID_LABEL_SKEW":
        # Divide classes unevenly (e.g., node 0 has mostly class 0, node 1 mostly class 1)
        classes = np.unique(y)
        for c in classes:
            c_indices = np.where(y == c)[0]
            np.random.shuffle(c_indices)
            # Allocate unevenly
            splits = np.array_split(c_indices, num_clients)
            for cid in range(num_clients):
                # Skew: rotate allocation
                assigned_chunk = splits[(cid + c) % num_clients]
                client_indices[cid].extend(assigned_chunk)
                
    else:  # NON_IID_DIRICHLET (Default & standard federated benchmark)
        classes = np.unique(y)
        for c in classes:
            c_indices = np.where(y == c)[0]
            np.random.shuffle(c_indices)
            
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
            proportions = proportions / proportions.sum()
            
            # Convert proportions into cumulative slice boundaries
            split_points = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]
            c_splits = np.split(c_indices, split_points)
            
            for cid in range(num_clients):
                client_indices[cid].extend(c_splits[cid])

    result = []
    for cid in range(num_clients):
        idxs = np.array(client_indices[cid])
        if len(idxs) == 0:
            # Fallback if Dirichlet allocated 0 samples to a client
            idxs = np.random.choice(num_samples, size=max(5, num_samples // (num_clients * 2)), replace=False)
        result.append((X[idxs], y[idxs]))

    return result


def prepare_federated_loaders(
    dataset_id: str,
    num_clients: int,
    distribution_type: str = "NON_IID_DIRICHLET",
    dirichlet_alpha: float = 0.5,
    test_ratio: float = 0.2,
    batch_size: int = 16,
    custom_samples: Optional[List[Dict[str, Any]]] = None,
    seed: int = 42
) -> Tuple[List[DataLoader], DataLoader, int, List[str], str]:
    """
    Complete pipeline to prepare:
    1. Client DataLoaders for each hospital institution
    2. Global Holdout Test DataLoader for unbiased validation
    3. Input dimension & feature metadata
    """
    if custom_samples and len(custom_samples) > 20:
        X = np.array([s["features"] for s in custom_samples], dtype=np.float32)
        y = np.array([s["label"] for s in custom_samples], dtype=np.int64)
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        target_name = "Target Class"
    else:
        X, y, feature_names, target_name = generate_synthetic_data(dataset_id, count=400, seed=seed)

    # Standardize features for stable neural network training
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train / Test split
    np.random.seed(seed)
    indices = np.random.permutation(len(y))
    test_size = int(len(y) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Partition training data across hospital clients
    client_splits = partition_dataset(
        X_train, y_train,
        num_clients=num_clients,
        distribution_type=distribution_type,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed
    )

    client_loaders = []
    for c_X, c_y in client_splits:
        dataset = ClinicalDataset(c_X, c_y)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
        client_loaders.append(loader)

    test_dataset = ClinicalDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return client_loaders, test_loader, X.shape[1], feature_names, target_name
