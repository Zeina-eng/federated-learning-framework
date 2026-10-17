"""
Configuration dataclasses and Pydantic schemas for the Federated Learning Framework.
"""
from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field


class DatasetSampleModel(BaseModel):
    id: int
    features: List[float]
    label: int


class DatasetConfigModel(BaseModel):
    datasetId: str = "heart_disease"
    targetVariable: str = "Heart Disease Present"
    numClients: int = 5
    distributionType: Literal["IID", "NON_IID_DIRICHLET", "NON_IID_LABEL_SKEW"] = "NON_IID_DIRICHLET"
    dirichletAlpha: float = 0.5


class FLConfigModel(BaseModel):
    numClients: int = 5
    commRounds: int = 12
    localEpochs: int = 3
    learningRate: float = 0.02
    aggregationStrategy: Literal["FedAvg", "FedProx"] = "FedAvg"
    fedProxMu: float = 0.05
    batchSize: int = 16
    modelType: Literal["logistic", "mlp"] = "logistic"


class SecurityConfigModel(BaseModel):
    baselineFL: bool = False
    differentialPrivacy: bool = True
    secureAggregation: bool = True
    anomalyDetection: bool = True
    proposedFramework: bool = True

    # DP Hyperparameters
    dpNoiseMultiplier: float = 0.5
    dpClippingBound: float = 3.0
    dpTargetDelta: float = 1e-5

    # SecAgg Hyperparameters
    secAggKeySize: int = 256
    secAggProtocol: Literal["Additive_Secret_Sharing", "DH_Masking"] = "DH_Masking"

    # Byzantine Anomaly Detection Algorithm
    anomalyAlgorithm: Literal["Krum", "Multi-Krum", "Trimmed_Mean", "Cosine_Similarity", "Median"] = "Cosine_Similarity"
    anomalyThreshold: float = 0.5


class AttackTypesModel(BaseModel):
    modelPoisoning: bool = True
    dataPoisoning: bool = False
    backdoorAttack: bool = False
    membershipInference: bool = True


class AttackConfigModel(BaseModel):
    enabled: bool = True
    maliciousClientRatio: float = 0.2
    attackTypes: AttackTypesModel = Field(default_factory=AttackTypesModel)
    modelPoisoningFactor: float = 3.0
    backdoorTargetClass: int = 1
    backdoorTriggerValue: float = 999.0


class ClientLogModel(BaseModel):
    clientId: int
    clientName: str
    isMalicious: bool
    isDetected: bool
    localLoss: float
    localAccuracy: float
    updateNorm: float
    masked: bool


class RoundResultModel(BaseModel):
    round: int
    globalAccuracy: float
    globalLoss: float
    globalPrecision: float
    globalSensitivity: float
    globalSpecificity: float
    globalF1: float
    globalAUC: float
    privacyEpsilon: float
    privacyDelta: float
    attackSuccessRate: float
    anomalyDetectionRate: float
    falsePositiveRate: float
    roundTimeMs: float
    commOverheadKB: float
    clientLogs: List[Dict[str, Any]]


class FrameworkMetricsModel(BaseModel):
    name: str
    accuracy: float
    precision: float
    sensitivity: float
    specificity: float
    f1Score: float
    auc: float
    privacyEpsilon: float
    privacyDelta: float
    attackSuccessRate: float
    anomalyDetectionRate: float
    falsePositiveRate: float
    trainingTimeSec: float
    commOverheadMB: float
    history: List[RoundResultModel]


class SimulationRequest(BaseModel):
    datasetConfig: DatasetConfigModel
    flConfig: FLConfigModel
    securityConfig: SecurityConfigModel
    attackConfig: AttackConfigModel
    customData: Optional[List[DatasetSampleModel]] = None
    customFeatureNames: Optional[List[str]] = None
