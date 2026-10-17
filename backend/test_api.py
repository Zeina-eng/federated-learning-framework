"""
API Integration Test using FastAPI TestClient.
Tests /api/health, /api/datasets, and /api/simulate without binding network ports.
"""
import unittest
from fastapi.testclient import TestClient
from app import app


class TestFLApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["engine"], "PyTorch Python Engine")
        self.assertIn("torch_version", data)

    def test_datasets_endpoint(self):
        response = self.client.get("/api/datasets")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("datasets", data)
        self.assertGreaterEqual(len(data["datasets"]), 4)

    def test_simulate_endpoint(self):
        payload = {
            "datasetConfig": {
                "datasetId": "heart_disease",
                "targetVariable": "Heart Disease Present",
                "numClients": 3,
                "distributionType": "NON_IID_DIRICHLET",
                "dirichletAlpha": 0.5
            },
            "flConfig": {
                "numClients": 3,
                "commRounds": 2,
                "localEpochs": 1,
                "learningRate": 0.05,
                "aggregationStrategy": "FedAvg",
                "fedProxMu": 0.05,
                "batchSize": 16,
                "modelType": "logistic"
            },
            "securityConfig": {
                "baselineFL": False,
                "differentialPrivacy": True,
                "secureAggregation": True,
                "anomalyDetection": True,
                "proposedFramework": True,
                "dpNoiseMultiplier": 0.5,
                "dpClippingBound": 3.0,
                "dpTargetDelta": 1e-5,
                "secAggKeySize": 256,
                "secAggProtocol": "DH_Masking",
                "anomalyAlgorithm": "Cosine_Similarity",
                "anomalyThreshold": 0.5
            },
            "attackConfig": {
                "enabled": True,
                "maliciousClientRatio": 0.33,
                "attackTypes": {
                    "modelPoisoning": True,
                    "dataPoisoning": False,
                    "backdoorAttack": False,
                    "membershipInference": True
                },
                "modelPoisoningFactor": 3.0,
                "backdoorTargetClass": 1,
                "backdoorTriggerValue": 999.0
            }
        }
        response = self.client.post("/api/simulate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "Proposed Combined Framework")
        self.assertEqual(len(data["history"]), 2)
        self.assertGreater(data["accuracy"], 30.0)


if __name__ == "__main__":
    unittest.main()
