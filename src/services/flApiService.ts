/**
 * Client API Service to interface with the Python PyTorch Backend Server.
 * Supports health telemetry, real-time WebSocket round streaming, and REST benchmark calls.
 */
import {
  DatasetConfig,
  FLConfig,
  SecurityConfig,
  AttackConfig,
  RoundResult,
  FrameworkMetrics,
  DatasetSample
} from '../types/flTypes';

const API_BASE_URL = 'http://localhost:8000';
const WS_BASE_URL = 'ws://localhost:8000';

export interface PythonHealthStatus {
  healthy: boolean;
  engine?: string;
  torch_version?: string;
  device?: string;
  device_name?: string;
}

/** Check if Python FastAPI server is currently running and active. */
export async function checkPythonBackendHealth(): Promise<PythonHealthStatus> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1500);

    const res = await fetch(`${API_BASE_URL}/api/health`, {
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      return {
        healthy: true,
        engine: data.engine,
        torch_version: data.torch_version,
        device: data.device,
        device_name: data.device_name
      };
    }
    return { healthy: false };
  } catch {
    return { healthy: false };
  }
}

/**
 * Run PyTorch FL Simulation via WebSocket.
 * Streams each round result directly from Python PyTorch to React UI in real-time.
 */
export function runPythonSimulationViaWebSocket(
  datasetConfig: DatasetConfig,
  flConfig: FLConfig,
  securityConfig: SecurityConfig,
  attackConfig: AttackConfig,
  customData: DatasetSample[] | undefined,
  onRound: (result: RoundResult) => void,
  onComplete: (metrics: FrameworkMetrics) => void,
  onError: (err: string) => void
): { cancel: () => void } {
  let ws: WebSocket | null = null;
  let isCancelled = false;

  try {
    ws = new WebSocket(`${WS_BASE_URL}/ws/fl-simulate`);

    ws.onopen = () => {
      if (isCancelled) {
        ws?.close();
        return;
      }
      const payload = {
        datasetConfig,
        flConfig,
        securityConfig,
        attackConfig,
        customData: customData || null
      };
      ws?.send(JSON.stringify(payload));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'ROUND_RESULT') {
          onRound(msg.data as RoundResult);
        } else if (msg.type === 'SIMULATION_COMPLETE') {
          onComplete(msg.data as FrameworkMetrics);
          ws?.close();
        } else if (msg.type === 'ERROR') {
          onError(msg.message || 'Error occurred during Python simulation.');
          ws?.close();
        }
      } catch (err: any) {
        onError(`Failed parsing WebSocket message: ${err.message}`);
      }
    };

    ws.onerror = () => {
      onError('WebSocket connection to Python PyTorch backend failed.');
    };

    ws.onclose = () => {
      // closed
    };
  } catch (err: any) {
    onError(`Could not open WebSocket: ${err.message}`);
  }

  return {
    cancel: () => {
      isCancelled = true;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    }
  };
}

/**
 * Run 5-Framework Comparative Benchmark via Python backend.
 */
export async function runPythonBenchmark(
  datasetConfig: DatasetConfig,
  flConfig: FLConfig,
  attackConfig: AttackConfig,
  customData?: DatasetSample[]
): Promise<Record<string, FrameworkMetrics>> {
  const payload = {
    datasetConfig,
    flConfig,
    securityConfig: {
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
    },
    attackConfig,
    customData: customData || null
  };

  const res = await fetch(`${API_BASE_URL}/api/benchmark/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    throw new Error(`Python benchmark failed: ${res.statusText}`);
  }

  const data = await res.json();
  return data.benchmarks as Record<string, FrameworkMetrics>;
}
