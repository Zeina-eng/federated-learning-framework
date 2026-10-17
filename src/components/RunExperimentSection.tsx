import React, { useEffect, useRef } from 'react';
import { Rocket, Play, Pause, Terminal, Activity } from 'lucide-react';
import { RoundResult, DatasetConfig, SecurityConfig, AttackConfig } from '../types/flTypes';

interface RunExperimentSectionProps {
  onStartExperiment: () => void;
  onPauseExperiment: () => void;
  isTraining: boolean;
  isPaused: boolean;
  currentRound: number;
  totalRounds: number;
  latestRoundResult: RoundResult | null;
  datasetConfig: DatasetConfig;
  securityConfig: SecurityConfig;
  attackConfig?: AttackConfig;
  logMessages: string[];
}

export const RunExperimentSection: React.FC<RunExperimentSectionProps> = ({
  onStartExperiment,
  onPauseExperiment,
  isTraining,
  isPaused,
  currentRound,
  totalRounds,
  latestRoundResult,
  datasetConfig,
  securityConfig,
  attackConfig: _attackConfig,
  logMessages
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll terminal log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logMessages]);

  // Animated Topology Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let particleOffset = 0;

    const renderTopology = () => {
      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;
      const numClients = datasetConfig.numClients;
      const radius = Math.min(width, height) * 0.35;

      ctx.clearRect(0, 0, width, height);

      // 1. Draw Network Connections & Transmitting Particles
      for (let i = 0; i < numClients; i++) {
        const angle = (i * 2 * Math.PI) / numClients - Math.PI / 2;
        const nodeX = centerX + radius * Math.cos(angle);
        const nodeY = centerY + radius * Math.sin(angle);

        const clientLog = latestRoundResult?.clientLogs.find(c => c.clientId === i + 1);
        const isMalicious = clientLog ? clientLog.isMalicious : false;
        const isDetected = clientLog ? clientLog.isDetected : false;

        // Line style
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(nodeX, nodeY);
        ctx.lineWidth = 1.5;

        if (isDetected) {
          ctx.strokeStyle = '#ef4444'; // Red isolated
          ctx.setLineDash([4, 4]);
        } else if (isMalicious) {
          ctx.strokeStyle = '#f59e0b'; // Amber suspicious
          ctx.setLineDash([]);
        } else if (securityConfig.secureAggregation || securityConfig.proposedFramework) {
          ctx.strokeStyle = '#38bdf8'; // Cyan encrypted
          ctx.setLineDash([]);
        } else {
          ctx.strokeStyle = '#334155'; // Slate normal
          ctx.setLineDash([]);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Animated packet dots
        if (isTraining && !isPaused) {
          const progress = (particleOffset + i * 0.2) % 1.0;
          const px = centerX + (nodeX - centerX) * progress;
          const py = centerY + (nodeY - centerY) * progress;

          ctx.beginPath();
          ctx.arc(px, py, 4, 0, 2 * Math.PI);
          ctx.fillStyle = isDetected ? '#ef4444' : isMalicious ? '#f59e0b' : '#38bdf8';
          ctx.fill();
        }
      }

      // 2. Draw Central Server Node
      ctx.beginPath();
      ctx.arc(centerX, centerY, 28, 0, 2 * Math.PI);
      const serverGrad = ctx.createRadialGradient(centerX, centerY, 5, centerX, centerY, 28);
      serverGrad.addColorStop(0, '#06b6d4');
      serverGrad.addColorStop(1, '#0f172a');
      ctx.fillStyle = serverGrad;
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#38bdf8';
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('SERVER', centerX, centerY - 2);
      ctx.font = '9px monospace';
      ctx.fillStyle = '#38bdf8';
      ctx.fillText(`R ${currentRound}/${totalRounds}`, centerX, centerY + 10);

      // 3. Draw Hospital Client Nodes
      for (let i = 0; i < numClients; i++) {
        const angle = (i * 2 * Math.PI) / numClients - Math.PI / 2;
        const nodeX = centerX + radius * Math.cos(angle);
        const nodeY = centerY + radius * Math.sin(angle);

        const clientLog = latestRoundResult?.clientLogs.find(c => c.clientId === i + 1);
        const isMalicious = clientLog ? clientLog.isMalicious : false;
        const isDetected = clientLog ? clientLog.isDetected : false;

        ctx.beginPath();
        ctx.arc(nodeX, nodeY, 18, 0, 2 * Math.PI);

        if (isDetected) {
          ctx.fillStyle = '#450a0a';
          ctx.strokeStyle = '#ef4444';
        } else if (isMalicious) {
          ctx.fillStyle = '#451a03';
          ctx.strokeStyle = '#f59e0b';
        } else {
          ctx.fillStyle = '#064e3b';
          ctx.strokeStyle = '#10b981';
        }
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px monospace';
        ctx.fillText(`H${i + 1}`, nodeX, nodeY);

        // Hospital Label
        ctx.font = '9px sans-serif';
        ctx.fillStyle = isDetected ? '#fca5a5' : isMalicious ? '#fcd34d' : '#94a3b8';
        ctx.fillText(`Hosp ${i + 1}`, nodeX, nodeY + 28);
      }

      particleOffset += 0.015;
      animationFrameId = requestAnimationFrame(renderTopology);
    };

    renderTopology();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [datasetConfig, securityConfig, latestRoundResult, isTraining, isPaused, currentRound, totalRounds]);

  const progressPercent = totalRounds > 0 ? (currentRound / totalRounds) * 100 : 0;

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800/80 shadow-xl mb-6">
      
      {/* Action Header & Launch Button */}
      <div className="flex flex-col md:flex-row items-center justify-between gap-4 mb-6 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-teal-500 text-slate-950 font-bold">
            <Rocket className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              5. Federated Learning Execution Engine
            </h2>
            <p className="text-xs text-slate-400">
              Run real-time multi-institutional FL training, compute differential privacy budgets, and observe Byzantine defense responses.
            </p>
          </div>
        </div>

        {/* Primary Run / Pause Button */}
        <div className="flex items-center gap-3">
          {!isTraining ? (
            <button
              type="button"
              onClick={onStartExperiment}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-400 text-slate-950 font-black text-sm tracking-wide shadow-lg shadow-cyan-500/25 hover:brightness-110 active:scale-95 transition flex items-center gap-2 glow-cyan"
            >
              <Rocket className="w-4 h-4 fill-slate-950" /> 🚀 Run Federated Learning Experiment
            </button>
          ) : (
            <button
              type="button"
              onClick={onPauseExperiment}
              className="px-6 py-3 rounded-xl bg-amber-500 text-slate-950 font-extrabold text-sm shadow-lg shadow-amber-500/20 hover:bg-amber-400 transition flex items-center gap-2"
            >
              {isPaused ? <Play className="w-4 h-4 fill-slate-950" /> : <Pause className="w-4 h-4 fill-slate-950" />}
              {isPaused ? 'Resume Experiment' : 'Pause Training'}
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar & Status */}
      <div className="mb-6 bg-slate-900/90 rounded-xl p-4 border border-slate-800">
        <div className="flex items-center justify-between text-xs font-semibold text-slate-300 mb-2">
          <span className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
            Training Progress Status: <span className="text-white font-mono">{isTraining ? (isPaused ? 'PAUSED' : 'TRAINING IN PROGRESS') : 'READY TO RUN'}</span>
          </span>
          <span className="font-mono text-cyan-400">
            Round {currentRound} of {totalRounds} ({Math.round(progressPercent)}%)
          </span>
        </div>
        <div className="w-full h-3 bg-slate-950 rounded-full overflow-hidden border border-slate-800 p-0.5">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 via-teal-400 to-emerald-400 rounded-full transition-all duration-300 shadow-sm"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Grid: Animated Canvas Topology + Terminal Log Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Topology Visualizer */}
        <div className="lg:col-span-6 bg-slate-950/80 rounded-xl p-4 border border-slate-800 flex flex-col items-center justify-center">
          <div className="w-full flex items-center justify-between mb-2 text-xs font-semibold text-slate-300 border-b border-slate-800/80 pb-2">
            <span>Federated Network Topology</span>
            <span className="text-[10px] text-cyan-400 font-mono">
              {datasetConfig.numClients} Hospital Nodes
            </span>
          </div>
          <canvas
            ref={canvasRef}
            width={400}
            height={320}
            className="w-full max-w-[400px] h-[320px] rounded-lg"
          />
          <div className="flex items-center justify-center gap-4 text-[11px] font-medium text-slate-400 mt-3 flex-wrap">
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span> Benign Hosp</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span> Malicious Update</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span> Rejected Outlier</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span> SecAgg Masked</span>
          </div>
        </div>

        {/* Live Terminal Log Stream */}
        <div className="lg:col-span-6 bg-slate-950/90 rounded-xl p-4 border border-slate-800 font-mono text-xs text-slate-300 flex flex-col justify-between h-[390px]">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800 text-slate-400">
            <span className="flex items-center gap-1.5 font-sans font-bold text-white text-xs">
              <Terminal className="w-3.5 h-3.5 text-cyan-400" /> FL Protocol Terminal Stream
            </span>
            <span className="text-[10px] text-emerald-400">● LIVE LOGS</span>
          </div>

          <div className="flex-1 overflow-y-auto py-2 space-y-1.5 scrollbar-thin text-[11px]">
            {logMessages.map((msg, idx) => (
              <div key={idx} className="leading-relaxed border-b border-slate-900/60 pb-1">
                {msg.includes('ERROR') || msg.includes('ATTACK') ? (
                  <span className="text-rose-400">{msg}</span>
                ) : msg.includes('ANOMALY') || msg.includes('REJECTED') ? (
                  <span className="text-amber-300">{msg}</span>
                ) : msg.includes('SUCCESS') || msg.includes('ROUND') ? (
                  <span className="text-cyan-300 font-semibold">{msg}</span>
                ) : (
                  <span className="text-slate-400">{msg}</span>
                )}
              </div>
            ))}
            <div ref={logEndRef} />
          </div>

          <div className="pt-2 border-t border-slate-800/80 text-[10px] text-slate-400 flex justify-between font-sans">
            <span>Aggregator: {securityConfig.proposedFramework ? 'Combined Defense' : securityConfig.anomalyDetection ? securityConfig.anomalyAlgorithm : 'FedAvg'}</span>
            <span>Privacy Accountant: RDP Active</span>
          </div>
        </div>

      </div>
    </div>
  );
};
