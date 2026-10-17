import React from 'react';
import { Shield, Key, Eye, Sparkles, AlertTriangle, CheckSquare, Square } from 'lucide-react';
import { SecurityConfig } from '../types/flTypes';

interface SecuritySectionProps {
  securityConfig: SecurityConfig;
  onChangeConfig: (newConfig: SecurityConfig) => void;
  isTraining: boolean;
}

export const SecuritySection: React.FC<SecuritySectionProps> = ({
  securityConfig,
  onChangeConfig,
  isTraining
}) => {

  const handleSelectMode = (mode: 'baseline' | 'dp' | 'secagg' | 'anomaly' | 'proposed') => {
    if (isTraining) return;

    if (mode === 'baseline') {
      onChangeConfig({
        ...securityConfig,
        baselineFL: true,
        differentialPrivacy: false,
        secureAggregation: false,
        anomalyDetection: false,
        proposedFramework: false
      });
    } else if (mode === 'dp') {
      onChangeConfig({
        ...securityConfig,
        baselineFL: false,
        differentialPrivacy: true,
        secureAggregation: false,
        anomalyDetection: false,
        proposedFramework: false
      });
    } else if (mode === 'secagg') {
      onChangeConfig({
        ...securityConfig,
        baselineFL: false,
        differentialPrivacy: false,
        secureAggregation: true,
        anomalyDetection: false,
        proposedFramework: false
      });
    } else if (mode === 'anomaly') {
      onChangeConfig({
        ...securityConfig,
        baselineFL: false,
        differentialPrivacy: false,
        secureAggregation: false,
        anomalyDetection: true,
        proposedFramework: false
      });
    } else if (mode === 'proposed') {
      onChangeConfig({
        ...securityConfig,
        baselineFL: false,
        differentialPrivacy: true,
        secureAggregation: true,
        anomalyDetection: true,
        proposedFramework: true
      });
    }
  };

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800/80 shadow-xl mb-6">
      {/* Section Header */}
      <div className="flex items-center gap-3 mb-5 border-b border-slate-800 pb-3">
        <div className="p-2.5 rounded-xl bg-cyan-950/80 border border-cyan-500/30 text-cyan-400">
          <Shield className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            3. Security Experiments & Privacy Defenses
          </h2>
          <p className="text-xs text-slate-400">
            Select security framework configuration to test Differential Privacy (DP), Secure Aggregation (SecAgg), and Byzantine Anomaly Defense against malicious vectors.
          </p>
        </div>
      </div>

      {/* Security Modes Radio/Checkbox Options */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
        
        {/* 1. Baseline FL */}
        <button
          type="button"
          onClick={() => handleSelectMode('baseline')}
          disabled={isTraining}
          className={`p-4 rounded-xl border text-left transition flex flex-col justify-between ${
            securityConfig.baselineFL && !securityConfig.proposedFramework
              ? 'bg-slate-900 border-slate-600 ring-2 ring-slate-500 text-white'
              : 'bg-slate-950/70 border-slate-800 text-slate-400 hover:border-slate-700'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="font-bold text-xs">Baseline FL</span>
            {securityConfig.baselineFL && !securityConfig.proposedFramework ? (
              <CheckSquare className="w-4 h-4 text-slate-300" />
            ) : (
              <Square className="w-4 h-4 text-slate-600" />
            )}
          </div>
          <p className="text-[11px] text-slate-400 mt-2 font-normal">
            Standard FedAvg. Zero differential privacy, zero encryption, zero anomaly defense.
          </p>
        </button>

        {/* 2. FL + Differential Privacy */}
        <button
          type="button"
          onClick={() => handleSelectMode('dp')}
          disabled={isTraining}
          className={`p-4 rounded-xl border text-left transition flex flex-col justify-between ${
            securityConfig.differentialPrivacy && !securityConfig.proposedFramework
              ? 'bg-cyan-950/80 border-cyan-500 ring-2 ring-cyan-500/50 text-cyan-200'
              : 'bg-slate-950/70 border-slate-800 text-slate-400 hover:border-slate-700'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="font-bold text-xs text-cyan-300">FL + DP</span>
            {securityConfig.differentialPrivacy && !securityConfig.proposedFramework ? (
              <CheckSquare className="w-4 h-4 text-cyan-400" />
            ) : (
              <Square className="w-4 h-4 text-slate-600" />
            )}
          </div>
          <p className="text-[11px] text-slate-400 mt-2 font-normal">
            Gradient norm clipping $C$ & Gaussian noise addition for ($\epsilon, \delta$) guarantees.
          </p>
        </button>

        {/* 3. FL + Secure Aggregation */}
        <button
          type="button"
          onClick={() => handleSelectMode('secagg')}
          disabled={isTraining}
          className={`p-4 rounded-xl border text-left transition flex flex-col justify-between ${
            securityConfig.secureAggregation && !securityConfig.proposedFramework
              ? 'bg-indigo-950/80 border-indigo-500 ring-2 ring-indigo-500/50 text-indigo-200'
              : 'bg-slate-950/70 border-slate-800 text-slate-400 hover:border-slate-700'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="font-bold text-xs text-indigo-300">FL + SecAgg</span>
            {securityConfig.secureAggregation && !securityConfig.proposedFramework ? (
              <CheckSquare className="w-4 h-4 text-indigo-400" />
            ) : (
              <Square className="w-4 h-4 text-slate-600" />
            )}
          </div>
          <p className="text-[11px] text-slate-400 mt-2 font-normal">
            Pairwise Diffie-Hellman secret sharing masks. Server never sees unmasked client weights.
          </p>
        </button>

        {/* 4. FL + Anomaly Detection */}
        <button
          type="button"
          onClick={() => handleSelectMode('anomaly')}
          disabled={isTraining}
          className={`p-4 rounded-xl border text-left transition flex flex-col justify-between ${
            securityConfig.anomalyDetection && !securityConfig.proposedFramework
              ? 'bg-emerald-950/80 border-emerald-500 ring-2 ring-emerald-500/50 text-emerald-200'
              : 'bg-slate-950/70 border-slate-800 text-slate-400 hover:border-slate-700'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="font-bold text-xs text-emerald-300">FL + Anomaly Def.</span>
            {securityConfig.anomalyDetection && !securityConfig.proposedFramework ? (
              <CheckSquare className="w-4 h-4 text-emerald-400" />
            ) : (
              <Square className="w-4 h-4 text-slate-600" />
            )}
          </div>
          <p className="text-[11px] text-slate-400 mt-2 font-normal">
            Krum / Trimmed Mean / Cosine distance filtering to reject Byzantine malicious client updates.
          </p>
        </button>

        {/* 5. Proposed Combined Framework */}
        <button
          type="button"
          onClick={() => handleSelectMode('proposed')}
          disabled={isTraining}
          className={`p-4 rounded-xl border text-left transition flex flex-col justify-between ${
            securityConfig.proposedFramework
              ? 'bg-gradient-to-br from-cyan-950 via-teal-950 to-emerald-950 border-cyan-400 ring-2 ring-cyan-400 text-white shadow-lg shadow-cyan-950/50'
              : 'bg-slate-950/70 border-slate-800 text-slate-400 hover:border-slate-700'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="font-bold text-xs text-cyan-200 flex items-center gap-1">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Proposed System
            </span>
            {securityConfig.proposedFramework ? (
              <CheckSquare className="w-4 h-4 text-emerald-400" />
            ) : (
              <Square className="w-4 h-4 text-slate-600" />
            )}
          </div>
          <p className="text-[11px] text-slate-300 mt-2 font-medium">
            Synergistic integration of DP + SecAgg + Anomaly Defense Matrix for maximum protection.
          </p>
        </button>

      </div>

      {/* Hyperparameter Settings Panel for Active Security Features */}
      <div className="bg-slate-900/90 rounded-xl p-4 border border-slate-800 grid grid-cols-1 md:grid-cols-3 gap-5">
        
        {/* Differential Privacy Parameters */}
        <div className={`p-3.5 rounded-lg border transition ${
          securityConfig.differentialPrivacy || securityConfig.proposedFramework
            ? 'bg-cyan-950/40 border-cyan-500/40'
            : 'bg-slate-950/40 border-slate-800/60 opacity-60'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-cyan-300 flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5" /> DP Noise Scale ($\sigma$) & Clipping ($C$)
            </span>
            <span className="text-[10px] font-mono text-cyan-400 font-bold">
              $\sigma = {securityConfig.dpNoiseMultiplier}$ | $C = {securityConfig.dpClippingBound}$
            </span>
          </div>

          <div className="space-y-2">
            <div>
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>Noise Scale ($\sigma$):</span>
                <span>{securityConfig.dpNoiseMultiplier}</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={2.0}
                step={0.1}
                value={securityConfig.dpNoiseMultiplier}
                onChange={e => onChangeConfig({ ...securityConfig, dpNoiseMultiplier: parseFloat(e.target.value) })}
                disabled={isTraining || (!securityConfig.differentialPrivacy && !securityConfig.proposedFramework)}
                className="w-full h-1 bg-slate-800 rounded appearance-none accent-cyan-400"
              />
            </div>
            <div>
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>Clipping Bound ($C$):</span>
                <span>{securityConfig.dpClippingBound}</span>
              </div>
              <input
                type="range"
                min={1.0}
                max={10.0}
                step={0.5}
                value={securityConfig.dpClippingBound}
                onChange={e => onChangeConfig({ ...securityConfig, dpClippingBound: parseFloat(e.target.value) })}
                disabled={isTraining || (!securityConfig.differentialPrivacy && !securityConfig.proposedFramework)}
                className="w-full h-1 bg-slate-800 rounded appearance-none accent-cyan-400"
              />
            </div>
          </div>
        </div>

        {/* Secure Aggregation Parameters */}
        <div className={`p-3.5 rounded-lg border transition ${
          securityConfig.secureAggregation || securityConfig.proposedFramework
            ? 'bg-indigo-950/40 border-indigo-500/40'
            : 'bg-slate-950/40 border-slate-800/60 opacity-60'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-indigo-300 flex items-center gap-1.5">
              <Key className="w-3.5 h-3.5" /> SecAgg Protocol & Key Size
            </span>
            <span className="text-[10px] font-mono text-indigo-400 font-bold">
              {securityConfig.secAggKeySize} Bits
            </span>
          </div>

          <div className="space-y-2.5">
            <div>
              <label className="block text-[10px] text-slate-400 mb-1">Secret Masking Key Bit Length</label>
              <select
                value={securityConfig.secAggKeySize}
                onChange={e => onChangeConfig({ ...securityConfig, secAggKeySize: parseInt(e.target.value) })}
                disabled={isTraining || (!securityConfig.secureAggregation && !securityConfig.proposedFramework)}
                className="w-full bg-slate-950 border border-slate-700 rounded p-1.5 text-xs font-mono text-indigo-300"
              >
                <option value={128}>128-bit AES-GCM (Lightweight)</option>
                <option value={256}>256-bit Diffie-Hellman Masking (Standard)</option>
                <option value={512}>512-bit High Security Modular Sharing</option>
              </select>
            </div>
            <p className="text-[10px] text-slate-400">
              Adds zero-sum pairwise secret masks (sum M_ij = 0) to obstruct individual hospital updates from untrusted server inspectors.
            </p>
          </div>
        </div>

        {/* Anomaly Detection Algorithm Parameters */}
        <div className={`p-3.5 rounded-lg border transition ${
          securityConfig.anomalyDetection || securityConfig.proposedFramework
            ? 'bg-emerald-950/40 border-emerald-500/40'
            : 'bg-slate-950/40 border-slate-800/60 opacity-60'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-emerald-300 flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5" /> Anomaly Defense Algorithm
            </span>
          </div>

          <div className="space-y-2">
            <div>
              <label className="block text-[10px] text-slate-400 mb-1">Byzantine Aggregator Strategy</label>
              <select
                value={securityConfig.anomalyAlgorithm}
                onChange={e => onChangeConfig({ ...securityConfig, anomalyAlgorithm: e.target.value as any })}
                disabled={isTraining || (!securityConfig.anomalyDetection && !securityConfig.proposedFramework)}
                className="w-full bg-slate-950 border border-slate-700 rounded p-1.5 text-xs font-mono text-emerald-300"
              >
                <option value="Cosine_Similarity">Cosine Similarity Threshold Filtering</option>
                <option value="Krum">Krum Euclidean Distance Defense</option>
                <option value="Multi-Krum">Multi-Krum Multi-Node Averaging</option>
                <option value="Trimmed_Mean">Coordinate-wise Trimmed Mean</option>
              </select>
            </div>
            {securityConfig.anomalyAlgorithm === 'Cosine_Similarity' && (
              <div>
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>Cosine Threshold ($\theta$):</span>
                  <span className="text-emerald-400">{securityConfig.anomalyThreshold}</span>
                </div>
                <input
                  type="range"
                  min={0.2}
                  max={0.9}
                  step={0.05}
                  value={securityConfig.anomalyThreshold}
                  onChange={e => onChangeConfig({ ...securityConfig, anomalyThreshold: parseFloat(e.target.value) })}
                  disabled={isTraining || (!securityConfig.anomalyDetection && !securityConfig.proposedFramework)}
                  className="w-full h-1 bg-slate-800 rounded appearance-none accent-emerald-400"
                />
              </div>
            )}
          </div>
        </div>

      </div>

    </div>
  );
};
