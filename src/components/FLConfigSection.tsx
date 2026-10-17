import React from 'react';
import { Cpu, RefreshCw, Repeat, Zap, GitCommit } from 'lucide-react';
import { FLConfig } from '../types/flTypes';

interface FLConfigSectionProps {
  flConfig: FLConfig;
  onChangeConfig: (newConfig: FLConfig) => void;
  isTraining: boolean;
}

export const FLConfigSection: React.FC<FLConfigSectionProps> = ({
  flConfig,
  onChangeConfig,
  isTraining
}) => {
  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800/80 shadow-xl mb-6">
      {/* Section Title */}
      <div className="flex items-center gap-3 mb-5 border-b border-slate-800 pb-3">
        <div className="p-2.5 rounded-xl bg-purple-950/80 border border-purple-500/30 text-purple-400">
          <Cpu className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-white">
            2. Federated Learning Hyperparameters
          </h2>
          <p className="text-xs text-slate-400">
            Configure global aggregation rounds, local hospital epochs, learning rate, and optimization algorithm (FedAvg vs FedProx).
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        
        {/* 1. Communication Rounds */}
        <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <RefreshCw className="w-3.5 h-3.5 text-purple-400" /> Comm. Rounds
            </label>
            <span className="px-2 py-0.5 rounded bg-purple-950 text-purple-300 font-mono text-xs font-bold border border-purple-800">
              {flConfig.commRounds} Rounds
            </span>
          </div>
          <input
            type="range"
            min={3}
            max={30}
            step={1}
            value={flConfig.commRounds}
            onChange={e => onChangeConfig({ ...flConfig, commRounds: parseInt(e.target.value) })}
            disabled={isTraining}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-400"
          />
          <p className="text-[10px] text-slate-400 mt-1.5 font-mono">
            Global model updates back and forth
          </p>
        </div>

        {/* 2. Local Epochs */}
        <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Repeat className="w-3.5 h-3.5 text-cyan-400" /> Local Epochs (E)
            </label>
            <span className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 font-mono text-xs font-bold border border-cyan-800">
              {flConfig.localEpochs} Epochs
            </span>
          </div>
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={flConfig.localEpochs}
            onChange={e => onChangeConfig({ ...flConfig, localEpochs: parseInt(e.target.value) })}
            disabled={isTraining}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
          />
          <p className="text-[10px] text-slate-400 mt-1.5 font-mono">
            Local SGD passes per round
          </p>
        </div>

        {/* 3. Learning Rate */}
        <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-400" /> Learning Rate (&eta;)
            </label>
            <span className="px-2 py-0.5 rounded bg-amber-950 text-amber-300 font-mono text-xs font-bold border border-amber-800">
              {flConfig.learningRate}
            </span>
          </div>
          <input
            type="range"
            min={0.001}
            max={0.1}
            step={0.005}
            value={flConfig.learningRate}
            onChange={e => onChangeConfig({ ...flConfig, learningRate: parseFloat(e.target.value) })}
            disabled={isTraining}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-400"
          />
          <p className="text-[10px] text-slate-400 mt-1.5 font-mono">
            Gradient step size
          </p>
        </div>

        {/* 4. Aggregation Strategy (FedAvg vs FedProx) */}
        <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800 space-y-2">
          <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
            <GitCommit className="w-3.5 h-3.5 text-emerald-400" /> Aggregation Strategy
          </label>
          <div className="grid grid-cols-2 gap-1.5">
            <button
              type="button"
              onClick={() => onChangeConfig({ ...flConfig, aggregationStrategy: 'FedAvg' })}
              disabled={isTraining}
              className={`py-1.5 px-2 rounded-lg text-xs font-semibold transition ${
                flConfig.aggregationStrategy === 'FedAvg'
                  ? 'bg-emerald-950 border border-emerald-500 text-emerald-300'
                  : 'bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800'
              }`}
            >
              FedAvg
            </button>
            <button
              type="button"
              onClick={() => onChangeConfig({ ...flConfig, aggregationStrategy: 'FedProx' })}
              disabled={isTraining}
              className={`py-1.5 px-2 rounded-lg text-xs font-semibold transition ${
                flConfig.aggregationStrategy === 'FedProx'
                  ? 'bg-teal-950 border border-teal-500 text-teal-300'
                  : 'bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800'
              }`}
            >
              FedProx
            </button>
          </div>
          {flConfig.aggregationStrategy === 'FedProx' && (
            <div className="pt-1">
              <div className="flex justify-between text-[10px] font-mono text-slate-400">
                <span>Proximal &mu;:</span>
                <span className="text-teal-400">{flConfig.fedProxMu}</span>
              </div>
              <input
                type="range"
                min={0.01}
                max={0.5}
                step={0.01}
                value={flConfig.fedProxMu}
                onChange={e => onChangeConfig({ ...flConfig, fedProxMu: parseFloat(e.target.value) })}
                disabled={isTraining}
                className="w-full h-1 bg-slate-800 rounded appearance-none accent-teal-400"
              />
            </div>
          )}
        </div>

      </div>
    </div>
  );
};
