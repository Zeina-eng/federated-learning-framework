import React from 'react';
import { X, Server, Activity, ShieldCheck, AlertTriangle, Database } from 'lucide-react';
import { RoundResult } from '../types/flTypes';

interface HospitalNodeModalProps {
  clientId: number | null;
  latestRoundResult: RoundResult | null;
  onClose: () => void;
}

export const HospitalNodeModal: React.FC<HospitalNodeModalProps> = ({
  clientId,
  latestRoundResult,
  onClose
}) => {
  if (clientId === null || !latestRoundResult) return null;

  const clientLog = latestRoundResult.clientLogs.find(c => c.clientId === clientId);
  if (!clientLog) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl border ${
              clientLog.isDetected
                ? 'bg-rose-950/80 border-rose-500 text-rose-400'
                : clientLog.isMalicious
                ? 'bg-amber-950/80 border-amber-500 text-amber-400'
                : 'bg-emerald-950/80 border-emerald-500 text-emerald-400'
            }`}>
              <Server className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">
                {clientLog.clientName}
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                Healthcare Institution Client #{clientLog.clientId}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Status Badges */}
        <div className="flex items-center gap-2 flex-wrap">
          {clientLog.isDetected ? (
            <span className="px-2.5 py-1 rounded-full bg-rose-950 border border-rose-500 text-rose-300 text-xs font-semibold flex items-center gap-1">
              <AlertTriangle className="w-3.5 h-3.5" /> Isolated BYZANTINE Malicious Node
            </span>
          ) : clientLog.isMalicious ? (
            <span className="px-2.5 py-1 rounded-full bg-amber-950 border border-amber-500 text-amber-300 text-xs font-semibold flex items-center gap-1">
              <AlertTriangle className="w-3.5 h-3.5" /> Suspicious Malicious Update
            </span>
          ) : (
            <span className="px-2.5 py-1 rounded-full bg-emerald-950 border border-emerald-500 text-emerald-300 text-xs font-semibold flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5" /> Verified Benign Healthcare Node
            </span>
          )}

          {clientLog.masked && (
            <span className="px-2.5 py-1 rounded-full bg-cyan-950 border border-cyan-500 text-cyan-300 text-xs font-semibold font-mono">
              SecAgg Key Masked
            </span>
          )}
        </div>

        {/* Local Clinical Metrics Grid */}
        <div className="grid grid-cols-2 gap-3 bg-slate-950 rounded-xl p-4 border border-slate-800">
          <div>
            <span className="text-[11px] text-slate-400 font-medium">Local Training Accuracy</span>
            <div className="text-lg font-bold text-cyan-300 font-mono">
              {clientLog.localAccuracy}%
            </div>
          </div>
          <div>
            <span className="text-[11px] text-slate-400 font-medium">Local Training Loss</span>
            <div className="text-lg font-bold text-rose-400 font-mono">
              {clientLog.localLoss}
            </div>
          </div>
          <div>
            <span className="text-[11px] text-slate-400 font-medium">Gradient Update Norm</span>
            <div className="text-base font-bold text-slate-200 font-mono">
              {clientLog.updateNorm}
            </div>
          </div>
          <div>
            <span className="text-[11px] text-slate-400 font-medium">Anomaly Defense Status</span>
            <div className="text-base font-bold font-mono">
              {clientLog.isDetected ? (
                <span className="text-rose-400">REJECTED</span>
              ) : (
                <span className="text-emerald-400">PASSED</span>
              )}
            </div>
          </div>
        </div>

        <div className="pt-2 text-right">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs transition"
          >
            Close Inspector
          </button>
        </div>

      </div>
    </div>
  );
};
