import React from 'react';
import { ShieldCheck, Activity, Database, Download, Sparkles } from 'lucide-react';

interface NavbarProps {
  onLoadPreset: (presetKey: string) => void;
  onExportReport: () => void;
  isTraining: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ onLoadPreset, onExportReport, isTraining }) => {
  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80 px-6 py-3.5 mb-6">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        
        {/* Brand Logo & Title */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-11 h-11 rounded-xl bg-gradient-to-tr from-cyan-600 via-teal-500 to-emerald-400 shadow-lg shadow-cyan-500/20 text-slate-950 font-bold">
            <ShieldCheck className="w-6 h-6 text-slate-950" />
            <span className="absolute -bottom-1 -right-1 flex h-3.5 w-3.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-emerald-500 border-2 border-slate-900"></span>
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-extrabold bg-gradient-to-r from-cyan-400 via-teal-200 to-white bg-clip-text text-transparent tracking-tight">
                Healthcare Privacy-Preserving FL Framework
              </h1>
              <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-cyan-950/80 border border-cyan-500/30 text-cyan-400">
                v2.4 Production
              </span>
            </div>
            <p className="text-xs text-slate-400 font-medium">
              Federated Data Analytics, Differential Privacy (DP), Secure Aggregation & Byzantine Defense
            </p>
          </div>
        </div>

        {/* Action Controls & Presets */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center bg-slate-900/90 border border-slate-700/60 rounded-lg p-1 text-xs font-medium">
            <span className="px-2 text-slate-400 flex items-center gap-1">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Presets:
            </span>
            <button
              onClick={() => onLoadPreset('clinical_standard')}
              disabled={isTraining}
              className="px-2.5 py-1 rounded-md text-slate-300 hover:text-white hover:bg-slate-800 transition text-xs font-medium disabled:opacity-50"
            >
              Clinical Standard
            </button>
            <button
              onClick={() => onLoadPreset('high_privacy_dp')}
              disabled={isTraining}
              className="px-2.5 py-1 rounded-md text-cyan-400 hover:bg-cyan-950/50 transition text-xs font-medium disabled:opacity-50"
            >
              High DP Privacy
            </button>
            <button
              onClick={() => onLoadPreset('adversarial_attack')}
              disabled={isTraining}
              className="px-2.5 py-1 rounded-md text-rose-400 hover:bg-rose-950/50 transition text-xs font-medium disabled:opacity-50"
            >
              Byzantine Attack Test
            </button>
            <button
              onClick={() => onLoadPreset('proposed_full')}
              disabled={isTraining}
              className="px-2.5 py-1 rounded-md bg-gradient-to-r from-cyan-600 to-teal-600 text-white font-semibold transition text-xs shadow-sm disabled:opacity-50"
            >
              Proposed Combined
            </button>
          </div>

          <button
            onClick={onExportReport}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-semibold transition shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" /> Export Report
          </button>
        </div>

      </div>
    </header>
  );
};
