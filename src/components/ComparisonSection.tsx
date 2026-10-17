import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, LineChart, Line
} from 'recharts';
import { Layers, Play, CheckCircle2, Shield, Eye, AlertTriangle, Download } from 'lucide-react';
import { FrameworkMetrics } from '../types/flTypes';

interface ComparisonSectionProps {
  comparisonResults: Record<string, FrameworkMetrics>;
  onRunAllComparisons: () => void;
  isComparing: boolean;
}

export const ComparisonSection: React.FC<ComparisonSectionProps> = ({
  comparisonResults,
  onRunAllComparisons,
  isComparing
}) => {
  const resultKeys = Object.keys(comparisonResults);
  const hasData = resultKeys.length > 0;

  // Format data for comparative bar charts
  const chartData = [
    {
      metric: 'Accuracy (%)',
      ...Object.fromEntries(resultKeys.map(k => [k, comparisonResults[k].accuracy]))
    },
    {
      metric: 'Sensitivity (%)',
      ...Object.fromEntries(resultKeys.map(k => [k, comparisonResults[k].sensitivity]))
    },
    {
      metric: 'F1-Score (%)',
      ...Object.fromEntries(resultKeys.map(k => [k, comparisonResults[k].f1Score]))
    },
    {
      metric: 'Attack Success Rate (%)',
      ...Object.fromEntries(resultKeys.map(k => [k, comparisonResults[k].attackSuccessRate]))
    },
    {
      metric: 'Anomaly Detect Rate (%)',
      ...Object.fromEntries(resultKeys.map(k => [k, comparisonResults[k].anomalyDetectionRate]))
    }
  ];

  // Colors for each configuration
  const COLOR_MAP: Record<string, string> = {
    'Baseline FL': '#94a3b8',
    'FL + Differential Privacy': '#38bdf8',
    'FL + Secure Aggregation': '#818cf8',
    'FL + Anomaly Detection': '#34d399',
    'Proposed Combined Framework': '#f59e0b'
  };

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800/80 shadow-xl mb-6">
      
      {/* Section Header & Benchmark Button */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 border-b border-slate-800 pb-3 mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-amber-950/80 border border-amber-500/30 text-amber-400">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              7. Security Framework Comparison & Benchmark Analysis
            </h2>
            <p className="text-xs text-slate-400">
              Comparative matrix evaluating trade-offs between Baseline FL, Differential Privacy, Secure Aggregation, Anomaly Defense, and Proposed Framework.
            </p>
          </div>
        </div>

        <button
          onClick={onRunAllComparisons}
          disabled={isComparing}
          className="px-4 py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs shadow-md transition flex items-center gap-2 disabled:opacity-50"
        >
          {isComparing ? (
            <>Running 5 Security Benchmarks...</>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-slate-950" /> Run Full 5-Framework Comparison
            </>
          )}
        </button>
      </div>

      {!hasData ? (
        <div className="bg-slate-950/60 rounded-xl p-8 border border-slate-800 text-center">
          <Layers className="w-10 h-10 text-slate-600 mx-auto mb-2" />
          <h4 className="text-sm font-bold text-slate-300">Comparative Benchmark Not Run Yet</h4>
          <p className="text-xs text-slate-400 mt-1 max-w-lg mx-auto">
            Click <span className="text-amber-400 font-semibold font-mono">Run Full 5-Framework Comparison</span> to execute automated benchmark tests across all security architectures under identical clinical conditions.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          
          {/* Side-by-Side Comparison Matrix Table */}
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-xs text-left text-slate-300">
              <thead className="text-[11px] font-mono text-slate-400 uppercase bg-slate-900 border-b border-slate-800">
                <tr>
                  <th className="px-4 py-3">Framework Configuration</th>
                  <th className="px-3 py-3 text-cyan-400">Accuracy</th>
                  <th className="px-3 py-3">Precision</th>
                  <th className="px-3 py-3">Sensitivity</th>
                  <th className="px-3 py-3">F1-Score</th>
                  <th className="px-3 py-3">AUC</th>
                  <th className="px-3 py-3 text-cyan-300">Privacy Budget ($\epsilon$)</th>
                  <th className="px-3 py-3 text-rose-400">Attack Success (ASR)</th>
                  <th className="px-3 py-3 text-emerald-400">Anomaly Detect (ADR)</th>
                  <th className="px-3 py-3">Time (s)</th>
                  <th className="px-3 py-3">Overhead (MB)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80 bg-slate-950/70">
                {resultKeys.map((key) => {
                  const item = comparisonResults[key];
                  const isProposed = key.includes('Proposed');
                  return (
                    <tr
                      key={key}
                      className={isProposed ? 'bg-amber-950/20 font-semibold border-amber-500/30' : 'hover:bg-slate-900/50'}
                    >
                      <td className="px-4 py-3 flex items-center gap-2 font-bold text-white">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLOR_MAP[key] || '#38bdf8' }}></span>
                        {key}
                        {isProposed && <span className="px-1.5 py-0.5 rounded bg-amber-400/20 text-amber-300 text-[10px] border border-amber-400/40">OUR PROPOSAL</span>}
                      </td>
                      <td className="px-3 py-3 font-mono font-bold text-cyan-300">{item.accuracy}%</td>
                      <td className="px-3 py-3 font-mono">{item.precision}%</td>
                      <td className="px-3 py-3 font-mono">{item.sensitivity}%</td>
                      <td className="px-3 py-3 font-mono">{item.f1Score}%</td>
                      <td className="px-3 py-3 font-mono">{item.auc}</td>
                      <td className="px-3 py-3 font-mono text-cyan-400">
                        {item.privacyEpsilon === Infinity ? '∞ (No DP)' : `ε=${item.privacyEpsilon}`}
                      </td>
                      <td className="px-3 py-3 font-mono text-rose-400">{item.attackSuccessRate}%</td>
                      <td className="px-3 py-3 font-mono text-emerald-400">{item.anomalyDetectionRate}%</td>
                      <td className="px-3 py-3 font-mono text-slate-400">{item.trainingTimeSec}s</td>
                      <td className="px-3 py-3 font-mono text-slate-400">{item.commOverheadMB} MB</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Comparative Multi-Bar Chart */}
          <div className="bg-slate-950/80 rounded-xl p-4 border border-slate-800">
            <h3 className="text-xs font-bold text-white mb-3 flex items-center gap-1.5">
              <Layers className="w-4 h-4 text-amber-400" /> Multi-Framework Performance Metrics Comparison
            </h3>
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="metric" stroke="#64748b" tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 100]} stroke="#64748b" tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }} />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                  {resultKeys.map(k => (
                    <Bar key={k} dataKey={k} fill={COLOR_MAP[k] || '#38bdf8'} radius={[4, 4, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>
      )}
    </div>
  );
};
