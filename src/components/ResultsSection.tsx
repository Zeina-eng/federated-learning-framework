import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import {
  Award, Target, Shield, Eye, AlertTriangle, Clock, HardDrive, TrendingUp, CheckCircle, Crosshair
} from 'lucide-react';
import { FrameworkMetrics } from '../types/flTypes';
import { generateROCCurveData } from '../utils/metrics';

interface ResultsSectionProps {
  metrics: FrameworkMetrics | null;
}

export const ResultsSection: React.FC<ResultsSectionProps> = ({ metrics }) => {
  if (!metrics || metrics.history.length === 0) {
    return (
      <div className="glass-card rounded-2xl p-8 border border-slate-800 text-center mb-6">
        <Award className="w-12 h-12 text-slate-600 mx-auto mb-3 animate-pulse" />
        <h3 className="text-base font-bold text-slate-300">No Experiment Results Available</h3>
        <p className="text-xs text-slate-400 mt-1">
          Click the <span className="text-cyan-400 font-semibold font-mono">🚀 Run Federated Learning Experiment</span> button above to initiate training and generate metric analytics.
        </p>
      </div>
    );
  }

  const rocData = generateROCCurveData(metrics.auc);

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800/80 shadow-xl mb-6">
      
      {/* Section Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-emerald-950/80 border border-emerald-500/30 text-emerald-400">
            <Award className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              6. Experiment Analytics & Performance Results
            </h2>
            <p className="text-xs text-slate-400">
              Automatic metrics extraction covering classification power, differential privacy budget consumption, Byzantine resilience, and resource efficiency.
            </p>
          </div>
        </div>

        <span className="px-3 py-1 rounded-full bg-emerald-950 border border-emerald-500/40 text-emerald-300 font-mono text-xs font-bold">
          {metrics.name}
        </span>
      </div>

      {/* 12 Metric KPI Cards Grid (Matches all required metrics from prompt) */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-8">
        
        {/* 1. Accuracy */}
        <div className="bg-slate-900/90 rounded-xl p-3.5 border border-slate-800 glass-card-hover">
          <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
            <span>Accuracy</span>
            <Target className="w-3.5 h-3.5 text-cyan-400" />
          </div>
          <div className="text-xl font-black text-cyan-300 font-mono">
            {metrics.accuracy}%
          </div>
          <span className="text-[10px] text-slate-400">Global Test Correctness</span>
        </div>

        {/* 2. Precision */}
        <div className="bg-slate-900/90 rounded-xl p-3.5 border border-slate-800 glass-card-hover">
          <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
            <span>Precision</span>
            <Crosshair className="w-3.5 h-3.5 text-teal-400" />
          </div>
          <div className="text-xl font-black text-teal-300 font-mono">
            {metrics.precision}%
          </div>
          <span className="text-[10px] text-slate-400">Positive Predictive Value</span>
        </div>

        {/* 3. Sensitivity (Recall) */}
        <div className="bg-slate-900/90 rounded-xl p-3.5 border border-slate-800 glass-card-hover">
          <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
            <span>Sensitivity</span>
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-xl font-black text-emerald-300 font-mono">
            {metrics.sensitivity}%
          </div>
          <span className="text-[10px] text-slate-400">True Positive Rate</span>
        </div>

        {/* 4. Specificity */}
        <div className="bg-slate-900/90 rounded-xl p-3.5 border border-slate-800 glass-card-hover">
          <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
            <span>Specificity</span>
            <Shield className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <div className="text-xl font-black text-indigo-300 font-mono">
            {metrics.specificity}%
          </div>
          <span className="text-[10px] text-slate-400">True Negative Rate</span>
        </div>

        {/* 5. F1-Score */}
        <div className="bg-slate-900/90 rounded-xl p-3.5 border border-slate-800 glass-card-hover">
          <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
            <span>F1-Score</span>
            <Award className="w-3.5 h-3.5 text-purple-400" />
          </div>
          <div className="text-xl font-black text-purple-300 font-mono">
            {metrics.f1Score}%
          </div>
          <span className="text-[10px] text-slate-400">Harmonic Mean</span>
        </div>

        {/* 6. AUC */}
        <div className="bg-slate-900/90 rounded-xl p-3.5 border border-slate-800 glass-card-hover">
          <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
            <span>AUC ROC</span>
            <TrendingUp className="w-3.5 h-3.5 text-amber-400" />
          </div>
          <div className="text-xl font-black text-amber-300 font-mono">
            {metrics.auc}
          </div>
          <span className="text-[10px] text-slate-400">Area Under Curve</span>
        </div>

        {/* 7. Privacy Budget ε, δ */}
        <div className="bg-slate-900/90 rounded-xl p-3.5 border border-slate-800 glass-card-hover">
          <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
            <span>Privacy Budget</span>
            <Eye className="w-3.5 h-3.5 text-cyan-400" />
          </div>
          <div className="text-lg font-black text-cyan-300 font-mono">
            ε={metrics.privacyEpsilon}
          </div>
          <span className="text-[10px] text-slate-400 font-mono">δ={metrics.privacyDelta}</span>
        </div>

        {/* 8. Attack Success Rate */}
        <div className="bg-slate-900/90 rounded-xl p-3.5 border border-slate-800 glass-card-hover">
          <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
            <span>Attack Success</span>
            <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
          </div>
          <div className="text-xl font-black text-rose-400 font-mono">
            {metrics.attackSuccessRate}%
          </div>
          <span className="text-[10px] text-slate-400">Byzantine Impact (ASR)</span>
        </div>

        {/* 9. Anomaly Detection Rate */}
        <div className="bg-slate-900/90 rounded-xl p-3.5 border border-slate-800 glass-card-hover">
          <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
            <span>Anomaly Detect</span>
            <Shield className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-xl font-black text-emerald-400 font-mono">
            {metrics.anomalyDetectionRate}%
          </div>
          <span className="text-[10px] text-slate-400">Malicious Caught (ADR)</span>
        </div>

        {/* 10. False Positive Rate */}
        <div className="bg-slate-900/90 rounded-xl p-3.5 border border-slate-800 glass-card-hover">
          <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
            <span>False Positive</span>
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
          </div>
          <div className="text-xl font-black text-amber-400 font-mono">
            {metrics.falsePositiveRate}%
          </div>
          <span className="text-[10px] text-slate-400">Benign Flagged (FPR)</span>
        </div>

        {/* 11. Training Time */}
        <div className="bg-slate-900/90 rounded-xl p-3.5 border border-slate-800 glass-card-hover">
          <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
            <span>Training Time</span>
            <Clock className="w-3.5 h-3.5 text-sky-400" />
          </div>
          <div className="text-xl font-black text-sky-300 font-mono">
            {metrics.trainingTimeSec}s
          </div>
          <span className="text-[10px] text-slate-400">Total Compute Duration</span>
        </div>

        {/* 12. Communication Overhead */}
        <div className="bg-slate-900/90 rounded-xl p-3.5 border border-slate-800 glass-card-hover">
          <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
            <span>Comm. Overhead</span>
            <HardDrive className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <div className="text-xl font-black text-indigo-300 font-mono">
            {metrics.commOverheadMB} MB
          </div>
          <span className="text-[10px] text-slate-400">Transferred Payload</span>
        </div>

      </div>

      {/* Visual Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Accuracy & Loss Progress Chart */}
        <div className="lg:col-span-8 bg-slate-950/80 rounded-xl p-4 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xs font-bold text-white flex items-center gap-1.5">
              <TrendingUp className="w-4 h-4 text-cyan-400" /> Global Model Accuracy & Loss Over Communication Rounds
            </h3>
            <div className="flex items-center gap-3 text-[10px] font-mono">
              <span className="flex items-center gap-1 text-cyan-400">● Accuracy (%)</span>
              <span className="flex items-center gap-1 text-rose-400">● Loss</span>
            </div>
          </div>
          
          <div className="h-[250px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics.history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="round" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="left" domain={[0, 100]} stroke="#38bdf8" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="right" orientation="right" stroke="#f43f5e" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                />
                <Line yAxisId="left" type="monotone" dataKey="globalAccuracy" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 3 }} />
                <Line yAxisId="right" type="monotone" dataKey="globalLoss" stroke="#f43f5e" strokeWidth={2} dot={{ r: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ROC Curve Graph */}
        <div className="lg:col-span-4 bg-slate-950/80 rounded-xl p-4 border border-slate-800 flex flex-col justify-between">
          <div>
            <h3 className="text-xs font-bold text-white mb-1">Receiver Operating Characteristic (ROC Curve)</h3>
            <p className="text-[10px] text-slate-400 mb-3">AUC Score = <span className="text-amber-300 font-bold font-mono">{metrics.auc}</span></p>
          </div>

          <div className="h-[200px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={rocData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="fpr" stroke="#64748b" tick={{ fontSize: 9 }} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 9 }} />
                <YAxis domain={[0, 1]} stroke="#64748b" tick={{ fontSize: 9 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }} />
                <Area type="monotone" dataKey="tpr" stroke="#fbbf24" fill="#fbbf24" fillOpacity={0.2} strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
};
