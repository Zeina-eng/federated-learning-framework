import React, { useRef } from 'react';
import { Database, Upload, Server, Sliders, CheckCircle2, Layers, AlertCircle } from 'lucide-react';
import { DatasetConfig, DatasetInfo } from '../types/flTypes';
import { BUILTIN_DATASETS, parseCSVData } from '../data/datasets';

interface DatasetSectionProps {
  datasetConfig: DatasetConfig;
  onChangeConfig: (newConfig: DatasetConfig) => void;
  selectedDatasetInfo: DatasetInfo;
  onCustomDatasetUploaded: (info: DatasetInfo) => void;
  isTraining: boolean;
}

export const DatasetSection: React.FC<DatasetSectionProps> = ({
  datasetConfig,
  onChangeConfig,
  selectedDatasetInfo,
  onCustomDatasetUploaded,
  isTraining
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDatasetSelect = (id: string) => {
    const builtin = BUILTIN_DATASETS.find(d => d.id === id);
    if (builtin) {
      onChangeConfig({
        ...datasetConfig,
        datasetId: id,
        targetVariable: builtin.targetName,
        customData: undefined
      });
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const parsed = parseCSVData(text);
        const customInfo: DatasetInfo = {
          id: 'custom_csv_' + Date.now(),
          name: `Custom Healthcare File: ${file.name}`,
          description: `User-uploaded CSV containing ${parsed.data.length} records across ${parsed.featureNames.length} features.`,
          sampleCount: parsed.data.length,
          featureNames: parsed.featureNames,
          targetName: parsed.targetName,
          data: parsed.data
        };

        onCustomDatasetUploaded(customInfo);
        onChangeConfig({
          ...datasetConfig,
          datasetId: customInfo.id,
          targetVariable: parsed.targetName,
          customData: parsed.data,
          customFeatureNames: parsed.featureNames
        });
      } catch (err: any) {
        alert('CSV Parsing Error: ' + err.message);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800/80 shadow-xl mb-6">
      {/* Section Title */}
      <div className="flex items-center gap-3 mb-5 border-b border-slate-800 pb-3">
        <div className="p-2.5 rounded-xl bg-cyan-950/80 border border-cyan-500/30 text-cyan-400">
          <Database className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            1. Healthcare Dataset & Distribution Setup
          </h2>
          <p className="text-xs text-slate-400">
            Select or upload healthcare dataset, define target clinical outcome, and configure institutional data distribution across hospitals.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Dataset Selection & Upload */}
        <div className="lg:col-span-6 space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-2 flex items-center justify-between">
              <span>Select Healthcare Dataset</span>
              <span className="text-[10px] text-cyan-400 font-mono">
                {selectedDatasetInfo.sampleCount} Patient Records
              </span>
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {BUILTIN_DATASETS.map(d => {
                const isSelected = datasetConfig.datasetId === d.id;
                return (
                  <button
                    key={d.id}
                    onClick={() => handleDatasetSelect(d.id)}
                    disabled={isTraining}
                    className={`p-3 rounded-xl text-left border transition flex flex-col justify-between ${
                      isSelected
                        ? 'bg-cyan-950/60 border-cyan-500/80 shadow-md shadow-cyan-950/50 text-cyan-200'
                        : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-1">
                      <span className="font-semibold text-xs text-white line-clamp-1">{d.name}</span>
                      {isSelected && <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />}
                    </div>
                    <span className="text-[11px] text-slate-400 line-clamp-2 mt-1 font-normal">
                      {d.description}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Upload Custom CSV Button */}
          <div className="pt-1">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".csv,.txt"
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isTraining}
              className="w-full py-2.5 px-4 rounded-xl border border-dashed border-cyan-500/40 bg-cyan-950/30 hover:bg-cyan-900/40 text-cyan-300 text-xs font-semibold flex items-center justify-center gap-2 transition"
            >
              <Upload className="w-4 h-4 text-cyan-400" /> Upload Custom Healthcare CSV Dataset
            </button>
          </div>

          {/* Target Variable Selector */}
          <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800">
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">
              Select Target Variable (Classification Output)
            </label>
            <div className="flex items-center gap-3">
              <select
                value={datasetConfig.targetVariable}
                onChange={e => onChangeConfig({ ...datasetConfig, targetVariable: e.target.value })}
                disabled={isTraining}
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-cyan-300 focus:outline-none focus:border-cyan-500"
              >
                {selectedDatasetInfo.featureNames.map(f => (
                  <option key={f} value={f}>{f}</option>
                ))}
                <option value={selectedDatasetInfo.targetName}>{selectedDatasetInfo.targetName} (Default Target)</option>
              </select>
            </div>
            <p className="text-[11px] text-slate-400 mt-1.5 flex items-center gap-1">
              <AlertCircle className="w-3 h-3 text-amber-400" /> Target column for binary medical risk classification (0 / 1).
            </p>
          </div>
        </div>

        {/* Right Column: Institutions & Data Distribution */}
        <div className="lg:col-span-6 space-y-4">
          
          {/* Healthcare Institutions Count */}
          <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <Server className="w-4 h-4 text-teal-400" /> Number of Healthcare Institutions (Clients)
              </label>
              <span className="px-2.5 py-0.5 rounded-md bg-teal-950 border border-teal-500/40 text-teal-300 text-xs font-mono font-bold">
                {datasetConfig.numClients} Hospitals
              </span>
            </div>
            <input
              type="range"
              min={2}
              max={10}
              step={1}
              value={datasetConfig.numClients}
              onChange={e => onChangeConfig({ ...datasetConfig, numClients: parseInt(e.target.value) })}
              disabled={isTraining}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal-400"
            />
            <div className="flex justify-between text-[10px] text-slate-400 font-mono mt-1">
              <span>2 Clients</span>
              <span>5 Clients</span>
              <span>10 Clients</span>
            </div>
          </div>

          {/* IID / Non-IID Distribution Toggle & Dirichlet Slider */}
          <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800 space-y-3">
            <label className="block text-xs font-semibold text-slate-300">
              Data Distribution Scheme Across Institutions
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => onChangeConfig({ ...datasetConfig, distributionType: 'IID' })}
                disabled={isTraining}
                className={`py-2 px-3 rounded-lg border text-xs font-semibold flex items-center justify-center gap-2 transition ${
                  datasetConfig.distributionType === 'IID'
                    ? 'bg-emerald-950/80 border-emerald-500 text-emerald-300'
                    : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                <Layers className="w-3.5 h-3.5" /> IID Distribution
              </button>
              <button
                type="button"
                onClick={() => onChangeConfig({ ...datasetConfig, distributionType: 'NON_IID_DIRICHLET' })}
                disabled={isTraining}
                className={`py-2 px-3 rounded-lg border text-xs font-semibold flex items-center justify-center gap-2 transition ${
                  datasetConfig.distributionType !== 'IID'
                    ? 'bg-amber-950/80 border-amber-500 text-amber-300'
                    : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                <Sliders className="w-3.5 h-3.5" /> Non-IID (Dirichlet Skew)
              </button>
            </div>

            {datasetConfig.distributionType !== 'IID' && (
              <div className="pt-2 border-t border-slate-800/80">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[11px] font-medium text-slate-300">
                    Dirichlet Heterogeneity Concentration (&alpha;)
                  </span>
                  <span className="font-mono text-xs text-amber-400 font-bold">
                    &alpha; = {datasetConfig.dirichletAlpha}
                  </span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={5.0}
                  step={0.1}
                  value={datasetConfig.dirichletAlpha}
                  onChange={e => onChangeConfig({ ...datasetConfig, dirichletAlpha: parseFloat(e.target.value) })}
                  disabled={isTraining}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-400"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono mt-1">
                  <span>0.1 (Extreme Skew)</span>
                  <span>1.0 (Moderate)</span>
                  <span>5.0 (Near IID)</span>
                </div>
              </div>
            )}
          </div>

          {/* Dataset Feature Summary Pill */}
          <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/60 flex items-center justify-between text-xs">
            <span className="text-slate-400 font-medium">Included Clinical Features:</span>
            <span className="text-slate-200 font-mono line-clamp-1 max-w-[280px]">
              {selectedDatasetInfo.featureNames.join(', ')}
            </span>
          </div>

        </div>

      </div>
    </div>
  );
};
