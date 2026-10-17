import React from 'react';
import { Skull, AlertOctagon, Bug, ShieldAlert, CheckSquare, Square, Users } from 'lucide-react';
import { AttackConfig } from '../types/flTypes';

interface AttackSectionProps {
  attackConfig: AttackConfig;
  onChangeConfig: (newConfig: AttackConfig) => void;
  isTraining: boolean;
}

export const AttackSection: React.FC<AttackSectionProps> = ({
  attackConfig,
  onChangeConfig,
  isTraining
}) => {
  const toggleAttackType = (type: keyof AttackConfig['attackTypes']) => {
    if (isTraining) return;
    const current = attackConfig.attackTypes[type];
    const newTypes = { ...attackConfig.attackTypes, [type]: !current };
    
    // Auto toggle master enable state
    const anyActive = Object.values(newTypes).some(v => v);
    onChangeConfig({
      ...attackConfig,
      enabled: anyActive,
      attackTypes: newTypes
    });
  };

  const handleMasterToggle = (enable: boolean) => {
    if (isTraining) return;
    onChangeConfig({
      ...attackConfig,
      enabled: enable,
      attackTypes: {
        modelPoisoning: enable,
        dataPoisoning: enable,
        backdoorAttack: enable,
        membershipInference: enable
      }
    });
  };

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800/80 shadow-xl mb-6">
      {/* Section Title */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-rose-950/80 border border-rose-500/30 text-rose-400">
            <Skull className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              4. Adversarial Attack Simulation Engine
            </h2>
            <p className="text-xs text-slate-400">
              Simulate malicious hospital clients launching Byzantine gradient corruption, local data poisoning, backdoor triggers, and Membership Inference Attacks (MIA).
            </p>
          </div>
        </div>

        {/* Master Enable / Disable Toggle */}
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-700/80 p-1 rounded-xl">
          <button
            type="button"
            onClick={() => handleMasterToggle(false)}
            disabled={isTraining}
            className={`px-3 py-1 rounded-lg text-xs font-semibold transition ${
              !attackConfig.enabled
                ? 'bg-slate-800 text-slate-200'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Disable Attacks
          </button>
          <button
            type="button"
            onClick={() => handleMasterToggle(true)}
            disabled={isTraining}
            className={`px-3 py-1 rounded-lg text-xs font-semibold transition ${
              attackConfig.enabled
                ? 'bg-rose-950 border border-rose-500 text-rose-300 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Simulate Attacks
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Attack Vector Selection Checkboxes */}
        <div className="lg:col-span-8 grid grid-cols-1 sm:grid-cols-2 gap-3">
          
          {/* 1. Model Poisoning */}
          <div
            onClick={() => toggleAttackType('modelPoisoning')}
            className={`p-3.5 rounded-xl border cursor-pointer transition flex items-start gap-3 ${
              attackConfig.attackTypes.modelPoisoning && attackConfig.enabled
                ? 'bg-rose-950/40 border-rose-500/80 text-rose-200'
                : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700'
            }`}
          >
            <div className="mt-0.5">
              {attackConfig.attackTypes.modelPoisoning && attackConfig.enabled ? (
                <CheckSquare className="w-4 h-4 text-rose-400" />
              ) : (
                <Square className="w-4 h-4 text-slate-600" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-1.5 font-bold text-xs text-white">
                <AlertOctagon className="w-3.5 h-3.5 text-rose-400" /> Model Poisoning Attack
              </div>
              <p className="text-[11px] text-slate-400 mt-1 font-normal">
                Malicious clients flip weight signs (-w_i) and inject massive gradients to sabotage server convergence.
              </p>
            </div>
          </div>

          {/* 2. Data Poisoning */}
          <div
            onClick={() => toggleAttackType('dataPoisoning')}
            className={`p-3.5 rounded-xl border cursor-pointer transition flex items-start gap-3 ${
              attackConfig.attackTypes.dataPoisoning && attackConfig.enabled
                ? 'bg-rose-950/40 border-rose-500/80 text-rose-200'
                : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700'
            }`}
          >
            <div className="mt-0.5">
              {attackConfig.attackTypes.dataPoisoning && attackConfig.enabled ? (
                <CheckSquare className="w-4 h-4 text-rose-400" />
              ) : (
                <Square className="w-4 h-4 text-slate-600" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-1.5 font-bold text-xs text-white">
                <Bug className="w-3.5 h-3.5 text-amber-400" /> Data Poisoning Attack
              </div>
              <p className="text-[11px] text-slate-400 mt-1 font-normal">
                Corrupts local patient record labels (y to 1-y) and adds noisy clinical features before training.
              </p>
            </div>
          </div>

          {/* 3. Backdoor Attack */}
          <div
            onClick={() => toggleAttackType('backdoorAttack')}
            className={`p-3.5 rounded-xl border cursor-pointer transition flex items-start gap-3 ${
              attackConfig.attackTypes.backdoorAttack && attackConfig.enabled
                ? 'bg-rose-950/40 border-rose-500/80 text-rose-200'
                : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700'
            }`}
          >
            <div className="mt-0.5">
              {attackConfig.attackTypes.backdoorAttack && attackConfig.enabled ? (
                <CheckSquare className="w-4 h-4 text-rose-400" />
              ) : (
                <Square className="w-4 h-4 text-slate-600" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-1.5 font-bold text-xs text-white">
                <ShieldAlert className="w-3.5 h-3.5 text-purple-400" /> Backdoor Trigger Attack
              </div>
              <p className="text-[11px] text-slate-400 mt-1 font-normal">
                Embeds hidden synthetic trigger markers into patient records to force specific target medical diagnoses.
              </p>
            </div>
          </div>

          {/* 4. Membership Inference Attack (MIA) */}
          <div
            onClick={() => toggleAttackType('membershipInference')}
            className={`p-3.5 rounded-xl border cursor-pointer transition flex items-start gap-3 ${
              attackConfig.attackTypes.membershipInference && attackConfig.enabled
                ? 'bg-rose-950/40 border-rose-500/80 text-rose-200'
                : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700'
            }`}
          >
            <div className="mt-0.5">
              {attackConfig.attackTypes.membershipInference && attackConfig.enabled ? (
                <CheckSquare className="w-4 h-4 text-rose-400" />
              ) : (
                <Square className="w-4 h-4 text-slate-600" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-1.5 font-bold text-xs text-white">
                <Users className="w-3.5 h-3.5 text-cyan-400" /> Membership Inference (MIA)
              </div>
              <p className="text-[11px] text-slate-400 mt-1 font-normal">
                Probes confidence distributions to determine if specific target patient records were used in training.
              </p>
            </div>
          </div>

        </div>

        {/* Malicious Client Ratio & Intensity Controls */}
        <div className="lg:col-span-4 bg-slate-900/90 rounded-xl p-4 border border-slate-800 space-y-4">
          
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold text-slate-300">
                Compromised Institutions Ratio
              </label>
              <span className="font-mono text-xs text-rose-400 font-bold">
                {Math.round(attackConfig.maliciousClientRatio * 100)}% Malicious
              </span>
            </div>
            <input
              type="range"
              min={0.1}
              max={0.5}
              step={0.05}
              value={attackConfig.maliciousClientRatio}
              onChange={e => onChangeConfig({ ...attackConfig, maliciousClientRatio: parseFloat(e.target.value) })}
              disabled={isTraining || !attackConfig.enabled}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-400 mt-1">
              <span>10% Nodes</span>
              <span>30% Nodes</span>
              <span>50% Max</span>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[11px] text-slate-400">Poisoning Gradient Scale Factor:</span>
              <span className="text-xs font-mono text-amber-400 font-bold">{attackConfig.modelPoisoningFactor}x</span>
            </div>
            <input
              type="range"
              min={1.0}
              max={5.0}
              step={0.5}
              value={attackConfig.modelPoisoningFactor}
              onChange={e => onChangeConfig({ ...attackConfig, modelPoisoningFactor: parseFloat(e.target.value) })}
              disabled={isTraining || !attackConfig.enabled}
              className="w-full h-1 bg-slate-800 rounded appearance-none accent-amber-400"
            />
          </div>

        </div>

      </div>
    </div>
  );
};
