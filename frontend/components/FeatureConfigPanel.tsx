'use client';

import { Plus, X, Calculator } from 'lucide-react';
import { ModelConfig, ColumnInfo, FeatureConfig, TemporalFeatureConfig, ExogenousFeatureConfig, DerivedFeatureConfig } from '../types/forecasting';

interface FeatureConfigPanelProps {
  model: ModelConfig;
  updateModelParams: (modelId: string, paramsOrFn: any) => void;
  availableColumns: ColumnInfo[];
}

/**
 * Feature Engineering Panel for ML models (XGBoost, Linear Regression).
 * Allows configuration of temporal features, exogenous variables, and derived features.
 */
export function FeatureConfigPanel({ model, updateModelParams, availableColumns }: FeatureConfigPanelProps) {
  const params = model.params as any;
  const featureConfig: FeatureConfig = params.feature_config || {
    target_lags: params.lags || [1, 7],
    temporal: { month: false, day_of_week: false, day_of_month: false, week_of_year: false, year: false, hour_of_day: false, minute_of_day: false },
    exogenous: [],
    derived: []
  };

  const updateConfig = (updater: (prev: FeatureConfig) => FeatureConfig) => {
    updateModelParams(model.id, (prev: any) => {
      const currentConfig: FeatureConfig = prev.feature_config || {
        target_lags: prev.lags || [1, 7],
        temporal: { month: false, day_of_week: false, day_of_month: false, week_of_year: false, year: false, hour_of_day: false, minute_of_day: false },
        exogenous: [],
        derived: []
      };
      return {
        feature_config: updater(currentConfig)
      };
    });
  };

  const parseLags = (str: string): number[] => {
    return str.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n >= 0);
  };

  const numericColumns = availableColumns.filter(c => c.dtype === 'numeric');

  // Build list of all available features for derived feature selection
  const allFeatures: string[] = [
    // Raw columns
    ...numericColumns.map(c => c.name),
    // Target lags
    ...(featureConfig.target_lags || []).map((lag: number) => `target_lag_${lag}`),
    // Exogenous lags
    ...(featureConfig.exogenous || []).flatMap((ex: ExogenousFeatureConfig) => 
      (ex.lags || []).map((lag: number) => `${ex.column}_lag_${lag}`)
    )
  ];

  return (
    <div className="space-y-4">
      <h4 className="text-sm font-semibold text-white flex items-center gap-2">
        <Calculator size={14} className="text-amber-500" />
        Feature Engineering
      </h4>
      
      {/* Temporal Features - Compact Row */}
      <div className="bg-black/20 p-3 rounded-lg border border-white/5">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-slate-400 font-medium">Temporal Features</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {[
            { key: 'month', label: 'Month' },
            { key: 'day_of_week', label: 'DoW' },
            { key: 'day_of_month', label: 'DoM' },
            { key: 'week_of_year', label: 'Week' },
            { key: 'year', label: 'Year' },
            { key: 'hour_of_day', label: 'Hour' },
            { key: 'minute_of_day', label: 'Minute' },
          ].map(({ key, label }) => (
            <button
              key={key}
              onClick={() => updateConfig(prev => ({
                ...prev,
                temporal: { ...prev.temporal, [key]: !prev.temporal?.[key as keyof TemporalFeatureConfig] }
              }))}
              className={`px-2 py-1 text-[10px] rounded border transition-all ${
                featureConfig.temporal?.[key as keyof TemporalFeatureConfig]
                  ? 'bg-amber-500 text-black border-amber-500 font-semibold'
                  : 'bg-white/5 text-slate-400 border-white/10 hover:border-white/20'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Exogenous Variables - Clean Table */}
      {numericColumns.length > 0 && (
        <div className="bg-black/20 p-3 rounded-lg border border-white/5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs text-slate-400 font-medium">Exogenous Variables</span>
            <span className="text-[10px] text-slate-500">Write lags: 0, 1, 7...</span>
          </div>
          <div className="space-y-2">
            {numericColumns.map(col => {
              const exogConfig = featureConfig.exogenous?.find((e: ExogenousFeatureConfig) => e.column === col.name);
              const isEnabled = !!exogConfig;
              
              return (
                <div key={col.name} className={`flex items-center gap-3 p-2 rounded-lg transition-all ${isEnabled ? 'bg-amber-500/5 border border-amber-500/20' : 'bg-black/20 border border-transparent'}`}>
                  <input 
                    type="checkbox" 
                    checked={isEnabled}
                    onChange={(e) => {
                      if (e.target.checked) {
                        updateConfig(prev => ({
                          ...prev,
                          exogenous: [...(prev.exogenous || []), { column: col.name, lags: [0, 1], use_actual: false }]
                        }));
                      } else {
                        updateConfig(prev => ({
                          ...prev,
                          exogenous: (prev.exogenous || []).filter((ex: ExogenousFeatureConfig) => ex.column !== col.name)
                        }));
                      }
                    }}
                    className="accent-amber-500 w-3.5 h-3.5" 
                  />
                  <span className={`text-xs font-medium w-24 truncate ${isEnabled ? 'text-amber-400' : 'text-slate-400'}`}>{col.name}</span>
                  
                  {isEnabled && (
                    <input
                      type="text"
                      placeholder="0, 1, 7"
                      defaultValue={exogConfig?.lags?.join(', ') || '0, 1'}
                      onBlur={(e) => {
                        const lags = parseLags(e.target.value);
                        updateConfig(prev => ({
                          ...prev,
                          exogenous: prev.exogenous.map((ex: ExogenousFeatureConfig) => 
                            ex.column === col.name ? { ...ex, lags } : ex
                          )
                        }));
                      }}
                      className="flex-1 bg-black/30 border border-white/10 rounded px-2 py-1 text-xs text-slate-200 placeholder:text-slate-600 focus:border-amber-500 outline-none font-mono"
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Derived Features - Simple List */}
      <div className="bg-black/20 p-3 rounded-lg border border-white/5">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs text-slate-400 font-medium">Derived Features</span>
        </div>
        
        {/* Existing derived features */}
        {(featureConfig.derived || []).length > 0 && (
          <div className="space-y-1.5 mb-3">
            {featureConfig.derived?.map((d: DerivedFeatureConfig, idx: number) => (
              <div key={idx} className="flex items-center justify-between px-2 py-1.5 bg-amber-500/5 rounded border border-amber-500/20">
                <span className="text-xs font-mono text-amber-300">
                  {d.alias || `${d.feature_a} ${d.operation === 'sum' ? '+' : d.operation === 'difference' ? '-' : d.operation === 'product' ? '×' : '÷'} ${d.feature_b}`}
                </span>
                <button onClick={() => updateConfig(prev => ({ ...prev, derived: prev.derived?.filter((_: DerivedFeatureConfig, i: number) => i !== idx) }))} className="text-slate-500 hover:text-red-400">
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Add new derived feature */}
        <div className="flex flex-col sm:flex-row gap-2">
          <div className="flex gap-2 flex-1 min-w-0">
            <select id={`featA-${model.id}`} className="flex-1 min-w-0 bg-slate-900 border border-white/10 rounded px-2 py-1.5 text-xs text-white outline-none truncate [&>option]:bg-slate-900 [&>option]:text-white">
              {allFeatures.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
            <select id={`op-${model.id}`} className="w-10 sm:w-14 bg-slate-900 border border-white/10 rounded px-1 py-1.5 text-xs text-white outline-none text-center [&>option]:bg-slate-900 [&>option]:text-white">
              <option value="sum">+</option>
              <option value="difference">−</option>
              <option value="product">×</option>
              <option value="ratio">÷</option>
            </select>
            <select id={`featB-${model.id}`} className="flex-1 min-w-0 bg-slate-900 border border-white/10 rounded px-2 py-1.5 text-xs text-white outline-none truncate [&>option]:bg-slate-900 [&>option]:text-white">
              {allFeatures.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
          <button 
            onClick={() => {
              const op = (document.getElementById(`op-${model.id}`) as HTMLSelectElement).value as DerivedFeatureConfig['operation'];
              const featA = (document.getElementById(`featA-${model.id}`) as HTMLSelectElement).value;
              const featB = (document.getElementById(`featB-${model.id}`) as HTMLSelectElement).value;
              
              updateConfig(prev => ({
                ...prev,
                derived: [...(prev.derived || []), {
                  operation: op,
                  feature_a: featA,
                  feature_b: featB,
                  alias: `${featA}_${op}_${featB}`
                }]
              }));
            }}
            className="px-3 py-1.5 bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 rounded text-xs font-medium transition-colors sm:w-auto w-full"
          >
            <Plus size={14} className="mx-auto sm:mx-0" />
          </button>
        </div>
      </div>
    </div>
  );
}
