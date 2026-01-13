'use client';

import { Plus, Trash2, ChevronDown, Activity, LineChart, Network, TrendingUp, Target, Settings, Zap, Play, X, Calculator } from 'lucide-react';
import { ModelConfig, ModelType, DateRange, ColumnInfo, LinearRegressionParams, FeatureConfig, TemporalFeatureConfig, ExogenousFeatureConfig, DerivedFeatureConfig } from '../types/forecasting';
import ForecastStrategyVisualizer from './ForecastStrategyVisualizer';
import { useState, useMemo } from 'react';

// Model Library Configuration
const MODEL_LIBRARY = [
  { id: 'LAG', name: 'Lag Baseline', desc: 'Simple persistence model', Icon: Activity, color: 'amber' },
  { id: 'LINEAR_REGRESSION', name: 'Linear Regression', desc: 'OLS with lag features', Icon: LineChart, color: 'amber' },
  { id: 'XGBOOST', name: 'XGBoost', desc: 'Gradient boosted trees', Icon: Network, color: 'amber' },
  { id: 'ARIMA', name: 'ARIMA', desc: 'Auto-regressive model', Icon: TrendingUp, color: 'blue' },
  { id: 'PROPHET', name: 'Prophet', desc: 'Trend + seasonality', Icon: Target, color: 'purple' },
] as const;

interface StrategyStepProps {
  // Data
  data: { dateColumn: string; targetColumn: string; frequency?: string } | null;
  fullData: Record<string, unknown>[];
  availableColumns: ColumnInfo[];
  
  // Models
  selectedModels: ModelConfig[];
  setSelectedModels: (models: ModelConfig[]) => void;
  addModel: (type: ModelType) => void;
  updateModelParams: (id: string, params: Record<string, unknown> | ((prev: Record<string, unknown>) => Record<string, unknown>)) => void;
  updateModelName: (id: string, name: string) => void;
  
  // Strategy
  trainingRanges: DateRange[];
  setTrainingRanges: (ranges: DateRange[]) => void;
  predictionRanges: DateRange[];
  setPredictionRanges: (ranges: DateRange[]) => void;
  forecastHorizon: number;
  setForecastHorizon: (h: number) => void;
  defaultLags: number[];
  setDefaultLags: (lags: number[]) => void;
  
  // Navigation
  onBack: () => void;
  onStartTraining: () => void;
}

// ============================================================================
// Inline Components
// ============================================================================

// TagInput for lags
function TagInput({ 
  values, 
  onChange, 
  placeholder 
}: { 
  values: number[]; 
  onChange: (v: number[]) => void; 
  placeholder?: string;
}) {
  const [input, setInput] = useState('');
  
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      const num = parseInt(input.trim());
      if (!isNaN(num) && num > 0 && !values.includes(num)) {
        onChange([...values, num].sort((a, b) => a - b));
      }
      setInput('');
    }
  };
  
  return (
    <div className="flex flex-wrap gap-1 p-2 bg-black/20 border border-white/10 rounded min-h-[36px]">
      {values.map((v) => (
        <span 
          key={v} 
          className="px-1.5 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded flex items-center gap-1"
        >
          {v}
          <button onClick={() => onChange(values.filter(x => x !== v))} className="hover:text-red-400">
            <X size={10} />
          </button>
        </span>
      ))}
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={values.length === 0 ? placeholder : ''}
        className="flex-1 min-w-[60px] bg-transparent text-xs text-white outline-none placeholder:text-slate-600"
      />
    </div>
  );
}

// Slider
function MiniSlider({
  value, min, max, step, onChange, label
}: {
  value: number; min: number; max: number; step: number; onChange: (v: number) => void; label: string;
}) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <span className="text-[10px] text-slate-400">{label}</span>
        <span className="text-[10px] text-amber-400 font-mono">{value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 appearance-none bg-slate-700 rounded cursor-pointer accent-amber-500"
      />
    </div>
  );
}

// Feature Config Panel
function FeatureConfigPanel({ 
  model, 
  updateModelParams, 
  availableColumns 
}: { 
  model: ModelConfig; 
  updateModelParams: (id: string, params: Record<string, unknown> | ((prev: Record<string, unknown>) => Record<string, unknown>)) => void;
  availableColumns: ColumnInfo[];
}) {
  const params = model.params as unknown as Record<string, unknown>;
  const featureConfig: FeatureConfig = (params.feature_config as FeatureConfig) || {
    target_lags: (params.lags as number[]) || [1, 7],
    temporal: { month: false, day_of_week: false, day_of_month: false, week_of_year: false, year: false, hour_of_day: false, minute_of_day: false },
    exogenous: [],
    derived: []
  };

  const updateConfig = (updater: (prev: FeatureConfig) => FeatureConfig) => {
    updateModelParams(model.id, (prev: Record<string, unknown>) => {
      const currentConfig: FeatureConfig = (prev.feature_config as FeatureConfig) || {
        target_lags: (prev.lags as number[]) || [1, 7],
        temporal: { month: false, day_of_week: false, day_of_month: false, week_of_year: false, year: false, hour_of_day: false, minute_of_day: false },
        exogenous: [],
        derived: []
      };
      return { feature_config: updater(currentConfig) };
    });
  };

  const parseLags = (str: string): number[] => {
    return str.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n >= 0);
  };

  const numericColumns = availableColumns.filter(c => c.dtype === 'numeric');

  return (
    <div className="space-y-3 mt-3 pt-3 border-t border-white/10">
      <h5 className="text-[11px] text-slate-400 uppercase tracking-wider flex items-center gap-1">
        <Calculator size={10} className="text-amber-500" /> Feature Engineering
      </h5>
      
      {/* Temporal Features */}
      <div className="bg-black/20 p-2 rounded border border-white/5">
        <span className="text-[10px] text-slate-500 mb-1.5 block">Temporal</span>
        <div className="flex flex-wrap gap-1">
          {[
            { key: 'month', label: 'Mo' },
            { key: 'day_of_week', label: 'DoW' },
            { key: 'day_of_month', label: 'DoM' },
            { key: 'week_of_year', label: 'Wk' },
            { key: 'hour_of_day', label: 'Hr' },
          ].map(({ key, label }) => (
            <button
              key={key}
              onClick={() => updateConfig(prev => ({
                ...prev,
                temporal: { ...prev.temporal, [key]: !prev.temporal?.[key as keyof TemporalFeatureConfig] }
              }))}
              className={`px-1.5 py-0.5 text-[9px] rounded border transition-all ${
                featureConfig.temporal?.[key as keyof TemporalFeatureConfig]
                  ? 'bg-amber-500 text-black border-amber-500 font-semibold'
                  : 'bg-white/5 text-slate-500 border-white/10 hover:border-white/20'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Exogenous Variables */}
      {numericColumns.length > 0 && (
        <div className="bg-black/20 p-2 rounded border border-white/5">
          <span className="text-[10px] text-slate-500 mb-1.5 block">Exogenous Variables</span>
          <div className="space-y-1">
            {numericColumns.slice(0, 4).map(col => {
              const exogConfig = featureConfig.exogenous?.find((e: ExogenousFeatureConfig) => e.column === col.name);
              const isEnabled = !!exogConfig;
              
              return (
                <div key={col.name} className="flex items-center gap-2">
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
                    className="accent-amber-500 w-3 h-3" 
                  />
                  <span className={`text-[10px] truncate flex-1 ${isEnabled ? 'text-amber-400' : 'text-slate-500'}`}>{col.name}</span>
                  {isEnabled && (
                    <input
                      type="text"
                      placeholder="0, 1, 7"
                      defaultValue={exogConfig?.lags?.join(', ') || '0, 1'}
                      onBlur={(e) => {
                        const lags = parseLags(e.target.value);
                        updateConfig(prev => ({
                          ...prev,
                          exogenous: (prev.exogenous || []).map((ex: ExogenousFeatureConfig) => 
                            ex.column === col.name ? { ...ex, lags } : ex
                          )
                        }));
                      }}
                      className="w-16 bg-black/30 border border-white/10 rounded px-1.5 py-0.5 text-[10px] text-slate-200 font-mono"
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// Model Form Component
function ModelForm({ 
  model, 
  defaultLags,
  onRemove,
  onUpdateParams,
  availableColumns 
}: { 
  model: ModelConfig;
  defaultLags: number[];
  onRemove: () => void;
  onUpdateParams: (id: string, params: Record<string, unknown> | ((prev: Record<string, unknown>) => Record<string, unknown>)) => void;
  availableColumns: ColumnInfo[];
}) {
  const [isExpanded, setIsExpanded] = useState(true);
  const params = model.params as unknown as Record<string, unknown>;
  const modelLags = params.lags as number[] | undefined;
  const isUsingDefaultLags = !modelLags || JSON.stringify(modelLags) === JSON.stringify(defaultLags);
  const isMLModel = ['LAG', 'LINEAR_REGRESSION', 'XGBOOST'].includes(model.type);
  
  const modelMeta = MODEL_LIBRARY.find(m => m.id === model.type);
  const Icon = modelMeta?.Icon || Activity;
  const colorClass = modelMeta?.color === 'blue' ? 'blue' : modelMeta?.color === 'purple' ? 'purple' : 'amber';

  return (
    <div className={`border rounded-lg overflow-hidden bg-black/20 border-${colorClass}-500/30`}>
      {/* Header */}
      <div 
        className="flex items-center justify-between p-3 cursor-pointer hover:bg-white/5 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <div className={`w-6 h-6 rounded bg-${colorClass}-500/20 flex items-center justify-center text-${colorClass}-400`}>
            <Icon size={12} />
          </div>
          <span className="text-sm text-white font-medium">{modelMeta?.name || model.type}</span>
          {isMLModel && model.type !== 'LAG' && (
            <span className={`text-[9px] px-1.5 py-0.5 rounded ${
              isUsingDefaultLags 
                ? 'bg-slate-700/50 text-slate-400' 
                : 'bg-amber-500/20 text-amber-400'
            }`}>
              {isUsingDefaultLags ? 'default lags' : `[${(modelLags || []).join(',')}]`}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <ChevronDown size={14} className={`text-slate-500 transition-transform ${isExpanded ? '' : '-rotate-90'}`} />
          <button 
            onClick={(e) => { e.stopPropagation(); onRemove(); }}
            className="text-slate-500 hover:text-red-400 p-1"
          >
            <X size={14} />
          </button>
        </div>
      </div>
      
      {/* Content */}
      {isExpanded && (
        <div className="px-3 pb-3 space-y-3">
          
          {/* LAG */}
          {model.type === 'LAG' && (
            <div>
              <label className="text-[10px] text-slate-500 mb-1 block">Lag Period</label>
              <input
                type="number"
                min={1}
                value={(params.lag as number) ?? 1}
                onChange={(e) => onUpdateParams(model.id, { lag: parseInt(e.target.value) || 1 })}
                className="w-full bg-black/30 border border-white/10 rounded px-2 py-1.5 text-sm text-white"
              />
            </div>
          )}

          {/* LINEAR REGRESSION */}
          {model.type === 'LINEAR_REGRESSION' && (() => {
            const lrParams = params as unknown as LinearRegressionParams;
            return (
              <>
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-[10px] text-slate-500">Target Lags</label>
                    {!isUsingDefaultLags && (
                      <button 
                        onClick={() => onUpdateParams(model.id, { lags: defaultLags })}
                        className="text-[9px] text-amber-500 hover:text-amber-400"
                      >
                        Reset
                      </button>
                    )}
                  </div>
                  <TagInput
                    values={lrParams.lags ?? defaultLags}
                    onChange={(lags) => onUpdateParams(model.id, { lags })}
                    placeholder="1, 7, 14"
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[10px] text-slate-500 mb-1 block">Mode</label>
                    <div className="flex gap-1">
                      {['raw', 'residual'].map((mode) => (
                        <button
                          key={mode}
                          onClick={() => onUpdateParams(model.id, { target_mode: mode })}
                          className={`flex-1 py-1 text-[10px] rounded border ${
                            (lrParams.target_mode ?? 'raw') === mode
                              ? 'bg-amber-500/20 border-amber-500/50 text-amber-400'
                              : 'bg-black/20 border-white/10 text-slate-500'
                          }`}
                        >
                          {mode}
                        </button>
                      ))}
                    </div>
                  </div>
                  {lrParams.target_mode === 'residual' ? (
                    <div>
                      <label className="text-[10px] text-slate-500 mb-1 block">Residual Lag</label>
                      <input
                        type="number"
                        min={1}
                        value={(lrParams as unknown as { residual_lag?: number }).residual_lag ?? 1}
                        onChange={(e) => {
                          const val = parseInt(e.target.value);
                          if (!isNaN(val) && val >= 1) {
                            onUpdateParams(model.id, { residual_lag: val });
                          }
                        }}
                        className="w-full bg-black/30 border border-white/10 rounded px-2 py-1 text-[10px] text-white"
                      />
                    </div>
                  ) : (
                    <div>
                      <label className="text-[10px] text-slate-500 mb-1 block">Standardize</label>
                      <button
                        onClick={() => onUpdateParams(model.id, { standardize: !lrParams.standardize })}
                        className={`w-full py-1 text-[10px] rounded border ${
                          lrParams.standardize
                            ? 'bg-amber-500/20 border-amber-500/50 text-amber-400'
                            : 'bg-black/20 border-white/10 text-slate-500'
                        }`}
                      >
                        {lrParams.standardize ? 'Yes' : 'No'}
                      </button>
                    </div>
                  )}
                </div>
                
                {lrParams.target_mode === 'residual' && (
                  <div>
                    <label className="text-[10px] text-slate-500 mb-1 block">Standardize</label>
                    <button
                      onClick={() => onUpdateParams(model.id, { standardize: !lrParams.standardize })}
                      className={`w-full py-1 text-[10px] rounded border ${
                        lrParams.standardize
                          ? 'bg-amber-500/20 border-amber-500/50 text-amber-400'
                          : 'bg-black/20 border-white/10 text-slate-500'
                      }`}
                    >
                      {lrParams.standardize ? 'Yes' : 'No'}
                    </button>
                  </div>
                )}
                
                <FeatureConfigPanel model={model} updateModelParams={onUpdateParams} availableColumns={availableColumns} />
              </>
            );
          })()}

          {/* XGBOOST */}
          {model.type === 'XGBOOST' && (() => {
            const xgbParams = params as { lags?: number[]; target_mode?: string; n_estimators?: number; max_depth?: number; learning_rate?: number };
            return (
              <>
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-[10px] text-slate-500">Target Lags</label>
                    {!isUsingDefaultLags && (
                      <button 
                        onClick={() => onUpdateParams(model.id, { lags: defaultLags })}
                        className="text-[9px] text-amber-500 hover:text-amber-400"
                      >
                        Reset
                      </button>
                    )}
                  </div>
                  <TagInput
                    values={(params.lags as number[]) ?? defaultLags}
                    onChange={(lags) => onUpdateParams(model.id, { lags })}
                    placeholder="1, 7, 14"
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[10px] text-slate-500 mb-1 block">Mode</label>
                    <div className="flex gap-1">
                      {['raw', 'residual'].map((mode) => (
                        <button
                          key={mode}
                          onClick={() => onUpdateParams(model.id, { target_mode: mode })}
                          className={`flex-1 py-1 text-[10px] rounded border ${
                            (xgbParams.target_mode ?? 'raw') === mode
                              ? 'bg-amber-500/20 border-amber-500/50 text-amber-400'
                              : 'bg-black/20 border-white/10 text-slate-500'
                          }`}
                        >
                          {mode}
                        </button>
                      ))}
                    </div>
                  </div>
                  {xgbParams.target_mode === 'residual' ? (
                    <div>
                      <label className="text-[10px] text-slate-500 mb-1 block">Residual Lag</label>
                      <input
                        type="number"
                        min={1}
                        value={(xgbParams as unknown as { residual_lag?: number }).residual_lag ?? 1}
                        onChange={(e) => {
                          const val = parseInt(e.target.value);
                          if (!isNaN(val) && val >= 1) {
                            onUpdateParams(model.id, { residual_lag: val });
                          }
                        }}
                        className="w-full bg-black/30 border border-white/10 rounded px-2 py-1 text-[10px] text-white"
                      />
                    </div>
                  ) : (
                    <MiniSlider
                      value={(params.n_estimators as number) ?? 100}
                      min={10} max={500} step={10}
                      onChange={(v) => onUpdateParams(model.id, { n_estimators: v })}
                      label="Trees"
                    />
                  )}
                </div>
                
                {xgbParams.target_mode === 'residual' && (
                  <MiniSlider
                    value={(params.n_estimators as number) ?? 100}
                    min={10} max={500} step={10}
                    onChange={(v) => onUpdateParams(model.id, { n_estimators: v })}
                    label="Trees"
                  />
                )}
                
                <div className="grid grid-cols-2 gap-2">
                  <MiniSlider
                    value={(params.max_depth as number) ?? 6}
                    min={1} max={15} step={1}
                    onChange={(v) => onUpdateParams(model.id, { max_depth: v })}
                    label="Depth"
                  />
                  <div>
                    <label className="text-[10px] text-slate-500 mb-1 block">Learning Rate</label>
                    <input 
                      type="text" 
                      inputMode="decimal"
                      value={(params.learning_rate as number) ?? 0.1}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val) && val > 0 && val <= 1) {
                          onUpdateParams(model.id, { learning_rate: val });
                        }
                      }}
                      className="w-full bg-black/30 border border-white/10 rounded px-2 py-1 text-sm text-white"
                    />
                  </div>
                </div>
                
                <FeatureConfigPanel model={model} updateModelParams={onUpdateParams} availableColumns={availableColumns} />
              </>
            );
          })()}

          {/* ARIMA */}
          {model.type === 'ARIMA' && (
            <div className="grid grid-cols-3 gap-2">
              <MiniSlider value={(params.p as number) ?? 1} min={0} max={5} step={1} onChange={(v) => onUpdateParams(model.id, { p: v })} label="P (AR)" />
              <MiniSlider value={(params.d as number) ?? 1} min={0} max={2} step={1} onChange={(v) => onUpdateParams(model.id, { d: v })} label="D" />
              <MiniSlider value={(params.q as number) ?? 1} min={0} max={5} step={1} onChange={(v) => onUpdateParams(model.id, { q: v })} label="Q (MA)" />
            </div>
          )}

          {/* PROPHET */}
          {model.type === 'PROPHET' && (
            <>
              <div>
                <label className="text-[10px] text-slate-500 mb-1.5 block">Seasonality</label>
                <div className="flex gap-1">
                  {['daily', 'weekly', 'yearly'].map((s) => (
                    <button
                      key={s}
                      onClick={() => onUpdateParams(model.id, { [`${s}_seasonality`]: !(params[`${s}_seasonality`] as boolean ?? s !== 'daily') })}
                      className={`flex-1 py-1 text-[10px] rounded border capitalize ${
                        (params[`${s}_seasonality`] as boolean ?? s !== 'daily')
                          ? 'bg-purple-500/20 border-purple-500/50 text-purple-400'
                          : 'bg-black/20 border-white/10 text-slate-500'
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-[10px] text-slate-500 mb-1 block">Mode</label>
                <div className="flex gap-1">
                  {['additive', 'multiplicative'].map((mode) => (
                    <button
                      key={mode}
                      onClick={() => onUpdateParams(model.id, { seasonality_mode: mode })}
                      className={`flex-1 py-1 text-[10px] rounded border capitalize ${
                        ((params.seasonality_mode as string) ?? 'additive') === mode
                          ? 'bg-purple-500/20 border-purple-500/50 text-purple-400'
                          : 'bg-black/20 border-white/10 text-slate-500'
                      }`}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function StrategyStep({
  data,
  fullData,
  availableColumns,
  selectedModels,
  setSelectedModels,
  addModel,
  updateModelParams,
  trainingRanges,
  setTrainingRanges,
  predictionRanges,
  setPredictionRanges,
  forecastHorizon,
  setForecastHorizon,
  defaultLags,
  setDefaultLags,
  onBack,
  onStartTraining
}: StrategyStepProps) {
  
  // Calculate max horizon based on actual data points in validation window
  const maxHorizon = useMemo(() => {
    if (predictionRanges.length === 0 || !predictionRanges[0].start || !predictionRanges[0].end) {
      return 30; // Default fallback
    }
    try {
      const startDate = new Date(predictionRanges[0].start);
      const endDate = new Date(predictionRanges[0].end);
      // Count actual data points in the validation range (inclusive)
      if (fullData && fullData.length > 0 && data?.dateColumn) {
        const validationPoints = fullData.filter(row => {
          const rowDate = new Date(row[data.dateColumn] as string);
          return rowDate >= startDate && rowDate <= endDate;
        }).length;
        // The max forecast horizon should be exactly the number of validation points (inclusive)
        if (validationPoints > 0) {
          return Math.max(1, Math.min(365, validationPoints));
        }
      }
      // Fallback: estimate based on frequency if no data
      const diffTime = Math.abs(endDate.getTime() - startDate.getTime());
      const frequency = data?.frequency || 'D';
      let estimatedPoints: number;
      switch (frequency) {
        case 'min':
          estimatedPoints = Math.floor(diffTime / (1000 * 60)) + 1;
          break;
        case 'H':
          estimatedPoints = Math.floor(diffTime / (1000 * 60 * 60)) + 1;
          break;
        case 'D':
          estimatedPoints = Math.floor(diffTime / (1000 * 60 * 60 * 24)) + 1;
          break;
        case 'W':
          estimatedPoints = Math.floor(diffTime / (1000 * 60 * 60 * 24 * 7)) + 1;
          break;
        case 'M':
          estimatedPoints = Math.floor(diffTime / (1000 * 60 * 60 * 24 * 30)) + 1;
          break;
        default:
          estimatedPoints = Math.floor(diffTime / (1000 * 60 * 60 * 24)) + 1;
      }
      // The +1 ensures the interval is inclusive
      return Math.max(1, Math.min(365, estimatedPoints));
    } catch {
      return 30;
    }
  }, [predictionRanges, fullData, data?.dateColumn, data?.frequency]);
  
  return (
    <div className="flex flex-col lg:flex-row gap-4 h-full animate-in fade-in slide-in-from-bottom-4 duration-500 overflow-hidden">
      
      {/* LEFT: Model Library */}
      <div className="lg:w-64 flex-shrink-0 bg-white/5 border border-white/10 rounded-xl p-4 overflow-y-auto custom-scrollbar">
        <h4 className="font-semibold text-sm text-white mb-4 flex items-center gap-2">
          <Settings size={14} className="text-amber-500" />
          Model Library
        </h4>
        
        <div className="space-y-2">
          {MODEL_LIBRARY.map((m) => {
            const Icon = m.Icon;
            const count = selectedModels.filter(sm => sm.type === m.id).length;
            const colorClass = m.color === 'blue' ? 'blue' : m.color === 'purple' ? 'purple' : 'amber';
            
            return (
              <button
                key={m.id}
                onClick={() => addModel(m.id as ModelType)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-all hover:scale-[1.02] bg-${colorClass}-500/5 border-${colorClass}-500/20 hover:border-${colorClass}-500/50`}
              >
                <div className={`w-8 h-8 rounded-lg bg-${colorClass}-500/20 flex items-center justify-center text-${colorClass}-400`}>
                  <Icon size={16} />
                </div>
                <div className="flex-1 text-left">
                  <div className="text-sm text-white font-medium">{m.name}</div>
                  <div className="text-[10px] text-slate-500">{m.desc}</div>
                </div>
                {count > 0 && (
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-full bg-${colorClass}-500/30 text-${colorClass}-300`}>
                    {count}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* RIGHT: Configuration */}
      <div className="flex-1 flex flex-col gap-4 overflow-y-auto custom-scrollbar pr-1">
        
        {/* Validation Strategy (Train/Test + Horizon) */}
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <h4 className="font-semibold text-sm text-white mb-3 flex items-center gap-2">
            <Zap size={14} className="text-amber-500" />
            Validation Strategy
          </h4>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Training Period */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[10px] text-slate-500 uppercase">Training</label>
                <button 
                  onClick={() => setTrainingRanges([...trainingRanges, { start: '', end: '' }])}
                  className="text-[10px] text-amber-500 hover:text-amber-400"
                >
                  <Plus size={10} />
                </button>
              </div>
              <div className="space-y-1.5">
                {trainingRanges.map((range, idx) => (
                  <div key={idx} className="flex items-center gap-1">
                    <input 
                      type="date" 
                      value={range.start}
                      onChange={(e) => {
                        const newRanges = [...trainingRanges];
                        newRanges[idx].start = e.target.value;
                        setTrainingRanges(newRanges);
                      }}
                      className="flex-1 bg-black/20 border border-white/10 rounded px-1.5 py-1 text-[10px] text-white min-w-0"
                    />
                    <span className="text-slate-600 text-[10px]">→</span>
                    <input 
                      type="date" 
                      value={range.end}
                      onChange={(e) => {
                        const newRanges = [...trainingRanges];
                        newRanges[idx].end = e.target.value;
                        setTrainingRanges(newRanges);
                      }}
                      className="flex-1 bg-black/20 border border-white/10 rounded px-1.5 py-1 text-[10px] text-white min-w-0"
                    />
                    <button onClick={() => setTrainingRanges(trainingRanges.filter((_, i) => i !== idx))} className="text-slate-600 hover:text-red-400">
                      <Trash2 size={10} />
                    </button>
                  </div>
                ))}
                {trainingRanges.length === 0 && <p className="text-[9px] text-slate-600 italic">Click + to add</p>}
              </div>
            </div>

            {/* Validation Period */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[10px] text-slate-500 uppercase">Validation</label>
                <button 
                  onClick={() => setPredictionRanges([...predictionRanges, { start: '', end: '' }])}
                  className="text-[10px] text-amber-500 hover:text-amber-400"
                >
                  <Plus size={10} />
                </button>
              </div>
              <div className="space-y-1.5">
                {predictionRanges.map((range, idx) => (
                  <div key={idx} className="flex items-center gap-1">
                    <input 
                      type="date" 
                      value={range.start}
                      onChange={(e) => {
                        const newRanges = [...predictionRanges];
                        newRanges[idx].start = e.target.value;
                        setPredictionRanges(newRanges);
                      }}
                      className="flex-1 bg-black/20 border border-white/10 rounded px-1.5 py-1 text-[10px] text-white min-w-0"
                    />
                    <span className="text-slate-600 text-[10px]">→</span>
                    <input 
                      type="date" 
                      value={range.end}
                      onChange={(e) => {
                        const newRanges = [...predictionRanges];
                        newRanges[idx].end = e.target.value;
                        setPredictionRanges(newRanges);
                      }}
                      className="flex-1 bg-black/20 border border-white/10 rounded px-1.5 py-1 text-[10px] text-white min-w-0"
                    />
                    <button onClick={() => setPredictionRanges(predictionRanges.filter((_, i) => i !== idx))} className="text-slate-600 hover:text-red-400">
                      <Trash2 size={10} />
                    </button>
                  </div>
                ))}
                {predictionRanges.length === 0 && <p className="text-[9px] text-slate-600 italic">Click + to add</p>}
              </div>
            </div>

            {/* Horizon */}
            <div>
              <label className="text-[10px] text-slate-500 uppercase mb-1.5 block">Forecast Horizon</label>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setForecastHorizon(Math.max(1, forecastHorizon - 1))}
                  className="p-1 bg-black/20 border border-white/10 rounded hover:border-amber-500/50 text-slate-400"
                >
                  <ChevronDown size={10} className="rotate-90" />
                </button>
                <input
                  type="number"
                  min={1}
                  max={maxHorizon}
                  value={forecastHorizon}
                  onChange={(e) => setForecastHorizon(Math.max(1, Math.min(maxHorizon, parseInt(e.target.value) || 1)))}
                  className="w-14 bg-black/20 border border-white/10 rounded px-1.5 py-1 text-center text-[10px] font-mono text-white focus:border-amber-500 outline-none"
                />
                <button
                  onClick={() => setForecastHorizon(Math.min(maxHorizon, forecastHorizon + 1))}
                  className="p-1 bg-black/20 border border-white/10 rounded hover:border-amber-500/50 text-slate-400"
                >
                  <ChevronDown size={10} className="-rotate-90" />
                </button>
                <button
                  onClick={() => setForecastHorizon(maxHorizon)}
                  className="text-[9px] text-slate-500 ml-1 hover:text-amber-400 transition-colors cursor-pointer"
                  title="Set horizon to full validation window"
                >
                  / <span className="underline decoration-dotted">{maxHorizon}</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Forecast Strategy Visualizer */}
        <ForecastStrategyVisualizer
          lags={defaultLags}
          horizon={forecastHorizon}
          onLagsChange={setDefaultLags}
          sampleData={fullData.slice(-15).map(d => (d[data?.targetColumn || ''] as number)).filter(v => typeof v === 'number')}
          frequency={data?.frequency || 'D'}
        />

        {/* Selected Models Configuration */}
        {selectedModels.length > 0 && (
          <div className="space-y-3">
            <h4 className="font-semibold text-sm text-white flex items-center gap-2">
              <Settings size={14} className="text-amber-500" />
              Model Configuration ({selectedModels.length})
            </h4>
            
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
              {selectedModels.map((model) => (
                <ModelForm
                  key={model.id}
                  model={model}
                  defaultLags={defaultLags}
                  onRemove={() => setSelectedModels(selectedModels.filter(m => m.id !== model.id))}
                  onUpdateParams={updateModelParams}
                  availableColumns={availableColumns}
                />
              ))}
            </div>
          </div>
        )}
        
        {selectedModels.length === 0 && (
          <div className="flex-1 flex items-center justify-center border-2 border-dashed border-white/10 rounded-xl">
            <p className="text-slate-500 text-sm">← Select models from the library</p>
          </div>
        )}

        {/* Footer */}
        <div className="flex justify-between items-center pt-4 border-t border-white/10 mt-auto">
          <button onClick={onBack} className="text-slate-400 hover:text-white transition-colors text-sm">
            ← Back to Data
          </button>
          <button 
            onClick={onStartTraining}
            disabled={selectedModels.length === 0}
            className="flex items-center gap-2 px-6 py-3 bg-amber-500 hover:bg-amber-400 disabled:opacity-50 disabled:cursor-not-allowed text-black font-bold rounded-lg shadow-[0_0_20px_rgba(245,158,11,0.3)] transition-all"
          >
            <Play size={16} fill="currentColor" /> Start Training ({selectedModels.length})
          </button>
        </div>
      </div>
    </div>
  );
}
