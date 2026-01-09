'use client';

import { useState } from 'react';
import { ChevronDown, ChevronRight, X, Activity, LineChart, Network, Target, TrendingUp } from 'lucide-react';
import { ModelConfig, ModelType, LinearRegressionParams, ColumnInfo } from '../types/forecasting';

interface ModelCardProps {
  model: ModelConfig;
  defaultLags: number[];
  onRemove: () => void;
  onUpdateParams: (params: Record<string, unknown>) => void;
  availableColumns: ColumnInfo[];
}

const modelIcons: Record<ModelType, React.ElementType> = {
  'LAG': Activity,
  'LINEAR_REGRESSION': LineChart,
  'XGBOOST': Network,
  'ARIMA': TrendingUp,
  'PROPHET': Target,
  'NBEATS': Activity
};

const modelColors: Record<string, { bg: string; text: string; border: string }> = {
  'LAG': { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/30' },
  'LINEAR_REGRESSION': { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/30' },
  'XGBOOST': { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/30' },
  'ARIMA': { bg: 'bg-blue-500/10', text: 'text-blue-400', border: 'border-blue-500/30' },
  'PROPHET': { bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500/30' },
  'NBEATS': { bg: 'bg-green-500/10', text: 'text-green-400', border: 'border-green-500/30' }
};

const modelLabels: Record<ModelType, string> = {
  'LAG': 'Lag Baseline',
  'LINEAR_REGRESSION': 'Linear Regression',
  'XGBOOST': 'XGBoost',
  'ARIMA': 'ARIMA',
  'PROPHET': 'Prophet',
  'NBEATS': 'N-BEATS'
};

// ============================================================================
// Inline Input Components
// ============================================================================

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
          <button 
            onClick={() => onChange(values.filter(x => x !== v))}
            className="hover:text-red-400"
          >
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

function MiniSlider({
  value,
  min,
  max,
  step,
  onChange,
  label
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  label: string;
}) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <span className="text-[10px] text-slate-400">{label}</span>
        <span className="text-[10px] text-amber-400 font-mono">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 appearance-none bg-slate-700 rounded cursor-pointer accent-amber-500"
      />
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function ModelCard({
  model,
  defaultLags,
  onRemove,
  onUpdateParams,
}: ModelCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const Icon = modelIcons[model.type] || Activity;
  const colors = modelColors[model.type] || { bg: 'bg-slate-500/10', text: 'text-slate-400', border: 'border-slate-500/30' };
  
  // Check if model uses custom lags (different from default)
  const modelLags = (model.params as unknown as Record<string, unknown>).lags as number[] | undefined;
  const isUsingDefaultLags = !modelLags || JSON.stringify(modelLags) === JSON.stringify(defaultLags);
  const isMLModel = ['LAG', 'LINEAR_REGRESSION', 'XGBOOST'].includes(model.type);

  return (
    <div className={`border rounded-lg overflow-hidden transition-all ${colors.border} ${isExpanded ? 'bg-black/20' : 'bg-black/10 hover:bg-black/15'}`}>
      {/* Header - Always visible */}
      <div 
        className="flex items-center justify-between p-3 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <button className="text-slate-500">
            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
          <div className={`w-6 h-6 rounded ${colors.bg} flex items-center justify-center ${colors.text}`}>
            <Icon size={12} />
          </div>
          <span className="text-sm text-white font-medium">{modelLabels[model.type]}</span>
          
          {/* Badges */}
          {isMLModel && model.type !== 'LAG' && isUsingDefaultLags && (
            <span className="text-[10px] px-1.5 py-0.5 bg-slate-700/50 text-slate-400 rounded">
              default lags
            </span>
          )}
          {isMLModel && model.type !== 'LAG' && !isUsingDefaultLags && modelLags && (
            <span className="text-[10px] px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded">
              [{modelLags.join(', ')}]
            </span>
          )}
        </div>
        
        <button 
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
          className="text-slate-500 hover:text-red-400 p-1 transition-colors"
        >
          <X size={14} />
        </button>
      </div>
      
      {/* Expanded Content - Model-specific forms */}
      {isExpanded && (
        <div className="px-3 pb-3 border-t border-white/5 pt-3 space-y-3">
          
          {/* LAG Model */}
          {model.type === 'LAG' && (
            <div>
              <label className="text-[11px] text-slate-400 mb-1.5 block">Lag Period</label>
              <input
                type="number"
                min={1}
                value={((model.params as unknown as Record<string, unknown>).lag as number) ?? 1}
                onChange={(e) => onUpdateParams({ lag: parseInt(e.target.value) || 1 })}
                className="w-full bg-black/20 border border-white/10 rounded px-3 py-2 text-sm text-white focus:border-amber-500 outline-none"
              />
              <p className="text-[10px] text-slate-500 mt-1">
                Predict y(t) = y(t-lag). Uses value from {((model.params as unknown as Record<string, unknown>).lag as number) ?? 1} period(s) ago.
              </p>
            </div>
          )}

          {/* LINEAR REGRESSION Model */}
          {model.type === 'LINEAR_REGRESSION' && (() => {
            const params = model.params as LinearRegressionParams;
            return (
              <div className="space-y-3">
                {/* Target Lags */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-[11px] text-slate-400">Target Lags</label>
                    {!isUsingDefaultLags && (
                      <button 
                        onClick={() => onUpdateParams({ lags: defaultLags })}
                        className="text-[10px] text-amber-500 hover:text-amber-400"
                      >
                        Reset to default
                      </button>
                    )}
                  </div>
                  <TagInput
                    values={params.lags ?? defaultLags}
                    onChange={(lags: number[]) => onUpdateParams({ lags })}
                    placeholder="e.g., 1, 7, 14"
                  />
                </div>
                
                {/* Target Mode */}
                <div>
                  <label className="text-[11px] text-slate-400 mb-1.5 block">Target Mode</label>
                  <div className="flex gap-1">
                    {['raw', 'residual'].map((mode) => (
                      <button
                        key={mode}
                        onClick={() => onUpdateParams({ target_mode: mode })}
                        className={`flex-1 py-1.5 text-xs rounded border transition-all ${
                          (params.target_mode ?? 'raw') === mode
                            ? 'bg-amber-500/20 border-amber-500/50 text-amber-400'
                            : 'bg-black/20 border-white/10 text-slate-400 hover:border-white/30'
                        }`}
                      >
                        {mode === 'raw' ? 'Raw (y)' : 'Residual (Δy)'}
                      </button>
                    ))}
                  </div>
                </div>
                
                {/* Residual Lag */}
                {params.target_mode === 'residual' && (
                  <div>
                    <label className="text-[11px] text-slate-400 mb-1.5 block">Residual Lag</label>
                    <input
                      type="number"
                      min={1}
                      value={params.residual_lag ?? 1}
                      onChange={(e) => onUpdateParams({ residual_lag: parseInt(e.target.value) || 1 })}
                      className="w-full bg-black/20 border border-white/10 rounded px-3 py-2 text-sm text-white focus:border-amber-500 outline-none"
                    />
                  </div>
                )}
                
                {/* Standardize */}
                <label className="flex items-center gap-2 p-2 bg-black/20 rounded border border-white/5 cursor-pointer hover:border-amber-500/30">
                  <input 
                    type="checkbox" 
                    checked={params.standardize ?? false}
                    onChange={(e) => onUpdateParams({ standardize: e.target.checked })}
                    className="accent-amber-500 w-3.5 h-3.5" 
                  />
                  <span className="text-xs text-slate-300">Standardize features</span>
                </label>
              </div>
            );
          })()}

          {/* XGBOOST Model */}
          {model.type === 'XGBOOST' && (() => {
            const params = model.params as unknown as Record<string, unknown>;
            return (
              <div className="space-y-3">
                {/* Target Lags */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-[11px] text-slate-400">Target Lags</label>
                    {!isUsingDefaultLags && (
                      <button 
                        onClick={() => onUpdateParams({ lags: defaultLags })}
                        className="text-[10px] text-amber-500 hover:text-amber-400"
                      >
                        Reset to default
                      </button>
                    )}
                  </div>
                  <TagInput
                    values={(params.lags as number[]) ?? defaultLags}
                    onChange={(lags: number[]) => onUpdateParams({ lags })}
                    placeholder="e.g., 1, 7, 14"
                  />
                </div>
                
                {/* XGBoost Params */}
                <div className="grid grid-cols-2 gap-3">
                  <MiniSlider
                    value={(params.n_estimators as number) ?? 100}
                    min={10}
                    max={500}
                    step={10}
                    onChange={(v: number) => onUpdateParams({ n_estimators: v })}
                    label="Trees"
                  />
                  <MiniSlider
                    value={(params.max_depth as number) ?? 6}
                    min={1}
                    max={15}
                    step={1}
                    onChange={(v: number) => onUpdateParams({ max_depth: v })}
                    label="Max Depth"
                  />
                </div>
                
                <div>
                  <label className="text-[11px] text-slate-400 mb-1.5 block">Learning Rate</label>
                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      inputMode="decimal"
                      value={(params.learning_rate as number) ?? 0.1}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val) && val > 0 && val <= 1) {
                          onUpdateParams({ learning_rate: val });
                        }
                      }}
                      className="flex-1 bg-black/20 border border-white/10 rounded px-2 py-1.5 text-sm text-white focus:border-amber-500 outline-none" 
                    />
                    <select
                      value=""
                      onChange={(e) => e.target.value && onUpdateParams({ learning_rate: parseFloat(e.target.value) })}
                      className="bg-black/30 border border-white/10 px-2 rounded text-xs text-slate-400"
                    >
                      <option value="">Quick</option>
                      <option value="0.01">0.01</option>
                      <option value="0.1">0.1</option>
                      <option value="0.3">0.3</option>
                    </select>
                  </div>
                </div>
              </div>
            );
          })()}

          {/* ARIMA Model */}
          {model.type === 'ARIMA' && (() => {
            const params = model.params as unknown as Record<string, unknown>;
            return (
              <div className="space-y-3">
                <div className="grid grid-cols-3 gap-2">
                  <MiniSlider
                    value={(params.p as number) ?? 1}
                    min={0}
                    max={5}
                    step={1}
                    onChange={(v: number) => onUpdateParams({ p: v })}
                    label="P (AR)"
                  />
                  <MiniSlider
                    value={(params.d as number) ?? 1}
                    min={0}
                    max={2}
                    step={1}
                    onChange={(v: number) => onUpdateParams({ d: v })}
                    label="D (Diff)"
                  />
                  <MiniSlider
                    value={(params.q as number) ?? 1}
                    min={0}
                    max={5}
                    step={1}
                    onChange={(v: number) => onUpdateParams({ q: v })}
                    label="Q (MA)"
                  />
                </div>
                <p className="text-[10px] text-blue-400/80 p-2 bg-blue-500/10 border border-blue-500/20 rounded">
                  ARIMA({(params.p as number) ?? 1},{(params.d as number) ?? 1},{(params.q as number) ?? 1}) — univariate, uses only target history
                </p>
              </div>
            );
          })()}

          {/* PROPHET Model */}
          {model.type === 'PROPHET' && (() => {
            const params = model.params as unknown as Record<string, unknown>;
            return (
              <div className="space-y-3">
                {/* Seasonality Toggles */}
                <div>
                  <label className="text-[11px] text-slate-400 mb-1.5 block">Seasonality</label>
                  <div className="flex gap-2">
                    {['daily', 'weekly', 'yearly'].map((s) => (
                      <label key={s} className="flex items-center gap-1.5 p-2 bg-black/20 rounded border border-white/5 cursor-pointer hover:border-purple-500/30 flex-1 justify-center">
                        <input 
                          type="checkbox" 
                          checked={(params[`${s}_seasonality`] as boolean) ?? (s !== 'daily')}
                          onChange={(e) => onUpdateParams({ [`${s}_seasonality`]: e.target.checked })}
                          className="accent-purple-500 w-3.5 h-3.5" 
                        />
                        <span className="text-xs text-slate-300 capitalize">{s}</span>
                      </label>
                    ))}
                  </div>
                </div>
                
                {/* Seasonality Mode */}
                <div>
                  <label className="text-[11px] text-slate-400 mb-1.5 block">Mode</label>
                  <div className="flex gap-1">
                    {['additive', 'multiplicative'].map((mode) => (
                      <button
                        key={mode}
                        onClick={() => onUpdateParams({ seasonality_mode: mode })}
                        className={`flex-1 py-1.5 text-xs rounded border transition-all capitalize ${
                          ((params.seasonality_mode as string) ?? 'additive') === mode
                            ? 'bg-purple-500/20 border-purple-500/50 text-purple-400'
                            : 'bg-black/20 border-white/10 text-slate-400 hover:border-white/30'
                        }`}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                </div>
                
                <p className="text-[10px] text-purple-400/80 p-2 bg-purple-500/10 border border-purple-500/20 rounded">
                  Prophet uses trend + seasonality decomposition (direct forecasting)
                </p>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}
