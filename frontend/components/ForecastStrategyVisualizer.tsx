'use client';

import { useState, useEffect, useMemo } from 'react';
import { ChevronLeft, ChevronRight, Play, Pause, AlertCircle, CheckCircle2, Zap, RotateCcw, Info, X, Plus, TrendingUp, BarChart3 } from 'lucide-react';

interface ForecastStrategyVisualizerProps {
  lags: number[];
  horizon: number;
  onLagsChange: (lags: number[]) => void;
  sampleData?: number[];
  frequency?: string;
}

export default function ForecastStrategyVisualizer({
  lags,
  horizon,
  onLagsChange,
  sampleData = [],
  frequency = 'D'
}: ForecastStrategyVisualizerProps) {
  const [currentStep, setCurrentStep] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showTooltip, setShowTooltip] = useState<number | null>(null);
  const [newLagInput, setNewLagInput] = useState('');

  // Sort lags
  const sortedLags = useMemo(() => [...lags].sort((a, b) => a - b), [lags]);
  const minLag = sortedLags.length > 0 ? sortedLags[0] : 1;
  const maxLag = sortedLags.length > 0 ? sortedLags[sortedLags.length - 1] : 1;

  // Determine forecasting mode for ML models
  const mlForecastMode = useMemo(() => {
    if (lags.length === 0) return 'none' as const;
    if (horizon <= minLag) return 'direct' as const;
    return 'recursive' as const;
  }, [horizon, minLag, lags.length]);

  // History bars count - same for both views, based on max lag
  const historyBarsCount = useMemo(() => Math.max(4, Math.min(8, maxLag + 2)), [maxLag]);

  // Generate sample data
  const displayData = useMemo(() => {
    if (sampleData.length >= historyBarsCount) {
      return sampleData.slice(-historyBarsCount);
    }
    const generated: number[] = [];
    for (let i = 0; i < historyBarsCount; i++) {
      const trend = 100 + i * 2;
      const seasonal = 10 * Math.sin(i * Math.PI / 3.5);
      generated.push(Math.round((trend + seasonal) * 10) / 10);
    }
    return generated;
  }, [sampleData, historyBarsCount]);

  // Auto-play animation
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= horizon) {
          setIsPlaying(false);
          return 1;
        }
        return prev + 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [isPlaying, horizon]);

  // Reset step when horizon changes
  useEffect(() => {
    setCurrentStep(1);
    setIsPlaying(false);
  }, [horizon]);

  // Add a new lag
  const addLag = (value: string) => {
    const num = parseInt(value.trim());
    if (!isNaN(num) && num > 0 && !lags.includes(num)) {
      onLagsChange([...lags, num].sort((a, b) => a - b));
      setNewLagInput('');
    }
  };

  // Remove a lag
  const removeLag = (lag: number) => {
    onLagsChange(lags.filter(l => l !== lag));
  };

  // Get frequency label
  const freqLabel = { 'D': 'day', 'H': 'hour', 'min': 'minute', 'W': 'week', 'M': 'month' }[frequency] || 'step';

  // Calculate which values are used for current prediction step
  const lagsUsedForCurrentStep = useMemo(() => {
    return sortedLags.map(lag => {
      const sourceStep = currentStep - lag;
      return {
        lag,
        sourceStep,
        isFromPrediction: sourceStep > 0, // If positive, it's from a prediction
        historyIndex: sourceStep <= 0 ? displayData.length + sourceStep - 1 : -1 // Index in history array
      };
    });
  }, [displayData.length, currentStep, sortedLags]);

  // Max predictions to show before truncating
  const maxVisiblePredictions = 10;

  return (
    <div className="bg-gradient-to-br from-slate-900/80 to-slate-800/50 border border-white/10 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-white/10 bg-black/20">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-white flex items-center gap-2">
            <Zap size={14} className="text-amber-500" />
            Forecast Strategy
          </h4>
          <div className="text-xs text-slate-400">
            Horizon: <span className="text-amber-400 font-mono">{horizon}</span> {freqLabel}{horizon > 1 ? 's' : ''}
          </div>
        </div>
      </div>

      {/* Two columns: ML models (with lags) and Statistical models */}
      <div className="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-white/10">
        
        {/* Left: ML Models (with features/lags) */}
        <div className="p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 rounded bg-amber-500/20 flex items-center justify-center">
              <BarChart3 size={12} className="text-amber-400" />
            </div>
            <div>
              <h5 className="text-xs font-semibold text-white">ML Models</h5>
              <p className="text-[10px] text-slate-500">Lag, Linear Regression, XGBoost</p>
            </div>
            <div className={`ml-auto px-2 py-0.5 rounded-full text-[10px] font-semibold flex items-center gap-1 ${
              mlForecastMode === 'direct' 
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
                : mlForecastMode === 'recursive'
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                  : 'bg-slate-500/20 text-slate-400 border border-slate-500/30'
            }`}>
              {mlForecastMode === 'direct' ? (
                <><CheckCircle2 size={10} /> Direct</>
              ) : mlForecastMode === 'recursive' ? (
                <><RotateCcw size={10} /> Recursive</>
              ) : (
                <>No lags</>
              )}
            </div>
          </div>

          {/* Lag Input */}
          <div className="mb-3">
            <label className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5 block">Target Lags (used as features)</label>
            <div className="flex flex-wrap gap-1.5 p-2 bg-black/30 rounded-lg border border-white/10 min-h-[38px]">
              {sortedLags.map((lag) => (
                <span 
                  key={lag} 
                  className={`inline-flex items-center gap-1 px-2 py-1 text-xs font-mono rounded transition-all ${
                    lag === minLag 
                      ? 'bg-amber-500 text-black font-semibold' 
                      : 'bg-white/10 text-slate-300'
                  }`}
                >
                  {lag}
                  {lag === minLag && <span className="text-[9px] opacity-70">(min)</span>}
                  <button
                    onClick={() => removeLag(lag)}
                    className="ml-0.5 hover:text-red-400 transition-colors"
                  >
                    <X size={12} />
                  </button>
                </span>
              ))}
              <input
                type="text"
                inputMode="numeric"
                value={newLagInput}
                onChange={(e) => setNewLagInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && newLagInput) {
                    addLag(newLagInput);
                  }
                }}
                onBlur={() => newLagInput && addLag(newLagInput)}
                placeholder={lags.length === 0 ? "Add lag (e.g., 1, 7)" : "+"}
                className="flex-1 min-w-[40px] bg-transparent outline-none text-xs text-white placeholder:text-slate-600"
              />
            </div>
          </div>

          {/* Mode explanation */}
          <div className={`p-2 rounded-lg text-[10px] mb-3 ${
            mlForecastMode === 'direct' 
              ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-300'
              : mlForecastMode === 'recursive'
                ? 'bg-amber-500/10 border border-amber-500/20 text-amber-300'
                : 'bg-slate-500/10 border border-slate-500/20 text-slate-400'
          }`}>
            {mlForecastMode === 'direct' ? (
              <span>✓ Horizon ({horizon}) ≤ min lag ({minLag}) → Uses only real historical values</span>
            ) : mlForecastMode === 'recursive' ? (
              <span>⚠ Horizon ({horizon}) &gt; min lag ({minLag}) → Uses predicted values as features (errors propagate)</span>
            ) : (
              <span>Add lags to enable ML models</span>
            )}
          </div>

          {/* Mini Timeline Visualization */}
          {lags.length > 0 && (
            <div className="bg-black/20 rounded-lg p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] text-slate-500">Step-by-step visualization</span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
                    disabled={currentStep <= 1}
                    className="p-0.5 text-slate-400 hover:text-white disabled:opacity-30 transition-all"
                  >
                    <ChevronLeft size={14} />
                  </button>
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className={`p-1 rounded transition-all ${isPlaying ? 'bg-amber-500 text-black' : 'bg-white/10 text-white hover:bg-white/20'}`}
                  >
                    {isPlaying ? <Pause size={12} /> : <Play size={12} />}
                  </button>
                  <button
                    onClick={() => setCurrentStep(Math.min(horizon, currentStep + 1))}
                    disabled={currentStep >= horizon}
                    className="p-0.5 text-slate-400 hover:text-white disabled:opacity-30 transition-all"
                  >
                    <ChevronRight size={14} />
                  </button>
                  <span className="text-[10px] text-slate-500 ml-1">{currentStep}/{horizon}</span>
                </div>
              </div>

              {/* Timeline bars - harmonized */}
              <div className="flex items-end gap-0.5 overflow-x-auto pb-3 scrollbar-thin scrollbar-thumb-white/10">
                {/* History bars */}
                {displayData.map((val, idx) => {
                  // t goes from -(N-1) to 0, so last element is t=0 (present)
                  const t = idx - displayData.length + 1;
                  const isUsed = lagsUsedForCurrentStep.some(l => l.historyIndex === idx);
                  
                  return (
                    <div key={`h-${idx}`} className="flex flex-col items-center flex-shrink-0">
                      <div
                        className={`w-5 sm:w-6 rounded-t transition-all ${
                          isUsed ? 'bg-emerald-400 ring-1 ring-emerald-500' : 'bg-slate-600/50'
                        }`}
                        style={{ height: `${Math.max(14, (val / Math.max(...displayData)) * 36)}px` }}
                      />
                      <span className="text-[7px] sm:text-[8px] text-slate-600 mt-0.5">
                        {t === 0 ? 't₀' : `t${t}`}
                      </span>
                    </div>
                  );
                })}
                
                {/* Separator at t=0 */}
                <div className="w-px h-10 bg-gradient-to-b from-amber-500 via-amber-500/50 to-transparent mx-0.5 flex-shrink-0" />
                
                {/* Predictions - truncate if too many */}
                {Array.from({ length: Math.min(horizon, maxVisiblePredictions) }, (_, h) => {
                  const step = h + 1;
                  const isCurrent = step === currentStep;
                  const isUsed = lagsUsedForCurrentStep.some(l => l.isFromPrediction && l.sourceStep === step);
                  const isPast = step < currentStep;
                  
                  return (
                    <div key={`p-${h}`} className="flex flex-col items-center flex-shrink-0">
                      <div
                        className={`w-5 sm:w-6 rounded-t transition-all ${
                          isCurrent
                            ? 'bg-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.5)]'
                            : isUsed
                              ? 'bg-amber-400/80 ring-1 ring-amber-500'
                              : isPast
                                ? 'bg-amber-500/30'
                                : 'bg-amber-500/20 border border-amber-500/30 border-dashed'
                        }`}
                        style={{ height: '28px' }}
                      />
                      <span className={`text-[7px] sm:text-[8px] mt-0.5 ${
                        isCurrent ? 'text-amber-400 font-bold' : 'text-slate-500'
                      }`}>
                        t+{step}
                      </span>
                    </div>
                  );
                })}
                {horizon > maxVisiblePredictions && (
                  <div className="flex flex-col items-center flex-shrink-0 px-1">
                    <div className="text-amber-400 text-[10px]">...</div>
                    <span className="text-[7px] text-slate-500 mt-0.5">+{horizon - maxVisiblePredictions}</span>
                  </div>
                )}
              </div>

              {/* Formula */}
              <div className="mt-2 p-2 bg-black/30 rounded text-[9px] sm:text-[10px] font-mono overflow-x-auto">
                <span className="text-amber-400">ŷ<sub>t+{currentStep}</sub></span>
                <span className="text-slate-500"> = f(</span>
                {lagsUsedForCurrentStep.map((l, idx) => {
                  const refStep = currentStep - l.lag;
                  const label = refStep === 0 ? 't₀' : refStep > 0 ? `t+${refStep}` : `t${refStep}`;
                  return (
                    <span key={l.lag}>
                      {idx > 0 && <span className="text-slate-600">, </span>}
                      <span className={l.isFromPrediction ? 'text-amber-400' : 'text-emerald-400'}>
                        {l.isFromPrediction ? 'ŷ' : 'y'}<sub>{label}</sub>
                      </span>
                    </span>
                  );
                })}
                <span className="text-slate-500">)</span>
              </div>
            </div>
          )}
        </div>

        {/* Right: Statistical Models (no lags) */}
        <div className="p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 rounded bg-blue-500/20 flex items-center justify-center">
              <TrendingUp size={12} className="text-blue-400" />
            </div>
            <div>
              <h5 className="text-xs font-semibold text-white">Statistical Models</h5>
              <p className="text-[10px] text-slate-500">ARIMA, Prophet</p>
            </div>
            <div className="ml-auto px-2 py-0.5 rounded-full text-[10px] font-semibold flex items-center gap-1 bg-blue-500/20 text-blue-400 border border-blue-500/30">
              <CheckCircle2 size={10} /> Direct
            </div>
          </div>

          <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-2 sm:p-3 text-[10px] sm:text-[11px] text-blue-300 mb-3">
            <p className="flex items-start gap-2">
              <Info size={14} className="flex-shrink-0 mt-0.5 hidden sm:block" />
              <span>
                Predict all <strong>{horizon}</strong> steps directly using internal dynamics.
              </span>
            </p>
          </div>

          {/* Visual representation for statistical models - HARMONIZED with ML */}
          <div className="bg-black/20 rounded-lg p-2 sm:p-3">
            <div className="flex items-end gap-0.5 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-white/10">
              {/* History - same displayData as ML for consistency */}
              {displayData.map((val, idx) => {
                // t goes from -(N-1) to 0, so last element is t=0 (present)
                const t = idx - displayData.length + 1;
                return (
                  <div key={`sh-${idx}`} className="flex flex-col items-center flex-shrink-0">
                    <div
                      className="w-5 sm:w-6 rounded-t bg-slate-600/50"
                      style={{ height: `${Math.max(14, (val / Math.max(...displayData)) * 36)}px` }}
                    />
                    <span className="text-[7px] sm:text-[8px] text-slate-600 mt-0.5">
                      {t === 0 ? 't₀' : `t${t}`}
                    </span>
                  </div>
                );
              })}
              
              <div className="w-px h-10 bg-gradient-to-b from-blue-500 via-blue-500/50 to-transparent mx-0.5 flex-shrink-0" />
              
              {/* All predictions - truncate same as ML */}
              {Array.from({ length: Math.min(horizon, maxVisiblePredictions) }, (_, h) => (
                <div key={`sp-${h}`} className="flex flex-col items-center flex-shrink-0">
                  <div
                    className="w-5 sm:w-6 rounded-t bg-blue-500/40 border border-blue-500/50"
                    style={{ height: '28px' }}
                  />
                  <span className="text-[7px] sm:text-[8px] text-blue-400 mt-0.5">t+{h + 1}</span>
                </div>
              ))}
              {horizon > maxVisiblePredictions && (
                <div className="flex flex-col items-center flex-shrink-0 px-1">
                  <div className="text-blue-400 text-[10px]">...</div>
                  <span className="text-[7px] text-slate-500 mt-0.5">+{horizon - maxVisiblePredictions}</span>
                </div>
              )}
            </div>
            
            <div className="mt-2 p-2 bg-black/30 rounded text-[9px] sm:text-[10px] hidden sm:block">
              <span className="text-blue-400">model.forecast</span>
              <span className="text-slate-400">(h={horizon})</span>
            </div>
          </div>
        </div>
      </div>

      {/* Validation Explanation */}
      <div className="px-3 sm:px-4 py-2.5 border-t border-white/10 bg-gradient-to-r from-slate-900/50 to-slate-800/30">
        <div className="flex items-start gap-2">
          <Info size={12} className="text-slate-500 flex-shrink-0 mt-0.5" />
          <div className="text-[9px] sm:text-[10px] text-slate-500">
            <span className="text-slate-400 font-medium">Validation:</span>{' '}
            Predictions are made in blocks of <span className="text-amber-400 font-mono">{horizon}</span> steps.
            Within each block, ML models use recursive predictions; between blocks, they reset to actual values.
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="px-3 sm:px-4 py-2 border-t border-white/10 bg-black/20 flex flex-wrap items-center justify-center gap-3 sm:gap-4 text-[9px] sm:text-[10px]">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded bg-slate-600/50" />
          <span className="text-slate-500">History</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded bg-emerald-400" />
          <span className="text-slate-500">Used (real)</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded bg-amber-400" />
          <span className="text-slate-500">Used (pred)</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded bg-amber-500 shadow-[0_0_5px_rgba(245,158,11,0.5)]" />
          <span className="text-slate-500">Target</span>
        </span>
      </div>
    </div>
  );
}
