'use client';

import { useState, useEffect, useMemo } from 'react';
import { ChevronLeft, ChevronRight, Play, Pause, AlertCircle, CheckCircle2, ArrowRight, Zap, RotateCcw, Info } from 'lucide-react';

interface ForecastHorizonVisualizerProps {
  lags: number[];
  horizon: number;
  onHorizonChange: (horizon: number) => void;
  sampleData?: number[]; // Optional sample of real data to display
  frequency?: string; // 'D', 'H', 'min', etc.
  onLagsChange?: (lags: number[]) => void;
}

interface TimeStep {
  index: number;
  value: number | null;
  isPrediction: boolean;
  isCurrentTarget: boolean;
  usedByLags: number[]; // Which lags use this value
  recursiveStep?: number; // Which step of recursion (1 = direct, 2+ = recursive)
}

export default function ForecastHorizonVisualizer({
  lags,
  horizon,
  onHorizonChange,
  sampleData = [],
  frequency = 'D',
  onLagsChange
}: ForecastHorizonVisualizerProps) {
  const [currentStep, setCurrentStep] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showTooltip, setShowTooltip] = useState<number | null>(null);

  // Sort lags to find the minimum lag
  const sortedLags = useMemo(() => [...lags].sort((a, b) => a - b), [lags]);
  const minLag = sortedLags.length > 0 ? sortedLags[0] : 1;
  const maxLag = sortedLags.length > 0 ? sortedLags[sortedLags.length - 1] : 1;

  // Determine forecasting mode
  const forecastMode = useMemo(() => {
    if (horizon <= minLag) {
      return 'direct' as const;
    }
    return 'recursive' as const;
  }, [horizon, minLag]);

  // Calculate number of recursive steps needed
  const recursiveSteps = useMemo(() => {
    if (forecastMode === 'direct') return 1;
    return Math.ceil(horizon / minLag);
  }, [forecastMode, horizon, minLag]);

  // Generate sample data if not provided
  const displayData = useMemo(() => {
    if (sampleData.length >= 10) {
      return sampleData.slice(-10);
    }
    // Generate synthetic data with trend + seasonality
    const generated: number[] = [];
    for (let i = 0; i < 10; i++) {
      const trend = 100 + i * 2;
      const seasonal = 10 * Math.sin(i * Math.PI / 3.5);
      const noise = (Math.random() - 0.5) * 5;
      generated.push(Math.round((trend + seasonal + noise) * 10) / 10);
    }
    return generated;
  }, [sampleData]);

  // Build timeline with predictions
  const timeline = useMemo(() => {
    const steps: TimeStep[] = [];
    const historyLength = displayData.length;
    const totalLength = historyLength + horizon;

    // Historical data
    for (let i = 0; i < historyLength; i++) {
      steps.push({
        index: i - historyLength, // Negative indices for history (t-n, t-n+1, ...)
        value: displayData[i],
        isPrediction: false,
        isCurrentTarget: false,
        usedByLags: []
      });
    }

    // Predictions
    for (let h = 1; h <= horizon; h++) {
      const recursiveStep = Math.ceil(h / minLag);
      steps.push({
        index: h - 1, // 0-indexed future (t, t+1, t+2, ...)
        value: null,
        isPrediction: true,
        isCurrentTarget: h === currentStep,
        usedByLags: [],
        recursiveStep
      });
    }

    return steps;
  }, [displayData, horizon, currentStep, minLag]);

  // Calculate which historical values are used for current prediction step
  const lagsUsedForCurrentStep = useMemo(() => {
    const historyLength = displayData.length;
    const usedIndices: { index: number; lag: number; isFromPrediction: boolean }[] = [];

    // For each lag, determine which value it references
    for (const lag of sortedLags) {
      const sourceIndex = historyLength - 1 + currentStep - lag; // Index in timeline
      const relativeIndex = currentStep - lag; // Relative to t=0 (first prediction)
      
      if (relativeIndex <= 0) {
        // Using actual historical value
        usedIndices.push({
          index: sourceIndex,
          lag,
          isFromPrediction: false
        });
      } else {
        // Using a predicted value (recursive)
        usedIndices.push({
          index: sourceIndex,
          lag,
          isFromPrediction: true
        });
      }
    }

    return usedIndices;
  }, [displayData.length, currentStep, sortedLags]);

  // Auto-play animation
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= horizon) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1200);

    return () => clearInterval(interval);
  }, [isPlaying, horizon]);

  // Reset current step when horizon changes
  useEffect(() => {
    setCurrentStep(1);
    setIsPlaying(false);
  }, [horizon]);

  // Get frequency label
  const getFrequencyLabel = (freq: string) => {
    const labels: Record<string, string> = {
      'D': 'day',
      'H': 'hour',
      'min': 'minute',
      'W': 'week',
      'M': 'month'
    };
    return labels[freq] || 'step';
  };

  const freqLabel = getFrequencyLabel(frequency);

  return (
    <div className="bg-gradient-to-br from-slate-900/80 to-slate-800/50 border border-white/10 rounded-xl p-4 sm:p-5">
      {/* Header with Mode Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-3">
          <h4 className="text-sm font-semibold text-white flex items-center gap-2">
            <Zap size={14} className="text-amber-500" />
            Forecast Strategy
          </h4>
          <div className={`px-2.5 py-1 rounded-full text-[10px] font-semibold flex items-center gap-1.5 ${
            forecastMode === 'direct' 
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
              : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
          }`}>
            {forecastMode === 'direct' ? (
              <>
                <CheckCircle2 size={10} />
                Direct
              </>
            ) : (
              <>
                <RotateCcw size={10} />
                Recursive ({recursiveSteps} steps)
              </>
            )}
          </div>
        </div>

        {/* Horizon Selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Horizon:</span>
          <div className="flex items-center bg-black/30 rounded-lg border border-white/10 overflow-hidden">
            <button
              onClick={() => onHorizonChange(Math.max(1, horizon - 1))}
              className="px-2 py-1.5 text-slate-400 hover:text-white hover:bg-white/5 transition-all"
            >
              <ChevronLeft size={14} />
            </button>
            <input
              type="number"
              min={1}
              max={30}
              value={horizon}
              onChange={(e) => onHorizonChange(Math.max(1, Math.min(30, parseInt(e.target.value) || 1)))}
              className="w-12 bg-transparent text-center text-sm font-mono text-white border-x border-white/10 py-1 outline-none"
            />
            <button
              onClick={() => onHorizonChange(Math.min(30, horizon + 1))}
              className="px-2 py-1.5 text-slate-400 hover:text-white hover:bg-white/5 transition-all"
            >
              <ChevronRight size={14} />
            </button>
          </div>
          <span className="text-xs text-slate-500">{freqLabel}{horizon > 1 ? 's' : ''}</span>
        </div>
      </div>

      {/* Mode Explanation */}
      <div className={`mb-4 p-3 rounded-lg text-xs ${
        forecastMode === 'direct' 
          ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-300'
          : 'bg-amber-500/10 border border-amber-500/20 text-amber-300'
      }`}>
        {forecastMode === 'direct' ? (
          <p className="flex items-start gap-2">
            <Info size={14} className="flex-shrink-0 mt-0.5" />
            <span>
              <strong>Direct forecasting:</strong> Horizon ({horizon}) ≤ min lag ({minLag}). 
              All predictions use only actual historical values — no recursive dependency.
            </span>
          </p>
        ) : (
          <p className="flex items-start gap-2">
            <Info size={14} className="flex-shrink-0 mt-0.5" />
            <span>
              <strong>Recursive forecasting:</strong> Horizon ({horizon}) &gt; min lag ({minLag}). 
              Later predictions depend on earlier predictions. Errors can accumulate.
            </span>
          </p>
        )}
      </div>

      {/* Interactive Timeline */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-slate-500">Timeline visualization</span>
          {forecastMode === 'recursive' && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
                disabled={currentStep <= 1}
                className="p-1 text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-all"
              >
                <ChevronLeft size={16} />
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`p-1.5 rounded transition-all ${
                  isPlaying 
                    ? 'bg-amber-500 text-black' 
                    : 'bg-white/10 text-white hover:bg-white/20'
                }`}
              >
                {isPlaying ? <Pause size={14} /> : <Play size={14} />}
              </button>
              <button
                onClick={() => setCurrentStep(Math.min(horizon, currentStep + 1))}
                disabled={currentStep >= horizon}
                className="p-1 text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-all"
              >
                <ChevronRight size={16} />
              </button>
              <span className="text-xs text-slate-400 ml-1">
                Step {currentStep}/{horizon}
              </span>
            </div>
          )}
        </div>

        {/* Timeline Bar */}
        <div className="relative bg-black/30 rounded-lg p-3 overflow-x-auto custom-scrollbar">
          <div className="flex items-end gap-1 min-w-max pb-6">
            {timeline.map((step, idx) => {
              const isUsedByLag = lagsUsedForCurrentStep.some(l => l.index === idx);
              const lagInfo = lagsUsedForCurrentStep.find(l => l.index === idx);
              const historyLength = displayData.length;
              const isCurrentTarget = step.isPrediction && step.index + 1 === currentStep;

              return (
                <div 
                  key={idx} 
                  className="relative flex flex-col items-center"
                  onMouseEnter={() => setShowTooltip(idx)}
                  onMouseLeave={() => setShowTooltip(null)}
                >
                  {/* Value bar */}
                  <div className="relative">
                    {/* Connection line for used lags */}
                    {isUsedByLag && lagInfo && !isCurrentTarget && (
                      <div className="absolute -top-1 left-1/2 transform -translate-x-1/2">
                        <div className={`w-px h-4 ${lagInfo.isFromPrediction ? 'bg-amber-500' : 'bg-emerald-500'}`} />
                        <div className={`absolute -top-1 left-1/2 transform -translate-x-1/2 w-2 h-2 rounded-full ${
                          lagInfo.isFromPrediction ? 'bg-amber-500' : 'bg-emerald-500'
                        }`} />
                      </div>
                    )}
                    
                    <div
                      className={`w-8 rounded-t transition-all duration-300 ${
                        isCurrentTarget
                          ? 'bg-amber-500 shadow-[0_0_15px_rgba(245,158,11,0.5)]'
                          : step.isPrediction
                            ? step.recursiveStep === 1
                              ? 'bg-emerald-500/40 border border-emerald-500/50'
                              : 'bg-amber-500/30 border border-amber-500/40 border-dashed'
                            : isUsedByLag
                              ? lagInfo?.isFromPrediction
                                ? 'bg-amber-400 ring-2 ring-amber-500/50'
                                : 'bg-emerald-400 ring-2 ring-emerald-500/50'
                              : 'bg-slate-600/50'
                      }`}
                      style={{
                        height: step.value !== null 
                          ? `${Math.max(20, Math.min(60, (step.value / Math.max(...displayData)) * 50))}px`
                          : '40px'
                      }}
                    />

                    {/* Tooltip */}
                    {showTooltip === idx && (
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 z-10">
                        <div className="bg-slate-900 border border-white/20 rounded-lg px-2 py-1 text-xs whitespace-nowrap shadow-xl">
                          {step.isPrediction ? (
                            <span className="text-amber-400">
                              t+{step.index + 1} 
                              {step.recursiveStep && step.recursiveStep > 1 && (
                                <span className="text-amber-500/70"> (recursive)</span>
                              )}
                            </span>
                          ) : (
                            <span className="text-slate-300">
                              t{step.index} = {step.value?.toFixed(1)}
                            </span>
                          )}
                          {lagInfo && (
                            <span className={`ml-1 ${lagInfo.isFromPrediction ? 'text-amber-400' : 'text-emerald-400'}`}>
                              ← lag {lagInfo.lag}
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Time label */}
                  <span className={`text-[9px] mt-1 font-mono ${
                    step.isPrediction 
                      ? isCurrentTarget 
                        ? 'text-amber-400 font-bold' 
                        : 'text-slate-500'
                      : 'text-slate-600'
                  }`}>
                    {step.isPrediction ? `t+${step.index + 1}` : `t${step.index}`}
                  </span>

                  {/* Separator between history and predictions */}
                  {idx === historyLength - 1 && (
                    <div className="absolute right-0 top-0 bottom-0 w-px bg-gradient-to-b from-amber-500 via-amber-500/50 to-transparent" />
                  )}
                </div>
              );
            })}
          </div>

          {/* Legend */}
          <div className="absolute bottom-0 left-0 right-0 flex items-center justify-center gap-4 text-[10px] pt-2 border-t border-white/5">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-slate-600/50" />
              <span className="text-slate-500">History</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-emerald-500/40 border border-emerald-500/50" />
              <span className="text-slate-500">Direct</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-amber-500/30 border border-amber-500/40 border-dashed" />
              <span className="text-slate-500">Recursive</span>
            </span>
            {forecastMode === 'recursive' && (
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]" />
                <span className="text-slate-500">Current target</span>
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Lags Display */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-slate-400">Using lags:</span>
        {sortedLags.map((lag, idx) => {
          const isMinLag = lag === minLag;
          return (
            <span
              key={lag}
              className={`px-2 py-1 rounded text-xs font-mono transition-all ${
                isMinLag
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                  : 'bg-white/5 text-slate-300 border border-white/10'
              }`}
            >
              t-{lag}
              {isMinLag && (
                <span className="ml-1 text-[9px] opacity-70">(min)</span>
              )}
            </span>
          );
        })}
        {lags.length === 0 && (
          <span className="text-xs text-red-400/80 flex items-center gap-1">
            <AlertCircle size={12} />
            No lags selected
          </span>
        )}
      </div>

      {/* Prediction Formula for Current Step */}
      {forecastMode === 'recursive' && (
        <div className="mt-4 p-3 bg-black/20 rounded-lg border border-white/5">
          <p className="text-xs text-slate-500 mb-2">Prediction formula for step {currentStep}:</p>
          <div className="font-mono text-sm flex flex-wrap items-center gap-1">
            <span className="text-amber-400">ŷ<sub>t+{currentStep}</sub></span>
            <span className="text-slate-500">=</span>
            <span className="text-slate-400">f(</span>
            {lagsUsedForCurrentStep.map((lagInfo, idx) => (
              <span key={lagInfo.lag} className="flex items-center gap-1">
                {idx > 0 && <span className="text-slate-600">,</span>}
                <span className={lagInfo.isFromPrediction ? 'text-amber-400' : 'text-emerald-400'}>
                  {lagInfo.isFromPrediction ? 'ŷ' : 'y'}<sub>t+{currentStep - lagInfo.lag}</sub>
                </span>
              </span>
            ))}
            <span className="text-slate-400">)</span>
          </div>
          {lagsUsedForCurrentStep.some(l => l.isFromPrediction) && (
            <p className="text-[10px] text-amber-400/70 mt-2 flex items-center gap-1">
              <AlertCircle size={10} />
              Uses {lagsUsedForCurrentStep.filter(l => l.isFromPrediction).length} predicted value(s) — errors may propagate
            </p>
          )}
        </div>
      )}
    </div>
  );
}
