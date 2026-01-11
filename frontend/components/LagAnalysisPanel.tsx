'use client';

import { useState } from 'react';
import { BarChart3, TrendingUp, AlertTriangle, Info, CheckCircle, Zap } from 'lucide-react';

interface LagAnalysis {
  suggested_lags: number[];
  acf: number[];
  pacf: number[];
  confidence_interval: number;
  significant_lags: { lag: number; pacf: number; significant: boolean }[];
  seasonality: {
    detected: boolean;
    period?: number;
    period_label?: string;
    strength?: number;
  };
  n_observations: number;
}

interface DataAlert {
  type: 'warning' | 'info' | 'error';
  category: string;
  message: string;
  details?: Record<string, any>;
}

interface LagAnalysisPanelProps {
  lagAnalysis: LagAnalysis | null;
  alerts: DataAlert[] | null;
  onApplyLags: (lags: number[]) => void;
  currentLags: number[];
}

export function LagAnalysisPanel({ lagAnalysis, alerts, onApplyLags, currentLags }: LagAnalysisPanelProps) {
  const [expanded, setExpanded] = useState(true);

  if (!lagAnalysis && (!alerts || alerts.length === 0)) {
    return null;
  }

  return (
    <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
      >
        <h4 className="text-sm font-semibold text-white flex items-center gap-2">
          <Zap size={16} className="text-amber-500" />
          Data Analysis & Lag Suggestions
        </h4>
        <span className={`text-slate-400 transition-transform ${expanded ? 'rotate-180' : ''}`}>
          ▼
        </span>
      </button>

      {expanded && (
        <div className="p-4 pt-0 space-y-4">
          {/* Alerts Section */}
          {alerts && alerts.length > 0 && (
            <div className="space-y-2">
              {alerts.map((alert, idx) => (
                <div
                  key={idx}
                  className={`flex items-start gap-2 p-3 rounded-lg border ${
                    alert.type === 'warning'
                      ? 'bg-orange-500/10 border-orange-500/30 text-orange-300'
                      : alert.type === 'error'
                      ? 'bg-red-500/10 border-red-500/30 text-red-300'
                      : 'bg-blue-500/10 border-blue-500/30 text-blue-300'
                  }`}
                >
                  {alert.type === 'warning' ? (
                    <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
                  ) : alert.type === 'error' ? (
                    <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
                  ) : (
                    <Info size={16} className="mt-0.5 flex-shrink-0" />
                  )}
                  <div>
                    <p className="text-sm">{alert.message}</p>
                    {alert.details?.recommendation && (
                      <p className="text-xs opacity-75 mt-1">{alert.details.recommendation}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Lag Suggestions */}
          {lagAnalysis && (
            <>
              {/* Suggested Lags Pills */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400">Suggested Lags (click to apply)</span>
                  <button
                    onClick={() => onApplyLags(lagAnalysis.suggested_lags)}
                    className="text-xs px-2 py-1 bg-amber-500/20 text-amber-400 rounded hover:bg-amber-500/30 transition-colors"
                  >
                    Apply All
                  </button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {lagAnalysis.suggested_lags.map((lag) => {
                    const isActive = currentLags.includes(lag);
                    const sigLag = lagAnalysis.significant_lags.find((s) => s.lag === lag);
                    
                    return (
                      <button
                        key={lag}
                        onClick={() => {
                          if (isActive) {
                            onApplyLags(currentLags.filter((l) => l !== lag));
                          } else {
                            onApplyLags([...currentLags, lag].sort((a, b) => a - b));
                          }
                        }}
                        className={`group relative px-3 py-1.5 rounded-lg border transition-all ${
                          isActive
                            ? 'bg-amber-500 text-black border-amber-500 font-bold'
                            : 'bg-white/5 text-slate-300 border-white/10 hover:border-amber-500/50'
                        }`}
                      >
                        <span className="text-sm">Lag {lag}</span>
                        {sigLag && (
                          <span
                            className={`ml-1.5 text-[10px] ${
                              isActive ? 'text-black/60' : 'text-slate-500'
                            }`}
                          >
                            ({sigLag.pacf > 0 ? '+' : ''}{sigLag.pacf.toFixed(2)})
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* PACF Chart */}
              <div className="space-y-2">
                <span className="text-xs text-slate-400">Partial Autocorrelation (PACF)</span>
                <div className="bg-black/20 rounded-lg p-3">
                  {/* Chart container with center line */}
                  <div className="relative h-32">
                    {/* Zero line */}
                    <div className="absolute left-0 right-0 top-1/2 h-px bg-slate-500/50" />
                    
                    {/* Confidence bounds */}
                    <div 
                      className="absolute left-0 right-0 border-t border-dashed border-blue-400/40"
                      style={{ top: `${50 - lagAnalysis.confidence_interval * 50}%` }}
                    />
                    <div 
                      className="absolute left-0 right-0 border-t border-dashed border-blue-400/40"
                      style={{ top: `${50 + lagAnalysis.confidence_interval * 50}%` }}
                    />
                    
                    {/* Bars */}
                    <div className="absolute inset-0 flex items-center gap-[2px] px-1">
                      {lagAnalysis.pacf.slice(1).map((value, idx) => {
                        const lag = idx + 1;
                        const isSignificant = Math.abs(value) > lagAnalysis.confidence_interval;
                        const isSuggested = lagAnalysis.suggested_lags.includes(lag);
                        // Clamp value to [-1, 1] for display
                        const clampedValue = Math.max(-1, Math.min(1, value));
                        const barHeight = Math.abs(clampedValue) * 50; // 50% max height (half of container)

                        return (
                          <div
                            key={lag}
                            className="flex-1 h-full relative group flex items-center"
                          >
                            {/* Bar - positioned from center */}
                            <div
                              className={`absolute left-0 right-0 rounded-sm transition-all ${
                                isSuggested
                                  ? 'bg-amber-500'
                                  : isSignificant
                                  ? value >= 0 ? 'bg-emerald-500/70' : 'bg-red-500/70'
                                  : 'bg-slate-500/50'
                              }`}
                              style={{
                                height: `${barHeight}%`,
                                top: value >= 0 ? `${50 - barHeight}%` : '50%',
                              }}
                            />

                            {/* Tooltip */}
                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-10 pointer-events-none">
                              <div className="bg-slate-800 text-white text-[10px] px-2 py-1 rounded shadow-lg whitespace-nowrap">
                                Lag {lag}: {value.toFixed(3)}
                                {isSignificant && ' ★'}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* X-axis labels */}
                  <div className="flex justify-between mt-2 text-[9px] text-slate-500">
                    <span>1</span>
                    <span>{Math.floor(lagAnalysis.pacf.length / 2)}</span>
                    <span>{lagAnalysis.pacf.length - 1}</span>
                  </div>
                </div>
              </div>

              {/* Seasonality Info */}
              {lagAnalysis.seasonality.detected && (
                <div className="flex items-center gap-2 p-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                  <TrendingUp size={14} className="text-emerald-400" />
                  <span className="text-xs text-emerald-300">
                    {lagAnalysis.seasonality.period_label} seasonality detected
                    {lagAnalysis.seasonality.period && ` (period: ${lagAnalysis.seasonality.period})`}
                  </span>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
