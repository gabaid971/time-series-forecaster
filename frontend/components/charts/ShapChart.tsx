'use client';

import { useState } from 'react';
import { X } from 'lucide-react';
import { ShapValue } from '../../types/forecasting';

interface ShapChartProps {
  shapValues: ShapValue[];
  featureName: string;
  onClose: () => void;
}

/**
 * SHAP Chart Component for visualizing SHAP values by temporal feature.
 */
export function ShapChart({ shapValues, featureName, onClose }: ShapChartProps) {
  const [useNormalized, setUseNormalized] = useState(true);

  const getLabel = (value: number): string => {
    switch (featureName) {
      case 'day_of_week':
        return ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][value] || String(value);
      case 'month':
        return ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][value - 1] || String(value);
      case 'hour_of_day':
        return `${value}h`;
      case 'minute_of_day':
        const h = Math.floor(value / 60);
        const m = value % 60;
        return `${h}:${String(m).padStart(2, '0')}`;
      default:
        return String(value);
    }
  };

  const getFeatureTitle = (): string => {
    switch (featureName) {
      case 'day_of_week': return 'Day of week';
      case 'month': return 'Month';
      case 'hour_of_day': return 'Hour of day';
      case 'minute_of_day': return 'Minute of day';
      default: return featureName;
    }
  };

  const values = shapValues.map(v => useNormalized ? v.shap_norm : v.shap);
  const maxAbsValue = Math.max(...values.map(v => Math.abs(v)), 0.001);

  return (
    <div
      className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-slate-900 border border-white/10 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-hidden shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-4 border-b border-white/10 flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-white">SHAP Values: {getFeatureTitle()}</h3>
            <p className="text-xs text-slate-400 mt-1">Impact of each value on the prediction</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-black/30 rounded-lg p-1">
              <button
                onClick={() => setUseNormalized(true)}
                className={`px-3 py-1 text-xs rounded transition-all ${
                  useNormalized ? 'bg-amber-500 text-black' : 'text-slate-400'
                }`}
              >
                Normalized
              </button>
              <button
                onClick={() => setUseNormalized(false)}
                className={`px-3 py-1 text-xs rounded transition-all ${
                  !useNormalized ? 'bg-amber-500 text-black' : 'text-slate-400'
                }`}
              >
                Raw
              </button>
            </div>
            <button onClick={onClose} className="text-slate-400 hover:text-white">
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Chart */}
        <div className="p-4 overflow-auto max-h-[60vh]">
          <div className="space-y-2">
            {shapValues.map((sv) => {
              const val = useNormalized ? sv.shap_norm : sv.shap;
              const isPositive = val >= 0;
              const barWidth = (Math.abs(val) / maxAbsValue) * 50;

              return (
                <div key={sv.value} className="flex items-center gap-2 h-8">
                  <span className="text-xs text-slate-400 w-12 text-right font-medium">
                    {getLabel(sv.value)}
                  </span>

                  <div className="flex-1 flex items-center">
                    <div className="w-1/2 flex justify-end">
                      {!isPositive && (
                        <div
                          className="h-6 bg-red-500/80 rounded-l"
                          style={{ width: `${barWidth}%` }}
                        />
                      )}
                    </div>
                    <div className="w-px h-8 bg-white/20" />
                    <div className="w-1/2">
                      {isPositive && (
                        <div
                          className="h-6 bg-emerald-500/80 rounded-r"
                          style={{ width: `${barWidth}%` }}
                        />
                      )}
                    </div>
                  </div>

                  <span
                    className={`text-xs font-mono w-16 text-right ${
                      isPositive ? 'text-emerald-400' : 'text-red-400'
                    }`}
                  >
                    {val >= 0 ? '+' : ''}
                    {val.toFixed(3)}
                  </span>

                  <span className="text-[10px] text-slate-600 w-10 text-right">
                    n={sv.count}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Legend */}
        <div className="p-3 border-t border-white/10 bg-black/20">
          <div className="flex items-center justify-center gap-6 text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 bg-red-500/80 rounded" />
              Negative impact
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 bg-emerald-500/80 rounded" />
              Positive impact
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
