'use client';

import { useState } from 'react';
import { ChevronDown, ChevronRight, X, Activity, LineChart, Network, Target, TrendingUp } from 'lucide-react';
import { ModelConfig, ModelType } from '../types/forecasting';

interface CollapsibleModelCardProps {
  model: ModelConfig;
  onRemove: () => void;
  onUpdateName: (name: string) => void;
  onUpdateParams: (params: any) => void;
  defaultLags: number[];  // Inherited from strategy
  children: React.ReactNode;  // Model-specific config
}

const modelIcons: Record<ModelType, React.ElementType> = {
  'LAG': Activity,
  'LINEAR_REGRESSION': LineChart,
  'XGBOOST': Network,
  'ARIMA': TrendingUp,
  'PROPHET': Target,
  'NBEATS': Activity
};

const modelColors: Record<ModelType, string> = {
  'LAG': 'amber',
  'LINEAR_REGRESSION': 'amber',
  'XGBOOST': 'amber',
  'ARIMA': 'blue',
  'PROPHET': 'blue',
  'NBEATS': 'purple'
};

export default function CollapsibleModelCard({
  model,
  onRemove,
  onUpdateName,
  onUpdateParams,
  defaultLags,
  children
}: CollapsibleModelCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const Icon = modelIcons[model.type] || Activity;
  const colorClass = modelColors[model.type] || 'slate';
  
  // Check if model uses custom lags (different from default)
  const modelLags = (model.params as any).lags || (model.params as any).feature_config?.target_lags;
  const hasCustomLags = modelLags && JSON.stringify(modelLags) !== JSON.stringify(defaultLags);

  return (
    <div className={`bg-white/5 border rounded-xl overflow-hidden transition-all ${
      isExpanded ? 'border-white/20' : 'border-white/10'
    }`}>
      {/* Header - Always visible */}
      <div 
        className="flex items-center gap-3 p-3 cursor-pointer hover:bg-white/5 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {/* Expand/Collapse icon */}
        <button className="text-slate-500">
          {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </button>
        
        {/* Model icon */}
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center bg-${colorClass}-500/10`}>
          <Icon size={16} className={`text-${colorClass}-400`} />
        </div>
        
        {/* Model name */}
        <div className="flex-1 min-w-0">
          <input
            type="text"
            value={model.name}
            onChange={(e) => onUpdateName(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            className="font-medium text-white bg-transparent border-b border-transparent hover:border-white/20 focus:border-amber-500 focus:outline-none w-full truncate text-sm"
          />
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-[10px] text-slate-500">{model.type}</span>
            {hasCustomLags && (
              <span className="text-[10px] px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded">
                Custom lags
              </span>
            )}
          </div>
        </div>
        
        {/* Remove button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          className="p-1.5 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors"
        >
          <X size={16} />
        </button>
      </div>
      
      {/* Expanded content */}
      {isExpanded && (
        <div className="px-4 pb-4 pt-2 border-t border-white/5 animate-in slide-in-from-top-2 duration-200">
          {children}
        </div>
      )}
    </div>
  );
}
