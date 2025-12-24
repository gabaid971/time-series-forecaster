'use client';

import { useState, useRef, useEffect } from 'react';
import { ModelConfig, ModelType, TimeSeriesData, ModelResult, DateRange, LinearRegressionParams, ColumnInfo, FeatureConfig, TemporalFeatureConfig, ExogenousFeatureConfig, DerivedFeatureConfig } from '../types/forecasting';
import { Upload, Activity, BarChart3, Settings, Play, Plus, X, ChevronRight, FileText, CheckCircle2, Trophy, Timer, Download, Trash2, LineChart, Network, Target, Calculator, Split, Percent } from 'lucide-react';
import Papa from 'papaparse';
import TimeSeriesChart from '../components/TimeSeriesChart';

// ============================================================================
// API CONFIGURATION
// ============================================================================
// Mode: 'direct' = appel direct au backend (contourne limite Vercel, clé exposée)
//       'proxy'  = passe par /api routes Vercel (clé cachée, limite 4.5MB)
const API_MODE = (process.env.NEXT_PUBLIC_API_MODE || 'direct') as 'direct' | 'proxy';

// URLs et clé API
const BACKEND_URL = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/$/, '');
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || '';

// Helper pour construire l'URL selon le mode
const getApiUrl = (endpoint: string) => {
  if (API_MODE === 'proxy') {
    return `/api/${endpoint}`;
  }
  return `${BACKEND_URL}/${endpoint}`;
};

// Helper pour construire les headers selon le mode
const getApiHeaders = (): HeadersInit => {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (API_MODE === 'direct' && API_KEY) {
    headers['X-API-Key'] = API_KEY;
  }
  return headers;
};

// Debug: log API config (check browser console)
if (typeof window !== 'undefined') {
  console.log('🔗 API Mode:', API_MODE);
  console.log('🔗 Backend URL:', BACKEND_URL);
}

// Tag Input Component for entering multiple integer values
const TagInput = ({ values, onChange, placeholder }: { values: number[], onChange: (values: number[]) => void, placeholder?: string }) => {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const addTag = (value: string) => {
    const num = parseInt(value.trim());
    if (!isNaN(num) && num >= 0 && !values.includes(num)) {
      onChange([...values, num].sort((a, b) => a - b));
      setInputValue('');
    }
  };

  const removeTag = (index: number) => {
    onChange(values.filter((_, i) => i !== index));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ',') && inputValue) {
      e.preventDefault();
      addTag(inputValue);
    } else if (e.key === 'Backspace' && !inputValue && values.length > 0) {
      onChange(values.slice(0, -1));
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    // Auto-add on comma (mobile friendly)
    if (val.includes(',')) {
      const numStr = val.replace(',', '').trim();
      if (numStr) addTag(numStr);
      else setInputValue('');
    } else {
      setInputValue(val);
    }
  };

  return (
    <div 
      onClick={() => inputRef.current?.focus()}
      className="flex flex-wrap gap-1.5 p-2 glass-input rounded-lg min-h-[42px] cursor-text"
    >
      {values.map((tag, idx) => (
        <span key={idx} className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-amber-500 text-black text-sm font-medium rounded">
          {tag}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              removeTag(idx);
            }}
            className="hover:bg-amber-600 rounded-sm p-1 transition-colors touch-manipulation"
            aria-label={`Remove ${tag}`}
          >
            <X size={14} />
          </button>
        </span>
      ))}
      <input
        ref={inputRef}
        type="text"
        inputMode="numeric"
        value={inputValue}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onBlur={() => inputValue && addTag(inputValue)}
        placeholder={values.length === 0 ? placeholder : ''}
        className="flex-1 min-w-[80px] bg-transparent outline-none text-sm text-white placeholder:text-slate-600"
      />
    </div>
  );
};

// Segmented Control Component for binary/ternary choices
const SegmentedControl = ({ value, options, onChange }: { 
  value: string, 
  options: { value: string, label: string }[], 
  onChange: (value: string) => void 
}) => {
  return (
    <div className="inline-flex bg-black/30 rounded-lg p-1 border border-white/10">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          className={`px-4 py-1.5 text-xs font-medium rounded transition-all ${
            value === option.value
              ? 'bg-amber-500 text-black shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
};

// Slider Component for numeric ranges
const Slider = ({ value, min, max, step, onChange, label }: { 
  value: number, 
  min: number, 
  max: number, 
  step?: number, 
  onChange: (value: number) => void,
  label?: string
}) => {
  const [localValue, setLocalValue] = useState(value);
  
  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400">{label}</span>
        <input
          type="number"
          value={localValue}
          onChange={(e) => {
            const val = parseInt(e.target.value) || min;
            setLocalValue(val);
            onChange(val);
          }}
          className="glass-input w-20 px-2 py-1 rounded text-xs text-center"
        />
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step || 1}
        value={localValue}
        onChange={(e) => {
          const val = parseInt(e.target.value);
          setLocalValue(val);
          onChange(val);
        }}
        className="w-full h-2 bg-black/20 rounded-lg appearance-none cursor-pointer slider-thumb"
        style={{
          background: `linear-gradient(to right, rgb(245, 158, 11) 0%, rgb(245, 158, 11) ${((localValue - min) / (max - min)) * 100}%, rgba(0,0,0,0.2) ${((localValue - min) / (max - min)) * 100}%, rgba(0,0,0,0.2) 100%)`
        }}
      />
      <div className="flex justify-between text-[10px] text-slate-600">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
};

// Single Number Input Component (like TagInput but for one value only)
const SingleNumberInput = ({ value, onChange, min = 0, placeholder }: {
  value: number,
  onChange: (value: number) => void,
  min?: number,
  placeholder?: string
}) => {
  const [inputValue, setInputValue] = useState(String(value));
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setInputValue(String(value));
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setInputValue(val);
    
    const num = parseInt(val);
    if (!isNaN(num) && num >= min) {
      onChange(num);
    }
  };

  const handleBlur = () => {
    const num = parseInt(inputValue);
    if (isNaN(num) || num < min) {
      setInputValue(String(value));
    }
  };

  return (
    <input
      ref={inputRef}
      type="text"
      inputMode="numeric"
      value={inputValue}
      onChange={handleChange}
      onBlur={handleBlur}
      placeholder={placeholder || String(value)}
      className="glass-input w-full p-2 rounded-lg text-sm text-white"
    />
  );
};

const FeatureConfigPanel = ({ model, updateModelParams, availableColumns }: { model: ModelConfig, updateModelParams: Function, availableColumns: ColumnInfo[] }) => {
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
                <button onClick={() => updateConfig(prev => ({ ...prev, derived: prev.derived?.filter((_, i) => i !== idx) }))} className="text-slate-500 hover:text-red-400">
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
};

export default function ForecastingPage() {
  const [step, setStep] = useState<number>(1);
  const [selectedModels, setSelectedModels] = useState<ModelConfig[]>([]);
  
  // Strategy State
  const [trainingRanges, setTrainingRanges] = useState<DateRange[]>([]);
  const [predictionRanges, setPredictionRanges] = useState<DateRange[]>([]);
  
  // Data State
  const [data, setData] = useState<TimeSeriesData | null>(null);
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [fullData, setFullData] = useState<any[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Available columns for exogenous features (from /analyze)
  const [availableColumns, setAvailableColumns] = useState<ColumnInfo[]>([]);
  
  // Dataset Stats State
  const [datasetStats, setDatasetStats] = useState<{
    date_min: string;
    date_max: string;
    total_rows: number;
    frequency: string;
    frequency_label: string;
    missing_dates: number;
    missing_values_target: number;
    value_min: number;
    value_max: number;
    value_mean: number;
  } | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Export forecasts to CSV
  const exportForecastsToCSV = () => {
    if (!results || results.length === 0 || !data) return;

    // Build a map of all dates and their data
    const dateMap = new Map<string, any>();

    // Collect all forecasts from all models
    results.forEach(result => {
      if (result.error) return; // Skip models with errors
      
      result.forecast.forEach((point: any) => {
        const dateKey = point[data.dateColumn];
        if (!dateMap.has(dateKey)) {
          dateMap.set(dateKey, {
            date: dateKey,
            actual: point[data.targetColumn]
          });
        }
        // Add this model's prediction
        dateMap.get(dateKey)[result.model_name] = point.prediction;
      });
    });

    // Convert to array and sort by date
    const rows = Array.from(dateMap.values()).sort((a, b) => 
      a.date.localeCompare(b.date)
    );

    // Create CSV header
    const modelNames = results.filter(r => !r.error).map(r => r.model_name);
    const headers = ['date', 'actual', ...modelNames];
    
    // Create CSV content
    let csvContent = headers.join(',') + '\n';
    rows.forEach(row => {
      const values = [
        row.date,
        row.actual ?? '',
        ...modelNames.map(name => row[name] ?? '')
      ];
      csvContent += values.join(',') + '\n';
    });

    // Download CSV
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `forecasts_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Results State
  const [isTraining, setIsTraining] = useState(false);
  const [results, setResults] = useState<ModelResult[]>([]);

  // Raw data from CSV (before backend normalization)
  const [rawData, setRawData] = useState<any[]>([]);

  const handleFile = (file: File) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        const columns = results.meta.fields || [];
        const firstRow = results.data[0] as any;
        
        // Simple heuristic to guess columns
        const dateCol = columns.find(c => c.toLowerCase().includes('date') || c.toLowerCase().includes('time')) || columns[0];
        const targetCol = columns.find(c => c !== dateCol && (typeof firstRow[c] === 'number')) || columns[1];

        // Store raw data for backend processing
        const filteredData = (results.data as any[]).filter((row: any) => row[dateCol]);
        setRawData(filteredData);

        setData({
          filename: file.name,
          columns: columns,
          dateColumn: dateCol,
          targetColumn: targetCol,
          frequency: 'D',
          exogenousFeatures: columns.filter(c => c !== dateCol && c !== targetCol)
        });
        setPreviewData(filteredData.slice(0, 5));
      },
      error: (error) => {
        console.error('Error parsing CSV:', error);
      }
    });
  };

  const startTraining = async () => {
    if (!data) return;
    setStep(3);
    setIsTraining(true);
    
    try {
      // Use rawData for training (contains all columns including exogenous)
      const payload = {
        data: rawData,
        data_config: {
          target_column: data.targetColumn,
          date_column: data.dateColumn,
          frequency: data.frequency,
          training_ranges: trainingRanges,
          prediction_ranges: predictionRanges
        },
        models: selectedModels
      };

      const response = await fetch(getApiUrl('train'), {
        method: 'POST',
        headers: getApiHeaders(),
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (result.status === 'success') {
        setResults(result.results);
      } else {
        console.error('Training failed:', result);
      }
    } catch (error) {
      console.error('Failed to connect to backend:', error);
      // Fallback or error state handling could go here
    } finally {
      setIsTraining(false);
    }
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  // Analyze dataset when data or column selection changes
  // Fallback: use local data if backend fails
  const useLocalFallback = () => {
    if (!data || !rawData.length) return;
    
    // Use raw CSV data directly for visualization
    const localData = rawData.map(row => ({
      [data.dateColumn]: row[data.dateColumn],
      [data.targetColumn]: row[data.targetColumn]
    }));
    setFullData(localData);
    
    // Calculate basic stats locally
    const values = rawData.map(r => r[data.targetColumn]).filter((v: unknown) => typeof v === 'number');
    const dates = rawData.map(r => String(r[data.dateColumn])).filter(Boolean);
    
    // Try to find min/max dates (works for various formats)
    const sortedDates = [...dates].sort();
    const dateMin = sortedDates[0] || '';
    const dateMax = sortedDates[sortedDates.length - 1] || '';
    
    // Set training/prediction ranges
    const splitIndex = Math.floor(localData.length * 0.8);
    const splitDate = sortedDates[splitIndex] || dateMax;
    
    setTrainingRanges([{ start: dateMin, end: splitDate }]);
    setPredictionRanges([{ start: splitDate, end: dateMax }]);
    
    // Set basic stats
    if (values.length > 0) {
      setDatasetStats({
        date_min: dateMin,
        date_max: dateMax,
        total_rows: rawData.length,
        frequency: 'D',
        frequency_label: 'Daily (assumed)',
        missing_dates: 0,
        missing_values_target: rawData.length - values.length,
        value_min: Math.min(...values),
        value_max: Math.max(...values),
        value_mean: values.reduce((a: number, b: number) => a + b, 0) / values.length
      });
    }
  };

  // Backend parses dates, returns normalized data with ISO format
  const analyzeDataset = async () => {
    if (!data || !rawData.length) return;
    
    setIsAnalyzing(true);
    try {
      const response = await fetch(getApiUrl('analyze'), {
        method: 'POST',
        headers: getApiHeaders(),
        body: JSON.stringify({
          data: rawData,
          date_column: data.dateColumn,
          target_column: data.targetColumn,
        }),
      });
      
      const result = await response.json();
      if (result.status === 'success' && result.stats) {
        setDatasetStats(result.stats);
        // Update data frequency with detected one
        setData(prev => prev ? { ...prev, frequency: result.stats.frequency } : null);
        
        // Store available columns for exogenous features
        if (result.available_columns) {
          setAvailableColumns(result.available_columns);
        }
        
        // Use normalized data from backend (dates in ISO format, values cleaned)
        if (result.normalized_data && result.normalized_data.length > 0) {
          // Convert to format expected by chart: keep full datetime for minute-level data
          const normalizedData = result.normalized_data.map((point: { date: string; value: number }) => ({
            [data.dateColumn]: point.date, // Keep full ISO datetime (YYYY-MM-DDTHH:MM:SS)
            [data.targetColumn]: point.value
          }));
          setFullData(normalizedData);
          
          // Set default training/prediction ranges from stats
          // For date inputs, extract just YYYY-MM-DD part
          const dateMin = result.stats.date_min.split('T')[0];
          const dateMax = result.stats.date_max.split('T')[0];
          const totalRows = normalizedData.length;
          const splitIndex = Math.floor(totalRows * 0.8);
          const splitDate = (normalizedData[splitIndex]?.[data.dateColumn] || dateMax).split('T')[0];
          
          setTrainingRanges([{ start: dateMin, end: splitDate }]);
          setPredictionRanges([{ start: splitDate, end: dateMax }]);
        }
      } else {
        // Backend returned error, use fallback
        console.warn('Backend analysis failed, using local fallback');
        useLocalFallback();
      }
    } catch (error) {
      console.error('Failed to analyze dataset:', error);
      // Use local fallback when backend is unreachable
      console.warn('Backend unreachable, using local fallback for visualization');
      useLocalFallback();
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Trigger analysis when data or columns change
  useEffect(() => {
    if (data && rawData.length > 0) {
      analyzeDataset();
    }
  }, [data?.dateColumn, data?.targetColumn, rawData.length]);

  const addModel = (type: ModelType) => {
    // Count existing models of this type to auto-number
    const existingCount = selectedModels.filter(m => m.type === type).length;
    const modelNumber = existingCount + 1;
    
    const typeNames: Record<ModelType, string> = {
      'LAG': 'Lag',
      'LINEAR_REGRESSION': 'Linear Regression',
      'ARIMA': 'ARIMA',
      'PROPHET': 'Prophet',
      'XGBOOST': 'XGBoost',
      'NBEATS': 'N-BEATS'
    };
    
    // Default params per model type
    const defaultParams: Record<ModelType, any> = {
      'LAG': { lag: 1 },
      'LINEAR_REGRESSION': { lags: [1, 7] },
      'XGBOOST': { lags: [1, 7], n_estimators: 100, max_depth: 3, learning_rate: 0.1 },
      'ARIMA': { p: 1, d: 1, q: 1 },
      'PROPHET': { daily_seasonality: false, weekly_seasonality: true, yearly_seasonality: true, seasonality_mode: 'additive' },
      'NBEATS': {}
    };
    
    const newModel: ModelConfig = {
      id: `${type}-${Date.now()}`,
      type: type,
      name: `${typeNames[type]} ${modelNumber}`,
      params: defaultParams[type]
    };
    setSelectedModels([...selectedModels, newModel]);
  };

  const updateModelName = (modelId: string, newName: string) => {
    setSelectedModels(prev => prev.map(m => 
      m.id === modelId ? { ...m, name: newName } : m
    ));
  };

  const updateModelParams = (modelId: string, newParamsOrFn: Partial<ModelConfig['params']> | ((prev: any) => Partial<ModelConfig['params']>)) => {
    setSelectedModels(prev => prev.map(m => {
      if (m.id !== modelId) return m;
      
      const newParams = typeof newParamsOrFn === 'function' 
        ? newParamsOrFn(m.params)
        : newParamsOrFn;
        
      return { ...m, params: { ...m.params, ...newParams } };
    }));
  };

  const parseLagsString = (lagsStr: string): number[] => {
    return lagsStr
      .split(',')
      .map(s => parseInt(s.trim()))
      .filter(n => !isNaN(n) && n > 0);
  };

  return (
    <div className="min-h-screen font-sans selection:bg-amber-500/30">
      {/* Ambient Background Effects */}
      <div className="fixed inset-0 pointer-events-none fog-gradient z-0" />
      <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-amber-500/10 blur-[120px] rounded-full pointer-events-none z-0" />

      <div className="relative z-10 max-w-6xl mx-auto p-3 sm:p-6">
        
        {/* Header */}
        <header className="flex items-center justify-between mb-12 pt-4">
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-gradient-to-br from-amber-400 to-orange-600 flex items-center justify-center shadow-lg shadow-amber-500/20">
              <Activity className="text-white w-5 h-5 sm:w-6 sm:h-6" />
            </div>
            <div>
              <h1 className="text-lg sm:text-2xl font-bold text-white tracking-tight">Time Series <span className="text-amber-500">Studio</span></h1>
              <p className="text-slate-400 text-[10px] sm:text-xs uppercase tracking-widest hidden sm:block">Forecasting Pipeline</p>
            </div>
          </div>
        </header>

        {/* Stepper */}
        <div className="mb-6 sm:mb-12">
          <div className="flex items-center justify-between relative px-4 sm:px-0">
            <div className="absolute left-0 top-1/2 w-full h-0.5 bg-white/10 -z-10" />
            {[1, 2, 3].map((i) => (
              <div 
                key={i}
                onClick={() => i < step ? setStep(i) : null}
                className={`relative flex flex-col items-center gap-1 sm:gap-2 cursor-pointer group ${step === i ? 'scale-105 sm:scale-110' : 'scale-100'} transition-all duration-300`}
              >
                <div className={`w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                  step >= i 
                    ? 'bg-amber-500 border-amber-500 text-black shadow-[0_0_20px_rgba(245,158,11,0.4)]' 
                    : 'bg-slate-900 border-white/20 text-slate-500 group-hover:border-white/40'
                }`}>
                  {i === 1 ? <Upload size={16} /> : i === 2 ? <Settings size={16} /> : <BarChart3 size={16} />}
                </div>
                <span className={`text-[10px] sm:text-xs font-medium tracking-wider ${step >= i ? 'text-amber-500' : 'text-slate-600'}`}>
                  {i === 1 ? 'DATA' : i === 2 ? 'STRATEGY' : 'RESULTS'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Main Content Area */}
        <div className="glass-panel rounded-xl sm:rounded-2xl p-1 min-h-[400px] sm:min-h-[600px] glow-box transition-all duration-500">
          <div className="bg-black/40 rounded-lg sm:rounded-xl min-h-[400px] sm:min-h-[600px] p-4 sm:p-8 backdrop-blur-sm">
            
            {/* STEP 1: DATA */}
            {step === 1 && (
              <div className="h-full flex flex-col animate-in fade-in slide-in-from-bottom-4 duration-500">
                {!data ? (
                  <div className="flex-1 flex flex-col items-center justify-center space-y-8">
                    <div className="w-full max-w-xl">
                      <input 
                        type="file" 
                        ref={fileInputRef}
                        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                        className="hidden" 
                        accept=".csv"
                      />
                      <div 
                        onDragOver={onDragOver}
                        onDragLeave={onDragLeave}
                        onDrop={onDrop}
                        onClick={() => fileInputRef.current?.click()}
                        className={`border-2 border-dashed rounded-xl sm:rounded-2xl p-8 sm:p-16 text-center transition-all cursor-pointer group ${
                          isDragging 
                            ? 'border-amber-500 bg-amber-500/10 scale-105' 
                            : 'border-white/10 hover:border-amber-500/50 hover:bg-white/5'
                        }`}
                      >
                        <div className="w-14 h-14 sm:w-20 sm:h-20 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-4 sm:mb-6 group-hover:scale-110 transition-transform duration-300">
                          <Upload className={`w-7 h-7 sm:w-10 sm:h-10 transition-colors ${isDragging ? 'text-amber-500' : 'text-slate-400 group-hover:text-amber-500'}`} />
                        </div>
                        <h3 className="text-lg sm:text-xl font-semibold text-white mb-2">Upload Time Series</h3>
                        <p className="text-slate-400 text-sm sm:text-base mb-4 sm:mb-8">Drag & drop CSV file here</p>
                        <button className="px-4 py-2 sm:px-6 sm:py-3 bg-white/10 hover:bg-white/20 text-white rounded-lg font-medium transition-all border border-white/10 text-sm sm:text-base">
                          Browse Files
                        </button>
                      </div>
                    </div>
                    <div className="flex gap-4 text-slate-500 text-sm">
                      <span>Supported: .csv</span>
                    </div>
                    
                    {/* Example Data Buttons */}
                    <div className="pt-8 border-t border-white/5 w-full max-w-xl">
                      <p className="text-xs text-slate-500 uppercase tracking-wider mb-4 text-center">Or try with example data</p>
                      <div className="flex justify-center">
                        <button 
                          onClick={() => {
                            // Generate Trend + Seasonality
                            const rows = [];
                            const now = new Date();
                            now.setHours(0, 0, 0, 0);
                            for (let i = 0; i < 365; i++) {
                              const date = new Date(now.getTime() - (365 - i) * 24 * 3600 * 1000);
                              const trend = i * 0.05;
                              const season = 10 * Math.sin(i * (2 * Math.PI / 7)); // Weekly
                              const val = 20 + trend + season + (Math.random() * 5);
                              rows.push({
                                date: date.toISOString().split('T')[0],
                                sales: parseFloat(val.toFixed(2)),
                                promotion: i % 7 === 0 ? 1 : 0
                              });
                            }
                            
                            setRawData(rows);
                            setData({
                              filename: 'sales_example.csv',
                              columns: ['date', 'sales', 'promotion'],
                              dateColumn: 'date',
                              targetColumn: 'sales',
                              frequency: 'D',
                              exogenousFeatures: ['promotion']
                            });
                            setPreviewData(rows.slice(0, 5));
                          }}
                          className="p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 hover:border-amber-500/30 transition-all text-left group max-w-xs"
                        >
                          <div className="font-medium text-slate-200 group-hover:text-amber-400 mb-1">Daily Sales</div>
                          <div className="text-xs text-slate-500">Trend + Weekly Seasonality</div>
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col gap-6">
                    <div className="flex items-center justify-between bg-white/5 p-4 rounded-xl border border-white/10">
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 bg-amber-500/20 rounded-lg flex items-center justify-center text-amber-500">
                          <FileText size={24} />
                        </div>
                        <div>
                          <h3 className="font-semibold text-white">{data.filename}</h3>
                          <p className="text-xs text-slate-400">{previewData.length}+ rows detected</p>
                        </div>
                      </div>
                      <button 
                        onClick={() => setData(null)}
                        className="text-slate-400 hover:text-white text-sm underline"
                      >
                        Change File
                      </button>
                    </div>

                    {/* Chart Preview */}
                    <TimeSeriesChart 
                      data={fullData} 
                      dateColumn={data.dateColumn} 
                      targetColumn={data.targetColumn} 
                    />

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
                      <div className="space-y-2">
                        <label className="text-xs sm:text-sm text-slate-400">Date Column</label>
                        <select 
                          value={data.dateColumn}
                          onChange={(e) => setData({...data, dateColumn: e.target.value})}
                          className="w-full bg-black/20 border border-white/10 rounded-lg p-2 sm:p-3 text-sm sm:text-base text-white focus:border-amber-500 outline-none [&>option]:bg-slate-900 [&>option]:text-white"
                        >
                          {data.columns.map(col => <option key={col} value={col}>{col}</option>)}
                        </select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-xs sm:text-sm text-slate-400">Target Column</label>
                        <select 
                          value={data.targetColumn}
                          onChange={(e) => setData({...data, targetColumn: e.target.value})}
                          className="w-full bg-black/20 border border-white/10 rounded-lg p-2 sm:p-3 text-sm sm:text-base text-white focus:border-amber-500 outline-none [&>option]:bg-slate-900 [&>option]:text-white"
                        >
                          {data.columns.map(col => <option key={col} value={col}>{col}</option>)}
                        </select>
                      </div>
                    </div>

                    {/* Dataset Stats Panel */}
                    <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                      <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                        <BarChart3 size={16} className="text-amber-500" />
                        Dataset Analysis
                        {isAnalyzing && <span className="text-xs text-slate-400 animate-pulse ml-2">Analyzing...</span>}
                      </h4>
                      {datasetStats ? (
                        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                          <div className="bg-black/20 rounded-lg p-3">
                            <p className="text-[10px] text-slate-500 uppercase">Frequency</p>
                            <p className="text-sm font-mono text-amber-400">{datasetStats.frequency_label}</p>
                          </div>
                          <div className="bg-black/20 rounded-lg p-3">
                            <p className="text-[10px] text-slate-500 uppercase">Date Range</p>
                            <p className="text-xs font-mono text-white">
                              {new Date(datasetStats.date_min).toLocaleDateString()} - {new Date(datasetStats.date_max).toLocaleDateString()}
                            </p>
                          </div>
                          <div className="bg-black/20 rounded-lg p-3">
                            <p className="text-[10px] text-slate-500 uppercase">Total Rows</p>
                            <p className="text-sm font-mono text-white">{datasetStats.total_rows.toLocaleString()}</p>
                          </div>
                          <div className="bg-black/20 rounded-lg p-3">
                            <p className="text-[10px] text-slate-500 uppercase">Missing Dates</p>
                            <p className={`text-sm font-mono ${datasetStats.missing_dates > 0 ? 'text-orange-400' : 'text-emerald-400'}`}>
                              {datasetStats.missing_dates}
                            </p>
                          </div>
                          <div className="bg-black/20 rounded-lg p-3">
                            <p className="text-[10px] text-slate-500 uppercase">Missing Values</p>
                            <p className={`text-sm font-mono ${datasetStats.missing_values_target > 0 ? 'text-orange-400' : 'text-emerald-400'}`}>
                              {datasetStats.missing_values_target}
                            </p>
                          </div>
                          <div className="bg-black/20 rounded-lg p-3">
                            <p className="text-[10px] text-slate-500 uppercase">Target Range</p>
                            <p className="text-xs font-mono text-white">
                              {datasetStats.value_min.toFixed(1)} - {datasetStats.value_max.toFixed(1)}
                            </p>
                          </div>
                        </div>
                      ) : (
                        <p className="text-slate-500 text-sm">Select date and target columns to analyze</p>
                      )}
                    </div>

                    <div className="flex-1 overflow-hidden border border-white/10 rounded-xl">
                      <div className="overflow-auto h-full custom-scrollbar">
                        <table className="w-full text-sm text-left text-slate-400">
                          <thead className="text-xs text-slate-200 uppercase bg-white/5 sticky top-0">
                            <tr>
                              {data.columns.map(col => (
                                <th key={col} className="px-6 py-3 font-medium">
                                  {col}
                                  {col === data.targetColumn && <span className="ml-2 text-amber-500">(Target)</span>}
                                  {col === data.dateColumn && <span className="ml-2 text-blue-400">(Date)</span>}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {previewData.map((row, idx) => (
                              <tr key={idx} className="border-b border-white/5 hover:bg-white/5">
                                {data.columns.map(col => (
                                  <td key={col} className="px-6 py-4 font-mono text-xs">
                                    {row[col]}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <div className="flex justify-end pt-4">
                      <button 
                        onClick={() => setStep(2)}
                        className="flex items-center gap-2 px-4 py-2 sm:px-8 sm:py-3 bg-amber-500 hover:bg-amber-400 text-black font-bold rounded-lg shadow-[0_0_20px_rgba(245,158,11,0.3)] transition-all text-sm sm:text-base"
                      >
                        Confirm Data <CheckCircle2 size={16} />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* STEP 2: MODELS & CONFIG */}
            {step === 2 && (
              <div className="flex flex-col lg:grid lg:grid-cols-12 gap-6 lg:gap-8 h-full animate-in fade-in slide-in-from-bottom-4 duration-500">
                
                {/* Left: Model Palette */}
                <div className="lg:col-span-4 space-y-3 lg:space-y-4 lg:border-r border-white/10 lg:pr-8">
                  <h3 className="text-xs sm:text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 lg:mb-6">Model Library</h3>
                  
                  <div className="grid grid-cols-2 lg:grid-cols-1 gap-2 lg:space-y-3">
                    {[
                      { id: 'LAG', name: 'Lag', desc: 'Naïve Baseline', Icon: Activity },
                      { id: 'LINEAR_REGRESSION', name: 'Linear Regression', desc: 'Simple Baseline', Icon: LineChart },
                      { id: 'ARIMA', name: 'ARIMA', desc: 'Statistical Baseline', Icon: Activity },
                      { id: 'PROPHET', name: 'Prophet', desc: 'Facebook model', Icon: Target },
                      { id: 'XGBOOST', name: 'XGBoost', desc: 'Gradient Boosting', Icon: Network },
                    ].map((m) => (
                      <button 
                        key={m.id}
                        onClick={() => addModel(m.id as ModelType)}
                        className="w-full text-left p-3 lg:p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 hover:border-amber-500/30 transition-all group"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 lg:gap-3">
                            <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center group-hover:bg-amber-500/10 transition-all">
                              <m.Icon size={18} className="text-slate-400 group-hover:text-amber-500 transition-colors" />
                            </div>
                            <div>
                              <div className="font-medium text-xs lg:text-base text-slate-200 group-hover:text-white">{m.name}</div>
                              <div className="text-[10px] lg:text-xs text-slate-500 hidden sm:block">{m.desc}</div>
                            </div>
                          </div>
                          <Plus size={14} className="text-slate-600 group-hover:text-amber-500" />
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Right: Configuration */}
                <div className="lg:col-span-8 flex flex-col">
                  <h3 className="text-xs sm:text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 lg:mb-6">Pipeline Configuration</h3>
                  
                  {/* Validation Strategy */}
                  <div className="bg-white/5 border border-white/10 rounded-xl p-3 sm:p-5 mb-4 lg:mb-6">
                    <h4 className="font-semibold text-sm sm:text-base text-white mb-3 sm:mb-4 flex items-center gap-2">
                      <Settings size={14} className="text-amber-500" /> Validation Strategy
                    </h4>
                    
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-8">
                      {/* Training Ranges */}
                      <div>
                        <div className="flex items-center justify-between mb-3">
                          <label className="text-xs sm:text-sm text-slate-400">Training Periods</label>
                          <button 
                            onClick={() => setTrainingRanges([...trainingRanges, { start: '', end: '' }])}
                            className="text-xs text-amber-500 hover:text-amber-400 flex items-center gap-1"
                          >
                            <Plus size={12} /> Add Period
                          </button>
                        </div>
                        <div className="space-y-2">
                          {trainingRanges.map((range, idx) => (
                            <div key={idx} className="flex items-center gap-2">
                              <input 
                                type="date" 
                                value={range.start}
                                onChange={(e) => {
                                  const newRanges = [...trainingRanges];
                                  newRanges[idx].start = e.target.value;
                                  setTrainingRanges(newRanges);
                                }}
                                className="bg-black/20 border border-white/10 rounded px-2 py-1 text-xs text-white w-full"
                              />
                              <span className="text-slate-500">-</span>
                              <input 
                                type="date" 
                                value={range.end}
                                onChange={(e) => {
                                  const newRanges = [...trainingRanges];
                                  newRanges[idx].end = e.target.value;
                                  setTrainingRanges(newRanges);
                                }}
                                className="bg-black/20 border border-white/10 rounded px-2 py-1 text-xs text-white w-full"
                              />
                              <button 
                                onClick={() => setTrainingRanges(trainingRanges.filter((_, i) => i !== idx))}
                                className="text-slate-500 hover:text-red-400"
                              >
                                <Trash2 size={14} />
                              </button>
                            </div>
                          ))}
                          {trainingRanges.length === 0 && (
                            <p className="text-xs text-slate-600 italic">No training periods defined.</p>
                          )}
                        </div>
                      </div>

                      {/* Prediction Ranges */}
                      <div>
                        <div className="flex items-center justify-between mb-3">
                          <label className="text-sm text-slate-400">Prediction Periods</label>
                          <button 
                            onClick={() => setPredictionRanges([...predictionRanges, { start: '', end: '' }])}
                            className="text-xs text-amber-500 hover:text-amber-400 flex items-center gap-1"
                          >
                            <Plus size={12} /> Add Period
                          </button>
                        </div>
                        <div className="space-y-2">
                          {predictionRanges.map((range, idx) => (
                            <div key={idx} className="flex items-center gap-2">
                              <input 
                                type="date" 
                                value={range.start}
                                onChange={(e) => {
                                  const newRanges = [...predictionRanges];
                                  newRanges[idx].start = e.target.value;
                                  setPredictionRanges(newRanges);
                                }}
                                className="bg-black/20 border border-white/10 rounded px-2 py-1 text-xs text-white w-full"
                              />
                              <span className="text-slate-500">-</span>
                              <input 
                                type="date" 
                                value={range.end}
                                onChange={(e) => {
                                  const newRanges = [...predictionRanges];
                                  newRanges[idx].end = e.target.value;
                                  setPredictionRanges(newRanges);
                                }}
                                className="bg-black/20 border border-white/10 rounded px-2 py-1 text-xs text-white w-full"
                              />
                              <button 
                                onClick={() => setPredictionRanges(predictionRanges.filter((_, i) => i !== idx))}
                                className="text-slate-500 hover:text-red-400"
                              >
                                <Trash2 size={14} />
                              </button>
                            </div>
                          ))}
                          {predictionRanges.length === 0 && (
                            <p className="text-xs text-slate-600 italic">No prediction periods defined.</p>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scrollbar">
                    {selectedModels.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center text-slate-500 border-2 border-dashed border-white/5 rounded-xl">
                        <Settings className="w-12 h-12 mb-4 opacity-20" />
                        <p>Select models from the library to configure them</p>
                      </div>
                    ) : (
                      selectedModels.map((model) => (
                        <div key={model.id} className="bg-white/5 border border-white/10 rounded-xl p-5 animate-in zoom-in-95 duration-300">
                          <div className="flex justify-between items-start mb-6 border-b border-white/5 pb-4">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 rounded bg-amber-500/10 flex items-center justify-center text-amber-500 font-bold text-xs">
                                {model.type.substring(0, 3)}
                              </div>
                              <div className="flex-1">
                                <input
                                  type="text"
                                  defaultValue={model.name}
                                  onBlur={(e) => updateModelName(model.id, e.target.value)}
                                  className="font-semibold text-white bg-transparent border-b border-transparent hover:border-white/20 focus:border-amber-500 focus:outline-none w-full transition-colors"
                                />
                                <p className="text-xs text-slate-500">ID: {model.id}</p>
                              </div>
                            </div>
                            <button 
                              onClick={() => setSelectedModels(selectedModels.filter(m => m.id !== model.id))}
                              className="text-slate-500 hover:text-red-400 transition-colors"
                            >
                              <X size={18} />
                            </button>
                          </div>

                          {/* Dynamic Forms */}
                          <div className="grid grid-cols-2 gap-6">
                            {model.type === 'LAG' && (
                              <div className="col-span-2">
                                <label className="text-xs text-slate-400 mb-2 block">Lag Period</label>
                                <SingleNumberInput
                                  value={(model.params as any).lag ?? 1}
                                  min={0}
                                  onChange={(value) => updateModelParams(model.id, { lag: value })}
                                  placeholder="1"
                                />
                                <p className="text-[10px] text-slate-500 mt-1">Predict using value from N periods ago (e.g., lag=1 uses previous period)</p>
                              </div>
                            )}

                            {model.type === 'LINEAR_REGRESSION' && (
                              <>
                                <div className="col-span-2">
                                  <label className="text-xs text-slate-400 mb-2 block">Target Lags</label>
                                  <TagInput
                                    values={'lags' in model.params ? model.params.lags : [1, 7]}
                                    onChange={(lags) => updateModelParams(model.id, { lags })}
                                    placeholder="Type a lag and press Enter (e.g., 1, 7, 14)"
                                  />
                                  <p className="text-[10px] text-slate-500 mt-1">Press Enter to add a lag period</p>
                                </div>
                                
                                {/* Target Mode */}
                                <div>
                                  <label className="text-xs text-slate-400 mb-2 block">Target Mode</label>
                                  <SegmentedControl
                                    value={(model.params as LinearRegressionParams).target_mode ?? 'raw'}
                                    options={[
                                      { value: 'raw', label: 'Raw (predict y)' },
                                      { value: 'residual', label: 'Residual (y - y_lag)' }
                                    ]}
                                    onChange={(value) => updateModelParams(model.id, { target_mode: value as 'raw' | 'residual' })}
                                  />
                                  <p className="text-[10px] text-slate-500 mt-1">Raw = predict target directly, Residual = predict difference</p>
                                </div>
                                
                                {/* Residual Lag (only visible when residual mode) */}
                                <div>
                                  <label className="text-xs text-slate-400 mb-2 block">
                                    Residual Lag {(model.params as LinearRegressionParams).target_mode === 'residual' ? '' : '(disabled)'}
                                  </label>
                                  {(model.params as LinearRegressionParams).target_mode === 'residual' ? (
                                    <SingleNumberInput
                                      value={(model.params as LinearRegressionParams).residual_lag ?? 1}
                                      min={0}
                                      onChange={(value) => updateModelParams(model.id, { residual_lag: value })}
                                      placeholder="1"
                                    />
                                  ) : (
                                    <div className="glass-input w-full p-2 rounded-lg text-sm opacity-40 text-slate-600 text-center">Disabled</div>
                                  )}
                                  <p className="text-[10px] text-slate-500 mt-1">Which lag to subtract (e.g., 1 = y - y_t-1)</p>
                                </div>
                                
                                {/* Standardize */}
                                <div className="col-span-2">
                                  <label className="flex items-center gap-3 p-3 bg-black/20 rounded-lg border border-white/5 cursor-pointer hover:border-amber-500/30">
                                    <input 
                                      type="checkbox" 
                                      checked={(model.params as LinearRegressionParams).standardize ?? false}
                                      onChange={(e) => updateModelParams(model.id, { standardize: e.target.checked })}
                                      className="accent-amber-500 w-4 h-4" 
                                    />
                                    <div>
                                      <span className="text-sm text-slate-300">Standardize Features</span>
                                      <p className="text-[10px] text-slate-500">Center and scale features (useful for regularized models)</p>
                                    </div>
                                  </label>
                                </div>
                                
                                {/* Feature Configuration Panel */}
                                <div className="col-span-2">
                                  <FeatureConfigPanel 
                                    model={model} 
                                    updateModelParams={updateModelParams} 
                                    availableColumns={availableColumns} 
                                  />
                                </div>
                              </>
                            )}

                            {model.type === 'XGBOOST' && (() => {
                              const params = model.params as any;
                              return (
                              <>
                                {/* Target Lags */}
                                <div className="col-span-2">
                                  <label className="text-xs text-slate-400 mb-2 block">Target Lags</label>
                                  <TagInput
                                    values={params.lags ?? [1, 7, 14, 30]}
                                    onChange={(lags) => updateModelParams(model.id, { lags } as any)}
                                    placeholder="Type a lag and press Enter (e.g., 1, 7, 14)"
                                  />
                                  <p className="text-[10px] text-slate-500 mt-1">Press Enter to add a lag period</p>
                                </div>
                                
                                {/* Target Mode */}
                                <div>
                                  <label className="text-xs text-slate-400 mb-2 block">Target Mode</label>
                                  <SegmentedControl
                                    value={params.target_mode ?? 'raw'}
                                    options={[
                                      { value: 'raw', label: 'Raw (predict y)' },
                                      { value: 'residual', label: 'Residual (y - y_lag)' }
                                    ]}
                                    onChange={(value) => updateModelParams(model.id, { target_mode: value } as any)}
                                  />
                                  <p className="text-[10px] text-slate-500 mt-1">Raw = predict target directly, Residual = predict difference</p>
                                </div>
                                
                                {/* Residual Lag */}
                                <div>
                                  <label className="text-xs text-slate-400 mb-2 block">
                                    Residual Lag {params.target_mode === 'residual' ? '' : '(disabled)'}
                                  </label>
                                  {params.target_mode === 'residual' ? (
                                    <SingleNumberInput
                                      value={params.residual_lag ?? 1}
                                      min={0}
                                      onChange={(value) => updateModelParams(model.id, { residual_lag: value } as any)}
                                      placeholder="1"
                                    />
                                  ) : (
                                    <div className="glass-input w-full p-2 rounded-lg text-sm opacity-40 text-slate-600 text-center">Disabled</div>
                                  )}
                                  <p className="text-[10px] text-slate-500 mt-1">Which lag to subtract</p>
                                </div>
                                
                                {/* XGBoost Specific Params */}
                                <div>
                                  <Slider
                                    value={params.n_estimators ?? 100}
                                    min={10}
                                    max={1000}
                                    step={10}
                                    onChange={(value) => updateModelParams(model.id, { n_estimators: value } as any)}
                                    label="N Estimators (trees in ensemble)"
                                  />
                                </div>
                                <div>
                                  <Slider
                                    value={params.max_depth ?? 6}
                                    min={1}
                                    max={20}
                                    step={1}
                                    onChange={(value) => updateModelParams(model.id, { max_depth: value } as any)}
                                    label="Max Depth (tree complexity)"
                                  />
                                </div>
                                <div>
                                  <label className="text-xs text-slate-400 mb-2 block">Learning Rate</label>
                                  <div className="flex flex-col sm:flex-row gap-2">
                                    <input 
                                      type="text" 
                                      inputMode="decimal"
                                      value={params.learning_rate ?? 0.1}
                                      onChange={(e) => {
                                        const val = parseFloat(e.target.value);
                                        if (!isNaN(val) && val > 0 && val <= 1) {
                                          updateModelParams(model.id, { learning_rate: val } as any);
                                        }
                                      }}
                                      className="glass-input flex-1 p-2 rounded-lg text-sm" 
                                    />
                                    <select
                                      value=""
                                      onChange={(e) => e.target.value && updateModelParams(model.id, { learning_rate: parseFloat(e.target.value) } as any)}
                                      className="glass-input px-2 rounded-lg text-xs bg-slate-900 text-white [&>option]:bg-slate-900 [&>option]:text-white"
                                    >
                                      <option value="">Presets</option>
                                      <option value="0.001">0.001 (slow)</option>
                                      <option value="0.01">0.01 (medium)</option>
                                      <option value="0.1">0.1 (fast)</option>
                                      <option value="0.3">0.3 (aggressive)</option>
                                    </select>
                                  </div>
                                </div>
                                
                                {/* Feature Configuration Panel */}
                                <div className="col-span-2">
                                  <FeatureConfigPanel 
                                    model={model} 
                                    updateModelParams={updateModelParams} 
                                    availableColumns={availableColumns} 
                                  />
                                </div>
                              </>
                              );
                            })()}

                            {model.type === 'ARIMA' && (() => {
                              const params = model.params as any;
                              return (
                              <>
                                <div>
                                  <Slider
                                    value={params.p ?? 1}
                                    min={0}
                                    max={10}
                                    step={1}
                                    onChange={(value) => updateModelParams(model.id, { p: value } as any)}
                                    label="P (Auto-regressive order)"
                                  />
                                </div>
                                <div>
                                  <Slider
                                    value={params.d ?? 1}
                                    min={0}
                                    max={3}
                                    step={1}
                                    onChange={(value) => updateModelParams(model.id, { d: value } as any)}
                                    label="D (Differencing order)"
                                  />
                                </div>
                                <div>
                                  <Slider
                                    value={params.q ?? 1}
                                    min={0}
                                    max={10}
                                    step={1}
                                    onChange={(value) => updateModelParams(model.id, { q: value } as any)}
                                    label="Q (Moving average order)"
                                  />
                                </div>
                                <div className="col-span-2 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                                  <p className="text-[10px] text-blue-300">
                                    💡 ARIMA is a univariate model - it uses only the target variable's history. 
                                    Common values: (1,1,1) for simple series, (5,1,0) for AR-heavy, (0,1,1) for MA-heavy.
                                  </p>
                                </div>
                              </>
                              );
                            })()}

                            {model.type === 'PROPHET' && (() => {
                              const params = model.params as any;
                              return (
                              <>
                                {/* Lag Regressors - Key for predictive performance */}
                                <div className="col-span-2">
                                  <div className="flex items-center justify-between mb-2">
                                    <label className="text-xs text-slate-400">Lag Regressors</label>
                                    <label className="flex items-center gap-2 cursor-pointer">
                                      <input 
                                        type="checkbox" 
                                        checked={params.use_lag_regressors ?? true}
                                        onChange={(e) => updateModelParams(model.id, { use_lag_regressors: e.target.checked } as any)}
                                        className="accent-amber-500 w-4 h-4" 
                                      />
                                      <span className="text-xs text-slate-300">Enable</span>
                                    </label>
                                  </div>
                                  {params.use_lag_regressors !== false ? (
                                    <TagInput
                                      values={params.lag_regressors ?? [1, 7]}
                                      onChange={(lags) => updateModelParams(model.id, { lag_regressors: lags } as any)}
                                      placeholder="Type a lag and press Enter (e.g., 1, 7, 14)"
                                    />
                                  ) : (
                                    <div className="glass-input w-full p-2 rounded-lg text-sm opacity-40 text-slate-600">Disabled</div>
                                  )}
                                  <p className="text-[10px] text-slate-500 mt-1">Past values to use as regressors (e.g., 1 = yesterday, 7 = last week)</p>
                                </div>
                                
                                {/* Seasonality Checkboxes */}
                                <div className="col-span-2">
                                  <label className="text-xs text-slate-400 mb-2 block">Seasonality Components</label>
                                  <div className="grid grid-cols-3 gap-3">
                                    <label className="flex items-center gap-2 p-3 bg-black/20 rounded-lg border border-white/5 cursor-pointer hover:border-amber-500/30">
                                      <input 
                                        type="checkbox" 
                                        checked={params.daily_seasonality ?? false}
                                        onChange={(e) => updateModelParams(model.id, { daily_seasonality: e.target.checked } as any)}
                                        className="accent-amber-500 w-4 h-4" 
                                      />
                                      <span className="text-sm text-slate-300">Daily</span>
                                    </label>
                                    <label className="flex items-center gap-2 p-3 bg-black/20 rounded-lg border border-white/5 cursor-pointer hover:border-amber-500/30">
                                      <input 
                                        type="checkbox" 
                                        checked={params.weekly_seasonality ?? true}
                                        onChange={(e) => updateModelParams(model.id, { weekly_seasonality: e.target.checked } as any)}
                                        className="accent-amber-500 w-4 h-4" 
                                      />
                                      <span className="text-sm text-slate-300">Weekly</span>
                                    </label>
                                    <label className="flex items-center gap-2 p-3 bg-black/20 rounded-lg border border-white/5 cursor-pointer hover:border-amber-500/30">
                                      <input 
                                        type="checkbox" 
                                        checked={params.yearly_seasonality ?? true}
                                        onChange={(e) => updateModelParams(model.id, { yearly_seasonality: e.target.checked } as any)}
                                        className="accent-amber-500 w-4 h-4" 
                                      />
                                      <span className="text-sm text-slate-300">Yearly</span>
                                    </label>
                                  </div>
                                </div>
                                
                                <div>
                                  <label className="text-xs text-slate-400 mb-2 block">Seasonality Mode</label>
                                  <SegmentedControl
                                    value={params.seasonality_mode ?? 'additive'}
                                    options={[
                                      { value: 'additive', label: 'Additive' },
                                      { value: 'multiplicative', label: 'Multiplicative' }
                                    ]}
                                    onChange={(value) => updateModelParams(model.id, { seasonality_mode: value } as any)}
                                  />
                                  <p className="text-[10px] text-slate-500 mt-1">Use multiplicative for data where seasonal effects grow with trend</p>
                                </div>
                                
                                <div className="col-span-2 p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                                  <p className="text-[10px] text-purple-300">
                                    💡 <strong>Lag regressors are key!</strong> Without them, Prophet only uses seasonality (smooth predictions). 
                                    With lag_1, it uses yesterday's value to predict today → much better for temperature forecasting.
                                  </p>
                                </div>
                              </>
                              );
                            })()}
                          </div>
                        </div>
                      ))
                    )}
                  </div>

                  <div className="mt-4 lg:mt-6 pt-4 lg:pt-6 border-t border-white/10 flex justify-between items-center">
                    <button onClick={() => setStep(1)} className="text-slate-400 hover:text-white transition-colors text-sm sm:text-base">Back</button>
                    <button 
                      onClick={startTraining}
                      disabled={selectedModels.length === 0}
                      className="flex items-center gap-2 px-4 py-2 sm:px-8 sm:py-3 bg-amber-500 hover:bg-amber-400 disabled:opacity-50 disabled:cursor-not-allowed text-black font-bold rounded-lg shadow-[0_0_20px_rgba(245,158,11,0.3)] transition-all text-sm sm:text-base"
                    >
                      <Play size={16} fill="currentColor" /> Start Training
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* STEP 3: RESULTS */}
            {step === 3 && (
              <div className="h-full animate-in fade-in slide-in-from-bottom-4 duration-500">
                {isTraining ? (
                  <div className="h-full flex flex-col items-center justify-center text-center px-4">
                    <div className="relative w-16 h-16 sm:w-24 sm:h-24 mb-6 sm:mb-8">
                      <div className="absolute inset-0 border-4 border-white/10 rounded-full"></div>
                      <div className="absolute inset-0 border-4 border-t-amber-500 rounded-full animate-spin"></div>
                      <Activity className="absolute inset-0 m-auto text-amber-500 w-6 h-6 sm:w-8 sm:h-8 animate-pulse" />
                    </div>
                    <h2 className="text-xl sm:text-2xl font-bold text-white mb-2">Training Models</h2>
                    <p className="text-slate-400 text-sm sm:text-base max-w-md">
                      Optimizing hyperparameters and generating forecasts. This might take a moment depending on your dataset size.
                    </p>
                  </div>
                ) : (
                  <div className="h-full flex flex-col gap-4 sm:gap-6">
                    {/* Best Model Card */}
                    {results.length > 0 && (() => {
                      // Find the model with minimum RMSE
                      const bestModel = results.reduce((best, current) => 
                        current.metrics.rmse < best.metrics.rmse ? current : best
                      , results[0]);
                      
                      return (
                        <div className="bg-gradient-to-r from-amber-500/20 to-orange-600/20 border border-amber-500/30 rounded-xl p-4 sm:p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                          <div className="flex items-center gap-3 sm:gap-4">
                            <div className="w-10 h-10 sm:w-12 sm:h-12 bg-amber-500 rounded-lg flex items-center justify-center text-black shadow-lg shadow-amber-500/20">
                              <Trophy size={20} fill="currentColor" />
                            </div>
                            <div>
                              <p className="text-amber-500 text-[10px] sm:text-xs font-bold uppercase tracking-wider">Best Performing Model</p>
                              <h3 className="text-lg sm:text-2xl font-bold text-white">{bestModel.model_name}</h3>
                            </div>
                          </div>
                          <div className="flex gap-6 sm:gap-8 sm:text-right">
                            <div>
                              <p className="text-slate-400 text-[10px] sm:text-xs uppercase">RMSE</p>
                              <p className="text-lg sm:text-xl font-mono font-bold text-white">{bestModel.metrics.rmse.toFixed(2)}</p>
                            </div>
                            <div>
                              <p className="text-slate-400 text-[10px] sm:text-xs uppercase">R² Score</p>
                              <p className="text-lg sm:text-xl font-mono font-bold text-emerald-400">{(bestModel.metrics.r2 * 100).toFixed(1)}%</p>
                            </div>
                          </div>
                        </div>
                      );
                    })()}

                    {/* Prediction Visualization */}
                    {results.length > 0 && data && (
                      <div className="bg-white/5 border border-white/10 rounded-xl p-3 sm:p-4">
                        <div className="flex justify-between items-center mb-3 sm:mb-4">
                          <h3 className="font-semibold text-sm sm:text-base text-white">Forecast Visualization</h3>
                          <button 
                            onClick={exportForecastsToCSV}
                            className="text-xs flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-400 hover:bg-amber-500/20 hover:text-amber-300 border border-amber-500/20 transition-all"
                          >
                            <Download size={14} /> Export CSV
                          </button>
                        </div>
                        <TimeSeriesChart 
                          data={results[0].forecast} 
                          dateColumn={data.dateColumn} 
                          targetColumn={data.targetColumn}
                          predictions={results.filter(r => !r.error).map(r => ({
                            name: r.model_name,
                            data: r.forecast
                          }))}
                          title="Model Comparison"
                        />
                      </div>
                    )}

                    {/* Metrics Table - Desktop */}
                    <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden hidden sm:block">
                      <div className="p-3 sm:p-4 border-b border-white/10">
                        <h3 className="font-semibold text-sm sm:text-base text-white">Model Leaderboard</h3>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm text-left text-slate-400">
                          <thead className="text-xs text-slate-200 uppercase bg-white/5">
                            <tr>
                              <th className="px-4 sm:px-6 py-3">Model</th>
                              <th className="px-4 sm:px-6 py-3">RMSE</th>
                              <th className="px-4 sm:px-6 py-3">MAE</th>
                              <th className="px-4 sm:px-6 py-3">MAPE</th>
                              <th className="px-4 sm:px-6 py-3">Time</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(() => {
                              const bestRmse = Math.min(...results.map(r => r.metrics.rmse));
                              return results.map((res) => {
                                const isBest = res.metrics.rmse === bestRmse;
                                
                                // Show error row if model failed
                                if (res.error) {
                                  return (
                                    <tr key={res.model_id} className="border-b border-white/5 bg-red-500/5">
                                      <td className="px-4 sm:px-6 py-3 font-medium text-red-400">{res.model_name}</td>
                                      <td colSpan={4} className="px-4 sm:px-6 py-3 text-xs sm:text-sm text-red-300">
                                        ❌ {res.error}
                                      </td>
                                    </tr>
                                  );
                                }
                                
                                return (
                                  <tr key={res.model_id} className={`border-b border-white/5 hover:bg-white/5 ${isBest ? 'bg-amber-500/5' : ''}`}>
                                    <td className="px-4 sm:px-6 py-3 font-medium text-white flex items-center gap-2">
                                      {isBest && <Trophy size={12} className="text-amber-500" />}
                                      <span className="truncate max-w-[100px] sm:max-w-none">{res.model_name}</span>
                                    </td>
                                    <td className="px-4 sm:px-6 py-3 font-mono text-xs sm:text-sm">{res.metrics.rmse.toFixed(2)}</td>
                                    <td className="px-4 sm:px-6 py-3 font-mono text-xs sm:text-sm">{res.metrics.mae.toFixed(2)}</td>
                                    <td className="px-4 sm:px-6 py-3 font-mono text-xs sm:text-sm">{(res.metrics.mape * 100).toFixed(1)}%</td>
                                    <td className="px-4 sm:px-6 py-3 font-mono text-xs sm:text-sm flex items-center gap-1">
                                      <Timer size={12} /> {res.metrics.execution_time.toFixed(1)}s
                                    </td>
                                  </tr>
                                );
                              });
                            })()}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Metrics Cards - Mobile */}
                    <div className="sm:hidden space-y-3">
                      <h3 className="font-semibold text-sm text-white">Model Leaderboard</h3>
                      {(() => {
                        const bestRmse = Math.min(...results.map(r => r.metrics.rmse));
                        return results.map((res) => {
                          const isBest = res.metrics.rmse === bestRmse;
                          
                          // Show error if model failed
                          if (res.error) {
                            return (
                              <div key={res.model_id} className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                                <div className="flex items-center gap-2 mb-2">
                                  <span className="font-medium text-red-400 text-sm">{res.model_name}</span>
                                </div>
                                <p className="text-xs text-red-300">❌ {res.error}</p>
                              </div>
                            );
                          }
                          
                          return (
                            <div key={res.model_id} className={`bg-white/5 border rounded-lg p-3 ${isBest ? 'border-amber-500/30 bg-amber-500/5' : 'border-white/10'}`}>
                              <div className="flex items-center gap-2 mb-2">
                                {isBest && <Trophy size={14} className="text-amber-500" />}
                                <span className="font-medium text-white text-sm">{res.model_name}</span>
                              </div>
                              <div className="grid grid-cols-4 gap-2 text-xs">
                                <div>
                                  <p className="text-slate-500">RMSE</p>
                                  <p className="font-mono text-white">{res.metrics.rmse.toFixed(1)}</p>
                                </div>
                                <div>
                                  <p className="text-slate-500">MAE</p>
                                  <p className="font-mono text-white">{res.metrics.mae.toFixed(1)}</p>
                                </div>
                                <div>
                                  <p className="text-slate-500">MAPE</p>
                                  <p className="font-mono text-white">{(res.metrics.mape * 100).toFixed(0)}%</p>
                                </div>
                                <div>
                                  <p className="text-slate-500">Time</p>
                                  <p className="font-mono text-white">{res.metrics.execution_time.toFixed(1)}s</p>
                                </div>
                              </div>
                            </div>
                          );
                        });
                      })()}
                    </div>

                    {/* Feature Importance Section */}
                    {results.some(r => r.feature_importance && r.feature_importance.length > 0) && (
                      <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                        <div className="p-4 border-b border-white/10">
                          <h3 className="font-semibold text-white">Feature Importance</h3>
                        </div>
                        <div className="p-4 space-y-4">
                          {results.filter(r => r.feature_importance && r.feature_importance.length > 0).map(res => (
                            <div key={res.model_id} className="space-y-2">
                              <h4 className="text-sm font-medium text-slate-300">{res.model_name}</h4>
                              <div className="space-y-2">
                                {res.feature_importance?.map((fi, idx) => {
                                  const maxImportance = Math.max(...(res.feature_importance?.map(f => f.importance) || [1]));
                                  const widthPercent = (fi.importance / maxImportance) * 100;
                                  return (
                                    <div key={fi.feature} className="flex items-center gap-3">
                                      <span className="text-xs text-slate-400 w-20 truncate">{fi.feature}</span>
                                      <div className="flex-1 bg-black/30 rounded-full h-4 overflow-hidden">
                                        <div 
                                          className={`h-full rounded-full ${idx === 0 ? 'bg-amber-500' : 'bg-slate-500'}`}
                                          style={{ width: `${widthPercent}%` }}
                                        />
                                      </div>
                                      <span className="text-xs font-mono text-slate-300 w-16 text-right">
                                        {fi.importance.toFixed(3)}
                                      </span>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="flex justify-end pt-4">
                      <button 
                        onClick={() => setStep(1)}
                        className="text-slate-400 hover:text-white transition-colors mr-4"
                      >
                        Start New Experiment
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  );
}
