'use client';

import { useState, useRef, useEffect } from 'react';
import { ModelConfig, ModelType, TimeSeriesData, ModelResult, DateRange, ShapValue, ColumnInfo } from '../types/forecasting';
import { Upload, Activity, BarChart3, Settings, ChevronDown, FileText, CheckCircle2, Trophy, Timer, Download, TrendingUp, TrendingDown, Info } from 'lucide-react';
import Papa from 'papaparse';

// Components
import TimeSeriesChart from '../components/TimeSeriesChart';
import StrategyStep from '../components/StrategyStep';
import { ShapChart } from '../components/charts';

// Utilities
import { getApiUrl, getApiHeaders, API_MODE, BACKEND_URL } from '../lib/api';
import { formatFeatureName, getShapKeyForFeature } from '../lib/formatters';

// Debug: log API config (check browser console)
if (typeof window !== 'undefined') {
  console.log('🔗 API Mode:', API_MODE);
  console.log('🔗 Backend URL:', BACKEND_URL);
}

export default function ForecastingPage() {
  const [step, setStep] = useState<number>(1);
  const [selectedModels, setSelectedModels] = useState<ModelConfig[]>([]);
  
  // Strategy State
  const [trainingRanges, setTrainingRanges] = useState<DateRange[]>([]);
  const [predictionRanges, setPredictionRanges] = useState<DateRange[]>([]);
  
  // Forecast Strategy State
  const [forecastHorizon, setForecastHorizon] = useState<number>(1);
  const [defaultLags, setDefaultLags] = useState<number[]>([1, 7]);
  
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
  
  // SHAP Modal State
  const [shapModal, setShapModal] = useState<{ 
    isOpen: boolean; 
    shapValues: ShapValue[]; 
    featureName: string; 
  }>({ isOpen: false, shapValues: [], featureName: '' });
  
  // Delta mode for forecast visualization
  const [isDeltaMode, setIsDeltaMode] = useState(false);

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
      // Get min lag from models to determine forecast mode
      const allLags: number[] = selectedModels.flatMap(m => {
        const params = m.params as any;
        return params.feature_config?.target_lags || params.lags || params.lag ? [params.lag] : [1];
      });
      const minLag = allLags.length > 0 ? Math.min(...allLags.filter(l => l > 0)) : 1;
      const forecastMode = forecastHorizon <= minLag ? 'direct' : 'recursive';
      
      // Use rawData for training (contains all columns including exogenous)
      const payload = {
        data: rawData,
        data_config: {
          target_column: data.targetColumn,
          date_column: data.dateColumn,
          frequency: data.frequency,
          training_ranges: trainingRanges,
          prediction_ranges: predictionRanges,
          forecast_strategy: {
            horizon: forecastHorizon,
            mode: forecastMode
          }
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
    
    // Default params per model type - ML models use defaultLags from strategy
    const defaultParams: Record<ModelType, any> = {
      'LAG': { lag: defaultLags[0] || 1 },
      'LINEAR_REGRESSION': { lags: [...defaultLags] },
      'XGBOOST': { lags: [...defaultLags], n_estimators: 100, max_depth: 3, learning_rate: 0.1 },
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

            {/* STEP 2: STRATEGY & MODELS */}
            {step === 2 && (
              <StrategyStep
                data={data}
                fullData={fullData}
                availableColumns={availableColumns}
                selectedModels={selectedModels}
                setSelectedModels={setSelectedModels}
                addModel={addModel}
                updateModelParams={updateModelParams}
                updateModelName={updateModelName}
                trainingRanges={trainingRanges}
                setTrainingRanges={setTrainingRanges}
                predictionRanges={predictionRanges}
                setPredictionRanges={setPredictionRanges}
                forecastHorizon={forecastHorizon}
                setForecastHorizon={setForecastHorizon}
                defaultLags={defaultLags}
                setDefaultLags={setDefaultLags}
                onBack={() => setStep(1)}
                onStartTraining={startTraining}
              />
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
                        <div className="flex justify-between items-center mb-3 sm:mb-4 gap-2">
                          <h3 className="font-semibold text-sm sm:text-base text-white">Forecast Visualization</h3>
                          <div className="flex items-center gap-2">
                            {/* Delta Mode Toggle */}
                            <div className="flex items-center gap-1.5 bg-black/30 rounded-lg p-1 border border-white/10">
                              <button
                                onClick={() => setIsDeltaMode(false)}
                                className={`px-2.5 py-1 text-[10px] sm:text-xs rounded transition-all ${
                                  !isDeltaMode ? 'bg-amber-500 text-black font-medium' : 'text-slate-400 hover:text-slate-200'
                                }`}
                              >
                                Values
                              </button>
                              <button
                                onClick={() => setIsDeltaMode(true)}
                                className={`px-2.5 py-1 text-[10px] sm:text-xs rounded transition-all ${
                                  isDeltaMode ? 'bg-amber-500 text-black font-medium' : 'text-slate-400 hover:text-slate-200'
                                }`}
                              >
                                Deltas
                              </button>
                            </div>
                            <button 
                              onClick={exportForecastsToCSV}
                              className="text-xs flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-400 hover:bg-amber-500/20 hover:text-amber-300 border border-amber-500/20 transition-all"
                            >
                              <Download size={14} /> <span className="hidden sm:inline">Export CSV</span>
                            </button>
                          </div>
                        </div>
                        <TimeSeriesChart 
                          data={isDeltaMode ? results[0].forecast.map(row => {
                            // In delta mode, show zero line for actual
                            const rowData = row as any;
                            return { ...rowData, [data.targetColumn]: 0 };
                          }) : results[0].forecast}
                          dateColumn={data.dateColumn} 
                          targetColumn={data.targetColumn}
                          predictions={results.filter(r => !r.error).map(r => ({
                            name: r.model_name,
                            data: isDeltaMode ? r.forecast.map(row => {
                              // Calculate delta: prediction - actual
                              const rowData = row as any;
                              const actual = rowData[data.targetColumn] || 0;
                              const pred = rowData['prediction'] !== undefined ? rowData['prediction'] : actual;
                              const delta = pred - actual;
                              return { ...rowData, prediction: delta };
                            }) : r.forecast
                          }))}
                          title={isDeltaMode ? "Prediction Errors (Pred - Actual)" : "Model Comparison"}
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
                              <th className="px-4 sm:px-6 py-3">MSLE</th>
                              <th className="px-4 sm:px-6 py-3">R²</th>
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
                                      <td colSpan={6} className="px-4 sm:px-6 py-3 text-xs sm:text-sm text-red-300">
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
                                    <td className="px-4 sm:px-6 py-3 font-mono text-xs sm:text-sm">{res.metrics.msle?.toFixed(4) ?? 'N/A'}</td>
                                    <td className="px-4 sm:px-6 py-3 font-mono text-xs sm:text-sm">{(res.metrics.r2 * 100).toFixed(1)}%</td>
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
                              <div className="grid grid-cols-3 gap-2 text-xs mb-2">
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
                              </div>
                              <div className="grid grid-cols-3 gap-2 text-xs">
                                <div>
                                  <p className="text-slate-500">MSLE</p>
                                  <p className="font-mono text-white">{res.metrics.msle?.toFixed(4) ?? 'N/A'}</p>
                                </div>
                                <div>
                                  <p className="text-slate-500">R²</p>
                                  <p className="font-mono text-white">{(res.metrics.r2 * 100).toFixed(0)}%</p>
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

                    {/* Horizon Metrics Section - Collapsible, collapsed by default */}
                    {results.some(r => r.metrics_by_horizon && r.metrics_by_horizon.length > 1) && (
                      <details className="bg-white/5 border border-white/10 rounded-xl overflow-hidden group">
                        <summary className="p-3 sm:p-4 cursor-pointer hover:bg-white/5 transition-colors flex items-center justify-between">
                          <h3 className="font-semibold text-sm sm:text-base text-white flex items-center gap-2">
                            <TrendingUp size={16} className="text-blue-400" />
                            Metrics by Horizon Step
                          </h3>
                          <ChevronDown size={16} className="text-slate-400 group-open:rotate-180 transition-transform" />
                        </summary>
                        <div className="p-3 sm:p-4 border-t border-white/10 space-y-4">
                          {results.filter(r => r.metrics_by_horizon && r.metrics_by_horizon.length > 1 && !r.error).map(res => {
                            const horizonMetrics = res.metrics_by_horizon!;
                            const rmseValues = horizonMetrics.map(h => h.rmse);
                            const minRmse = Math.min(...rmseValues);
                            const maxRmse = Math.max(...rmseValues);
                            const range = maxRmse - minRmse;
                            const maxBarHeight = 48; // pixels (total height minus labels space)
                            
                            return (
                              <div key={res.model_id} className="space-y-2">
                                <h4 className="text-sm font-medium text-slate-300">{res.model_name}</h4>
                                {/* Compact horizontal bar chart with amplified scale */}
                                <div className="flex items-end gap-1 h-20 bg-black/20 rounded-lg p-2 pt-4">
                                  {horizonMetrics.map(hm => {
                                    // Scale from 20% (min) to 100% (max) to show differences clearly
                                    const normalizedRatio = range > 0 
                                      ? 0.2 + ((hm.rmse - minRmse) / range) * 0.8 
                                      : 0.5;
                                    const barHeight = Math.round(normalizedRatio * maxBarHeight);
                                    return (
                                      <div 
                                        key={hm.horizon_step}
                                        className="flex-1 flex flex-col items-center justify-end h-full"
                                        title={`h=${hm.horizon_step}: RMSE=${hm.rmse.toFixed(2)}, MAE=${hm.mae.toFixed(2)}`}
                                      >
                                        <span className="text-[7px] text-slate-400 mb-0.5">{hm.rmse.toFixed(1)}</span>
                                        <div 
                                          className="w-full bg-gradient-to-t from-blue-500 to-blue-400 rounded-t transition-all"
                                          style={{ height: `${barHeight}px` }}
                                        />
                                        <span className="text-[8px] text-slate-500 mt-0.5">h{hm.horizon_step}</span>
                                      </div>
                                    );
                                  })}
                                </div>
                                <div className="flex justify-between text-[10px] text-slate-500">
                                  <span>h=1: RMSE {horizonMetrics[0]?.rmse.toFixed(2)}</span>
                                  <span>h={horizonMetrics.length}: RMSE {horizonMetrics[horizonMetrics.length - 1]?.rmse.toFixed(2)}</span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </details>
                    )}

                    {/* Feature Importance Section */}
                    {results.some(r => r.feature_importance && r.feature_importance.length > 0) && (
                      <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                        <div className="p-4 border-b border-white/10">
                          <h3 className="font-semibold text-white flex items-center gap-2">
                            <BarChart3 size={16} className="text-amber-500" />
                            Feature Importance
                          </h3>
                        </div>
                        <div className="p-4 space-y-6">
                          {results.filter(r => r.feature_importance && r.feature_importance.length > 0).map(res => {
                            const maxImportance = Math.max(...(res.feature_importance?.map(f => f.importance) || [1]));
                            const hasShapAnalysis = res.shap_analysis && Object.keys(res.shap_analysis.temporal).length > 0;
                            
                            return (
                              <div key={res.model_id} className="space-y-3">
                                <div className="flex items-center justify-between">
                                  <h4 className="text-sm font-medium text-slate-300 flex items-center gap-2">
                                    {res.model_name}
                                    {hasShapAnalysis && (
                                      <span className="px-2 py-0.5 text-[10px] bg-amber-500/20 text-amber-400 rounded-full">
                                        SHAP
                                      </span>
                                    )}
                                  </h4>
                                </div>
                                
                                <div className="space-y-1.5">
                                  {res.feature_importance?.slice(0, 12).map((fi, idx) => {
                                    const widthPercent = (fi.importance / maxImportance) * 100;
                                    const shapKey = hasShapAnalysis ? getShapKeyForFeature(fi.feature) : null;
                                    const shapData = shapKey && res.shap_analysis?.temporal 
                                      ? res.shap_analysis.temporal[shapKey as keyof typeof res.shap_analysis.temporal] 
                                      : null;
                                    
                                    return (
                                      <div key={fi.feature} className="group flex items-center gap-2">
                                        {/* Feature name with tooltip */}
                                        <div className="flex items-center gap-1.5 w-28 sm:w-36">
                                          <span 
                                            className="text-xs text-slate-400 truncate" 
                                            title={fi.feature}
                                          >
                                            {formatFeatureName(fi.feature)}
                                          </span>
                                          {shapData && shapKey && (
                                            <button
                                              onClick={() => {
                                                setShapModal({
                                                  isOpen: true,
                                                  shapValues: shapData,
                                                  featureName: shapKey
                                                });
                                              }}
                                              className="p-0.5 text-amber-500/60 hover:text-amber-400 transition-colors"
                                              title="Voir les valeurs SHAP"
                                            >
                                              <Info size={12} />
                                            </button>
                                          )}
                                        </div>
                                        
                                        {/* Bar */}
                                        <div className="flex-1 bg-black/30 rounded h-5 overflow-hidden">
                                          <div 
                                            className={`h-full rounded transition-all ${
                                              idx === 0 ? 'bg-gradient-to-r from-amber-500 to-amber-400' : 
                                              idx < 3 ? 'bg-amber-500/60' : 'bg-slate-600'
                                            }`}
                                            style={{ width: `${widthPercent}%` }}
                                          />
                                        </div>
                                        
                                        {/* Value */}
                                        <span className="text-xs font-mono text-slate-400 w-14 text-right">
                                          {(fi.importance * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                    );
                                  })}
                                  
                                  {/* Show more indicator */}
                                  {(res.feature_importance?.length || 0) > 12 && (
                                    <p className="text-xs text-slate-600 text-center pt-2">
                                      + {(res.feature_importance?.length || 0) - 12} more features
                                    </p>
                                  )}
                                </div>
                                
                                {/* SHAP Temporal Summary (if available) */}
                                {hasShapAnalysis && (
                                  <div className="mt-4 pt-3 border-t border-white/5">
                                    <p className="text-xs text-slate-500 mb-2 flex items-center gap-1.5">
                                      <TrendingUp size={12} />
                                      Click on <Info size={10} className="inline text-amber-500" /> to explore temporal impact
                                    </p>
                                    <div className="flex flex-wrap gap-2">
                                      {Object.entries(res.shap_analysis!.temporal).map(([key, values]) => {
                                        const featureLabels: Record<string, string> = {
                                          day_of_week: 'Day',
                                          month: 'Month',
                                          hour_of_day: 'Hour',
                                          minute_of_day: 'Minute'
                                        };
                                        const positiveCount = (values as ShapValue[]).filter(v => v.shap_norm > 0.3).length;
                                        const negativeCount = (values as ShapValue[]).filter(v => v.shap_norm < -0.3).length;
                                        
                                        return (
                                          <button
                                            key={key}
                                            onClick={() => setShapModal({
                                              isOpen: true,
                                              shapValues: values as ShapValue[],
                                              featureName: key
                                            })}
                                            className="flex items-center gap-2 px-3 py-1.5 bg-black/30 hover:bg-black/50 border border-white/10 hover:border-amber-500/30 rounded-lg transition-all text-xs"
                                          >
                                            <span className="text-slate-300">{featureLabels[key] || key}</span>
                                            <span className="flex items-center gap-1 text-[10px]">
                                              {positiveCount > 0 && (
                                                <span className="text-emerald-400 flex items-center">
                                                  <TrendingUp size={10} />{positiveCount}
                                                </span>
                                              )}
                                              {negativeCount > 0 && (
                                                <span className="text-red-400 flex items-center">
                                                  <TrendingDown size={10} />{negativeCount}
                                                </span>
                                              )}
                                            </span>
                                          </button>
                                        );
                                      })}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    
                    {/* SHAP Modal */}
                    {shapModal.isOpen && (
                      <ShapChart
                        shapValues={shapModal.shapValues}
                        featureName={shapModal.featureName}
                        onClose={() => setShapModal({ isOpen: false, shapValues: [], featureName: '' })}
                      />
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
