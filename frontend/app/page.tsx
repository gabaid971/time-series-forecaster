'use client';

import { useState, useRef, useEffect } from 'react';
import { ModelConfig, ModelType, TimeSeriesData, ModelResult, DateRange, LinearRegressionParams } from '../types/forecasting';
import { Upload, Activity, BarChart3, Settings, Play, Plus, X, ChevronRight, FileText, CheckCircle2, Trophy, Timer, Download, Trash2 } from 'lucide-react';
import Papa from 'papaparse';
import TimeSeriesChart from '../components/TimeSeriesChart';

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
      const payload = {
        data: fullData,
        data_config: {
          target_column: data.targetColumn,
          date_column: data.dateColumn,
          frequency: data.frequency,
          training_ranges: trainingRanges,
          prediction_ranges: predictionRanges
        },
        models: selectedModels
      };

      const response = await fetch('http://localhost:8000/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
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
  // Backend parses dates, returns normalized data with ISO format
  const analyzeDataset = async () => {
    if (!data || !rawData.length) return;
    
    setIsAnalyzing(true);
    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
        
        // Use normalized data from backend (dates in ISO format, values cleaned)
        if (result.normalized_data && result.normalized_data.length > 0) {
          // Convert to format expected by chart: [{date: 'YYYY-MM-DD', value: number}]
          const normalizedData = result.normalized_data.map((point: { date: string; value: number }) => ({
            [data.dateColumn]: point.date.split('T')[0], // Keep just YYYY-MM-DD
            [data.targetColumn]: point.value
          }));
          setFullData(normalizedData);
          
          // Set default training/prediction ranges from stats
          const dateMin = result.stats.date_min.split('T')[0];
          const dateMax = result.stats.date_max.split('T')[0];
          const totalRows = normalizedData.length;
          const splitIndex = Math.floor(totalRows * 0.8);
          const splitDate = normalizedData[splitIndex]?.[data.dateColumn] || dateMax;
          
          setTrainingRanges([{ start: dateMin, end: splitDate }]);
          setPredictionRanges([{ start: splitDate, end: dateMax }]);
        }
      }
    } catch (error) {
      console.error('Failed to analyze dataset:', error);
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
      'LINEAR_REGRESSION': 'Linear Regression',
      'ARIMA': 'ARIMA',
      'PROPHET': 'Prophet',
      'XGBOOST': 'XGBoost',
      'NBEATS': 'N-BEATS'
    };
    
    const newModel: ModelConfig = {
      id: `${type}-${Date.now()}`,
      type: type,
      name: `${typeNames[type]} ${modelNumber}`,
      params: type === 'LINEAR_REGRESSION'
        ? { lags: [1, 7] }
        : type === 'XGBOOST' 
        ? { lags: [1, 7], rolling_features: [], use_exogenous: true } 
        : type === 'ARIMA'
        ? { p: 1, d: 1, q: 1 } 
        : { daily_seasonality: true, weekly_seasonality: true, yearly_seasonality: true }
    };
    setSelectedModels([...selectedModels, newModel]);
  };

  const updateModelName = (modelId: string, newName: string) => {
    setSelectedModels(prev => prev.map(m => 
      m.id === modelId ? { ...m, name: newName } : m
    ));
  };

  const updateModelParams = (modelId: string, newParams: Partial<ModelConfig['params']>) => {
    setSelectedModels(prev => prev.map(m => 
      m.id === modelId 
        ? { ...m, params: { ...m.params, ...newParams } }
        : m
    ));
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

      <div className="relative z-10 max-w-6xl mx-auto p-6">
        
        {/* Header */}
        <header className="flex items-center justify-between mb-12 pt-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-400 to-orange-600 flex items-center justify-center shadow-lg shadow-amber-500/20">
              <Activity className="text-white w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">Time Series <span className="text-amber-500">Studio</span></h1>
              <p className="text-slate-400 text-xs uppercase tracking-widest">Forecasting Pipeline</p>
            </div>
          </div>
          <div className="flex gap-4 text-sm text-slate-400">
            <span className="hover:text-white cursor-pointer transition">Docs</span>
            <span className="hover:text-white cursor-pointer transition">History</span>
          </div>
        </header>

        {/* Stepper */}
        <div className="mb-12">
          <div className="flex items-center justify-between relative">
            <div className="absolute left-0 top-1/2 w-full h-0.5 bg-white/10 -z-10" />
            {[1, 2, 3].map((i) => (
              <div 
                key={i}
                onClick={() => i < step ? setStep(i) : null}
                className={`relative flex flex-col items-center gap-2 cursor-pointer group ${step === i ? 'scale-110' : 'scale-100'} transition-all duration-300`}
              >
                <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                  step >= i 
                    ? 'bg-amber-500 border-amber-500 text-black shadow-[0_0_20px_rgba(245,158,11,0.4)]' 
                    : 'bg-slate-900 border-white/20 text-slate-500 group-hover:border-white/40'
                }`}>
                  {i === 1 ? <Upload size={18} /> : i === 2 ? <Settings size={18} /> : <BarChart3 size={18} />}
                </div>
                <span className={`text-xs font-medium tracking-wider ${step >= i ? 'text-amber-500' : 'text-slate-600'}`}>
                  {i === 1 ? 'DATA' : i === 2 ? 'STRATEGY' : 'RESULTS'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Main Content Area */}
        <div className="glass-panel rounded-2xl p-1 min-h-[600px] glow-box transition-all duration-500">
          <div className="bg-black/40 rounded-xl min-h-[600px] p-8 backdrop-blur-sm">
            
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
                        className={`border-2 border-dashed rounded-2xl p-16 text-center transition-all cursor-pointer group ${
                          isDragging 
                            ? 'border-amber-500 bg-amber-500/10 scale-105' 
                            : 'border-white/10 hover:border-amber-500/50 hover:bg-white/5'
                        }`}
                      >
                        <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300">
                          <Upload className={`w-10 h-10 transition-colors ${isDragging ? 'text-amber-500' : 'text-slate-400 group-hover:text-amber-500'}`} />
                        </div>
                        <h3 className="text-xl font-semibold text-white mb-2">Upload Time Series</h3>
                        <p className="text-slate-400 mb-8">Drag & drop CSV file here</p>
                        <button className="px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-lg font-medium transition-all border border-white/10">
                          Browse Files
                        </button>
                      </div>
                    </div>
                    <div className="flex gap-4 text-slate-500 text-sm">
                      <span>Supported: .csv</span>
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

                    <div className="grid grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label className="text-sm text-slate-400">Date Column</label>
                        <select 
                          value={data.dateColumn}
                          onChange={(e) => setData({...data, dateColumn: e.target.value})}
                          className="w-full bg-black/20 border border-white/10 rounded-lg p-3 text-white focus:border-amber-500 outline-none [&>option]:bg-slate-900 [&>option]:text-white"
                        >
                          {data.columns.map(col => <option key={col} value={col}>{col}</option>)}
                        </select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm text-slate-400">Target Column</label>
                        <select 
                          value={data.targetColumn}
                          onChange={(e) => setData({...data, targetColumn: e.target.value})}
                          className="w-full bg-black/20 border border-white/10 rounded-lg p-3 text-white focus:border-amber-500 outline-none [&>option]:bg-slate-900 [&>option]:text-white"
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
                        className="flex items-center gap-2 px-8 py-3 bg-amber-500 hover:bg-amber-400 text-black font-bold rounded-lg shadow-[0_0_20px_rgba(245,158,11,0.3)] transition-all"
                      >
                        Confirm Data <CheckCircle2 size={18} />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* STEP 2: MODELS & CONFIG */}
            {step === 2 && (
              <div className="grid grid-cols-12 gap-8 h-full animate-in fade-in slide-in-from-bottom-4 duration-500">
                
                {/* Left: Model Palette */}
                <div className="col-span-4 space-y-4 border-r border-white/10 pr-8">
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-6">Model Library</h3>
                  
                  <div className="space-y-3">
                    {[
                      { id: 'LINEAR_REGRESSION', name: 'Linear Regression', desc: 'Simple Baseline', icon: '📏' },
                      { id: 'ARIMA', name: 'ARIMA', desc: 'Statistical Baseline', icon: '📊' },
                      { id: 'PROPHET', name: 'Prophet', desc: 'Facebook / Meta', icon: '📈' },
                      { id: 'XGBOOST', name: 'XGBoost', desc: 'Gradient Boosting', icon: '🚀' },
                      { id: 'NBEATS', name: 'N-BEATS', desc: 'Deep Learning SOTA', icon: '🧠' },
                    ].map((m) => (
                      <button 
                        key={m.id}
                        onClick={() => addModel(m.id as ModelType)}
                        className="w-full text-left p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 hover:border-amber-500/30 transition-all group"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <span className="text-xl grayscale group-hover:grayscale-0 transition-all">{m.icon}</span>
                            <div>
                              <div className="font-medium text-slate-200 group-hover:text-white">{m.name}</div>
                              <div className="text-xs text-slate-500">{m.desc}</div>
                            </div>
                          </div>
                          <Plus size={16} className="text-slate-600 group-hover:text-amber-500" />
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Right: Configuration */}
                <div className="col-span-8 flex flex-col">
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-6">Pipeline Configuration</h3>
                  
                  {/* Validation Strategy */}
                  <div className="bg-white/5 border border-white/10 rounded-xl p-5 mb-6">
                    <h4 className="font-semibold text-white mb-4 flex items-center gap-2">
                      <Settings size={16} className="text-amber-500" /> Validation Strategy
                    </h4>
                    
                    <div className="grid grid-cols-2 gap-8">
                      {/* Training Ranges */}
                      <div>
                        <div className="flex items-center justify-between mb-3">
                          <label className="text-sm text-slate-400">Training Periods</label>
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
                            {model.type === 'LINEAR_REGRESSION' && (
                              <>
                                <div className="col-span-2">
                                  <label className="text-xs text-slate-400 mb-2 block">Lag Features</label>
                                  <input 
                                    type="text" 
                                    defaultValue={'lags' in model.params ? model.params.lags.join(', ') : '1, 7'}
                                    onBlur={(e) => {
                                      const lags = parseLagsString(e.target.value);
                                      if (lags.length > 0) {
                                        updateModelParams(model.id, { lags });
                                      }
                                    }}
                                    className="glass-input w-full p-2 rounded-lg text-sm" 
                                    placeholder="1, 7, 14"
                                  />
                                  <p className="text-[10px] text-slate-500 mt-1">Comma separated lag periods (e.g., 1, 7, 14)</p>
                                </div>
                                
                                {/* Target Mode */}
                                <div>
                                  <label className="text-xs text-slate-400 mb-2 block">Target Mode</label>
                                  <select
                                    value={(model.params as LinearRegressionParams).target_mode ?? 'raw'}
                                    onChange={(e) => updateModelParams(model.id, { target_mode: e.target.value as 'raw' | 'residual' })}
                                    className="glass-input w-full p-2 rounded-lg text-sm bg-black/30"
                                  >
                                    <option value="raw">Raw (predict y)</option>
                                    <option value="residual">Residual (predict y - y_lag)</option>
                                  </select>
                                  <p className="text-[10px] text-slate-500 mt-1">Raw = predict target directly, Residual = predict difference</p>
                                </div>
                                
                                {/* Residual Lag (only visible when residual mode) */}
                                <div>
                                  <label className="text-xs text-slate-400 mb-2 block">
                                    Residual Lag {(model.params as LinearRegressionParams).target_mode === 'residual' ? '' : '(disabled)'}
                                  </label>
                                  <input 
                                    type="number" 
                                    min="1"
                                    value={(model.params as LinearRegressionParams).residual_lag ?? 1}
                                    onChange={(e) => updateModelParams(model.id, { residual_lag: parseInt(e.target.value) || 1 })}
                                    disabled={(model.params as LinearRegressionParams).target_mode !== 'residual'}
                                    className="glass-input w-full p-2 rounded-lg text-sm disabled:opacity-40 disabled:cursor-not-allowed" 
                                  />
                                  <p className="text-[10px] text-slate-500 mt-1">Which lag to subtract (e.g., 1 = y - y_{'{t-1}'})</p>
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
                                
                                <div className="col-span-2">
                                  <label className="text-xs text-slate-400 mb-2 block">Exogenous Variables</label>
                                  <div className="flex flex-wrap gap-2">
                                    {data?.columns.filter(c => c !== data.dateColumn && c !== data.targetColumn).map(col => (
                                      <label key={col} className="flex items-center gap-2 p-2 bg-black/20 rounded-lg border border-white/5 cursor-pointer hover:border-amber-500/30">
                                        <input type="checkbox" className="accent-amber-500 w-4 h-4" />
                                        <span className="text-sm text-slate-300">{col}</span>
                                      </label>
                                    ))}
                                  </div>
                                  <p className="text-[10px] text-slate-500 mt-1">Select additional features to include</p>
                                </div>
                              </>
                            )}

                            {model.type === 'XGBOOST' && (
                              <>
                                <div className="col-span-2">
                                  <label className="text-xs text-slate-400 mb-2 block">Lag Features</label>
                                  <input 
                                    type="text" 
                                    defaultValue={'lags' in model.params ? model.params.lags.join(', ') : '1, 7, 14, 30'}
                                    onBlur={(e) => {
                                      const lags = parseLagsString(e.target.value);
                                      if (lags.length > 0) {
                                        updateModelParams(model.id, { lags });
                                      }
                                    }}
                                    className="glass-input w-full p-2 rounded-lg text-sm" 
                                  />
                                  <p className="text-[10px] text-slate-500 mt-1">Comma separated lag periods</p>
                                </div>
                                <div className="flex items-center gap-3 p-3 bg-black/20 rounded-lg border border-white/5">
                                  <input type="checkbox" defaultChecked className="accent-amber-500 w-4 h-4" />
                                  <span className="text-sm text-slate-300">Use Exogenous Features</span>
                                </div>
                              </>
                            )}

                            {model.type === 'ARIMA' && (
                              <>
                                <div>
                                  <label className="text-xs text-slate-400 mb-2 block">P (Auto-regressive)</label>
                                  <input type="number" defaultValue={1} className="glass-input w-full p-2 rounded-lg text-sm" />
                                </div>
                                <div>
                                  <label className="text-xs text-slate-400 mb-2 block">D (Integrated)</label>
                                  <input type="number" defaultValue={1} className="glass-input w-full p-2 rounded-lg text-sm" />
                                </div>
                                <div>
                                  <label className="text-xs text-slate-400 mb-2 block">Q (Moving Avg)</label>
                                  <input type="number" defaultValue={1} className="glass-input w-full p-2 rounded-lg text-sm" />
                                </div>
                              </>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                  </div>

                  <div className="mt-6 pt-6 border-t border-white/10 flex justify-between items-center">
                    <button onClick={() => setStep(1)} className="text-slate-400 hover:text-white transition-colors">Back</button>
                    <button 
                      onClick={startTraining}
                      disabled={selectedModels.length === 0}
                      className="flex items-center gap-2 px-8 py-3 bg-amber-500 hover:bg-amber-400 disabled:opacity-50 disabled:cursor-not-allowed text-black font-bold rounded-lg shadow-[0_0_20px_rgba(245,158,11,0.3)] transition-all"
                    >
                      <Play size={18} fill="currentColor" /> Start Training
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* STEP 3: RESULTS */}
            {step === 3 && (
              <div className="h-full animate-in fade-in slide-in-from-bottom-4 duration-500">
                {isTraining ? (
                  <div className="h-full flex flex-col items-center justify-center text-center">
                    <div className="relative w-24 h-24 mb-8">
                      <div className="absolute inset-0 border-4 border-white/10 rounded-full"></div>
                      <div className="absolute inset-0 border-4 border-t-amber-500 rounded-full animate-spin"></div>
                      <Activity className="absolute inset-0 m-auto text-amber-500 w-8 h-8 animate-pulse" />
                    </div>
                    <h2 className="text-2xl font-bold text-white mb-2">Training Models</h2>
                    <p className="text-slate-400 max-w-md">
                      Optimizing hyperparameters and generating forecasts. This might take a moment depending on your dataset size.
                    </p>
                  </div>
                ) : (
                  <div className="h-full flex flex-col gap-6">
                    {/* Best Model Card */}
                    {results.length > 0 && (() => {
                      // Find the model with minimum RMSE
                      const bestModel = results.reduce((best, current) => 
                        current.metrics.rmse < best.metrics.rmse ? current : best
                      , results[0]);
                      
                      return (
                        <div className="bg-gradient-to-r from-amber-500/20 to-orange-600/20 border border-amber-500/30 rounded-xl p-6 flex items-center justify-between">
                          <div className="flex items-center gap-4">
                            <div className="w-12 h-12 bg-amber-500 rounded-lg flex items-center justify-center text-black shadow-lg shadow-amber-500/20">
                              <Trophy size={24} fill="currentColor" />
                            </div>
                            <div>
                              <p className="text-amber-500 text-xs font-bold uppercase tracking-wider">Best Performing Model</p>
                              <h3 className="text-2xl font-bold text-white">{bestModel.model_name}</h3>
                            </div>
                          </div>
                          <div className="flex gap-8 text-right">
                            <div>
                              <p className="text-slate-400 text-xs uppercase">RMSE</p>
                              <p className="text-xl font-mono font-bold text-white">{bestModel.metrics.rmse.toFixed(2)}</p>
                            </div>
                            <div>
                              <p className="text-slate-400 text-xs uppercase">R² Score</p>
                              <p className="text-xl font-mono font-bold text-emerald-400">{(bestModel.metrics.r2 * 100).toFixed(1)}%</p>
                            </div>
                          </div>
                        </div>
                      );
                    })()}

                    {/* Prediction Visualization */}
                    {results.length > 0 && data && (
                      <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                        <h3 className="font-semibold text-white mb-4">Forecast Visualization (Test Set)</h3>
                        <TimeSeriesChart 
                          data={results[0].forecast} 
                          dateColumn={data.dateColumn} 
                          targetColumn={data.targetColumn}
                          predictions={results.map(r => ({
                            name: r.model_name,
                            data: r.forecast
                          }))}
                          title="Model Comparison"
                        />
                      </div>
                    )}

                    {/* Metrics Table */}
                    <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
                      <div className="p-4 border-b border-white/10 flex justify-between items-center">
                        <h3 className="font-semibold text-white">Model Leaderboard</h3>
                        <button className="text-xs flex items-center gap-2 text-slate-400 hover:text-white transition">
                          <Download size={14} /> Export CSV
                        </button>
                      </div>
                      <table className="w-full text-sm text-left text-slate-400">
                        <thead className="text-xs text-slate-200 uppercase bg-white/5">
                          <tr>
                            <th className="px-6 py-3">Model Name</th>
                            <th className="px-6 py-3">RMSE</th>
                            <th className="px-6 py-3">MAE</th>
                            <th className="px-6 py-3">MAPE</th>
                            <th className="px-6 py-3">Time (s)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(() => {
                            const bestRmse = Math.min(...results.map(r => r.metrics.rmse));
                            return results.map((res) => {
                              const isBest = res.metrics.rmse === bestRmse;
                              return (
                                <tr key={res.model_id} className={`border-b border-white/5 hover:bg-white/5 ${isBest ? 'bg-amber-500/5' : ''}`}>
                                  <td className="px-6 py-4 font-medium text-white flex items-center gap-2">
                                    {isBest && <Trophy size={14} className="text-amber-500" />}
                                    {res.model_name}
                                  </td>
                                  <td className="px-6 py-4 font-mono">{res.metrics.rmse.toFixed(2)}</td>
                                  <td className="px-6 py-4 font-mono">{res.metrics.mae.toFixed(2)}</td>
                                  <td className="px-6 py-4 font-mono">{(res.metrics.mape * 100).toFixed(1)}%</td>
                                  <td className="px-6 py-4 font-mono flex items-center gap-1">
                                    <Timer size={12} /> {res.metrics.execution_time.toFixed(1)}s
                                  </td>
                                </tr>
                              );
                            });
                          })()}
                        </tbody>
                      </table>
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
