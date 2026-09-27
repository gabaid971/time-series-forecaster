/**
 * Application state (zustand), persisted in the browser (IndexedDB) so that a page
 * refresh keeps the dataset, the configuration and the last results.
 */

import { del, get, set } from 'idb-keyval';
import Papa from 'papaparse';
import { useEffect, useState } from 'react';
import { create } from 'zustand';
import { createJSONStorage, persist, StateStorage } from 'zustand/middleware';

import { API_HEADERS, getApiUrl, getErrorMessage } from './api';
import { ExampleDataset } from './examples';
import { nextColorIndex } from './modelColors';
import { catalogEntry, defaultParams, featureConfigOf, recommendedTypes } from './models';
import { incompatibility } from './recipes';
import type {
  ColumnInfo,
  DataAlert,
  DatasetStats,
  DateRange,
  FeatureConfig,
  ForecastOutput,
  LagAnalysis,
  ModelConfig,
  ModelResult,
  ModelType,
  Recipe,
  TimeSeriesData,
} from '../types/forecasting';

export type Theme = 'dark' | 'light';
export type Mode = 'simple' | 'advanced';
export type Space = 'experiment' | 'forecast';
export type Step = 1 | 2 | 3;
export type Row = Record<string, unknown>;

/** What the last training used: results stay consistent if the configuration changes afterwards. */
export interface RunSnapshot {
  models: ModelConfig[];
  horizon: number;
  trainingRanges: DateRange[];
  predictionRanges: DateRange[];
}

interface AppState {
  // Interface
  mode: Mode;
  space: Space;
  step: Step;

  // Data
  data: TimeSeriesData | null;
  rawData: Row[];
  fullData: Row[];               // Normalized series (ISO dates) for charts
  availableColumns: ColumnInfo[];
  datasetStats: DatasetStats | null;
  lagAnalysis: LagAnalysis | null;
  dataAlerts: DataAlert[] | null;
  analyzedKey: string | null;    // Dataset + columns of the last analysis
  isAnalyzing: boolean;
  analyzeError: string | null;

  // Evaluation strategy and models
  trainingRanges: DateRange[];
  predictionRanges: DateRange[];
  forecastHorizon: number;
  defaultLags: number[];
  selectedModels: ModelConfig[];

  // Results
  results: ModelResult[];
  lastRun: RunSnapshot | null;
  isTraining: boolean;
  trainError: string | null;

  // Forecast space
  library: Recipe[];
  forecastSelection: string[];
  forecastSteps: number;
  intervalLevel: 0.8 | 0.95;
  forecastOutput: ForecastOutput | null;
  isForecasting: boolean;
  forecastError: string | null;

  // Actions
  setMode: (mode: Mode) => void;
  setSpace: (space: Space) => void;
  setStep: (step: Step) => void;
  loadCsvFile: (file: File) => void;
  loadExample: (example: ExampleDataset) => Promise<void>;
  loadRows: (filename: string, columns: string[], rows: Row[], dateColumn?: string, targetColumn?: string) => void;
  setColumns: (columns: { dateColumn?: string; targetColumn?: string }) => void;
  clearData: () => void;
  analyze: () => Promise<void>;
  setTrainingRanges: (ranges: DateRange[]) => void;
  setPredictionRanges: (ranges: DateRange[]) => void;
  setForecastHorizon: (horizon: number) => void;
  setDefaultLags: (lags: number[]) => void;
  setSelectedModels: (models: ModelConfig[]) => void;
  addModel: (type: ModelType) => void;
  addRecommendedModels: () => void;
  removeModel: (modelId: string) => void;
  setSplit: (splitDate: string) => void;
  updateModelName: (modelId: string, name: string) => void;
  updateModelParams: (modelId: string, params: Row | ((prev: Row) => Row)) => void;
  train: () => Promise<void>;
  addRecipe: (recipe: Recipe) => void;
  removeRecipe: (id: string) => void;
  renameRecipe: (id: string, name: string) => void;
  importRecipes: (recipes: Recipe[]) => void;
  toggleRecipeSelection: (id: string) => void;
  setForecastSteps: (steps: number) => void;
  setIntervalLevel: (level: 0.8 | 0.95) => void;
  runForecast: () => Promise<void>;
}

const STATE_VERSION = 2;

const analysisKey = (data: TimeSeriesData | null, rows: Row[]) =>
  data ? `${data.filename}|${rows.length}|${data.dateColumn}|${data.targetColumn}` : null;

const INTRADAY = new Set(['s', 'min', 'H']);

/** Range bound: the date only, or the full timestamp for intraday data. */
const rangeBound = (iso: string, frequency: string) => (INTRADAY.has(frequency) ? iso : iso.split('T')[0]);

/** Split: training up to `splitDate` (excluded), validation from `splitDate` to the end. */
function splitRanges(dates: string[], splitDate: string, frequency: string): { training: DateRange[]; prediction: DateRange[] } {
  return {
    training: [{ start: rangeBound(dates[0], frequency), end: rangeBound(splitDate, frequency) }],
    prediction: [{ start: rangeBound(splitDate, frequency), end: rangeBound(dates[dates.length - 1], frequency) }],
  };
}

/** Default split: the first 80% of the series for training, the rest for validation. */
function defaultRanges(dates: string[], frequency = 'D'): { training: DateRange[]; prediction: DateRange[] } {
  const dateMin = rangeBound(dates[0], frequency);
  const dateMax = rangeBound(dates[dates.length - 1], frequency);
  const splitDate = rangeBound(dates[Math.floor(dates.length * 0.8)] || dates[dates.length - 1], frequency);
  return {
    training: [{ start: dateMin, end: splitDate }],
    prediction: [{ start: splitDate, end: dateMax }],
  };
}

// Everything that belongs to a dataset: reset when another file is loaded.
// The recipe library is kept on purpose: reload an updated file to forecast what follows.
const emptyData = {
  data: null,
  rawData: [],
  fullData: [],
  availableColumns: [],
  datasetStats: null,
  lagAnalysis: null,
  dataAlerts: null,
  analyzedKey: null,
  analyzeError: null,
  trainingRanges: [],
  predictionRanges: [],
  results: [],
  lastRun: null,
  trainError: null,
  selectedModels: [] as ModelConfig[],
  forecastHorizon: 1,
  forecastOutput: null,
  forecastError: null,
};

// Results computed for other columns: cleared when the date or target column changes
const emptyResults = { results: [], lastRun: null, trainError: null, forecastOutput: null, forecastError: null };

// IndexedDB: no 5 MB limit like localStorage, datasets fit
const idbStorage: StateStorage = {
  getItem: async (name) => (await get(name)) ?? null,
  setItem: (name, value) => set(name, value),
  removeItem: (name) => del(name),
};

export const useAppStore = create<AppState>()(
  persist(
    (setState, getState) => ({
      mode: 'simple',
      space: 'experiment',
      step: 1,
      ...emptyData,
      isAnalyzing: false,
      isTraining: false,
      forecastHorizon: 1,
      defaultLags: [1, 7],
      selectedModels: [],
      library: [],
      forecastSelection: [],
      forecastSteps: 30,
      intervalLevel: 0.8,
      forecastOutput: null,
      isForecasting: false,
      forecastError: null,

      setMode: (mode) => setState({ mode }),
      setSpace: (space) => setState({ space }),
      setStep: (step) => setState({ step }),

      loadCsvFile: (file) => {
        Papa.parse<Row>(file, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (parsed) => getState().loadRows(file.name, parsed.meta.fields || [], parsed.data),
          error: (error) => setState({ analyzeError: `Cannot read the CSV file: ${error.message}` }),
        });
      },

      loadExample: async (example) => {
        try {
          const response = await fetch(example.file);
          const parsed = Papa.parse<Row>(await response.text(), { header: true, dynamicTyping: true, skipEmptyLines: true });
          const filename = example.file.split('/').pop() || example.id;
          getState().loadRows(filename, parsed.meta.fields || [], parsed.data, example.dateColumn, example.targetColumn);
        } catch {
          setState({ analyzeError: `Cannot load the example "${example.title}"` });
        }
      },

      loadRows: (filename, columns, rows, dateColumn, targetColumn) => {
        // Simple heuristic to guess columns
        const dateCol = dateColumn
          ?? columns.find(c => c.toLowerCase().includes('date') || c.toLowerCase().includes('time'))
          ?? columns[0];
        const targetCol = targetColumn
          ?? columns.find(c => c !== dateCol && typeof rows[0]?.[c] === 'number')
          ?? columns[1];
        setState({
          ...emptyData,
          step: 1,
          rawData: rows.filter(row => row[dateCol]),
          data: {
            filename,
            columns,
            dateColumn: dateCol,
            targetColumn: targetCol,
            frequency: 'D',
            exogenousFeatures: columns.filter(c => c !== dateCol && c !== targetCol),
          },
        });
      },

      setColumns: (columns) => {
        const { data, selectedModels } = getState();
        if (!data) return;
        const next = { ...data, ...columns };
        // A column that became the date or the target cannot stay an extra variable
        const models = selectedModels.map(model => {
          const params = model.params as unknown as Row;
          if (!params.feature_config) return model;
          const features = featureConfigOf(params);
          const exogenous = features.exogenous.filter(e => e.column !== next.dateColumn && e.column !== next.targetColumn);
          return { ...model, params: { ...params, feature_config: { ...(params.feature_config as object), exogenous } } as unknown as ModelConfig['params'] };
        });
        setState({ data: next, selectedModels: models, ...emptyResults });
      },

      clearData: () => setState({ ...emptyData, step: 1 }),

      analyze: async () => {
        const { data, rawData } = getState();
        if (!data || !rawData.length) return;
        setState({ isAnalyzing: true, analyzeError: null, lagAnalysis: null, dataAlerts: null, analyzedKey: analysisKey(data, rawData) });

        try {
          const response = await fetch(getApiUrl('analyze'), {
            method: 'POST',
            headers: API_HEADERS,
            body: JSON.stringify({ data: rawData, date_column: data.dateColumn, target_column: data.targetColumn }),
          });
          if (!response.ok) {
            setState({ analyzeError: await getErrorMessage(response) });
            applyLocalFallback();
            return;
          }

          const result = await response.json();
          const normalizedData: Row[] = (result.normalized_data || []).map((point: { date: string; value: number }) => ({
            [data.dateColumn]: point.date, // Full ISO datetime (keeps minute-level data)
            [data.targetColumn]: point.value,
          }));
          const suggestedLags: number[] = result.lag_analysis?.suggested_lags || [];
          const ranges = normalizedData.length > 0
            ? defaultRanges(normalizedData.map(row => String(row[data.dateColumn])), result.stats.frequency)
            : null;

          setState({
            datasetStats: result.stats,
            data: { ...data, frequency: result.stats.frequency },
            availableColumns: result.available_columns || [],
            lagAnalysis: result.lag_analysis || null,
            dataAlerts: result.alerts || null,
            ...(suggestedLags.length > 0 ? { defaultLags: suggestedLags } : {}),
            ...(normalizedData.length > 0 ? { fullData: normalizedData } : {}),
            ...(ranges ? { trainingRanges: ranges.training, predictionRanges: ranges.prediction } : {}),
          });
        } catch (error) {
          console.error('Failed to analyze dataset:', error);
          setState({ analyzeError: 'Cannot reach the backend. Is it running?' });
          applyLocalFallback();
        } finally {
          setState({ isAnalyzing: false });
        }

        // Backend unavailable: basic stats and chart computed locally
        function applyLocalFallback() {
          const { data, rawData } = getState();
          if (!data) return;
          const values = rawData.map(r => r[data.targetColumn]).filter((v): v is number => typeof v === 'number');
          const dates = rawData.map(r => String(r[data.dateColumn])).filter(Boolean).sort();
          const ranges = dates.length > 0 ? defaultRanges(dates) : null;
          setState({
            fullData: rawData.map(row => ({ [data.dateColumn]: row[data.dateColumn], [data.targetColumn]: row[data.targetColumn] })),
            ...(ranges ? { trainingRanges: ranges.training, predictionRanges: ranges.prediction } : {}),
            datasetStats: values.length > 0 ? {
              date_min: dates[0] || '',
              date_max: dates[dates.length - 1] || '',
              total_rows: rawData.length,
              frequency: 'D',
              frequency_label: 'Daily (assumed)',
              missing_dates: 0,
              missing_values_target: rawData.length - values.length,
              value_min: Math.min(...values),
              value_max: Math.max(...values),
              value_mean: values.reduce((a, b) => a + b, 0) / values.length,
            } : null,
          });
        }
      },

      setTrainingRanges: (trainingRanges) => setState({ trainingRanges }),
      setPredictionRanges: (predictionRanges) => setState({ predictionRanges }),
      setForecastHorizon: (forecastHorizon) => {
        const previous = getState().forecastHorizon;
        // Unknown exogenous variables whose lag followed the horizon keep following it
        const selectedModels = getState().selectedModels.map(model => {
          const params = model.params as unknown as Row;
          if (!params.feature_config) return model;
          const features = featureConfigOf(params);
          const exogenous = features.exogenous.map(e =>
            !e.known_in_advance && e.lags.length === 1 && e.lags[0] === previous ? { ...e, lags: [forecastHorizon] } : e
          );
          return { ...model, params: { ...params, feature_config: { ...(params.feature_config as object), exogenous } } as unknown as ModelConfig['params'] };
        });
        setState({ forecastHorizon, selectedModels });
      },

      setSplit: (splitDate) => {
        const { data, fullData, datasetStats } = getState();
        if (!data || fullData.length === 0) return;
        const dates = fullData.map(row => String(row[data.dateColumn]));
        const ranges = splitRanges(dates, splitDate, datasetStats?.frequency || 'D');
        setState({ trainingRanges: ranges.training, predictionRanges: ranges.prediction });
      },
      setDefaultLags: (defaultLags) => setState({ defaultLags }),
      setSelectedModels: (selectedModels) => setState({ selectedModels }),

      addModel: (type) => {
        const { selectedModels, defaultLags, lagAnalysis } = getState();
        const sameType = selectedModels.filter(m => m.type === type).length;
        const name = catalogEntry(type).name;
        const model: ModelConfig = {
          id: `${type}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          type,
          name: sameType === 0 ? name : `${name} ${sameType + 1}`,
          colorIndex: nextColorIndex(selectedModels.map(m => m.colorIndex)),
          params: defaultParams(type, defaultLags, lagAnalysis) as unknown as ModelConfig['params'],
        };
        setState({ selectedModels: [...selectedModels, model] });
      },

      addRecommendedModels: () => {
        const { selectedModels, datasetStats, addModel } = getState();
        recommendedTypes(datasetStats?.frequency || 'D')
          .filter(type => !selectedModels.some(m => m.type === type))
          .forEach(type => addModel(type));
      },

      removeModel: (modelId) => setState({ selectedModels: getState().selectedModels.filter(m => m.id !== modelId) }),

      updateModelName: (modelId, name) => setState({
        selectedModels: getState().selectedModels.map(m => (m.id === modelId ? { ...m, name } : m)),
      }),

      updateModelParams: (modelId, paramsOrFn) => setState({
        selectedModels: getState().selectedModels.map(m => {
          if (m.id !== modelId) return m;
          const current = m.params as unknown as Row;
          const changes = typeof paramsOrFn === 'function' ? paramsOrFn(current) : paramsOrFn;
          return { ...m, params: { ...current, ...changes } as unknown as ModelConfig['params'] };
        }),
      }),

      train: async () => {
        const { data, rawData, selectedModels, trainingRanges, predictionRanges, forecastHorizon } = getState();
        if (!data) return;
        setState({
          step: 3, isTraining: true, results: [], trainError: null,
          lastRun: { models: selectedModels, horizon: forecastHorizon, trainingRanges, predictionRanges },
        });

        try {
          // Send only the columns the models use (date, target, selected exogenous variables)
          const usedColumns = new Set([data.dateColumn, data.targetColumn]);
          selectedModels.forEach(m => {
            const exogenous = (m.params as { feature_config?: FeatureConfig }).feature_config?.exogenous || [];
            exogenous.forEach(e => usedColumns.add(e.column));
          });
          const response = await fetch(getApiUrl('train'), {
            method: 'POST',
            headers: API_HEADERS,
            body: JSON.stringify({
              data: rawData.map(row => Object.fromEntries(Array.from(usedColumns, col => [col, row[col]]))),
              data_config: {
                target_column: data.targetColumn,
                date_column: data.dateColumn,
                frequency: data.frequency,
                training_ranges: trainingRanges,
                prediction_ranges: predictionRanges,
                forecast_strategy: { horizon: forecastHorizon },
              },
              models: selectedModels,
            }),
          });

          if (!response.ok) {
            setState({ trainError: await getErrorMessage(response) });
            return;
          }
          const result = await response.json();
          if (result.status === 'success') {
            setState({ results: result.results });
          } else {
            setState({ trainError: result.message || 'Training failed' });
          }
        } catch (error) {
          console.error('Failed to connect to backend:', error);
          setState({ trainError: 'Cannot reach the backend. Is it running?' });
        } finally {
          setState({ isTraining: false });
        }
      },

      addRecipe: (recipe) => {
        const { library, forecastSelection } = getState();
        const colorIndex = nextColorIndex(library.map(r => r.colorIndex));
        setState({ library: [...library, { ...recipe, colorIndex }], forecastSelection: [...forecastSelection, recipe.id] });
      },

      removeRecipe: (id) => setState({
        library: getState().library.filter(r => r.id !== id),
        forecastSelection: getState().forecastSelection.filter(x => x !== id),
      }),

      renameRecipe: (id, name) => setState({ library: getState().library.map(r => (r.id === id ? { ...r, name } : r)) }),

      importRecipes: (recipes) => {
        const { library } = getState();
        const known = new Set(library.map(r => r.id));
        const added = recipes.filter(r => r?.id && r.model?.type && !known.has(r.id));
        const withColors = added.reduce<Recipe[]>((acc, r) => [
          ...acc, { ...r, colorIndex: nextColorIndex([...library, ...acc].map(x => x.colorIndex)) },
        ], []);
        setState({ library: [...library, ...withColors] });
      },

      toggleRecipeSelection: (id) => {
        const { forecastSelection } = getState();
        setState({ forecastSelection: forecastSelection.includes(id) ? forecastSelection.filter(x => x !== id) : [...forecastSelection, id] });
      },

      setForecastSteps: (forecastSteps) => setState({ forecastSteps }),
      setIntervalLevel: (intervalLevel) => setState({ intervalLevel }),

      runForecast: async () => {
        const { data, rawData, datasetStats, library, forecastSelection, forecastSteps } = getState();
        // Only recipes that can apply to the loaded data (columns, frequency)
        const recipes = library.filter(r => forecastSelection.includes(r.id) && !incompatibility(r, data, datasetStats?.frequency));
        if (!data || recipes.length === 0) return;
        setState({ isForecasting: true, forecastError: null });

        try {
          // One request per (date, target) pair: recipes of the same dataset go together
          const groups = new Map<string, Recipe[]>();
          recipes.forEach(r => {
            const key = `${r.dataset.dateColumn}|${r.dataset.targetColumn}`;
            groups.set(key, [...(groups.get(key) || []), r]);
          });

          const results: ForecastOutput['results'] = [];
          let info = { frequency: '', lastDate: '' };
          for (const group of Array.from(groups.values())) {
            const { dateColumn, targetColumn } = group[0].dataset;
            const usedColumns = new Set([dateColumn, targetColumn]);
            group.forEach(r => featureConfigOf(r.model.params as unknown as Row).exogenous.forEach(e => usedColumns.add(e.column)));
            const response = await fetch(getApiUrl('forecast'), {
              method: 'POST',
              headers: API_HEADERS,
              body: JSON.stringify({
                data: rawData.map(row => Object.fromEntries(Array.from(usedColumns, col => [col, row[col]]))),
                date_column: dateColumn,
                target_column: targetColumn,
                steps: forecastSteps,
                models: group.map(r => ({ ...r.model, id: r.id, name: r.name })),
              }),
            });
            if (!response.ok) {
              setState({ forecastError: await getErrorMessage(response) });
              return;
            }
            const body = await response.json();
            info = { frequency: body.frequency, lastDate: body.last_date };
            body.results.forEach((r: { model_id: string; model_name: string; forecast: ForecastOutput['results'][number]['forecast']; warning?: string; error?: string }) => {
              results.push({ recipeId: r.model_id, name: r.model_name, forecast: r.forecast || [], warning: r.warning ?? undefined, error: r.error ?? undefined });
            });
          }
          setState({ forecastOutput: { ...info, steps: forecastSteps, results } });
        } catch (error) {
          console.error('Failed to connect to backend:', error);
          setState({ forecastError: 'Cannot reach the backend. Is it running?' });
        } finally {
          setState({ isForecasting: false });
        }
      },
    }),
    {
      name: 'tss-app-state',
      // Bump when the shape of persisted data changes (e.g. new /analyze fields):
      // migrate() then drops what is outdated instead of feeding it to new code.
      version: STATE_VERSION,
      migrate: (persisted, version) => {
        const state = persisted as Partial<AppState>;
        if (version < 2) {
          // v2: /analyze returns seasonalities and suggested_temporal -> analyze again
          return { ...state, analyzedKey: null, lagAnalysis: null, dataAlerts: null };
        }
        return state;
      },
      storage: createJSONStorage(() => idbStorage),
      // Transient flags are not persisted (a refresh must not stay "analyzing")
      partialize: ({ isAnalyzing, isTraining, isForecasting, ...rest }) => rest,
    }
  )
);

export const analysisKeyOf = analysisKey;

/** True once the persisted state has been loaded from IndexedDB (render nothing stale before). */
export function useHydrated(): boolean {
  const [hydrated, setHydrated] = useState(() => useAppStore.persist.hasHydrated());
  useEffect(() => {
    const unsubscribe = useAppStore.persist.onFinishHydration(() => setHydrated(true));
    setHydrated(useAppStore.persist.hasHydrated());
    return unsubscribe;
  }, []);
  return hydrated;
}
