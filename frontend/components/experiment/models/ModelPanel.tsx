'use client';

import { AlertTriangle, X } from 'lucide-react';

import { seriesColor } from '../../../lib/modelColors';
import { catalogEntry, featureConfigOf, modelSummary, Row, shortExogenousLags, temporalFeaturesFor } from '../../../lib/models';
import { useAppStore } from '../../../lib/store';
import type { ExogenousFeatureConfig, FeatureConfig, ModelConfig, TemporalFeatureConfig } from '../../../types/forecasting';
import { Field, LagInput, Segmented, SliderField, ToggleChip } from '../../ui/fields';

const INPUT = 'rounded-lg border border-ink/15 bg-panel px-2 py-1 text-sm text-fg tabular-nums focus:border-accent focus:outline-none';

/** One selected model: identity, summary (simple mode) or full settings (advanced mode). */
export function ModelPanel({ model }: { model: ModelConfig }) {
  const { mode, removeModel, updateModelName, updateModelParams, availableColumns, datasetStats, forecastHorizon } = useAppStore();
  const params = model.params as unknown as Row;
  const entry = catalogEntry(model.type);
  const features = featureConfigOf(params);
  const frequency = datasetStats?.frequency || 'D';
  const numericColumns = availableColumns.filter(c => c.dtype === 'numeric');

  const update = (changes: Row) => updateModelParams(model.id, changes);
  const updateFeatures = (changes: Partial<FeatureConfig>) =>
    updateModelParams(model.id, prev => ({ feature_config: { ...featureConfigOf(prev), ...changes } }));
  const toggleTemporal = (key: keyof TemporalFeatureConfig) =>
    updateFeatures({ temporal: { ...features.temporal, [key]: !features.temporal[key] } });
  const toggleExogenous = (column: string) => {
    const enabled = features.exogenous.some(e => e.column === column);
    updateFeatures({
      exogenous: enabled
        ? features.exogenous.filter(e => e.column !== column)
        : [...features.exogenous, { column, lags: [forecastHorizon], use_actual: false, known_in_advance: false }],
    });
  };
  const updateExogenous = (column: string, changes: Partial<ExogenousFeatureConfig>) =>
    updateFeatures({ exogenous: features.exogenous.map(e => (e.column === column ? { ...e, ...changes } : e)) });

  return (
    <article className="rounded-xl border border-ink/10 bg-panel">
      <header className="flex items-center gap-3 px-4 py-3 border-b border-ink/10">
        <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: seriesColor(model.colorIndex) }} aria-hidden />
        {mode === 'advanced' ? (
          <input
            value={model.name}
            onChange={(e) => updateModelName(model.id, e.target.value)}
            className="min-w-0 flex-1 bg-transparent font-semibold text-fg outline-none border-b border-transparent focus:border-accent"
            aria-label="Model name"
          />
        ) : (
          <h4 className="min-w-0 flex-1 truncate font-semibold text-fg">{model.name}</h4>
        )}
        {model.name !== entry.name && <span className="text-xs text-fg-subtle">{entry.name}</span>}
        <button type="button" onClick={() => removeModel(model.id)} className="text-fg-subtle hover:text-negative" aria-label={`Remove ${model.name}`}>
          <X size={16} />
        </button>
      </header>

      <div className="p-4 space-y-4">
        {mode === 'simple' && <Summary model={model} frequency={frequency} />}

        {/* Extra variables: essential, so available in both modes */}
        {entry.tabular && numericColumns.length > 0 && (
          <Field
            label="Extra variables"
            help="Other columns of the file used to help the forecast. By default their future is unknown: only values from at least one horizon earlier are used."
          >
            <div className="flex flex-wrap gap-2">
              {numericColumns.map(col => (
                <ToggleChip key={col.name} active={features.exogenous.some(e => e.column === col.name)} onClick={() => toggleExogenous(col.name)}>
                  {col.name}
                </ToggleChip>
              ))}
            </div>
            {mode === 'advanced' && features.exogenous.length > 0 && (
              <ExogenousTable exogenous={features.exogenous} horizon={forecastHorizon} onChange={updateExogenous} />
            )}
          </Field>
        )}

        {mode === 'advanced' && (
          <>
            {model.type === 'LAG' && (
              <Field label="Lag" help="The forecast is the value observed this many steps earlier (1 = last value, 7 = same day last week for daily data).">
                <input type="number" min={1} value={(params.lag as number) ?? 1} onChange={(e) => update({ lag: Math.max(1, parseInt(e.target.value, 10) || 1) })} className={`${INPUT} w-24`} />
              </Field>
            )}

            {entry.tabular && (
              <>
                <Field label="Past values (lags)" help="Which past values of the series are used as inputs: 1 = previous step, 7 = seven steps before…">
                  <LagInput values={(params.lags as number[]) || []} onChange={(lags) => update({ lags })} />
                </Field>
                <Field label="Calendar features" help="Position in the calendar, encoded as cycles (e.g. month captures a yearly pattern).">
                  <div className="flex flex-wrap gap-2">
                    {temporalFeaturesFor(frequency).map(f => (
                      <ToggleChip key={f.key} active={!!features.temporal[f.key]} onClick={() => toggleTemporal(f.key)}>{f.label}</ToggleChip>
                    ))}
                  </div>
                </Field>
                <div className="flex flex-wrap gap-6">
                  <Field label="Target" help="Raw: predict the value. Residual: predict the change since a past value (often better on trending series).">
                    <Segmented
                      options={[{ value: 'raw', label: 'Raw' }, { value: 'residual', label: 'Residual' }]}
                      value={(params.target_mode as string) || 'raw'}
                      onChange={(target_mode) => update({ target_mode })}
                    />
                  </Field>
                  {params.target_mode === 'residual' && (
                    <Field label="Change since" help="The model predicts y(t) − y(t − this lag).">
                      <input type="number" min={1} value={(params.residual_lag as number) ?? 1} onChange={(e) => update({ residual_lag: Math.max(1, parseInt(e.target.value, 10) || 1) })} className={`${INPUT} w-20`} />
                    </Field>
                  )}
                  {model.type === 'LINEAR_REGRESSION' && (
                    <Field label="Standardize" help="Scale features to the same range. Does not change predictions, only makes coefficients comparable.">
                      <Segmented
                        options={[{ value: 'no', label: 'No' }, { value: 'yes', label: 'Yes' }]}
                        value={params.standardize ? 'yes' : 'no'}
                        onChange={(v) => update({ standardize: v === 'yes' })}
                      />
                    </Field>
                  )}
                </div>
              </>
            )}

            {model.type === 'XGBOOST' && (
              <div className="grid gap-4">
                <SliderField label="Trees" help="More trees fit more detail, and take longer." value={(params.n_estimators as number) ?? 100} min={10} max={1000} step={10} onChange={(n_estimators) => update({ n_estimators })} />
                <SliderField label="Depth" help="Complexity of each tree. Deeper trees capture interactions but overfit more easily." value={(params.max_depth as number) ?? 3} min={1} max={12} onChange={(max_depth) => update({ max_depth })} />
                <SliderField label="Learning rate" help="Contribution of each tree. Lower is more careful and needs more trees." value={(params.learning_rate as number) ?? 0.1} min={0.01} max={1} step={0.01} onChange={(learning_rate) => update({ learning_rate })} />
              </div>
            )}

            {model.type === 'ARIMA' && (
              <div className="grid gap-4">
                <SliderField label="p (autoregressive)" help="How many past values directly influence the next one." value={(params.p as number) ?? 1} min={0} max={10} onChange={(p) => update({ p })} />
                <SliderField label="d (differencing)" help="0 for a stable series, 1 to model changes instead of levels (trending series), 2 rarely." value={(params.d as number) ?? 1} min={0} max={2} onChange={(d) => update({ d })} />
                <SliderField label="q (moving average)" help="How many past forecast errors influence the next value." value={(params.q as number) ?? 1} min={0} max={10} onChange={(q) => update({ q })} />
              </div>
            )}

            {model.type === 'PROPHET' && (
              <div className="flex flex-wrap gap-6">
                <Field label="Seasonality" help="Cycles Prophet looks for. Pre-selected from the cycles found in your data.">
                  <div className="flex flex-wrap gap-2">
                    {(['yearly', 'weekly', 'daily'] as const).map(s => (
                      <ToggleChip key={s} active={!!params[`${s}_seasonality`]} onClick={() => update({ [`${s}_seasonality`]: !params[`${s}_seasonality`] })}>
                        {s[0].toUpperCase() + s.slice(1)}
                      </ToggleChip>
                    ))}
                  </div>
                </Field>
                <Field label="Seasonality mode" help="Additive: seasonal swings of constant size. Multiplicative: swings proportional to the level.">
                  <Segmented
                    options={[{ value: 'additive', label: 'Additive' }, { value: 'multiplicative', label: 'Multiplicative' }]}
                    value={(params.seasonality_mode as string) || 'additive'}
                    onChange={(seasonality_mode) => update({ seasonality_mode })}
                  />
                </Field>
              </div>
            )}
          </>
        )}
      </div>
    </article>
  );
}

/** Key settings in plain words (simple mode). */
function Summary({ model, frequency }: { model: ModelConfig; frequency: string }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {modelSummary(model, frequency).map(item => (
        <span key={item} className="rounded-md bg-ink/5 px-2 py-0.5 text-sm text-fg-muted">{item}</span>
      ))}
    </div>
  );
}

/** Lags and "known in advance" of the selected extra variables (advanced mode). */
function ExogenousTable({ exogenous, horizon, onChange }: {
  exogenous: ExogenousFeatureConfig[]; horizon: number; onChange: (column: string, changes: Partial<ExogenousFeatureConfig>) => void;
}) {
  return (
    <div className="mt-3 space-y-2">
      {exogenous.map(exog => {
        const short = shortExogenousLags(exog, horizon);
        return (
          <div key={exog.column} className="rounded-lg bg-ink/5 px-3 py-2">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-sm font-medium text-fg min-w-[6rem]">{exog.column}</span>
              <div className="flex-1 min-w-[10rem]">
                <LagInput values={exog.lags} onChange={(lags) => onChange(exog.column, { lags })} placeholder={`≥ ${horizon}`} min={0} />
              </div>
              <label className="flex items-center gap-2 text-sm text-fg-muted" title="Future values are known when forecasting (calendar, planned promotions…). Then any lag, including 0, is allowed.">
                <input type="checkbox" checked={!!exog.known_in_advance} onChange={() => onChange(exog.column, { known_in_advance: !exog.known_in_advance })} className="accent-accent" />
                Known in advance
              </label>
            </div>
            {short.length > 0 && (
              <p className="mt-1.5 flex items-start gap-1.5 text-xs text-negative">
                <AlertTriangle size={14} className="shrink-0" />
                Lag {short.join(', ')} &lt; horizon {horizon}: these values are not known when forecasting. Use lags ≥ {horizon} or mark the variable as known in advance.
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
