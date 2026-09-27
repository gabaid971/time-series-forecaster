'use client';

import { AlertTriangle, BookMarked, CalendarRange, FlaskConical, ShieldCheck } from 'lucide-react';

import { incompatibility } from '../../lib/recipes';
import { useAppStore } from '../../lib/store';
import { Button } from '../ui/Button';
import { ForecastResultsCard } from './ForecastResultsCard';
import { ForecastSettings } from './ForecastSettings';
import { LibraryCard } from './LibraryCard';

/**
 * Forecast space: pick validated recipes from the library, retrain them on all the
 * data, forecast the dates after it with confidence intervals.
 */
export default function ForecastSpace() {
  const { library, forecastSelection, data, datasetStats, forecastError, setSpace, setStep, results } = useAppStore();
  const usable = library.filter(r => forecastSelection.includes(r.id) && !incompatibility(r, data, datasetStats?.frequency));
  const validatedHorizon = usable.length > 0 ? Math.min(...usable.map(r => r.validation.horizon)) : null;

  if (library.length === 0) {
    const steps = [
      { Icon: FlaskConical, title: 'Validate models', text: 'In the Experiment space, compare models on a validation period.' },
      { Icon: BookMarked, title: 'Save the good ones', text: 'On the Results page, "Save to library" keeps a model with its validation metrics.' },
      { Icon: CalendarRange, title: 'Forecast the future', text: 'Here, the model is retrained on all your data and forecasts the dates after it.' },
      { Icon: ShieldCheck, title: 'With intervals', text: 'Derived from the errors measured on the validation period.' },
    ];
    return (
      <div className="flex flex-col items-center py-12 text-center">
        <h2 className="text-2xl font-bold text-fg mb-2">Forecast the future</h2>
        <p className="text-fg-muted max-w-lg mb-8">Your library is empty. Save validated models from the Results page to use them here.</p>
        <ol className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 max-w-5xl w-full mb-8">
          {steps.map(({ Icon, title, text }, i) => (
            <li key={title} className="rounded-xl border border-ink/10 bg-panel p-5 text-left">
              <div className="flex items-center gap-2 mb-2">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent/15 text-xs font-semibold text-accent-text">{i + 1}</span>
                <Icon className="text-accent-text" size={18} />
              </div>
              <h3 className="font-semibold text-fg mb-1">{title}</h3>
              <p className="text-sm text-fg-muted">{text}</p>
            </li>
          ))}
        </ol>
        <Button variant="primary" onClick={() => { setSpace('experiment'); setStep(results.length > 0 ? 3 : 1); }}>
          <FlaskConical size={16} /> {results.length > 0 ? 'Go to results' : 'Start an experiment'}
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {!data && (
        <div role="alert" className="flex items-start gap-3 rounded-xl border border-warning/30 bg-warning/10 px-4 py-3 text-sm">
          <AlertTriangle size={18} className="text-warning shrink-0 mt-0.5" />
          <p className="text-fg">
            No dataset loaded. Load your data (with its most recent values) in the{' '}
            <button onClick={() => { setSpace('experiment'); setStep(1); }} className="text-accent-text underline">Experiment space</button> first.
          </p>
        </div>
      )}
      <LibraryCard />
      <ForecastSettings validatedHorizon={validatedHorizon} selectedCount={usable.length} />
      {forecastError && (
        <div role="alert" className="flex items-start gap-3 rounded-xl border border-negative/30 bg-negative/10 px-4 py-3 text-sm">
          <AlertTriangle size={18} className="text-negative shrink-0 mt-0.5" />
          <p className="text-fg">{forecastError}</p>
        </div>
      )}
      <ForecastResultsCard />
    </div>
  );
}
