'use client';

import { AlertTriangle, ArrowLeft, Boxes, Play } from 'lucide-react';

import { configurationIssues } from '../../lib/models';
import { countInRanges } from '../../lib/ranges';
import { useAppStore } from '../../lib/store';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';
import { EvaluationCard } from './models/EvaluationCard';
import { ModelPanel } from './models/ModelPanel';
import { ModelPicker } from './models/ModelPicker';

/** Step 2: how to evaluate (split, horizon) and which models to compare. */
export default function ModelsStep() {
  const { data, fullData, predictionRanges, forecastHorizon, selectedModels, mode, isTraining, setStep, train } = useAppStore();
  if (!data) return null;

  const dates = fullData.map(row => String(row[data.dateColumn]));
  const validationPoints = countInRanges(dates, predictionRanges, true);
  const issues = configurationIssues(selectedModels, forecastHorizon, validationPoints);

  return (
    <div className="space-y-5">
      <EvaluationCard validationPoints={validationPoints} />

      <Card
        title="Models"
        subtitle={mode === 'simple'
          ? 'Settings are pre-filled from the analysis of your data. Switch to advanced mode to fine-tune them.'
          : 'Each model can be fine-tuned; add several variants of the same model to compare settings.'}
        icon={<Boxes size={18} />}
      >
        <ModelPicker />
        {selectedModels.length > 0 && (
          <div className="mt-5 grid gap-4 lg:grid-cols-2 items-start">
            {selectedModels.map(model => <ModelPanel key={model.id} model={model} />)}
          </div>
        )}
      </Card>

      {/* Action bar: stays visible on wide screens; on phones it would cover too much content */}
      <div className="sm:sticky sm:bottom-3 z-20 rounded-xl border border-ink/10 bg-panel/95 backdrop-blur px-4 py-3 shadow-xl">
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="ghost" onClick={() => setStep(1)}>
            <ArrowLeft size={16} /> Data
          </Button>
          <div className="flex-1 min-w-[12rem] text-sm">
            {issues.length > 0 ? (
              <ul className="space-y-0.5">
                {issues.map(issue => (
                  <li key={issue} className="flex items-start gap-1.5 text-warning">
                    <AlertTriangle size={14} className="mt-0.5 shrink-0" /> <span className="text-fg">{issue}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-fg-muted">
                <span className="text-fg font-medium">{selectedModels.length} model{selectedModels.length > 1 ? 's' : ''}</span>
                {' · '}horizon {forecastHorizon}{' · '}{validationPoints.toLocaleString('en-US')} validation points
              </p>
            )}
          </div>
          <Button variant="primary" size="lg" onClick={train} disabled={issues.length > 0 || isTraining}>
            <Play size={16} fill="currentColor" />
            Train {selectedModels.length > 0 ? `${selectedModels.length} model${selectedModels.length > 1 ? 's' : ''}` : 'models'}
          </Button>
        </div>
      </div>
    </div>
  );
}
