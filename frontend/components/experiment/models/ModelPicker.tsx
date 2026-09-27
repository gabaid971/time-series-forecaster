'use client';

import { Check, Plus, Sparkles } from 'lucide-react';

import { MODEL_CATALOG } from '../../../lib/models';
import { useAppStore } from '../../../lib/store';
import { Button } from '../../ui/Button';
import { HelpTip } from '../../ui/fields';

/**
 * Model library. Simple mode: one model per type, cards toggle it.
 * Advanced mode: cards add a new variant each time (compare settings of the same model).
 */
export function ModelPicker() {
  const { selectedModels, mode, addModel, removeModel, addRecommendedModels } = useAppStore();

  const onPick = (type: (typeof MODEL_CATALOG)[number]['type']) => {
    const existing = selectedModels.filter(m => m.type === type);
    if (mode === 'simple' && existing.length > 0) existing.forEach(m => removeModel(m.id));
    else addModel(type);
  };

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <p className="text-sm text-fg-muted">
          {mode === 'simple' ? 'Click a model to add or remove it.' : 'Click a model to add a variant; configure each one below.'}
        </p>
        <Button variant="secondary" size="sm" onClick={addRecommendedModels}>
          <Sparkles size={14} /> Add recommended models
        </Button>
      </div>
      <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-5">
        {MODEL_CATALOG.map(entry => {
          const count = selectedModels.filter(m => m.type === entry.type).length;
          const selected = count > 0;
          return (
            <button
              key={entry.type}
              type="button"
              onClick={() => onPick(entry.type)}
              aria-pressed={mode === 'simple' ? selected : undefined}
              className={`relative text-left rounded-xl border p-3.5 transition-all ${
                selected ? 'border-accent/60 bg-accent/10' : 'border-ink/10 bg-panel hover:border-ink/30'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <span className="font-semibold text-fg">{entry.name}</span>
                <span className={`shrink-0 flex h-6 min-w-6 items-center justify-center rounded-full px-1.5 text-xs font-semibold ${
                  selected ? 'bg-accent text-on-accent' : 'bg-ink/10 text-fg-muted'
                }`}>
                  {selected ? (mode === 'simple' ? <Check size={14} /> : count) : <Plus size={14} />}
                </span>
              </div>
              <p className="mt-1 text-xs sm:text-sm text-fg-muted">{entry.description}</p>
              <p className="mt-2 flex items-center gap-1 text-xs text-fg-subtle">
                When to use <HelpTip>{entry.whenToUse}</HelpTip>
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
