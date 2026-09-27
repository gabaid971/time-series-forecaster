'use client';

import { useRef, useState } from 'react';
import { AlertTriangle, BookMarked, Download, Trash2, Upload } from 'lucide-react';

import { formatDate } from '../../lib/formatters';
import { seriesColor } from '../../lib/modelColors';
import { catalogEntry, modelSummary } from '../../lib/models';
import { incompatibility } from '../../lib/recipes';
import { signedPercent } from '../../lib/results';
import { useAppStore } from '../../lib/store';
import type { Recipe } from '../../types/forecasting';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';
import { formatMetric, gainText } from '../experiment/results/types';

/** Saved recipes: select the ones to forecast with, rename, delete, export / import. */
export function LibraryCard() {
  const { library, forecastSelection, data, datasetStats, mode, toggleRecipeSelection, removeRecipe, renameRecipe, importRecipes } = useAppStore();
  const fileInput = useRef<HTMLInputElement>(null);
  const [importError, setImportError] = useState<string | null>(null);

  const exportLibrary = () => {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([JSON.stringify(library, null, 2)], { type: 'application/json' }));
    link.download = `recipes_${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const importLibrary = async (file: File) => {
    try {
      const parsed = JSON.parse(await file.text());
      if (!Array.isArray(parsed)) throw new Error('not a list');
      importRecipes(parsed as Recipe[]);
      setImportError(null);
    } catch {
      setImportError(`"${file.name}" is not a recipes export (a JSON file exported from this library).`);
    }
  };

  return (
    <Card
      title="Library"
      subtitle="Model configurations saved from the Results page, with how they performed on validation."
      icon={<BookMarked size={18} />}
      actions={
        <>
          <input ref={fileInput} type="file" accept=".json,application/json" className="hidden" onChange={(e) => e.target.files?.[0] && importLibrary(e.target.files[0])} />
          <Button variant="ghost" size="sm" onClick={() => fileInput.current?.click()} title="Import recipes from a JSON file">
            <Upload size={14} /> Import
          </Button>
          <Button variant="ghost" size="sm" onClick={exportLibrary} disabled={library.length === 0} title="Export the library as JSON">
            <Download size={14} /> Export
          </Button>
        </>
      }
    >
      {importError && <p className="mb-3 text-sm text-negative">{importError}</p>}
      <ul className="space-y-2">
        {library.map(recipe => {
          const problem = incompatibility(recipe, data, datasetStats?.frequency);
          const checked = forecastSelection.includes(recipe.id) && !problem;
          const v = recipe.validation;
          return (
            <li key={recipe.id} className={`rounded-lg border px-3 py-3 transition-colors ${checked ? 'border-accent/50 bg-accent/5' : 'border-ink/10'}`}>
              <div className="flex items-start gap-3">
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={!!problem}
                  onChange={() => toggleRecipeSelection(recipe.id)}
                  className="mt-1.5 h-4 w-4 accent-accent"
                  aria-label={`Use ${recipe.name}`}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: seriesColor(recipe.colorIndex) }} aria-hidden />
                    <input
                      value={recipe.name}
                      onChange={(e) => renameRecipe(recipe.id, e.target.value)}
                      className="min-w-0 flex-1 bg-transparent font-medium text-fg outline-none border-b border-transparent focus:border-accent"
                      aria-label="Recipe name"
                    />
                    {recipe.name !== catalogEntry(recipe.model.type)?.name && (
                      <span className="text-xs text-fg-subtle">{catalogEntry(recipe.model.type)?.name}</span>
                    )}
                  </div>
                  <p className="mt-1 text-sm text-fg-muted">
                    <span className={gainText(v.gain)}>{signedPercent(-v.gain)} vs naive</span>
                    {' · '}RMSE {formatMetric(v.rmse)}
                    {' · '}validated {v.horizon} step{v.horizon > 1 ? 's' : ''} ahead
                  </p>
                  {mode === 'advanced' && (
                    <p className="mt-0.5 text-xs text-fg-subtle">
                      {modelSummary(recipe.model, recipe.dataset.frequency).join(' · ')} — {recipe.dataset.filename}, saved {formatDate(recipe.createdAt)}
                    </p>
                  )}
                  {problem && (
                    <p className="mt-1 flex items-center gap-1.5 text-xs text-warning">
                      <AlertTriangle size={13} /> {problem}
                    </p>
                  )}
                </div>
                <button type="button" onClick={() => removeRecipe(recipe.id)} className="text-fg-subtle hover:text-negative" aria-label={`Delete ${recipe.name}`}>
                  <Trash2 size={16} />
                </button>
              </div>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}
