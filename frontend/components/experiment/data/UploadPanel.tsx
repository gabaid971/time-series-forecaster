'use client';

import { useRef, useState } from 'react';
import { FileSpreadsheet, Upload } from 'lucide-react';

import { EXAMPLES } from '../../../lib/examples';
import { useAppStore } from '../../../lib/store';
import { Button } from '../../ui/Button';

/** Empty state of the Data step: drop or browse a CSV, or start from an example. */
export function UploadPanel() {
  const { loadCsvFile, loadExample, analyzeError } = useAppStore();
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) loadCsvFile(file);
  };

  return (
    <div className="flex flex-col items-center gap-10 py-6">
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
        onDrop={onDrop}
        className={`w-full max-w-2xl rounded-2xl border-2 border-dashed p-8 sm:p-12 text-center transition-colors ${
          isDragging ? 'border-accent bg-accent/10' : 'border-ink/15 bg-panel'
        }`}
      >
        <div className="w-14 h-14 mx-auto mb-5 rounded-full bg-accent/15 text-accent-text flex items-center justify-center">
          <Upload size={26} />
        </div>
        <h2 className="text-xl font-semibold text-fg mb-2">Load a time series</h2>
        <p className="text-fg-muted mb-1">Drop a CSV file here, or browse your files.</p>
        <p className="text-sm text-fg-subtle mb-6">
          One row per date, with a header row: a date column and at least one numeric column.
        </p>
        <input
          type="file"
          ref={fileInputRef}
          accept=".csv"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && loadCsvFile(e.target.files[0])}
        />
        <Button variant="primary" onClick={() => fileInputRef.current?.click()}>
          <FileSpreadsheet size={16} /> Browse files
        </Button>
        {analyzeError && <p className="mt-4 text-sm text-negative">{analyzeError}</p>}
      </div>

      <div className="w-full">
        <p className="text-sm text-fg-muted text-center mb-4">Or start with an example dataset</p>
        <div className="grid gap-4 sm:grid-cols-3">
          {EXAMPLES.map(example => (
            <button
              key={example.id}
              onClick={() => loadExample(example)}
              className="group text-left rounded-xl border border-ink/10 bg-panel p-4 hover:border-accent/50 hover:shadow-md transition-all"
            >
              <h3 className="font-semibold text-fg group-hover:text-accent-text transition-colors">{example.title}</h3>
              <p className="text-sm text-fg-muted mt-1 mb-3">{example.description}</p>
              <div className="flex flex-wrap gap-1.5">
                {example.tags.map(tag => (
                  <span key={tag} className="text-xs px-2 py-0.5 rounded-full bg-ink/5 text-fg-muted">{tag}</span>
                ))}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
