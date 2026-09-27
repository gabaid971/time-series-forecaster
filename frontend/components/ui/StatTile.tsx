import type { ReactNode } from 'react';

interface StatTileProps {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  tone?: 'default' | 'warning';
}

/** A headline number: label (sentence case), value, optional detail line. */
export function StatTile({ label, value, detail, tone = 'default' }: StatTileProps) {
  return (
    <div className="rounded-xl border border-ink/10 bg-panel px-4 py-3">
      <p className="text-xs text-fg-muted">{label}</p>
      <p className={`mt-1 text-xl font-semibold ${tone === 'warning' ? 'text-warning' : 'text-fg'}`}>{value}</p>
      {detail && <p className="mt-0.5 text-xs text-fg-subtle truncate">{detail}</p>}
    </div>
  );
}
