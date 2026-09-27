'use client';

import { useState, type ReactNode } from 'react';
import { HelpCircle, X } from 'lucide-react';

/** "?" icon showing an explanation on hover or keyboard focus. */
export function HelpTip({ children }: { children: ReactNode }) {
  return (
    <span className="relative inline-flex group align-middle">
      {/* Not a <button>: help icons sit inside clickable cards, and buttons cannot be nested */}
      <span tabIndex={0} role="img" aria-label="Help" className="text-fg-subtle hover:text-fg-muted focus:text-fg-muted outline-none">
        <HelpCircle size={14} />
      </span>
      <span
        role="tooltip"
        // Not rendered until shown: an invisible tooltip must not widen the page on phones
        className="pointer-events-none absolute left-1/2 bottom-full z-30 mb-2 hidden w-64 max-w-[calc(100vw-2rem)] -translate-x-1/2 rounded-lg border border-ink/10 bg-panel px-3 py-2 text-xs font-normal normal-case tracking-normal text-fg shadow-xl group-hover:block group-focus-within:block"
      >
        {children}
      </span>
    </span>
  );
}

/** Label (with optional help) above a control. */
export function Field({ label, help, children, className = '' }: { label: string; help?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={className}>
      <div className="flex items-center gap-1.5 mb-1.5">
        <span className="text-xs font-medium text-fg-muted">{label}</span>
        {help && <HelpTip>{help}</HelpTip>}
      </div>
      {children}
    </div>
  );
}

/** Integers >= min as removable chips; type a number and press Enter or comma. */
export function LagInput({ values, onChange, placeholder = 'Add a lag', min = 1 }: { values: number[]; onChange: (v: number[]) => void; placeholder?: string; min?: number }) {
  const [input, setInput] = useState('');
  const commit = () => {
    const value = parseInt(input.trim(), 10);
    if (!isNaN(value) && value >= min && !values.includes(value)) onChange([...values, value].sort((a, b) => a - b));
    setInput('');
  };
  return (
    <div className="flex flex-wrap items-center gap-1.5 rounded-lg border border-ink/15 bg-panel px-2 py-1.5 focus-within:border-accent">
      {values.map(v => (
        <span key={v} className="inline-flex items-center gap-1 rounded-md bg-accent/15 px-2 py-0.5 text-sm font-medium text-accent-text tabular-nums">
          {v}
          <button type="button" onClick={() => onChange(values.filter(x => x !== v))} aria-label={`Remove lag ${v}`} className="hover:text-negative">
            <X size={12} />
          </button>
        </span>
      ))}
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); commit(); } }}
        onBlur={commit}
        inputMode="numeric"
        placeholder={placeholder}
        className="min-w-[5rem] flex-1 bg-transparent text-sm text-fg outline-none placeholder:text-fg-subtle"
      />
    </div>
  );
}

/** Range slider with its current value. */
export function SliderField({ label, help, value, min, max, step = 1, onChange }: {
  label: string; help?: ReactNode; value: number; min: number; max: number; step?: number; onChange: (v: number) => void;
}) {
  return (
    <Field label={label} help={help}>
      <div className="flex items-center gap-3">
        <input
          type="range" min={min} max={max} step={step} value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="h-1.5 flex-1 cursor-pointer accent-accent"
          aria-label={label}
        />
        <span className="w-10 text-right text-sm font-medium text-fg tabular-nums">{value}</span>
      </div>
    </Field>
  );
}

/** Mutually exclusive options. */
export function Segmented<T extends string | number>({ options, value, onChange, ariaLabel }: {
  options: { value: T; label: ReactNode }[]; value: T; onChange: (v: T) => void; ariaLabel?: string;
}) {
  return (
    <div className="inline-flex flex-wrap rounded-lg border border-ink/15 bg-ink/5 p-0.5" role="group" aria-label={ariaLabel}>
      {options.map(option => (
        <button
          key={String(option.value)}
          type="button"
          onClick={() => onChange(option.value)}
          aria-pressed={option.value === value}
          className={`rounded-md px-3 py-1 text-sm transition-colors ${
            option.value === value ? 'bg-panel text-fg font-medium shadow-sm' : 'text-fg-muted hover:text-fg'
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/** On/off chip. */
export function ToggleChip({ active, onClick, children, title }: { active: boolean; onClick: () => void; children: ReactNode; title?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      title={title}
      className={`rounded-full border px-3 py-1 text-sm transition-colors ${
        active ? 'border-accent/60 bg-accent/15 text-accent-text font-medium' : 'border-ink/15 text-fg-muted hover:border-ink/30 hover:text-fg'
      }`}
    >
      {children}
    </button>
  );
}
