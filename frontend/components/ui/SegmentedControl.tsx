'use client';

interface SegmentedControlProps {
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}

/**
 * Segmented Control Component for binary/ternary choices.
 */
export function SegmentedControl({ value, options, onChange }: SegmentedControlProps) {
  return (
    <div className="inline-flex bg-black/30 rounded-lg p-1 border border-white/10">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          className={`px-4 py-1.5 text-xs font-medium rounded transition-all ${
            value === option.value
              ? 'bg-amber-500 text-black shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
