'use client';

import { useState, useEffect } from 'react';

interface SliderProps {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  label?: string;
}

/**
 * Slider Component for numeric ranges with input field.
 */
export function Slider({ value, min, max, step, onChange, label }: SliderProps) {
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400">{label}</span>
        <input
          type="number"
          value={localValue}
          onChange={(e) => {
            const val = parseInt(e.target.value) || min;
            setLocalValue(val);
            onChange(val);
          }}
          className="glass-input w-20 px-2 py-1 rounded text-xs text-center"
        />
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step || 1}
        value={localValue}
        onChange={(e) => {
          const val = parseInt(e.target.value);
          setLocalValue(val);
          onChange(val);
        }}
        className="w-full h-2 bg-black/20 rounded-lg appearance-none cursor-pointer slider-thumb"
        style={{
          background: `linear-gradient(to right, rgb(245, 158, 11) 0%, rgb(245, 158, 11) ${
            ((localValue - min) / (max - min)) * 100
          }%, rgba(0,0,0,0.2) ${((localValue - min) / (max - min)) * 100}%, rgba(0,0,0,0.2) 100%)`,
        }}
      />
      <div className="flex justify-between text-[10px] text-slate-600">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
