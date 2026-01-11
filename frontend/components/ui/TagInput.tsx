'use client';

import { useState, useRef } from 'react';
import { X } from 'lucide-react';

interface TagInputProps {
  values: number[];
  onChange: (values: number[]) => void;
  placeholder?: string;
}

/**
 * Tag Input Component for entering multiple integer values.
 * Supports comma-separated input and backspace removal.
 */
export function TagInput({ values, onChange, placeholder }: TagInputProps) {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const addTag = (value: string) => {
    const num = parseInt(value.trim());
    if (!isNaN(num) && num >= 0 && !values.includes(num)) {
      onChange([...values, num].sort((a, b) => a - b));
      setInputValue('');
    }
  };

  const removeTag = (index: number) => {
    onChange(values.filter((_, i) => i !== index));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ',') && inputValue) {
      e.preventDefault();
      addTag(inputValue);
    } else if (e.key === 'Backspace' && !inputValue && values.length > 0) {
      onChange(values.slice(0, -1));
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    if (val.includes(',')) {
      const numStr = val.replace(',', '').trim();
      if (numStr) addTag(numStr);
      else setInputValue('');
    } else {
      setInputValue(val);
    }
  };

  return (
    <div
      onClick={() => inputRef.current?.focus()}
      className="flex flex-wrap gap-1.5 p-2 glass-input rounded-lg min-h-[42px] cursor-text"
    >
      {values.map((tag, idx) => (
        <span
          key={idx}
          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-amber-500 text-black text-sm font-medium rounded"
        >
          {tag}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              removeTag(idx);
            }}
            className="hover:bg-amber-600 rounded-sm p-1 transition-colors touch-manipulation"
            aria-label={`Remove ${tag}`}
          >
            <X size={14} />
          </button>
        </span>
      ))}
      <input
        ref={inputRef}
        type="text"
        inputMode="numeric"
        value={inputValue}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onBlur={() => inputValue && addTag(inputValue)}
        placeholder={values.length === 0 ? placeholder : ''}
        className="flex-1 min-w-[80px] bg-transparent outline-none text-sm text-white placeholder:text-slate-600"
      />
    </div>
  );
}
