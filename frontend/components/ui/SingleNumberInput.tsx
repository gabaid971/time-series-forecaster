'use client';

import { useState, useEffect, useRef } from 'react';

interface SingleNumberInputProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  placeholder?: string;
}

/**
 * Single Number Input Component (like TagInput but for one value only).
 */
export function SingleNumberInput({
  value,
  onChange,
  min = 0,
  placeholder,
}: SingleNumberInputProps) {
  const [inputValue, setInputValue] = useState(String(value));
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setInputValue(String(value));
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setInputValue(val);

    const num = parseInt(val);
    if (!isNaN(num) && num >= min) {
      onChange(num);
    }
  };

  const handleBlur = () => {
    const num = parseInt(inputValue);
    if (isNaN(num) || num < min) {
      setInputValue(String(value));
    }
  };

  return (
    <input
      ref={inputRef}
      type="text"
      inputMode="numeric"
      value={inputValue}
      onChange={handleChange}
      onBlur={handleBlur}
      placeholder={placeholder || String(value)}
      className="glass-input w-full p-2 rounded-lg text-sm text-white"
    />
  );
}
