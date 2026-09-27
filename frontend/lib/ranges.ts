/**
 * Date range helpers matching the backend: training ranges exclude their end date,
 * validation ranges include it. Dates are compared as ISO strings normalized to
 * "YYYY-MM-DDTHH:MM:SS" (a bare date means midnight, like the backend).
 */

import type { DateRange } from '../types/forecasting';

export function isoKey(value: string): string {
  if (value.length === 10) return `${value}T00:00:00`;
  if (value.length === 16) return `${value}:00`;
  return value.slice(0, 19);
}

const inRange = (date: string, range: DateRange, inclusiveEnd: boolean) => {
  const d = isoKey(date);
  const start = isoKey(range.start);
  const end = isoKey(range.end);
  return d >= start && (inclusiveEnd ? d <= end : d < end);
};

/** Number of dates inside the ranges. */
export function countInRanges(dates: string[], ranges: DateRange[], inclusiveEnd: boolean): number {
  return dates.filter(date => ranges.some(r => r.start && r.end && inRange(date, r, inclusiveEnd))).length;
}

/** Index of the first date at or after `value` (dates sorted). */
export function indexAtOrAfter(dates: string[], value: string): number {
  const key = isoKey(value);
  const index = dates.findIndex(d => isoKey(d) >= key);
  return index === -1 ? dates.length - 1 : index;
}

/** Value for a <input type="date|datetime-local">. */
export const inputValue = (value: string, intraday: boolean) => (intraday ? isoKey(value).slice(0, 16) : value.slice(0, 10));
