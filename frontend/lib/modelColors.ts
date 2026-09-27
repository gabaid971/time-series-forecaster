/**
 * Model colors: a fixed categorical palette, one slot per model.
 *
 * A model keeps its slot for its whole life (color follows the entity, never its rank),
 * slots are taken in order, and the values live in CSS variables (--series-N) so each
 * theme has its own validated steps. Palette validated for color-vision deficiencies
 * against both theme surfaces.
 */

export const SERIES_COUNT = 8;

/** CSS color of a model series, resolved by the current theme. */
export const seriesColor = (colorIndex: number | undefined): string =>
  `rgb(var(--series-${((colorIndex ?? 0) % SERIES_COUNT) + 1}))`;

/** First palette slot not used by the given models. */
export function nextColorIndex(used: (number | undefined)[]): number {
  for (let i = 0; i < SERIES_COUNT; i++) {
    if (!used.includes(i)) return i;
  }
  return used.length % SERIES_COUNT;
}
