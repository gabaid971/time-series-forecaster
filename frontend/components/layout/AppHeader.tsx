'use client';

import { Activity, FlaskConical, Moon, Sparkles, Sun } from 'lucide-react';

import { Mode, Space, useAppStore } from '../../lib/store';
import { useTheme } from '../../lib/theme';

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: 'simple', label: 'Simple', hint: 'Essential settings with smart defaults' },
  { id: 'advanced', label: 'Advanced', hint: 'All settings and secondary metrics' },
];

const SPACES: { id: Space; label: string; Icon: typeof FlaskConical }[] = [
  { id: 'experiment', label: 'Experiment', Icon: FlaskConical },
  { id: 'forecast', label: 'Forecast', Icon: Sparkles },
];

/** Brand, the two spaces of the app, simple/advanced mode and theme toggles. */
export default function AppHeader() {
  const { space, setSpace, mode, setMode } = useAppStore();
  const [theme, setTheme] = useTheme();

  return (
    <header className="flex flex-wrap items-center justify-between gap-4 mb-8 pt-4">
      <div className="flex items-center gap-2 sm:gap-3">
        <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-gradient-to-br from-accent to-orange-600 flex items-center justify-center shadow-lg shadow-accent/20">
          <Activity className="text-white w-5 h-5 sm:w-6 sm:h-6" />
        </div>
        <div>
          <h1 className="text-lg sm:text-2xl font-bold text-fg tracking-tight">
            Time Series <span className="text-accent-text">Studio</span>
          </h1>
          <p className="text-fg-muted text-xs uppercase tracking-widest hidden sm:block">Forecasting Pipeline</p>
        </div>
      </div>

      {/* Spaces */}
      <nav className="flex items-center gap-1 p-1 rounded-xl bg-ink/5 border border-ink/10" aria-label="Spaces">
        {SPACES.map(({ id, label, Icon }) => (
          <button
            key={id}
            onClick={() => setSpace(id)}
            aria-current={space === id ? 'page' : undefined}
            className={`flex items-center gap-2 px-3 sm:px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              space === id ? 'bg-accent text-on-accent shadow-md shadow-accent/20' : 'text-fg-muted hover:text-fg hover:bg-ink/5'
            }`}
          >
            <Icon size={16} /> {label}
          </button>
        ))}
      </nav>

      <div className="flex items-center gap-2">
        {/* Simple / advanced mode */}
        <div className="flex items-center p-1 rounded-lg bg-ink/5 border border-ink/10 text-xs" role="group" aria-label="Interface mode">
          {MODES.map(({ id, label, hint }) => (
            <button
              key={id}
              onClick={() => setMode(id)}
              aria-pressed={mode === id}
              title={hint}
              className={`px-2.5 py-1.5 rounded-md transition-all ${
                mode === id ? 'bg-ink/10 text-fg font-medium' : 'text-fg-muted hover:text-fg'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Theme */}
        <button
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className="w-9 h-9 flex items-center justify-center rounded-lg bg-ink/5 border border-ink/10 text-fg-muted hover:text-fg transition-colors"
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </header>
  );
}
