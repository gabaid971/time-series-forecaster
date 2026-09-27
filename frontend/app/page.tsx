'use client';

import { useEffect } from 'react';

import AppHeader from '../components/layout/AppHeader';
import Stepper from '../components/layout/Stepper';
import { ServerStatusBanner } from '../components/layout/ServerStatusBanner';
import DataStep from '../components/experiment/DataStep';
import ResultsStep from '../components/experiment/ResultsStep';
import ModelsStep from '../components/experiment/ModelsStep';
import ForecastSpace from '../components/forecast/ForecastSpace';
import { analysisKeyOf, useAppStore, useHydrated } from '../lib/store';

export default function ForecastingPage() {
  const hydrated = useHydrated();
  const { space, step, data, rawData, analyzedKey, analyze } = useAppStore();

  // Analyze the dataset when it or its columns change (not again after a page refresh)
  useEffect(() => {
    if (hydrated && data && rawData.length > 0 && analysisKeyOf(data, rawData) !== analyzedKey) {
      analyze();
    }
  }, [hydrated, data, rawData, analyzedKey, analyze]);

  return (
    <div className="min-h-screen font-sans selection:bg-accent/30">
      {/* Ambient Background Effects */}
      <div className="fixed inset-0 pointer-events-none fog-gradient z-0" />
      <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-accent/10 blur-[120px] rounded-full pointer-events-none z-0" />

      <div className="relative z-10 max-w-6xl mx-auto p-3 sm:p-6">
        <AppHeader />
        <ServerStatusBanner />
        {space === 'experiment' && <Stepper />}

        {/* Main Content Area */}
        <main className="glass-panel rounded-xl sm:rounded-2xl glow-box transition-all duration-500">
          <div className="rounded-xl sm:rounded-2xl min-h-[400px] sm:min-h-[600px] p-3 sm:p-6">
            {!hydrated ? (
              <p className="text-fg-muted text-sm animate-pulse">Loading…</p>
            ) : space === 'forecast' ? (
              <ForecastSpace />
            ) : step === 1 ? (
              <DataStep />
            ) : step === 2 ? (
              <ModelsStep />
            ) : (
              <ResultsStep />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
