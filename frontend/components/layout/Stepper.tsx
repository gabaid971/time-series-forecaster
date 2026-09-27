'use client';

import { BarChart3, Settings, Upload } from 'lucide-react';

import { Step, useAppStore } from '../../lib/store';

const STEPS: { id: Step; label: string; Icon: typeof Upload }[] = [
  { id: 1, label: 'Data', Icon: Upload },
  { id: 2, label: 'Models', Icon: Settings },
  { id: 3, label: 'Results', Icon: BarChart3 },
];

/** Steps of the Experiment space. A step can be visited once its inputs exist. */
export default function Stepper() {
  const { step, setStep, data, results, isTraining } = useAppStore();
  const reachable = (id: Step) => id === 1 || (id === 2 && !!data) || (id === 3 && (results.length > 0 || isTraining));

  return (
    <nav className="mb-6 sm:mb-10 relative" aria-label="Experiment steps">
      <div className="absolute left-0 top-4 sm:top-5 w-full h-0.5 bg-ink/10" aria-hidden />
      <ol className="flex items-center justify-between relative px-4 sm:px-0">
        {STEPS.map(({ id, label, Icon }) => {
          const done = step >= id;
          const canVisit = reachable(id) && id !== step;
          return (
            <li key={id}>
              <button
                onClick={() => canVisit && setStep(id)}
                disabled={!canVisit && id !== step}
                aria-current={step === id ? 'step' : undefined}
                className={`relative flex flex-col items-center gap-1 sm:gap-2 group transition-all duration-300 ${
                  step === id ? 'scale-105 sm:scale-110' : ''
                } ${canVisit ? 'cursor-pointer' : 'cursor-default'}`}
              >
                <span className={`w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                  done
                    ? 'bg-accent border-accent text-on-accent shadow-[0_0_20px_rgba(245,158,11,0.4)]'
                    : `bg-panel border-ink/20 text-fg-subtle ${canVisit ? 'group-hover:border-ink/40' : ''}`
                }`}>
                  <Icon size={16} />
                </span>
                <span className={`text-xs font-medium tracking-wider uppercase ${done ? 'text-accent-text' : 'text-fg-subtle'}`}>
                  {label}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
