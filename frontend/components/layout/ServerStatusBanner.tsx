'use client';

import { useEffect, useState } from 'react';
import { AlertTriangle, Loader2 } from 'lucide-react';

import { getApiUrl } from '../../lib/api';

type Status = 'checking' | 'waking' | 'ready' | 'down';

const WAKING_AFTER_MS = 2500;   // A running server answers /health well before this
const GIVE_UP_AFTER_MS = 120_000;

/**
 * The free backend instance sleeps after 15 minutes without visits and takes about
 * a minute to wake up: say so instead of looking frozen.
 */
export function ServerStatusBanner() {
  const [status, setStatus] = useState<Status>('checking');

  useEffect(() => {
    let done = false;
    const controller = new AbortController();
    const wakingTimer = setTimeout(() => { if (!done) setStatus('waking'); }, WAKING_AFTER_MS);
    const giveUpTimer = setTimeout(() => controller.abort(), GIVE_UP_AFTER_MS);

    fetch(getApiUrl('health'), { signal: controller.signal })
      .then(response => { done = true; setStatus(response.ok ? 'ready' : 'down'); })
      .catch(() => { done = true; setStatus('down'); })
      .finally(() => { clearTimeout(wakingTimer); clearTimeout(giveUpTimer); });

    return () => { done = true; controller.abort(); clearTimeout(wakingTimer); clearTimeout(giveUpTimer); };
  }, []);

  if (status === 'waking') {
    return (
      <div role="status" className="mb-4 flex items-start gap-3 rounded-xl border border-info/30 bg-info/10 px-4 py-3 text-sm">
        <Loader2 size={18} className="mt-0.5 shrink-0 animate-spin text-info" />
        <p className="text-fg">
          Waking up the server…
          <span className="text-fg-muted"> The free hosting sleeps after 15 minutes without visits; this takes about a minute.</span>
        </p>
      </div>
    );
  }
  if (status === 'down') {
    return (
      <div role="alert" className="mb-4 flex items-start gap-3 rounded-xl border border-negative/30 bg-negative/10 px-4 py-3 text-sm">
        <AlertTriangle size={18} className="mt-0.5 shrink-0 text-negative" />
        <p className="text-fg">
          The server cannot be reached.
          <span className="text-fg-muted"> Analyses and trainings will fail until it is back; try again in a few minutes.</span>
        </p>
      </div>
    );
  }
  return null;
}
