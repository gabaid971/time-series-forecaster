import type { ReactNode } from 'react';

interface CardProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}

/** Content container: panel surface, optional header with title, subtitle and actions. */
export function Card({ title, subtitle, icon, actions, children, className = '', bodyClassName = 'p-4 sm:p-5' }: CardProps) {
  return (
    <section className={`rounded-xl border border-ink/10 bg-panel shadow-sm ${className}`}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 px-4 sm:px-5 pt-4 sm:pt-5">
          <div className="flex items-start gap-2 min-w-0">
            {icon && <span className="mt-0.5 text-accent-text shrink-0">{icon}</span>}
            <div className="min-w-0">
              {title && <h3 className="font-semibold text-fg leading-tight">{title}</h3>}
              {subtitle && <p className="text-sm text-fg-muted mt-0.5">{subtitle}</p>}
            </div>
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}
