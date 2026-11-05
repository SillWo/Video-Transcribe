import type { PropsWithChildren, ReactNode } from 'react'

import { cn } from '../lib/utils'

type SectionCardProps = PropsWithChildren<{
  title: string
  description?: string
  action?: ReactNode
  className?: string
}>

export function SectionCard({
  title,
  description,
  action,
  className,
  children,
}: SectionCardProps) {
  return (
    <section
      className={cn(
        'min-w-0 rounded-2xl border border-stone-200 bg-white/90 p-5 shadow-[0_20px_60px_-30px_rgba(41,37,36,0.35)] backdrop-blur',
        className,
      )}
    >
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="break-words text-lg font-semibold text-stone-900">{title}</h2>
          {description ? (
            <p className="mt-1 text-sm text-stone-500">{description}</p>
          ) : null}
        </div>
        {action}
      </div>

      {children}
    </section>
  )
}
