import { ScrollText } from 'lucide-react'

import { useI18n } from '../i18n/useI18n'

type LogPanelProps = {
  logText: string
}

export function LogPanel({ logText }: LogPanelProps) {
  const { t } = useI18n()

  return (
    <details className="min-w-0 rounded-2xl border border-stone-200 bg-white/90 p-4 shadow-[0_12px_40px_-24px_rgba(41,37,36,0.35)]">
      <summary className="flex min-h-11 cursor-pointer list-none items-center justify-between gap-3">
        <span>
          <span className="block font-semibold text-stone-900">{t('panels.log.title')}</span>
          <span className="block text-sm text-stone-500">{t('panels.log.description')}</span>
        </span>
        <ScrollText className="h-5 w-5 shrink-0 text-stone-400" />
      </summary>
      <div className="mt-3 rounded-2xl bg-stone-950 p-4 text-sm text-stone-100">
        <pre className="max-h-[340px] overflow-auto whitespace-pre-wrap break-words font-mono leading-6">
          {logText || t('panels.log.empty')}
        </pre>
      </div>
    </details>
  )
}
