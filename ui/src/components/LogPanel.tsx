import { ScrollText } from 'lucide-react'

import { SectionCard } from './SectionCard'
import { useI18n } from '../i18n/useI18n'

type LogPanelProps = {
  logText: string
}

export function LogPanel({ logText }: LogPanelProps) {
  const { t } = useI18n()

  return (
    <SectionCard
      title={t('panels.log.title')}
      description={t('panels.log.description')}
      action={<ScrollText className="h-5 w-5 text-stone-400" />}
      className="h-full"
    >
      <div className="min-h-[280px] rounded-3xl bg-stone-950 p-4 text-sm text-stone-100">
        <pre className="max-h-[340px] overflow-auto whitespace-pre-wrap break-words font-mono leading-6">
          {logText || t('panels.log.empty')}
        </pre>
      </div>
    </SectionCard>
  )
}
