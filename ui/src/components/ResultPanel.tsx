import { Copy, Download } from 'lucide-react'

import { SectionCard } from './SectionCard'
import { useI18n } from '../i18n/useI18n'

type ResultPanelProps = {
  text: string
  rawPreview: string
  canDownload: boolean
  onCopy: () => void
  onDownload: () => void
}

export function ResultPanel({
  text,
  rawPreview,
  canDownload,
  onCopy,
  onDownload,
}: ResultPanelProps) {
  const { t } = useI18n()

  return (
    <SectionCard
      title={t('panels.transcript.title')}
      description={t('panels.transcript.description')}
      action={
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onCopy}
            className="inline-flex items-center gap-2 rounded-full border border-stone-200 bg-stone-50 px-3 py-2 text-sm font-medium text-stone-700 transition hover:bg-stone-100"
          >
            <Copy className="h-4 w-4" />
            {t('panels.transcript.copy')}
          </button>
          <button
            type="button"
            onClick={onDownload}
            disabled={!canDownload}
            className="inline-flex items-center gap-2 rounded-full bg-stone-900 px-3 py-2 text-sm font-medium text-white transition hover:bg-stone-800 disabled:cursor-not-allowed disabled:bg-stone-300"
          >
            <Download className="h-4 w-4" />
            {t('panels.transcript.download')}
          </button>
        </div>
      }
      className="h-full"
    >
      <div className="grid gap-4">
        <div className="min-h-[220px] rounded-3xl border border-stone-200 bg-stone-50 p-4">
          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
            {t('panels.transcript.plainText')}
          </p>
          <pre className="max-h-[240px] overflow-auto whitespace-pre-wrap break-words font-sans text-sm leading-7 text-stone-700">
            {text || t('panels.transcript.plainTextEmpty')}
          </pre>
        </div>

        <div className="min-h-[180px] rounded-3xl bg-stone-950 p-4 text-stone-100">
          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-stone-400">
            {t('panels.transcript.savedPreview')}
          </p>
          <pre className="max-h-[220px] overflow-auto whitespace-pre-wrap break-words font-mono text-sm leading-6">
            {rawPreview || t('panels.transcript.savedPreviewEmpty')}
          </pre>
        </div>
      </div>
    </SectionCard>
  )
}
