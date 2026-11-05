import { Copy, Download } from 'lucide-react'

import { SectionCard } from './SectionCard'
import { useI18n } from '../i18n/useI18n'
import { getDeviceDisplayLabel } from '../i18n/displayLabels'

type ResultPanelProps = {
  text: string
  rawPreview: string
  canDownload: boolean
  copyStatus: 'success' | 'error' | null
  device?: string | null
  engine?: string | null
  onCopy: () => void
  onDownload: () => void
}

export function ResultPanel({
  text,
  rawPreview,
  canDownload,
  copyStatus,
  device,
  engine,
  onCopy,
  onDownload,
}: ResultPanelProps) {
  const { locale, t } = useI18n()
  const deviceLabel = device ? getDeviceDisplayLabel(locale, device) : null

  return (
    <SectionCard
      title={t('panels.transcript.title')}
      description={t('panels.transcript.description')}
      action={
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onCopy}
            disabled={!text}
            className="inline-flex min-h-11 items-center gap-2 rounded-full border border-stone-200 bg-stone-50 px-3 py-2 text-sm font-medium text-stone-700 transition-[background-color,color] hover:bg-stone-100 disabled:cursor-not-allowed disabled:text-stone-400"
          >
            <Copy className="h-4 w-4" />
            {t('panels.transcript.copy')}
          </button>
          <button
            type="button"
            onClick={onDownload}
            disabled={!canDownload}
            className="inline-flex min-h-11 items-center gap-2 rounded-full bg-stone-900 px-3 py-2 text-sm font-medium text-white transition-[background-color] hover:bg-stone-800 disabled:cursor-not-allowed disabled:bg-stone-300"
          >
            <Download className="h-4 w-4" />
            {t('panels.transcript.download')}
          </button>
        </div>
      }
    >
      <div className="grid gap-4">
        {engine || device ? (
          <p className="text-xs text-stone-500">
            {engine ? <>{t('panels.transcript.engineUsed')}: {engine}</> : null}
            {engine && deviceLabel ? ' · ' : null}
            {deviceLabel ? <>{t('panels.transcript.deviceUsed')}: {deviceLabel}</> : null}
          </p>
        ) : null}
        {copyStatus ? (
          <p role="status" aria-live="polite" className="text-sm text-stone-600">
            {copyStatus === 'success' ? t('panels.transcript.copied') : t('panels.transcript.copyFailed')}
          </p>
        ) : null}
        <div className="min-h-0 rounded-2xl border border-stone-200 bg-stone-50 p-4">
          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
            {t('panels.transcript.plainText')}
          </p>
          <pre className="max-h-[240px] overflow-auto whitespace-pre-wrap break-words font-sans text-sm leading-7 text-stone-700">
            {text || t('panels.transcript.plainTextEmpty')}
          </pre>
        </div>

        <div className="min-h-0 rounded-2xl bg-stone-950 p-4 text-stone-100">
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
