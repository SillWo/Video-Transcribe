import { startTransition, useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Download, FolderOpen, HardDriveDownload, Trash2 } from 'lucide-react'

import { fetchJson } from '../lib/api'
import { cn } from '../lib/utils'

type ModelPanelStatus = 'downloaded' | 'not_downloaded' | 'unknown'
type ModelLanguageScope = 'multilingual' | 'english'
type ModelFamily = 'standard' | 'distil'

type ModelPanelItem = {
  id: string
  displayName: string
  backendValue: string
  hfRepoId: string
  languageScope: ModelLanguageScope
  family: ModelFamily
  enabled: boolean
  isDownloaded: boolean
  cacheLocation: string | null
  downloadedSizeBytes: number | null
  lastModified: string | null
  status: ModelPanelStatus
}

type ModelPanelResponse = {
  catalog: ModelPanelItem[]
  summary: {
    downloadedCount: number
    availableCount: number
    totalDownloadedSizeBytes: number
  }
}

const EMPTY_SUMMARY: ModelPanelResponse['summary'] = {
  downloadedCount: 0,
  availableCount: 0,
  totalDownloadedSizeBytes: 0,
}

const EMPTY_VALUE = '—'

function formatBytes(value: number | null) {
  if (value == null) {
    return EMPTY_VALUE
  }
  if (value === 0) {
    return '0 B'
  }

  const units = ['B', 'KB', 'MB', 'GB']
  const exponent = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1)
  const size = value / 1024 ** exponent

  return `${size.toFixed(size >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`
}

function formatDate(value: string | null) {
  if (!value) {
    return EMPTY_VALUE
  }

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return EMPTY_VALUE
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(parsed)
}

function getStatusLabel(status: ModelPanelStatus) {
  switch (status) {
    case 'downloaded':
      return 'downloaded'
    case 'not_downloaded':
      return 'not_downloaded'
    default:
      return 'unknown'
  }
}

function getStatusTone(status: ModelPanelStatus) {
  switch (status) {
    case 'downloaded':
      return 'border-emerald-200 bg-emerald-50 text-emerald-800'
    case 'not_downloaded':
      return 'border-stone-200 bg-stone-100 text-stone-700'
    default:
      return 'border-amber-200 bg-amber-50 text-amber-800'
  }
}

function getCardTone(status: ModelPanelStatus) {
  switch (status) {
    case 'downloaded':
      return 'border-emerald-200 bg-emerald-50/50'
    case 'not_downloaded':
      return 'border-stone-200 bg-stone-50/70'
    default:
      return 'border-amber-200 bg-amber-50/55'
  }
}

function getFamilyLabel(family: ModelFamily) {
  return family === 'distil' ? 'distil' : 'standard'
}

function getLanguageScopeLabel(languageScope: ModelLanguageScope) {
  return languageScope === 'english' ? 'english' : 'multilingual'
}

function SummaryCard(props: { label: string; value: string | number }) {
  return (
    <div className="rounded-3xl border border-stone-200 bg-stone-50 px-4 py-4">
      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
        {props.label}
      </div>
      <div className="mt-2 text-2xl font-semibold text-stone-900">{props.value}</div>
    </div>
  )
}

function InfoField(props: { label: string; value: string; title?: string; valueClassName?: string }) {
  return (
    <div className="rounded-2xl border border-stone-200 bg-white px-3 py-3">
      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500">
        {props.label}
      </div>
      <div
        className={cn('mt-1 text-sm font-medium text-stone-800', props.valueClassName)}
        title={props.title}
      >
        {props.value}
      </div>
    </div>
  )
}

export function ModelManagementPanel() {
  const [panelData, setPanelData] = useState<ModelPanelResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true

    async function loadPanel() {
      try {
        const payload = await fetchJson<ModelPanelResponse>('/api/models/panel')
        if (!active) {
          return
        }

        startTransition(() => {
          setPanelData(payload)
          setError(null)
        })
      } catch (loadError) {
        if (!active) {
          return
        }

        setError(loadError instanceof Error ? loadError.message : 'Unable to load model catalog.')
      } finally {
        if (active) {
          setIsLoading(false)
        }
      }
    }

    void loadPanel()
    return () => {
      active = false
    }
  }, [])

  const catalog = panelData?.catalog ?? []
  const summary = panelData?.summary ?? EMPTY_SUMMARY
  const shouldShowModelList = catalog.length > 0

  const stateMessage = useMemo(() => {
    if (isLoading) {
      return {
        title: 'Loading model catalog',
        description: 'Requesting /api/models/panel and preparing the Whisper model inventory.',
      }
    }

    if (error) {
      return {
        title: 'Model panel unavailable',
        description: error,
      }
    }

    return {
      title: 'No models in catalog',
      description: 'The backend returned an empty catalog. The panel is waiting for registry data.',
    }
  }, [error, isLoading])

  return (
    <section className="rounded-[28px] border border-stone-200 bg-white/90 p-5 shadow-[0_20px_60px_-30px_rgba(41,37,36,0.35)] backdrop-blur">
      <div className="grid gap-6">
        <header className="grid gap-4 border-b border-stone-200 pb-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="max-w-3xl">
              <div className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                Model Management
              </div>
              <h2 className="mt-2 text-2xl font-semibold text-stone-900">Whisper model inventory</h2>
              <p className="mt-2 text-sm text-stone-500">
                This panel reads the backend registry and the local Hugging Face cache. Download and
                delete actions stay disabled until the next tasks implement them.
              </p>
            </div>
            <div className="rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-stone-600">
              {isLoading ? 'Refreshing' : 'Panel ready'}
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <SummaryCard label="Downloaded" value={summary.downloadedCount} />
            <SummaryCard label="Available" value={summary.availableCount} />
            <SummaryCard label="Total size" value={formatBytes(summary.totalDownloadedSizeBytes)} />
          </div>
        </header>

        {shouldShowModelList ? (
          <div className="grid gap-4">
            {catalog.map((item) => (
              <article
                key={item.id}
                className={cn('rounded-3xl border px-4 py-4', getCardTone(item.status))}
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
                      {item.id}
                    </div>
                    <h3 className="mt-1 text-lg font-semibold text-stone-900">{item.displayName}</h3>
                    <p className="mt-1 text-sm text-stone-500">
                      backendValue: <span className="font-medium text-stone-700">{item.backendValue}</span>
                    </p>
                  </div>
                  <div
                    className={cn(
                      'rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]',
                      getStatusTone(item.status),
                    )}
                  >
                    {getStatusLabel(item.status)}
                  </div>
                </div>

                <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                  <InfoField label="Family" value={getFamilyLabel(item.family)} />
                  <InfoField label="Language scope" value={getLanguageScopeLabel(item.languageScope)} />
                  <InfoField label="Size" value={formatBytes(item.downloadedSizeBytes)} />
                  <InfoField label="Last modified" value={formatDate(item.lastModified)} />
                </div>

                <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1fr)_280px]">
                  <InfoField
                    label="HF repo"
                    value={item.hfRepoId || EMPTY_VALUE}
                    title={item.hfRepoId || undefined}
                    valueClassName="truncate"
                  />
                  <InfoField
                    label="Local path"
                    value={item.cacheLocation ?? EMPTY_VALUE}
                    title={item.cacheLocation ?? undefined}
                    valueClassName="truncate"
                  />
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  <button
                    type="button"
                    disabled
                    className="inline-flex items-center justify-center gap-2 rounded-2xl border border-stone-200 bg-white px-4 py-2 text-sm font-semibold text-stone-400"
                  >
                    <Download className="h-4 w-4" />
                    Download
                  </button>
                  <button
                    type="button"
                    disabled
                    className="inline-flex items-center justify-center gap-2 rounded-2xl border border-stone-200 bg-white px-4 py-2 text-sm font-semibold text-stone-400"
                  >
                    <Trash2 className="h-4 w-4" />
                    Delete
                  </button>
                </div>
              </article>
            ))}
          </div>
        ) : null}

        {!shouldShowModelList ? (
          <div className="rounded-3xl border border-dashed border-stone-300 bg-stone-50 px-5 py-6 text-stone-700">
            <div className="flex items-start gap-3">
              {error ? (
                <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-rose-500" />
              ) : (
                <HardDriveDownload className="mt-0.5 h-5 w-5 shrink-0 text-stone-500" />
              )}
              <div>
                <div className="text-base font-semibold text-stone-900">{stateMessage.title}</div>
                <p className="mt-2 text-sm text-stone-600">{stateMessage.description}</p>
                <div className="mt-4 flex flex-wrap gap-3 text-xs font-medium text-stone-500">
                  <span className="inline-flex items-center gap-2 rounded-full border border-stone-200 bg-white px-3 py-1">
                    <FolderOpen className="h-4 w-4" />
                    HF cache state
                  </span>
                  <span className="inline-flex items-center gap-2 rounded-full border border-stone-200 bg-white px-3 py-1">
                    <AlertTriangle className="h-4 w-4" />
                    Empty / Error State
                  </span>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </section>
  )
}
