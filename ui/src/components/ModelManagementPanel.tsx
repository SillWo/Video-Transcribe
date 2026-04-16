import { startTransition, useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FolderOpen,
  HardDriveDownload,
  LoaderCircle,
  Trash2,
} from 'lucide-react'

import { useI18n } from '../i18n/useI18n'
import { fetchJson } from '../lib/api'
import { cn } from '../lib/utils'

type ModelPanelStatus = 'downloaded' | 'not_downloaded' | 'unknown'
type ModelLanguageScope = 'multilingual' | 'english'
type ModelFamily = 'standard' | 'distil'
type DownloadJobStatus = 'queued' | 'running' | 'success' | 'error'
type ModelCardUiState =
  | 'downloaded'
  | 'not_downloaded'
  | 'unknown'
  | 'downloading'
  | 'download_error'
type NoticeKind = 'info' | 'success' | 'error'

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

type ModelDownloadJobResponse = {
  jobId: string
  modelId: string
  status: DownloadJobStatus
  progress: {
    percent: number
    label: string
  }
  message: string
  error: string | null
}

type Notice = {
  kind: NoticeKind
  text: string
}

const EMPTY_SUMMARY: ModelPanelResponse['summary'] = {
  downloadedCount: 0,
  availableCount: 0,
  totalDownloadedSizeBytes: 0,
}

const EMPTY_VALUE = '\u2014'

const COPY = {
  en: {
    headerEyebrow: 'Model Management',
    headerTitle: 'Whisper model inventory',
    headerDescription:
      'Download a Whisper model into the same local Hugging Face cache that faster-whisper will use later.',
    refreshing: 'Refreshing',
    ready: 'Panel ready',
    downloadingBusy: 'Download in progress',
    summaryDownloaded: 'Downloaded',
    summaryAvailable: 'Available',
    summaryTotalSize: 'Total size',
    loadingTitle: 'Loading model catalog',
    loadingDescription: 'Requesting /api/models/panel and preparing the Whisper model inventory.',
    errorTitle: 'Model panel unavailable',
    emptyTitle: 'No models in catalog',
    emptyDescription: 'The backend returned an empty catalog. The panel is waiting for registry data.',
    hfCacheState: 'HF cache state',
    emptyErrorState: 'Empty / Error State',
    backendValue: 'backendValue',
    family: 'Family',
    languageScope: 'Language scope',
    size: 'Size',
    lastModified: 'Last modified',
    hfRepo: 'HF repo',
    localPath: 'Local path',
    statusLabel: {
      downloaded: 'downloaded',
      not_downloaded: 'not_downloaded',
      unknown: 'unknown',
      downloading: 'downloading',
      download_error: 'download_error',
    },
    familyLabel: {
      standard: 'standard',
      distil: 'distil',
    },
    languageLabel: {
      multilingual: 'multilingual',
      english: 'english',
    },
    download: 'Download',
    downloadRunning: 'Downloading',
    delete: 'Delete',
    anotherDownload: 'Another model download is already running.',
    progress: 'Download progress',
    percent: 'Percent',
    downloadFailed: 'Download failed',
    downloadReady: 'Model is available locally.',
    queueMessage: 'The backend accepted the download job and started background processing.',
  },
  ru: {
    headerEyebrow: 'Управление моделями',
    headerTitle: 'Локальный каталог Whisper-моделей',
    headerDescription:
      'Скачайте модель Whisper заранее в тот же локальный Hugging Face cache, который потом использует faster-whisper.',
    refreshing: 'Обновление',
    ready: 'Панель готова',
    downloadingBusy: 'Идёт загрузка',
    summaryDownloaded: 'Загружено',
    summaryAvailable: 'Доступно',
    summaryTotalSize: 'Общий размер',
    loadingTitle: 'Загрузка каталога моделей',
    loadingDescription: 'Запрашиваем /api/models/panel и собираем локальный каталог Whisper-моделей.',
    errorTitle: 'Панель моделей недоступна',
    emptyTitle: 'Каталог моделей пуст',
    emptyDescription: 'Backend вернул пустой каталог. Панель ждёт данные из реестра моделей.',
    hfCacheState: 'Состояние HF cache',
    emptyErrorState: 'Пустое / ошибочное состояние',
    backendValue: 'backendValue',
    family: 'Семейство',
    languageScope: 'Языковой охват',
    size: 'Размер',
    lastModified: 'Изменено',
    hfRepo: 'HF repo',
    localPath: 'Локальный путь',
    statusLabel: {
      downloaded: 'downloaded',
      not_downloaded: 'not_downloaded',
      unknown: 'unknown',
      downloading: 'downloading',
      download_error: 'download_error',
    },
    familyLabel: {
      standard: 'standard',
      distil: 'distil',
    },
    languageLabel: {
      multilingual: 'multilingual',
      english: 'english',
    },
    download: 'Скачать',
    downloadRunning: 'Загрузка',
    delete: 'Удалить',
    anotherDownload: 'Сейчас уже выполняется загрузка другой модели.',
    progress: 'Прогресс загрузки',
    percent: 'Процент',
    downloadFailed: 'Загрузка завершилась ошибкой',
    downloadReady: 'Модель доступна локально.',
    queueMessage: 'Backend принял задачу и запустил фоновую загрузку модели.',
  },
} as const

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

function getUiState(item: ModelPanelItem, job?: ModelDownloadJobResponse): ModelCardUiState {
  if (job?.status === 'queued' || job?.status === 'running') {
    return 'downloading'
  }
  if (job?.status === 'error') {
    return 'download_error'
  }

  switch (item.status) {
    case 'downloaded':
      return 'downloaded'
    case 'unknown':
      return 'unknown'
    default:
      return 'not_downloaded'
  }
}

function getStateTone(state: ModelCardUiState) {
  switch (state) {
    case 'downloaded':
      return {
        card: 'border-emerald-200 bg-emerald-50/50',
        badge: 'border-emerald-200 bg-emerald-50 text-emerald-800',
      }
    case 'downloading':
      return {
        card: 'border-sky-200 bg-sky-50/55',
        badge: 'border-sky-200 bg-sky-50 text-sky-800',
      }
    case 'download_error':
      return {
        card: 'border-rose-200 bg-rose-50/55',
        badge: 'border-rose-200 bg-rose-50 text-rose-800',
      }
    case 'unknown':
      return {
        card: 'border-amber-200 bg-amber-50/55',
        badge: 'border-amber-200 bg-amber-50 text-amber-800',
      }
    default:
      return {
        card: 'border-stone-200 bg-stone-50/70',
        badge: 'border-stone-200 bg-stone-100 text-stone-700',
      }
  }
}

function getNoticeTone(kind: NoticeKind) {
  switch (kind) {
    case 'success':
      return 'border-emerald-200 bg-emerald-50 text-emerald-800'
    case 'error':
      return 'border-rose-200 bg-rose-50 text-rose-800'
    default:
      return 'border-sky-200 bg-sky-50 text-sky-800'
  }
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
  const { locale } = useI18n()
  const copy = locale === 'ru' ? COPY.ru : COPY.en
  const [panelData, setPanelData] = useState<ModelPanelResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [downloadJobs, setDownloadJobs] = useState<Record<string, ModelDownloadJobResponse>>({})
  const [modelNotices, setModelNotices] = useState<Record<string, Notice>>({})
  const [globalNotice, setGlobalNotice] = useState<Notice | null>(null)

  async function loadPanel(options?: { keepLoadingState?: boolean }) {
    if (!options?.keepLoadingState) {
      setIsLoading(true)
    }

    try {
      const payload = await fetchJson<ModelPanelResponse>('/api/models/panel')
      startTransition(() => {
        setPanelData(payload)
        setError(null)
      })
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Unable to load model catalog.')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    void loadPanel()
  }, [])

  useEffect(() => {
    if (!activeJobId) {
      return
    }

    let active = true
    const timer = window.setInterval(async () => {
      try {
        const job = await fetchJson<ModelDownloadJobResponse>(`/api/models/download/${activeJobId}`)
        if (!active) {
          return
        }

        startTransition(() => {
          setDownloadJobs((current) => ({
            ...current,
            [job.modelId]: job,
          }))
        })

        if (job.status === 'success') {
          setActiveJobId(null)
          setModelNotices((current) => ({
            ...current,
            [job.modelId]: { kind: 'success', text: job.message || copy.downloadReady },
          }))
          setGlobalNotice({ kind: 'success', text: job.message || copy.downloadReady })
          await loadPanel({ keepLoadingState: true })
        } else if (job.status === 'error') {
          setActiveJobId(null)
          const message = job.error || job.message || copy.downloadFailed
          setModelNotices((current) => ({
            ...current,
            [job.modelId]: { kind: 'error', text: message },
          }))
        }
      } catch (pollError) {
        if (!active) {
          return
        }

        setActiveJobId(null)
        setGlobalNotice({
          kind: 'error',
          text: pollError instanceof Error ? pollError.message : copy.downloadFailed,
        })
      }
    }, 1000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [activeJobId, copy.downloadFailed, copy.downloadReady])

  const catalog = panelData?.catalog ?? []
  const summary = panelData?.summary ?? EMPTY_SUMMARY
  const shouldShowModelList = catalog.length > 0
  const activeJob = activeJobId
    ? Object.values(downloadJobs).find((job) => job.jobId === activeJobId)
    : null

  const stateMessage = useMemo(() => {
    if (isLoading) {
      return {
        title: copy.loadingTitle,
        description: copy.loadingDescription,
      }
    }

    if (error) {
      return {
        title: copy.errorTitle,
        description: error,
      }
    }

    return {
      title: copy.emptyTitle,
      description: copy.emptyDescription,
    }
  }, [copy.emptyDescription, copy.emptyTitle, copy.errorTitle, copy.loadingDescription, copy.loadingTitle, error, isLoading])

  async function handleDownload(modelId: string) {
    setGlobalNotice(null)
    setModelNotices((current) => {
      const next = { ...current }
      delete next[modelId]
      return next
    })

    try {
      const job = await fetchJson<ModelDownloadJobResponse>('/api/models/download', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ modelId }),
      })

      startTransition(() => {
        setDownloadJobs((current) => ({
          ...current,
          [job.modelId]: job,
        }))
      })

      if (job.status === 'queued' || job.status === 'running') {
        setActiveJobId(job.jobId)
        setModelNotices((current) => ({
          ...current,
          [job.modelId]: { kind: 'info', text: job.message || copy.queueMessage },
        }))
        return
      }

      if (job.status === 'success') {
        setModelNotices((current) => ({
          ...current,
          [job.modelId]: { kind: 'success', text: job.message || copy.downloadReady },
        }))
        setGlobalNotice({ kind: 'success', text: job.message || copy.downloadReady })
        await loadPanel({ keepLoadingState: true })
        return
      }

      const message = job.error || job.message || copy.downloadFailed
      setModelNotices((current) => ({
        ...current,
        [job.modelId]: { kind: 'error', text: message },
      }))
    } catch (downloadError) {
      const message = downloadError instanceof Error ? downloadError.message : copy.downloadFailed
      setGlobalNotice({ kind: 'error', text: message })
      setModelNotices((current) => ({
        ...current,
        [modelId]: { kind: 'error', text: message },
      }))
    }
  }

  return (
    <section className="rounded-[28px] border border-stone-200 bg-white/90 p-5 shadow-[0_20px_60px_-30px_rgba(41,37,36,0.35)] backdrop-blur">
      <div className="grid gap-6">
        <header className="grid gap-4 border-b border-stone-200 pb-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="max-w-3xl">
              <div className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                {copy.headerEyebrow}
              </div>
              <h2 className="mt-2 text-2xl font-semibold text-stone-900">{copy.headerTitle}</h2>
              <p className="mt-2 text-sm text-stone-500">{copy.headerDescription}</p>
            </div>
            <div className="rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-stone-600">
              {isLoading ? copy.refreshing : activeJob ? copy.downloadingBusy : copy.ready}
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <SummaryCard label={copy.summaryDownloaded} value={summary.downloadedCount} />
            <SummaryCard label={copy.summaryAvailable} value={summary.availableCount} />
            <SummaryCard label={copy.summaryTotalSize} value={formatBytes(summary.totalDownloadedSizeBytes)} />
          </div>

          {globalNotice ? (
            <div className={cn('rounded-2xl border px-4 py-3 text-sm font-medium', getNoticeTone(globalNotice.kind))}>
              {globalNotice.text}
            </div>
          ) : null}
        </header>

        {shouldShowModelList ? (
          <div className="grid gap-4">
            {catalog.map((item) => {
              const downloadJob = downloadJobs[item.id]
              const uiState = getUiState(item, downloadJob)
              const tone = getStateTone(uiState)
              const canDownload = item.status !== 'downloaded'
              const isAnotherJobActive = activeJob != null && activeJob.modelId !== item.id
              const isThisJobRunning = uiState === 'downloading'
              const notice = modelNotices[item.id]

              return (
                <article key={item.id} className={cn('rounded-3xl border px-4 py-4', tone.card)}>
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <div className="text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
                        {item.id}
                      </div>
                      <h3 className="mt-1 text-lg font-semibold text-stone-900">{item.displayName}</h3>
                      <p className="mt-1 text-sm text-stone-500">
                        {copy.backendValue}:{' '}
                        <span className="font-medium text-stone-700">{item.backendValue}</span>
                      </p>
                    </div>
                    <div
                      className={cn(
                        'rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]',
                        tone.badge,
                      )}
                    >
                      {copy.statusLabel[uiState]}
                    </div>
                  </div>

                  <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                    <InfoField label={copy.family} value={copy.familyLabel[item.family]} />
                    <InfoField label={copy.languageScope} value={copy.languageLabel[item.languageScope]} />
                    <InfoField label={copy.size} value={formatBytes(item.downloadedSizeBytes)} />
                    <InfoField label={copy.lastModified} value={formatDate(item.lastModified)} />
                  </div>

                  <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1fr)_280px]">
                    <InfoField
                      label={copy.hfRepo}
                      value={item.hfRepoId || EMPTY_VALUE}
                      title={item.hfRepoId || undefined}
                      valueClassName="truncate"
                    />
                    <InfoField
                      label={copy.localPath}
                      value={item.cacheLocation ?? EMPTY_VALUE}
                      title={item.cacheLocation ?? undefined}
                      valueClassName="truncate"
                    />
                  </div>

                  {isThisJobRunning && downloadJob ? (
                    <div className="mt-4 rounded-2xl border border-sky-200 bg-sky-50 px-4 py-3">
                      <div className="flex items-center justify-between gap-3 text-sm font-medium text-sky-900">
                        <span>{downloadJob.progress.label || copy.progress}</span>
                        <span>
                          {copy.percent}: {Math.max(0, Math.min(100, downloadJob.progress.percent))}%
                        </span>
                      </div>
                      <div className="mt-3 h-2 overflow-hidden rounded-full bg-sky-100">
                        <div
                          className="h-full rounded-full bg-sky-500 transition-[width]"
                          style={{ width: `${Math.max(0, Math.min(100, downloadJob.progress.percent))}%` }}
                        />
                      </div>
                      <p className="mt-3 text-sm text-sky-800">{downloadJob.message}</p>
                    </div>
                  ) : null}

                  {notice ? (
                    <div className={cn('mt-4 rounded-2xl border px-4 py-3 text-sm font-medium', getNoticeTone(notice.kind))}>
                      {notice.text}
                    </div>
                  ) : null}

                  <div className="mt-4 flex flex-wrap gap-2">
                    {canDownload ? (
                      <button
                        type="button"
                        onClick={() => {
                          void handleDownload(item.id)
                        }}
                        disabled={isAnotherJobActive || isThisJobRunning}
                        className="inline-flex items-center justify-center gap-2 rounded-2xl border border-stone-200 bg-white px-4 py-2 text-sm font-semibold text-stone-800 transition hover:bg-stone-50 disabled:cursor-not-allowed disabled:border-stone-200 disabled:bg-stone-100 disabled:text-stone-400"
                      >
                        {isThisJobRunning ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                        {isThisJobRunning ? copy.downloadRunning : copy.download}
                      </button>
                    ) : (
                      <div className="inline-flex items-center justify-center gap-2 rounded-2xl border border-emerald-200 bg-white px-4 py-2 text-sm font-semibold text-emerald-700">
                        <CheckCircle2 className="h-4 w-4" />
                        {copy.downloadReady}
                      </div>
                    )}

                    <button
                      type="button"
                      disabled
                      className="inline-flex items-center justify-center gap-2 rounded-2xl border border-stone-200 bg-white px-4 py-2 text-sm font-semibold text-stone-400"
                    >
                      <Trash2 className="h-4 w-4" />
                      {copy.delete}
                    </button>
                  </div>

                  {isAnotherJobActive ? (
                    <p className="mt-3 text-sm text-stone-500">{copy.anotherDownload}</p>
                  ) : null}
                </article>
              )
            })}
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
                    {copy.hfCacheState}
                  </span>
                  <span className="inline-flex items-center gap-2 rounded-full border border-stone-200 bg-white px-3 py-1">
                    <AlertTriangle className="h-4 w-4" />
                    {copy.emptyErrorState}
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
