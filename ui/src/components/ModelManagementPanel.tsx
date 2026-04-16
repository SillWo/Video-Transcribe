import { startTransition, useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle,
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
type OperationJobStatus = 'queued' | 'running' | 'success' | 'error'
type OperationType = 'download' | 'delete'
type ModelCardUiState =
  | 'downloaded'
  | 'not_downloaded'
  | 'unknown'
  | 'downloading'
  | 'download_error'
  | 'deleting'
  | 'delete_error'
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

type ModelOperationJobResponse = {
  jobId: string
  modelId: string
  status: OperationJobStatus
  progress: {
    percent: number
    label: string
  }
  message: string
  error: string | null
}

type ActiveOperation = {
  operationType: OperationType
  jobId: string
  modelId: string
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
      'Preload and remove Whisper models in the same local Hugging Face cache used by faster-whisper.',
    refreshing: 'Refreshing',
    ready: 'Panel ready',
    operationBusy: 'Operation in progress',
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
    progress: 'Operation progress',
    percent: 'Percent',
    download: 'Download',
    downloadRunning: 'Downloading',
    downloadReady: 'Model is available locally.',
    downloadFailed: 'Download failed',
    delete: 'Delete',
    deleteRunning: 'Deleting',
    deleteDone: 'Model removed from local cache.',
    deleteFailed: 'Delete failed',
    deleteConfirm: (name: string) => `Delete the local model "${name}" from the Hugging Face cache?`,
    queueMessage: 'The backend accepted the operation and started background processing.',
    anotherOperation: 'Another model operation is already running.',
    deleteDisabled: 'Delete is only available for downloaded models.',
    statusLabel: {
      downloaded: 'downloaded',
      not_downloaded: 'not_downloaded',
      unknown: 'unknown',
      downloading: 'downloading',
      download_error: 'download_error',
      deleting: 'deleting',
      delete_error: 'delete_error',
    },
    familyLabel: {
      standard: 'standard',
      distil: 'distil',
    },
    languageLabel: {
      multilingual: 'multilingual',
      english: 'english',
    },
  },
  ru: {
    headerEyebrow: 'Управление моделями',
    headerTitle: 'Локальный каталог Whisper-моделей',
    headerDescription:
      'Предзагружайте и удаляйте Whisper-модели в том же локальном Hugging Face cache, который использует faster-whisper.',
    refreshing: 'Обновление',
    ready: 'Панель готова',
    operationBusy: 'Идёт операция',
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
    progress: 'Прогресс операции',
    percent: 'Процент',
    download: 'Скачать',
    downloadRunning: 'Загрузка',
    downloadReady: 'Модель доступна локально.',
    downloadFailed: 'Загрузка завершилась ошибкой',
    delete: 'Удалить',
    deleteRunning: 'Удаление',
    deleteDone: 'Модель удалена из локального cache.',
    deleteFailed: 'Удаление завершилось ошибкой',
    deleteConfirm: (name: string) => `Удалить локальную модель "${name}" из Hugging Face cache?`,
    queueMessage: 'Backend принял задачу и запустил фоновую операцию.',
    anotherOperation: 'Сейчас уже выполняется другая операция с моделями.',
    deleteDisabled: 'Удаление доступно только для скачанных моделей.',
    statusLabel: {
      downloaded: 'downloaded',
      not_downloaded: 'not_downloaded',
      unknown: 'unknown',
      downloading: 'downloading',
      download_error: 'download_error',
      deleting: 'deleting',
      delete_error: 'delete_error',
    },
    familyLabel: {
      standard: 'standard',
      distil: 'distil',
    },
    languageLabel: {
      multilingual: 'multilingual',
      english: 'english',
    },
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

function getUiState(
  item: ModelPanelItem,
  downloadJob?: ModelOperationJobResponse,
  deleteJob?: ModelOperationJobResponse,
): ModelCardUiState {
  if (deleteJob?.status === 'queued' || deleteJob?.status === 'running') {
    return 'deleting'
  }
  if (deleteJob?.status === 'error') {
    return 'delete_error'
  }
  if (downloadJob?.status === 'queued' || downloadJob?.status === 'running') {
    return 'downloading'
  }
  if (downloadJob?.status === 'error') {
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
    case 'deleting':
      return {
        card: 'border-orange-200 bg-orange-50/60',
        badge: 'border-orange-200 bg-orange-50 text-orange-800',
      }
    case 'delete_error':
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
  const [activeOperation, setActiveOperation] = useState<ActiveOperation | null>(null)
  const [downloadJobs, setDownloadJobs] = useState<Record<string, ModelOperationJobResponse>>({})
  const [deleteJobs, setDeleteJobs] = useState<Record<string, ModelOperationJobResponse>>({})
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
    if (!activeOperation) {
      return
    }

    let active = true
    const timer = window.setInterval(async () => {
      try {
        const endpoint =
          activeOperation.operationType === 'download'
            ? `/api/models/download/${activeOperation.jobId}`
            : `/api/models/delete/${activeOperation.jobId}`
        const job = await fetchJson<ModelOperationJobResponse>(endpoint)
        if (!active) {
          return
        }

        startTransition(() => {
          if (activeOperation.operationType === 'download') {
            setDownloadJobs((current) => ({
              ...current,
              [job.modelId]: job,
            }))
          } else {
            setDeleteJobs((current) => ({
              ...current,
              [job.modelId]: job,
            }))
          }
        })

        if (job.status === 'success') {
          const successMessage =
            activeOperation.operationType === 'download'
              ? job.message || copy.downloadReady
              : job.message || copy.deleteDone
          setActiveOperation(null)
          setModelNotices((current) => ({
            ...current,
            [job.modelId]: { kind: 'success', text: successMessage },
          }))
          setGlobalNotice({ kind: 'success', text: successMessage })
          await loadPanel({ keepLoadingState: true })
        } else if (job.status === 'error') {
          const failureMessage =
            job.error ||
            job.message ||
            (activeOperation.operationType === 'download' ? copy.downloadFailed : copy.deleteFailed)
          setActiveOperation(null)
          setModelNotices((current) => ({
            ...current,
            [job.modelId]: { kind: 'error', text: failureMessage },
          }))
        }
      } catch (pollError) {
        if (!active) {
          return
        }

        setActiveOperation(null)
        setGlobalNotice({
          kind: 'error',
          text:
            pollError instanceof Error
              ? pollError.message
              : activeOperation.operationType === 'download'
                ? copy.downloadFailed
                : copy.deleteFailed,
        })
      }
    }, 1000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [
    activeOperation,
    copy.deleteDone,
    copy.deleteFailed,
    copy.downloadFailed,
    copy.downloadReady,
  ])

  const catalog = panelData?.catalog ?? []
  const summary = panelData?.summary ?? EMPTY_SUMMARY
  const shouldShowModelList = catalog.length > 0

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

  function clearModelNotice(modelId: string) {
    setModelNotices((current) => {
      const next = { ...current }
      delete next[modelId]
      return next
    })
  }

  async function startOperation(modelId: string, operationType: OperationType) {
    clearModelNotice(modelId)
    setGlobalNotice(null)

    try {
      const job = await fetchJson<ModelOperationJobResponse>(`/api/models/${operationType}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ modelId }),
      })

      startTransition(() => {
        if (operationType === 'download') {
          setDownloadJobs((current) => ({
            ...current,
            [job.modelId]: job,
          }))
        } else {
          setDeleteJobs((current) => ({
            ...current,
            [job.modelId]: job,
          }))
        }
      })

      if (job.status === 'queued' || job.status === 'running') {
        setActiveOperation({
          operationType,
          jobId: job.jobId,
          modelId: job.modelId,
        })
        setModelNotices((current) => ({
          ...current,
          [job.modelId]: { kind: 'info', text: job.message || copy.queueMessage },
        }))
        return
      }

      if (job.status === 'success') {
        const successMessage =
          operationType === 'download' ? job.message || copy.downloadReady : job.message || copy.deleteDone
        setModelNotices((current) => ({
          ...current,
          [job.modelId]: { kind: 'success', text: successMessage },
        }))
        setGlobalNotice({ kind: 'success', text: successMessage })
        await loadPanel({ keepLoadingState: true })
        return
      }

      const failureMessage =
        job.error || job.message || (operationType === 'download' ? copy.downloadFailed : copy.deleteFailed)
      setModelNotices((current) => ({
        ...current,
        [job.modelId]: { kind: 'error', text: failureMessage },
      }))
    } catch (operationError) {
      const message =
        operationError instanceof Error
          ? operationError.message
          : operationType === 'download'
            ? copy.downloadFailed
            : copy.deleteFailed
      setGlobalNotice({ kind: 'error', text: message })
      setModelNotices((current) => ({
        ...current,
        [modelId]: { kind: 'error', text: message },
      }))
    }
  }

  async function handleDownload(modelId: string) {
    await startOperation(modelId, 'download')
  }

  async function handleDelete(model: ModelPanelItem) {
    if (!window.confirm(copy.deleteConfirm(model.displayName))) {
      return
    }

    await startOperation(model.id, 'delete')
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
              {isLoading ? copy.refreshing : activeOperation ? copy.operationBusy : copy.ready}
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
              const deleteJob = deleteJobs[item.id]
              const uiState = getUiState(item, downloadJob, deleteJob)
              const tone = getStateTone(uiState)
              const notice = modelNotices[item.id]
              const isOperationActiveForThisModel =
                activeOperation != null && activeOperation.modelId === item.id
              const isAnotherOperationActive =
                activeOperation != null && activeOperation.modelId !== item.id
              const operationJob =
                uiState === 'deleting' || uiState === 'delete_error' ? deleteJob : downloadJob
              const canDownload = item.status === 'not_downloaded' || item.status === 'unknown'
              const canDelete = item.status === 'downloaded'

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

                  {(uiState === 'downloading' || uiState === 'deleting') && operationJob ? (
                    <div className="mt-4 rounded-2xl border border-sky-200 bg-sky-50 px-4 py-3">
                      <div className="flex items-center justify-between gap-3 text-sm font-medium text-sky-900">
                        <span>{operationJob.progress.label || copy.progress}</span>
                        <span>
                          {copy.percent}: {Math.max(0, Math.min(100, operationJob.progress.percent))}%
                        </span>
                      </div>
                      <div className="mt-3 h-2 overflow-hidden rounded-full bg-sky-100">
                        <div
                          className="h-full rounded-full bg-sky-500 transition-[width]"
                          style={{ width: `${Math.max(0, Math.min(100, operationJob.progress.percent))}%` }}
                        />
                      </div>
                      <p className="mt-3 text-sm text-sky-800">{operationJob.message}</p>
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
                        disabled={isAnotherOperationActive || isOperationActiveForThisModel}
                        className="inline-flex items-center justify-center gap-2 rounded-2xl border border-stone-200 bg-white px-4 py-2 text-sm font-semibold text-stone-800 transition hover:bg-stone-50 disabled:cursor-not-allowed disabled:border-stone-200 disabled:bg-stone-100 disabled:text-stone-400"
                      >
                        {uiState === 'downloading' ? (
                          <LoaderCircle className="h-4 w-4 animate-spin" />
                        ) : (
                          <Download className="h-4 w-4" />
                        )}
                        {uiState === 'downloading' ? copy.downloadRunning : copy.download}
                      </button>
                    ) : (
                      <button
                        type="button"
                        disabled
                        className="inline-flex items-center justify-center gap-2 rounded-2xl border border-stone-200 bg-white px-4 py-2 text-sm font-semibold text-stone-400"
                      >
                        <Download className="h-4 w-4" />
                        {copy.download}
                      </button>
                    )}

                    <button
                      type="button"
                      onClick={() => {
                        void handleDelete(item)
                      }}
                      disabled={!canDelete || isAnotherOperationActive || isOperationActiveForThisModel}
                      title={!canDelete ? copy.deleteDisabled : undefined}
                      className="inline-flex items-center justify-center gap-2 rounded-2xl border border-stone-200 bg-white px-4 py-2 text-sm font-semibold text-stone-800 transition hover:bg-stone-50 disabled:cursor-not-allowed disabled:border-stone-200 disabled:bg-stone-100 disabled:text-stone-400"
                    >
                      {uiState === 'deleting' ? (
                        <LoaderCircle className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                      {uiState === 'deleting' ? copy.deleteRunning : copy.delete}
                    </button>
                  </div>

                  {isAnotherOperationActive ? (
                    <p className="mt-3 text-sm text-stone-500">{copy.anotherOperation}</p>
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
