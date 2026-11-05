import {
  AlertTriangle,
  Download,
  HardDriveDownload,
  LoaderCircle,
  Trash2,
} from 'lucide-react'

import { useModelManagement, type ModelPanelItem } from '../context/ModelManagementContext'
import { useI18n } from '../i18n/useI18n'
import {
  EMPTY_MODEL_VALUE,
  formatModelBytes,
  formatModelDate,
  getModelStateTone,
  getModelUiState,
  getNoticeTone,
} from '../lib/modelManagementUi'
import { cn } from '../lib/utils'

const COPY = {
  en: {
    refreshing: 'Refreshing',
    ready: 'Models are up to date',
    errorStatus: 'Could not load models',
    operationBusy: 'Operation in progress',
    summaryDownloaded: 'Downloaded',
    summaryAvailable: 'Available',
    summaryTotalSize: 'Total size',
    loadingTitle: 'Loading model catalog',
    loadingDescription: 'Loading the locally managed CPU and CUDA models.',
    errorTitle: 'Model panel unavailable',
    emptyTitle: 'No models available',
    emptyDescription: 'No locally managed models are available yet.',
    retry: 'Try again',
    vulkanNote: 'Vulkan models download automatically the first time you transcribe with a GPU through Vulkan.',
    backendValue: 'Model name',
    family: 'Family',
    languageScope: 'Language scope',
    size: 'Size',
    lastModified: 'Last modified',
    hfRepo: 'Model source',
    localPath: 'Saved location',
    progress: 'Operation progress',
    percent: 'Percent',
    modelFolder: 'Project model folder',
    download: 'Download',
    downloadRunning: 'Downloading',
    delete: 'Delete',
    deleteRunning: 'Deleting',
    deleteConfirm: (name: string) => `Remove the local model "${name}" from this project?`,
    anotherOperation: 'Another model operation is already running.',
    deleteDisabled: 'Delete is only available for downloaded models.',
    statusLabel: {
      downloaded: 'downloaded',
      not_downloaded: 'not downloaded',
      unknown: 'unknown',
      downloading: 'downloading',
      download_error: 'download failed',
      deleting: 'deleting',
      delete_error: 'delete failed',
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
    refreshing: 'Обновление',
    ready: 'Список моделей обновлён',
    errorStatus: 'Не удалось загрузить модели',
    operationBusy: 'Идёт операция',
    summaryDownloaded: 'Загружено',
    summaryAvailable: 'Доступно',
    summaryTotalSize: 'Общий размер',
    loadingTitle: 'Загрузка каталога моделей',
    loadingDescription: 'Загружаем модели Whisper для CPU и CUDA.',
    errorTitle: 'Не удалось загрузить список моделей',
    emptyTitle: 'Пока нет доступных моделей',
    emptyDescription: 'Список локально управляемых моделей пока пуст.',
    retry: 'Повторить',
    vulkanNote: 'Модели Vulkan автоматически загружаются при первой расшифровке через Vulkan на видеокарте.',
    backendValue: 'Название модели',
    family: 'Семейство',
    languageScope: 'Языковой охват',
    size: 'Размер',
    lastModified: 'Изменено',
    hfRepo: 'Источник модели',
    localPath: 'Место хранения',
    progress: 'Прогресс операции',
    percent: 'Процент',
    modelFolder: 'Папка моделей проекта',
    download: 'Скачать',
    downloadRunning: 'Загрузка',
    delete: 'Удалить',
    deleteRunning: 'Удаление',
    deleteConfirm: (name: string) => `Удалить модель "${name}" из папки проекта?`,
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
      standard: 'Обычная',
      distil: 'Дистиллированная',
    },
    languageLabel: {
      multilingual: 'Многоязычная',
      english: 'Английский',
    },
  },
} as const

function SummaryCard(props: { label: string; value: string | number }) {
  return (
    <div className="rounded-3xl border border-stone-200 bg-stone-50 px-4 py-4">
      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
        {props.label}
      </div>
      <div className="mt-2 text-2xl font-semibold tabular-nums text-stone-900">{props.value}</div>
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

export function ModelManagementDashboard() {
  const { locale } = useI18n()
  const copy = locale === 'ru' ? COPY.ru : COPY.en
  const modelStatusLabels =
    locale === 'ru'
      ? {
          downloaded: 'загружена',
          not_downloaded: 'не загружена',
          unknown: 'неизвестно',
          downloading: 'загружается',
          download_error: 'ошибка загрузки',
          deleting: 'удаляется',
          delete_error: 'ошибка удаления',
        }
      : copy.statusLabel
  const {
    catalog,
    summary,
    isLoading,
    error,
    activeOperation,
    downloadJobs,
    deleteJobs,
    modelNotices,
    globalNotice,
    startOperation,
    loadPanel,
    modelsDirectory,
  } = useModelManagement()

  const shouldShowModelList = catalog.length > 0

  const stateMessage = (() => {
    if (isLoading) {
      return {
        title: copy.loadingTitle,
        description: copy.loadingDescription,
      }
    }

    if (error) {
      return {
        title: copy.errorTitle,
        description: copy.errorStatus,
      }
    }

    return {
      title: copy.emptyTitle,
      description: copy.emptyDescription,
    }
  })()

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
          <div className="min-w-0 max-w-3xl">
              <div className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
              {locale === 'ru' ? 'Библиотека моделей' : 'Model library'}
              </div>
            <h2 className="mt-2 break-words text-2xl font-semibold text-stone-900">
              {locale === 'ru' ? 'Управление моделями' : 'Manage models'}
            </h2>
              <p className="mt-2 text-sm text-stone-500">
              {locale === 'ru'
                ? 'Скачивайте и удаляйте модели, используемые на CPU и CUDA.'
                : 'Download or remove models used with CPU and CUDA.'}
              </p>
            </div>
                    <div className="max-w-full break-words rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] text-stone-600">
              {isLoading ? copy.refreshing : error ? copy.errorStatus : activeOperation ? copy.operationBusy : copy.ready}
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <SummaryCard label={copy.summaryDownloaded} value={summary.downloadedCount} />
            <SummaryCard label={copy.summaryAvailable} value={summary.availableCount} />
            <SummaryCard label={copy.summaryTotalSize} value={formatModelBytes(summary.totalDownloadedSizeBytes)} />
          </div>

          <p className="rounded-2xl border border-orange-200 bg-orange-50 px-4 py-3 text-sm text-orange-900">
            {copy.vulkanNote}
          </p>
          {modelsDirectory ? (
            <p className="break-all rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 text-sm text-stone-600">
              {copy.modelFolder}: <span className="font-medium text-stone-800">{modelsDirectory}</span>
            </p>
          ) : null}

          {globalNotice ? (
            <div role="status" aria-live={globalNotice.kind === 'error' ? 'assertive' : 'polite'} className={cn('rounded-2xl border px-4 py-3 text-sm font-medium', getNoticeTone(globalNotice.kind))}>
              {globalNotice.text}
            </div>
          ) : null}
          {error && shouldShowModelList ? (
            <button
              type="button"
              onClick={() => { void loadPanel() }}
              className="min-h-11 justify-self-start rounded-xl bg-stone-900 px-4 py-2 text-sm font-semibold text-white hover:bg-stone-800"
            >
              {copy.retry}
            </button>
          ) : null}
        </header>

        {shouldShowModelList ? (
          <div className="grid gap-4">
            {catalog.map((item) => {
              const downloadJob = downloadJobs[item.id]
              const deleteJob = deleteJobs[item.id]
              const uiState = getModelUiState(item, downloadJob, deleteJob)
              const tone = getModelStateTone(uiState)
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
                      {modelStatusLabels[uiState]}
                    </div>
                  </div>

                  <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                    <InfoField label={copy.family} value={copy.familyLabel[item.family]} />
                    <InfoField label={copy.languageScope} value={copy.languageLabel[item.languageScope]} />
                    <InfoField label={copy.size} value={formatModelBytes(item.downloadedSizeBytes)} />
                    <InfoField label={copy.lastModified} value={formatModelDate(item.lastModified)} />
                  </div>

                  <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1fr)_280px]">
                    <InfoField
                      label={copy.hfRepo}
                      value={item.hfRepoId || EMPTY_MODEL_VALUE}
                      title={item.hfRepoId || undefined}
                      valueClassName="truncate"
                    />
                    <InfoField
                      label={copy.localPath}
                      value={item.cacheLocation ?? EMPTY_MODEL_VALUE}
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
                {error ? (
                  <button
                    type="button"
                    onClick={() => { void loadPanel() }}
                    className="mt-4 inline-flex min-h-11 items-center rounded-xl bg-stone-900 px-4 py-2 text-sm font-semibold text-white hover:bg-stone-800"
                  >
                    {copy.retry}
                  </button>
                ) : null}
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </section>
  )
}
