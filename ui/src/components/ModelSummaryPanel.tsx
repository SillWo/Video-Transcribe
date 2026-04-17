import {
  ArrowRight,
  Boxes,
  HardDriveDownload,
  LoaderCircle,
  PackageOpen,
  ServerCrash,
} from 'lucide-react'
import { Link } from 'react-router-dom'

import { useModelManagement } from '../context/ModelManagementContext'
import { useI18n } from '../i18n/useI18n'
import { formatModelBytes } from '../lib/modelManagementUi'

const COPY = {
  en: {
    eyebrow: 'Local Models',
    title: 'Local Whisper cache summary',
    description: 'A compact overview of downloaded local models from the backend panel payload.',
    loading: 'Loading backend panel',
    ready: 'Panel ready',
    busy: 'Operation in progress',
    downloaded: 'Downloaded',
    available: 'Available',
    totalSize: 'Total size',
    previewTitle: 'Downloaded models',
    previewDescription: 'Only locally downloaded models are shown here.',
    loadingPreview: 'Loading local model summary from /api/models/panel.',
    empty: 'No local models are downloaded yet.',
    error: 'Model summary is unavailable',
    moreModels: (count: number) => `${count} more downloaded`,
    openDashboard: 'Open model management',
    modelsLabel: 'models',
    stateLabel: {
      loading: 'loading',
      ready: 'ready',
      empty: 'empty',
      error: 'error',
    },
  },
  ru: {
    eyebrow: 'Локальные модели',
    title: 'Сводка локального кэша Whisper',
    description: 'Компактный обзор локально загруженных моделей из backend panel payload.',
    loading: 'Загрузка backend-панели',
    ready: 'Панель готова',
    busy: 'Идёт операция',
    downloaded: 'Загружено',
    available: 'Доступно',
    totalSize: 'Общий размер',
    previewTitle: 'Загруженные модели',
    previewDescription: 'Здесь показываются только локально загруженные модели.',
    loadingPreview: 'Загружаем сводку локальных моделей из /api/models/panel.',
    empty: 'Локально скачанных моделей пока нет.',
    error: 'Сводка моделей недоступна',
    moreModels: (count: number) => `Ещё загружено: ${count}`,
    openDashboard: 'Открыть управление моделями',
    modelsLabel: 'моделей',
    stateLabel: {
      loading: 'loading',
      ready: 'ready',
      empty: 'empty',
      error: 'error',
    },
  },
} as const

function SummaryMetric(props: { label: string; value: string | number }) {
  return (
    <div className="rounded-3xl border border-stone-200 bg-stone-50 px-4 py-4">
      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
        {props.label}
      </div>
      <div className="mt-2 text-2xl font-semibold text-stone-900">{props.value}</div>
    </div>
  )
}

function PreviewRow(props: { displayName: string; backendValue: string; sizeLabel: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-2xl border border-stone-200 bg-white px-4 py-3">
      <div className="min-w-0">
        <div className="truncate text-sm font-semibold text-stone-900">{props.displayName}</div>
        <div className="truncate text-xs text-stone-500">{props.backendValue}</div>
      </div>
      <div className="shrink-0 text-xs font-medium text-stone-600">{props.sizeLabel}</div>
    </div>
  )
}

export function ModelSummaryPanel() {
  const { locale } = useI18n()
  const copy = locale === 'ru' ? COPY.ru : COPY.en
  const { summary, isLoading, error, activeOperation, catalog } = useModelManagement()

  const downloadedModels = catalog.filter((item) => item.status === 'downloaded')
  const previewModels = downloadedModels.slice(0, 5)
  const hiddenDownloadedCount = Math.max(downloadedModels.length - previewModels.length, 0)
  const panelState =
    error != null
      ? 'error'
      : isLoading
        ? 'loading'
        : downloadedModels.length > 0
          ? 'ready'
          : 'empty'

  const statusLabel =
    panelState === 'error'
      ? `${copy.error} / ${copy.stateLabel.error}`
      : panelState === 'loading'
        ? `${copy.loading} / ${copy.stateLabel.loading}`
        : panelState === 'ready'
          ? `${activeOperation ? copy.busy : copy.ready} / ${copy.stateLabel.ready}`
          : `${copy.ready} / ${copy.stateLabel.empty}`

  return (
    <section className="rounded-[28px] border border-stone-200 bg-white/90 p-5 shadow-[0_20px_60px_-30px_rgba(41,37,36,0.35)] backdrop-blur">
      <div className="grid gap-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="max-w-3xl">
            <div className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
              {copy.eyebrow}
            </div>
            <h2 className="mt-2 text-2xl font-semibold text-stone-900">{copy.title}</h2>
            <p className="mt-2 text-sm text-stone-500">{copy.description}</p>
          </div>
          <div className="rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-stone-600">
            {statusLabel}
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          <SummaryMetric label={copy.downloaded} value={summary.downloadedCount} />
          <SummaryMetric label={copy.available} value={summary.availableCount} />
          <SummaryMetric label={copy.totalSize} value={formatModelBytes(summary.totalDownloadedSizeBytes)} />
        </div>

        <div className="grid gap-3 rounded-3xl border border-stone-200 bg-stone-50/80 px-4 py-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
                {copy.previewTitle}
              </div>
              <p className="mt-1 text-sm text-stone-500">{copy.previewDescription}</p>
            </div>
            <div className="rounded-full border border-stone-200 bg-white px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-600">
              {downloadedModels.length}
            </div>
          </div>

          {panelState === 'loading' ? (
            <div className="rounded-2xl border border-dashed border-stone-300 bg-white px-4 py-4 text-sm text-stone-600">
              <div className="inline-flex items-center gap-2">
                <LoaderCircle className="h-4 w-4 animate-spin" />
                {copy.loadingPreview}
              </div>
            </div>
          ) : null}

          {panelState === 'error' ? (
            <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-4 text-sm text-rose-700">
              <div className="inline-flex items-start gap-2">
                <ServerCrash className="mt-0.5 h-4 w-4 shrink-0" />
                <span>
                  {copy.error}: {error}
                </span>
              </div>
            </div>
          ) : null}

          {panelState === 'empty' ? (
            <div className="rounded-2xl border border-dashed border-stone-300 bg-white px-4 py-4 text-sm text-stone-600">
              <div className="inline-flex items-start gap-2">
                <PackageOpen className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{copy.empty}</span>
              </div>
            </div>
          ) : null}

          {panelState === 'ready' ? (
            <div className="grid gap-2">
              {previewModels.map((item) => (
                <PreviewRow
                  key={item.id}
                  displayName={item.displayName}
                  backendValue={item.backendValue}
                  sizeLabel={formatModelBytes(item.downloadedSizeBytes)}
                />
              ))}

              {hiddenDownloadedCount > 0 ? (
                <div className="px-1 pt-1 text-xs font-medium text-stone-500">
                  {copy.moreModels(hiddenDownloadedCount)}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-stone-200 pt-4">
          <div className="inline-flex items-center gap-2 text-sm text-stone-500">
            {isLoading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Boxes className="h-4 w-4" />}
            {activeOperation ? activeOperation.modelId : `${summary.availableCount} ${copy.modelsLabel}`}
          </div>
          <Link
            to="/models"
            className="inline-flex items-center justify-center gap-2 rounded-2xl border border-stone-200 bg-white px-4 py-2 text-sm font-semibold text-stone-800 transition hover:bg-stone-50"
          >
            <HardDriveDownload className="h-4 w-4" />
            {copy.openDashboard}
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </div>
    </section>
  )
}
