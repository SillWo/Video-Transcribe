import type {
  ModelOperationJobResponse,
  ModelPanelItem,
  NoticeKind,
} from '../context/ModelManagementContext'

export type ModelCardUiState =
  | 'downloaded'
  | 'not_downloaded'
  | 'unknown'
  | 'downloading'
  | 'download_error'
  | 'deleting'
  | 'delete_error'

export const EMPTY_MODEL_VALUE = '\u2014'

export function formatModelBytes(value: number | null) {
  if (value == null) {
    return EMPTY_MODEL_VALUE
  }
  if (value === 0) {
    return '0 B'
  }

  const units = ['B', 'KB', 'MB', 'GB']
  const exponent = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1)
  const size = value / 1024 ** exponent

  return `${size.toFixed(size >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`
}

export function formatModelDate(value: string | null) {
  if (!value) {
    return EMPTY_MODEL_VALUE
  }

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return EMPTY_MODEL_VALUE
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(parsed)
}

export function getModelUiState(
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

export function getModelStateTone(state: ModelCardUiState) {
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

export function getNoticeTone(kind: NoticeKind) {
  switch (kind) {
    case 'success':
      return 'border-emerald-200 bg-emerald-50 text-emerald-800'
    case 'error':
      return 'border-rose-200 bg-rose-50 text-rose-800'
    default:
      return 'border-sky-200 bg-sky-50 text-sky-800'
  }
}
