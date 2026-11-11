import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { AlertCircle, Check, Circle, Clipboard, FileText, LoaderCircle, Minus, Plus, Trash2, X } from 'lucide-react'
import { API_BASE, fetchJson, type FileSelection } from './lib/api'
import { DraftManager, SelectionGate, flushAndRead } from './lib/draftManager'
import { defaultDiarizationPrefs, diarizationRequestFields, sanitizeDiarizationPrefs, sanitizeStoredPrefs, stageOrderFor } from './lib/diarization'

type Profile = 'quality' | 'balanced' | 'economy'
type Device = 'cpu' | 'gpu'
type JobStatus = 'queued' | 'running' | 'cancelling' | 'completed' | 'failed' | 'interrupted' | 'cancelled'
type Stage = 'preparation' | 'source' | 'audio' | 'recognition' | 'diarization' | 'postprocess' | 'result'
type Checkpoint = { stage: Stage; status: 'pending' | 'active' | 'completed' | 'skipped' | 'failed' | 'cancelled'; startedAt: string | null; completedAt: string | null }
type JobProgress = { stage: Stage; percent: number | null; completedBytes: number | null; totalBytes: number | null; processedSeconds: number | null; totalSeconds: number | null; message: string | null; updatedAt: string }
type SimpleJob = {
  id: string; sourceType: 'url' | 'file'; source: string; profile: Profile; device: Device
  cpuThreads: number; restorePunctuation: boolean; diarizationEnabled?: boolean; diarizationDevice?: string
  diarization?: { status: string; requestedDevice: string; effectiveDevice: { id: string; name: string; backend: string } | null; turns?: { start: number; end: number; text: string; speakerIds: number[]; label: string }[] }
  status: JobStatus
  stage: Stage; progress?: JobProgress | null; checkpoints?: Checkpoint[]; createdAt: string
  finishedAt?: string | null; durationSeconds?: number | null
  updatedAt: string; detectedLanguage: string | null; error: string | null; notice: string | null
  revision: number; text: string | null; files: { format: 'txt' | 'md'; name: string; downloadUrl: string }[]
  startedAt?: string | null; queueWaitSeconds?: number | null; processingDurationSeconds?: number | null; errorCode?: string | null
}
type Summary = Omit<SimpleJob, 'text'> & { queuePosition?: number | null }
type HistoryItem = Omit<SimpleJob, 'text'> & { queuePosition?: number | null }
type QueueState = { paused: boolean; pauseReason: { code: string; message: string } | null; cancelling: boolean; activeJob: Summary | null; waitingJobs: Summary[]; waitingCount: number }
type Options = {
  profiles: { id: Profile; label: string; model: string }[]
  devices: { cpu: { available: boolean; name: string }; gpu: { available: boolean; name: string; reason: string | null } }
  defaultDevice: Device; maxCpuThreads: number
  diarization?: { enabledByDefault: boolean; defaultDevice: string; devices: { id: string; name: string; backend: string; available: boolean; reason: string | null }[]; ready: boolean }
}
type ApiError = Error & { status?: number }
type Prefs = { profile: Profile; device: Device; cpuThreads: number; restorePunctuation: boolean; diarizationEnabled: boolean; diarizationDevice: string }

const preferenceKey = 'video-transcribe-simple-preferences'
const tabsKey = 'video-transcribe-open-tabs'
const selectedTabKey = 'video-transcribe-selected-tab'
const ordinalKey = 'video-transcribe-tab-order'
let flushCurrentDraft: (() => Promise<void>) | null = null
let flushAllWorkspaceDrafts: (() => Promise<void>) | null = null
let clearWorkspaceJob: ((jobId: string) => void) | null = null
let openWorkspaceTask: ((jobId: string) => void) | null = null
let clearedLatestJob: { id: string; createdAt: string } | null = null
const defaults: Prefs = { profile: 'balanced', device: 'cpu', cpuThreads: 1, restorePunctuation: true, ...defaultDiarizationPrefs }
const stages: Record<Stage, string> = {
  preparation: 'Подготовка модели', source: 'Получение записи', audio: 'Подготовка аудио',
  recognition: 'Распознавание речи', diarization: 'Разделение по говорящим', postprocess: 'Пунктуация', result: 'Готово',
}
const statusLabels: Record<JobStatus, string> = {
  queued: 'Ожидает', running: 'Обрабатывается', cancelling: 'Отменяется', completed: 'Готово', failed: 'Ошибка', interrupted: 'Прервана', cancelled: 'Отменена',
}
const profileCopy: Record<Profile, string> = {
  quality: 'Более точное распознавание', balanced: 'Для обычного использования', economy: 'Меньше нагрузка на память',
}
const profileLabels: Record<Profile, string> = { quality: 'Качество', balanced: 'Баланс', economy: 'Экономный' }
const api = {
  options: () => fetchJson<Options>('/api/simple/options'),
  session: () => fetchJson<{ job: SimpleJob | null }>('/api/simple/session'),
  history: () => fetchJson<{ items: HistoryItem[] }>('/api/simple/history'),
  pick: () => fetchJson<FileSelection>('/api/simple/pick-files', { method: 'POST' }),
  job: (id: string, signal?: AbortSignal) => fetchJson<SimpleJob>(`/api/simple/jobs/${encodeURIComponent(id)}`, { signal }),
  cancel: (id: string) => fetchJson<SimpleJob>(`/api/simple/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
  queue: () => fetchJson<QueueState>('/api/simple/queue'),
  resume: () => fetchJson<QueueState>('/api/simple/queue/resume', { method: 'POST' }),
  cancelAll: () => fetchJson<QueueState>('/api/simple/queue/cancel', { method: 'POST' }),
  batch: (paths: string[], prefs: Prefs) => fetchJson<{ jobs: SimpleJob[]; queue: QueueState; rejected: { path: string; message: string }[] }>('/api/simple/jobs/batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ paths, ...prefs }) }),
}
function errorText(error: unknown) { return error instanceof Error ? error.message : 'Не удалось выполнить запрос. Проверьте соединение и повторите попытку.' }
function errorStatus(error: unknown) { return typeof error === 'object' && error !== null && 'status' in error ? (error as ApiError).status : undefined }
function readPrefs(): Partial<Prefs> {
  try {
    return sanitizeStoredPrefs(JSON.parse(localStorage.getItem(preferenceKey) ?? '{}'))
  } catch { return sanitizeStoredPrefs(null) }
}
function readClearedLatest() {
  try { return JSON.parse(localStorage.getItem('video-transcribe-cleared-latest') ?? 'null') as { id: string; createdAt: string } | null } catch { return null }
}
function formatDate(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.valueOf()) ? value : new Intl.DateTimeFormat('ru-RU', { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}
function sourceLabel(item: { sourceType: 'url' | 'file'; source: string }) { return item.source || (item.sourceType === 'file' ? 'Локальный файл' : 'Ссылка') }
function isActiveStatus(status: JobStatus) { return status === 'queued' || status === 'running' || status === 'cancelling' }
function isActive(job: SimpleJob | null): job is SimpleJob { return job !== null && isActiveStatus(job.status) }
function isRetryableCancellation(job: SimpleJob | null) { return job?.status === 'cancelling' && /не удалось остановить обработку|повторите отмену/i.test(job.error ?? '') }
function safeError(value: string | null) {
  if (!value || !/start\.ps1/i.test(value)) return value
  return /punct|пунктуац|sbert/i.test(value)
    ? 'Не удалось подготовить пунктуацию автоматически. Отключите её в настройках или повторите попытку после восстановления сети.'
    : 'Не удалось подготовить модель автоматически. Проверьте подключение, при проблеме с видеокартой выберите CPU и повторите попытку.'
}
function stageName(stage: Stage) { return stages[stage] }
function diarizationReason(reason: string | null | undefined) {
  if (!reason) return ''
  if (/model.*not ready|model is missing|model_unavailable/i.test(reason)) return 'Модель разделения говорящих не установлена или повреждена. Запустите подготовку приложения и повторите попытку.'
  if (/runtime is not installed|runtime is missing|not installed/i.test(reason)) return 'Компонент для этого устройства не установлен. Запустите подготовку приложения и повторите попытку.'
  if (/driver|runtime.*unavailable|incompatible/i.test(reason)) return 'Драйвер или компонент GPU недоступен либо несовместим.'
  if (/backend is unavailable|not exposed/i.test(reason)) return 'Этот GPU не поддерживается установленным компонентом разделения говорящих.'
  if (/no usable.*device/i.test(reason)) return 'Нет готового устройства для разделения говорящих. Проверьте доступность CPU или GPU.'
  if (/cpu.*not ready/i.test(reason)) return 'Разделение говорящих на процессоре пока не готово.'
  return 'Устройство для разделения говорящих сейчас недоступно.'
}
function diarizationDeviceLabel(id: string | null | undefined, devices: { id: string; name: string; backend: string }[] = []) {
  if (!id || id === 'auto') return 'Авто'
  if (id === 'cpu') return 'Процессор'
  const device = devices.find((item) => item.id === id)
  if (!device) return id
  const backend = device.backend === 'cuda' ? ' (CUDA)' : device.backend === 'vulkan' ? ' (Vulkan)' : ''
  return `${device.name}${backend}`
}
function checkpointsFor(job: SimpleJob): Checkpoint[] {
  if (job.status === 'queued') return []
  const known = Array.isArray(job.checkpoints) ? job.checkpoints : []
  if (known.length === 0 && job.status === 'completed') return []
  const order = stageOrderFor(job.diarizationEnabled === true) as readonly Stage[]
  return order.map((stage) => {
    const checkpoint = known.find((item) => item?.stage === stage)
    if (checkpoint) return checkpoint
    return { stage, status: stage === job.stage && isActive(job) ? 'active' as const : 'pending' as const, startedAt: null, completedAt: null }
  })
}
function elapsedSince(value: string | null | undefined, now: number) {
  if (!value) return null
  const start = Date.parse(value)
  return Number.isFinite(start) ? Math.max(0, Math.floor((now - start) / 1000)) : null
}
function clockText(seconds: number) {
  const safe = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(safe / 3600)
  const minutes = Math.floor((safe % 3600) / 60)
  const remainder = safe % 60
  return hours > 0
    ? `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
    : `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}
function terminalDurationSeconds(item: Pick<SimpleJob, 'status' | 'durationSeconds' | 'finishedAt' | 'createdAt'>) {
  if (isActiveStatus(item.status)) return null
  if (typeof item.durationSeconds === 'number' && Number.isFinite(item.durationSeconds) && item.durationSeconds >= 0) return Math.floor(item.durationSeconds)
  if (!item.finishedAt) return null
  const created = Date.parse(item.createdAt)
  const finished = Date.parse(item.finishedAt)
  return Number.isFinite(created) && Number.isFinite(finished) && finished >= created ? Math.floor((finished - created) / 1000) : null
}
function sourceHeading(item: SimpleJob) {
  if (item.sourceType === 'file') return item.source.split(/[\\/]/).filter(Boolean).pop() || 'Локальный файл'
  try { return new URL(item.source).hostname || 'Источник видео' } catch { return 'Источник видео' }
}
function assignOrdinals(current: Record<string, number>, ids: string[]) {
  const next = { ...current }
  let ordinal = Math.max(0, ...Object.values(next)) + 1
  for (const id of ids) if (!next[id]) next[id] = ordinal++
  return next
}
function stageProgressText(progress: JobProgress | null | undefined) {
  if (!progress) return null
  const parts: string[] = []
  if (Number.isFinite(progress.completedBytes) && progress.completedBytes !== null) {
    const received = (progress.completedBytes / 1_000_000).toFixed(1)
    parts.push(progress.totalBytes !== null && Number.isFinite(progress.totalBytes) ? `${received} / ${(progress.totalBytes / 1_000_000).toFixed(1)} МБ` : `${received} МБ получено`)
  }
  if (Number.isFinite(progress.processedSeconds) && progress.processedSeconds !== null) {
    const processed = clockText(Math.floor(progress.processedSeconds))
    parts.push(progress.totalSeconds !== null && Number.isFinite(progress.totalSeconds) ? `${processed} / ${clockText(Math.floor(progress.totalSeconds))}` : processed)
  }
  return parts.join(' · ')
}

export default function App() {
  const location = useLocation()
  const navigate = useNavigate()
  const go = async (event: React.MouseEvent<HTMLAnchorElement>, path: string) => {
    event.preventDefault()
    try { await (flushAllWorkspaceDrafts ?? flushCurrentDraft)?.(); navigate(path) } catch { /* keep the unsaved draft visible */ }
  }
  return <div className="app-frame">
    <header className="topbar">
      <Link className="brand" to="/" onClick={(event) => void go(event, '/')} aria-label="Video Transcribe — главная"><span className="brand-mark"><FileText size={18} strokeWidth={1.8} /></span><span>Video Transcribe</span></Link>
      <nav className="topnav" aria-label="Основная навигация">
        <Link className={location.pathname !== '/history' ? 'active' : ''} to="/" onClick={(event) => void go(event, '/')}>Транскрибация</Link>
        <Link className={location.pathname === '/history' ? 'active' : ''} to="/history" onClick={(event) => void go(event, '/history')}>История</Link>
      </nav>
    </header>
    <div className="workspace-preserver" style={{ display: location.pathname === '/history' ? 'none' : 'contents' }}><Workspace /></div>
    <Routes>
      <Route path="/" element={null} />
      <Route path="/history" element={<HistoryPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </div>
}

function Workspace() {
  const navigate = useNavigate()
  const storedPrefsRef = useRef<Partial<Prefs>>(readPrefs())
  const [options, setOptions] = useState<Options | null>(null)
  const [optionsError, setOptionsError] = useState('')
  const [preferencesReady, setPreferencesReady] = useState(false)
  const [job, setJob] = useState<SimpleJob | null>(null)
  const [queue, setQueue] = useState<QueueState>({ paused: false, pauseReason: null, cancelling: false, activeJob: null, waitingJobs: [], waitingCount: 0 })
  const [queueLoaded, setQueueLoaded] = useState(false)
  const [queueError, setQueueError] = useState('')
  const [taskSummaries, setTaskSummaries] = useState<Record<string, Summary>>({})
  const [openIds, setOpenIds] = useState<string[]>(() => { try { return JSON.parse(localStorage.getItem(tabsKey) ?? '[]') as string[] } catch { return [] } })
  const openIdsRef = useRef(openIds)
  openIdsRef.current = openIds
  const [ordinals, setOrdinals] = useState<Record<string, number>>(() => { try { return JSON.parse(localStorage.getItem(ordinalKey) ?? '{}') as Record<string, number> } catch { return {} } })
  const [fileBatch, setFileBatch] = useState<{ id: string; path: string; name: string }[]>([])
  const [rejectedFiles, setRejectedFiles] = useState<{ path: string; message: string }[]>([])
  const [dragActive, setDragActive] = useState(false)
  const dragDepthRef = useRef(0)
  const [cancelAllConfirm, setCancelAllConfirm] = useState(false)
  const [cancelAllError, setCancelAllError] = useState('')
  const [queueActionBusy, setQueueActionBusy] = useState(false)
  const queueActionBusyRef = useRef(false)
  const cancelAllTriggerRef = useRef<HTMLButtonElement>(null)
  const queueFocusFallbackRef = useRef<HTMLDivElement>(null)
  const [enqueueBusy, setEnqueueBusy] = useState(false)
  const [pendingTaskId, setPendingTaskId] = useState<string | null>(null)
  const tabStorageExists = useRef(localStorage.getItem(tabsKey) !== null)
  const selectedTabRef = useRef(localStorage.getItem(selectedTabKey))
  const historyReadyRef = useRef(false)
  const historyItemsRef = useRef<HistoryItem[]>([])
  const queueSignatureRef = useRef('')
  const [prefs, setPrefs] = useState<Prefs>(() => ({ ...defaults, ...storedPrefsRef.current }))
  const [sourceType, setSourceType] = useState<'url' | 'file'>('url')
  const [source, setSource] = useState('')
  const [pickerBusy, setPickerBusy] = useState(false)
  const [cancelBusy, setCancelBusy] = useState(false)
  const [retryableCancel, setRetryableCancel] = useState(false)
  const [cancelError, setCancelError] = useState('')
  const [formError, setFormError] = useState('')
  const [pollError, setPollError] = useState('')
  const [draft, setDraft] = useState('')
  const [clockNow, setClockNow] = useState(Date.now())
  const [notice, setNotice] = useState('')
  const [saveState, setSaveState] = useState<'saved' | 'saving' | 'failed'>('saved')
  const jobRef = useRef<SimpleJob | null>(null)
  const draftsRef = useRef(new DraftManager())
  const draftRef = useRef('')
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pollingGenerationRef = useRef(0)
  const cancelGenerationRef = useRef(0)
  const selectionGateRef = useRef(new SelectionGate())
  const tabButtonRefs = useRef(new Map<string, HTMLButtonElement>())
  useEffect(() => { if (preferencesReady) { try { localStorage.setItem(preferenceKey, JSON.stringify(prefs)) } catch { /* preferences are optional */ } } }, [prefs, preferencesReady])
  useEffect(() => { try { localStorage.setItem(tabsKey, JSON.stringify(openIds)); localStorage.setItem(ordinalKey, JSON.stringify(ordinals)); if (job?.id) localStorage.setItem(selectedTabKey, job.id) } catch { /* optional tab restore */ } }, [openIds, ordinals, job?.id])
  useEffect(() => {
    let active = true
    Promise.allSettled([api.options(), api.session()]).then(([optionsResult, sessionResult]) => {
      if (!active) return
      const stored = storedPrefsRef.current
      let nextPrefs: Prefs = { ...defaults, ...stored }
      if (optionsResult.status === 'fulfilled') {
        const value = optionsResult.value
        setOptions(value)
          nextPrefs = { ...nextPrefs, device: stored.device ?? value.defaultDevice, cpuThreads: clamp(stored.cpuThreads ?? value.maxCpuThreads, 1, value.maxCpuThreads) }
      } else setOptionsError(errorText(optionsResult.reason))

      if (sessionResult.status === 'rejected') setFormError(`Не удалось восстановить последнюю задачу. ${errorText(sessionResult.reason)}`)
      else if (sessionResult.value.job) {
        const restored = sessionResult.value.job
        const cleared = clearedLatestJob ?? readClearedLatest()
        const deletedLatest = Boolean(cleared && new Date(restored.createdAt).valueOf() <= new Date(cleared.createdAt).valueOf())
        clearedLatestJob = null
        if (cleared) { try { localStorage.removeItem('video-transcribe-cleared-latest') } catch { /* optional marker */ } }
        if (!deletedLatest && !tabStorageExists.current) {
          setClockNow(Date.now())
          jobRef.current = restored; setJob(restored)
          setOpenIds((ids) => ids.includes(restored.id) ? ids : [...ids, restored.id])
          setSourceType(restored.sourceType); setSource(restored.sourceType === 'url' ? restored.source : '')
          if (restored.status === 'completed') putDraft(restored.text ?? '', restored.revision)
          if (restored.notice) setNotice(restored.notice)
        }
      }
      setPrefs(nextPrefs)
      setPreferencesReady(true)
    })
    return () => { active = false }
  }, [])

  useEffect(() => {
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    let previousStatuses = new Map<string, JobStatus>()
    const poll = async () => {
      try {
        const state = await api.queue()
        if (stopped) return
        setQueueLoaded(true); setQueueError('')
        setQueue(state)
        const live = [state.activeJob, ...state.waitingJobs].filter((item): item is Summary => item !== null)
        const signature = JSON.stringify(live.map((item) => [item.id, item.status]))
        const shouldRefreshHistory = !historyReadyRef.current || signature !== queueSignatureRef.current
        queueSignatureRef.current = signature
        if (shouldRefreshHistory) {
          try { const history = await api.history(); historyItemsRef.current = history.items; historyReadyRef.current = true } catch { setPollError('Не удалось обновить историю очереди. Повторим попытку автоматически.') }
        }
        if (stopped) return
        const historyItems = historyItemsRef.current
        setTaskSummaries((current) => ({ ...current, ...Object.fromEntries([...historyItems, ...live].map((item) => [item.id, item])) }))
        const available = new Set([...live.map((item) => item.id), ...historyItems.map((item) => item.id)])
        setOrdinals((current) => assignOrdinals(current, [...openIdsRef.current, ...live.map((item) => item.id)]))
        if (historyReadyRef.current) setOpenIds((ids) => [...new Set([...ids.filter((id) => available.has(id)), ...live.map((item) => item.id)])])
        if (!jobRef.current) {
          const validOpenIds = openIdsRef.current.filter((id) => available.has(id))
          const restoreId = selectedTabRef.current && validOpenIds.includes(selectedTabRef.current) ? selectedTabRef.current : validOpenIds[0] ?? (!tabStorageExists.current ? live[0]?.id : undefined)
          if (restoreId) {
            const restored = await api.job(restoreId)
            if (!stopped && !jobRef.current) { jobRef.current = restored; setJob(restored); if (restored.status === 'completed') putDraft(restored.text ?? '', restored.revision) }
          }
        }
        const statusById = new Map([...historyItems.map((item) => [item.id, item.status] as const), ...live.map((item) => [item.id, item.status] as const)])
        const changed = [...statusById].some(([id, status]) => previousStatuses.has(id) && previousStatuses.get(id) !== status)
        previousStatuses = statusById
        if (changed) window.dispatchEvent(new Event('simple-history-changed'))
        if (state.paused || state.cancelling) setNotice(state.pauseReason?.message ?? '')
        timer = setTimeout(() => void poll(), 1000)
      } catch {
        if (!stopped) { setQueueLoaded(true); setQueueError('Нет связи с очередью. Проверьте соединение и повторите попытку.'); timer = setTimeout(() => void poll(), 2500) }
      }
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [])

  function putDraft(text: string, revision: number) {
    const current = jobRef.current
    if (current?.status === 'completed') draftsRef.current.hydrate(current.id, text, revision)
    draftRef.current = text
    setDraft(text); setSaveState('saved')
  }

  const saveSnapshot = useCallback((id: string, version: number) => {
    const record = draftsRef.current.get(id)
    if (!record) return Promise.resolve()
    const task = draftsRef.current.save(id, (snapshot, revision) => fetchJson<{ text: string; revision: number }>(`/api/simple/jobs/${encodeURIComponent(id)}/text`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: snapshot, baseRevision: revision }),
    }), version).then(() => {
      if (jobRef.current?.id === id) {
        const latest = draftsRef.current.get(id)!
        setJob((existing) => existing?.id === id ? { ...existing, text: latest.text, revision: latest.revision } : existing)
        setSaveState(latest.savedVersion >= latest.version ? 'saved' : 'saving')
      }
    }).catch((error: unknown) => {
      if (jobRef.current?.id === id) {
        if (errorStatus(error) === 409) setFormError('Текст изменился в другом окне. Ваш черновик сохранён на экране; скопируйте его перед обновлением страницы.')
        else setFormError(`Не удалось сохранить текст. Проверьте соединение и повторите попытку. ${errorText(error)}`)
        setSaveState('failed')
      }
      throw error
    })
    return task
  }, [])

  const flushDraftById = useCallback(async (id: string) => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = null
    const record = draftsRef.current.get(id)
    if (!record) return
    await draftsRef.current.flush(id, (text, revision) => fetchJson<{ text: string; revision: number }>(`/api/simple/jobs/${encodeURIComponent(id)}/text`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, baseRevision: revision }) }))
    if (jobRef.current?.id === id) {
      setJob((existing) => existing?.id === id ? { ...existing, text: record.text, revision: record.revision } : existing)
      setDraft(record.text)
      setSaveState(record.savedVersion >= record.version ? 'saved' : 'saving')
    }
  }, [])

  const flushDraft = useCallback(async () => {
    const currentJob = jobRef.current
    if (!currentJob || currentJob.status !== 'completed') {
      if (saveState === 'failed') throw new Error('Последние изменения не сохранены. Проверьте соединение; черновик останется в редакторе.')
      return
    }
    await flushDraftById(currentJob.id)
  }, [flushDraftById, saveState])

  const flushAllDrafts = useCallback(async () => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = null
    await draftsRef.current.flushAll((id, text, revision) => fetchJson<{ text: string; revision: number }>(`/api/simple/jobs/${encodeURIComponent(id)}/text`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, baseRevision: revision }) }))
    const current = jobRef.current
    const record = current ? draftsRef.current.get(current.id) : undefined
    if (current && record) {
      setJob((existing) => existing?.id === current.id ? { ...existing, text: record.text, revision: record.revision } : existing)
      setDraft(record.text)
      setSaveState(record.savedVersion >= record.version ? 'saved' : 'saving')
    }
  }, [])

  useEffect(() => {
    flushCurrentDraft = flushDraft
    return () => { if (flushCurrentDraft === flushDraft) flushCurrentDraft = null }
  }, [flushDraft])

  useEffect(() => {
    const unsubscribe = window.desktopApi?.onBeforeQuit(flushAllDrafts)
    return () => { unsubscribe?.() }
  }, [flushAllDrafts])
  useEffect(() => {
    flushAllWorkspaceDrafts = flushAllDrafts
    return () => { if (flushAllWorkspaceDrafts === flushAllDrafts) flushAllWorkspaceDrafts = null }
  }, [flushAllDrafts])

  useEffect(() => {
    clearWorkspaceJob = (jobId) => {
      setOpenIds((ids) => ids.filter((id) => id !== jobId))
      setTaskSummaries((items) => { const next = { ...items }; delete next[jobId]; return next })
      draftsRef.current.remove(jobId)
      try { if (localStorage.getItem(selectedTabKey) === jobId) localStorage.removeItem(selectedTabKey) } catch { /* optional tab restore */ }
      if (jobRef.current?.id !== jobId) return
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
      jobRef.current = null
      draftRef.current = ''
      setJob(null); setDraft(''); setSaveState('saved'); setNotice(''); setFormError('')
    }
    return () => { clearWorkspaceJob = null }
  }, [])

  const changeDraft = (value: string) => {
    const current = jobRef.current
    if (!current || current.status !== 'completed') return
    draftRef.current = value
    const version = draftsRef.current.edit(current.id, value)
    setDraft(value); setSaveState('saving'); setFormError('')
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = setTimeout(() => {
      void saveSnapshot(current.id, version).catch(() => undefined)
    }, 400)
  }

  useEffect(() => () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current) }, [])

  useEffect(() => {
    const currentJob = jobRef.current
    if (!isActive(currentJob)) return
    const generation = ++pollingGenerationRef.current
    const controller = new AbortController()
    let canceled = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const latest = await api.job(currentJob.id, controller.signal)
        if (canceled || generation !== pollingGenerationRef.current || jobRef.current?.id !== currentJob.id) return
        const previous = jobRef.current
        jobRef.current = latest; setJob(latest)
        setPollError('')
        if (!isActive(latest)) { setCancelError(''); setCancelBusy(false); setRetryableCancel(false) }
        else if (isRetryableCancellation(latest)) { setCancelBusy(false); setRetryableCancel(true); setCancelError(latest.error ?? 'Не удалось остановить обработку. Повторите отмену.') }
        if (latest.notice) setNotice(latest.notice)
        if (latest.status === 'completed' && previous.status !== 'completed') putDraft(latest.text ?? '', latest.revision)
      } catch (error) { if (!canceled && generation === pollingGenerationRef.current && !controller.signal.aborted) setPollError(`Не удалось обновить состояние. Черновик сохранён. ${errorText(error)}`) }
      if (!canceled && generation === pollingGenerationRef.current && isActive(jobRef.current)) timer = setTimeout(() => { void poll() }, 1000)
    }
    timer = setTimeout(() => { void poll() }, 0)
    return () => { canceled = true; controller.abort(); clearTimeout(timer); pollingGenerationRef.current += 1 }
  }, [job?.id, job?.status])

  const clockJobId = job && isActive(job) ? job.id : null
  useEffect(() => {
    if (!clockJobId) return
    const timer = window.setInterval(() => setClockNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [clockJobId])

  useEffect(() => {
    if (!job) return
    const handler = (event: BeforeUnloadEvent) => {
      const record = draftsRef.current.get(job.id)
      if ((record && record.version > record.savedVersion) || saveState === 'saving' || saveState === 'failed') {
        event.preventDefault()
        event.returnValue = ''
      }
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [job, saveState])

  const active = job !== null && isActiveStatus(job.status)
  const formLocked = enqueueBusy || queue.cancelling
  const canSubmit = Boolean(options && !formLocked && (sourceType === 'url' ? isHttpUrl(source.trim()) : fileBatch.length > 0))
  const showInitialLoading = !preferencesReady && !job

  const pickFile = async () => {
    setPickerBusy(true); setFormError('')
    try {
      const result = window.desktopApi ? await window.desktopApi.pickFiles() : await api.pick()
      appendSelection(result)
    } catch (error) { setFormError(errorText(error)) } finally { setPickerBusy(false) }
  }

  const appendSelection = (result: FileSelection) => {
    setFileBatch((rows) => [...rows, ...result.files.map((file) => ({ ...file, id: crypto.randomUUID() }))])
    setRejectedFiles(result.rejected)
    if (result.files.length || result.rejected.length) setSourceType('file')
  }
  const selectTask = useCallback(async (id: string) => {
    if (jobRef.current?.id === id) return
    const generation = selectionGateRef.current.begin()
    setPendingTaskId(id)
    try {
      await flushDraft()
      const latest = await api.job(id)
      if (!selectionGateRef.current.isCurrent(generation)) return
      cancelGenerationRef.current += 1
      setCancelBusy(false); setRetryableCancel(false); setCancelError('')
      setTaskSummaries((current) => ({ ...current, [id]: latest }))
      jobRef.current = latest; setJob(latest); setClockNow(Date.now())
      if (latest.status === 'completed') putDraft(latest.text ?? '', latest.revision)
      else putDraft('', latest.revision)
      setNotice(latest.notice ?? '')
      setFormError(''); setPollError('')
    } catch (error) { if (selectionGateRef.current.isCurrent(generation)) setFormError(`Не удалось открыть задачу. ${errorText(error)}`) }
    finally { if (selectionGateRef.current.isCurrent(generation)) setPendingTaskId(null) }
  }, [flushDraft])
  const handleTaskTabKey = (event: React.KeyboardEvent<HTMLButtonElement>, index: number) => {
    let targetIndex: number | null = null
    if (event.key === 'ArrowRight') targetIndex = (index + 1) % openIds.length
    else if (event.key === 'ArrowLeft') targetIndex = (index - 1 + openIds.length) % openIds.length
    else if (event.key === 'Home') targetIndex = 0
    else if (event.key === 'End') targetIndex = openIds.length - 1
    if (targetIndex === null || openIds.length === 0) return
    event.preventDefault()
    const id = openIds[targetIndex]
    const tab = tabButtonRefs.current.get(id)
    tab?.focus({ preventScroll: true })
    tab?.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'auto' })
    void selectTask(id)
  }
  const closeTask = async (id: string) => {
    const target = id === jobRef.current?.id ? jobRef.current : null
    if (target && isActiveStatus(target.status)) return
    if (target) { try { await flushDraft() } catch { return } }
    const next = openIds.filter((item) => item !== id)
    setOpenIds(next)
    if (target) {
      const nearest = next[Math.min(openIds.indexOf(id), next.length - 1)]
      if (nearest) await selectTask(nearest)
      else { jobRef.current = null; setJob(null); putDraft('', 0) }
    }
  }

  useEffect(() => {
    openWorkspaceTask = (id) => {
      setOpenIds((ids) => ids.includes(id) ? ids : [...ids, id])
      setOrdinals((current) => assignOrdinals(current, [id]))
      navigate('/')
      void selectTask(id)
    }
    return () => { openWorkspaceTask = null }
  }, [navigate, selectTask])

  useEffect(() => {
    const preventFileNavigation = (event: DragEvent) => {
      if (Array.from(event.dataTransfer?.types ?? []).includes('Files')) event.preventDefault()
    }
    const resetDropState = () => { dragDepthRef.current = 0; setDragActive(false) }
    const handleGlobalDrop = (event: DragEvent) => { preventFileNavigation(event); resetDropState() }
    window.addEventListener('dragover', preventFileNavigation)
    window.addEventListener('drop', handleGlobalDrop)
    window.addEventListener('dragend', resetDropState)
    window.addEventListener('blur', resetDropState)
    return () => { window.removeEventListener('dragover', preventFileNavigation); window.removeEventListener('drop', handleGlobalDrop); window.removeEventListener('dragend', resetDropState); window.removeEventListener('blur', resetDropState) }
  }, [])

  useEffect(() => {
    if (!formLocked) return
    dragDepthRef.current = 0
    const frame = requestAnimationFrame(() => setDragActive(false))
    return () => cancelAnimationFrame(frame)
  }, [formLocked])

  const enqueueCurrent = async () => {
    if (!canSubmit) return
    setEnqueueBusy(true); setFormError('')
    try {
      await flushDraft()
      if (sourceType === 'url') {
        const created = await fetchJson<SimpleJob>('/api/simple/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sourceType: 'url', url: source.trim(), ...prefs }) })
        setTaskSummaries((current) => ({ ...current, [created.id]: created }))
        setOpenIds((ids) => ids.includes(created.id) ? ids : [...ids, created.id])
        setOrdinals((current) => assignOrdinals(current, [created.id]))
        await selectTask(created.id)
      } else {
        const result = await api.batch(fileBatch.map((file) => file.path), prefs)
        setRejectedFiles(result.rejected)
        setFileBatch([])
        if (result.jobs.length) {
          setTaskSummaries((current) => ({ ...current, ...Object.fromEntries(result.jobs.map((item) => [item.id, item])) }))
          setOpenIds((ids) => [...new Set([...ids, ...result.jobs.map((item) => item.id)])])
          setOrdinals((current) => assignOrdinals(current, result.jobs.map((item) => item.id)))
          await selectTask(result.jobs[0].id)
        }
      }
      setQueue(await api.queue())
    } catch (error) { setFormError(`Не удалось добавить задачу. ${errorText(error)}`) } finally { setEnqueueBusy(false) }
  }

  const resumeQueue = async () => {
    if (queueActionBusyRef.current) return
    queueActionBusyRef.current = true; setQueueActionBusy(true); setFormError('')
    try { setQueue(await api.resume()) } catch (error) { setFormError(`Не удалось продолжить очередь. ${errorText(error)}`) }
    finally { queueActionBusyRef.current = false; setQueueActionBusy(false) }
  }
  const stopQueue = async () => {
    if (queueActionBusyRef.current) return
    queueActionBusyRef.current = true; setQueueActionBusy(true); setCancelAllError('')
    try { setQueue(await api.cancelAll()); setCancelAllConfirm(false) }
    catch (error) { setCancelAllError(`Не удалось отменить очередь. ${errorText(error)}`) }
    finally { queueActionBusyRef.current = false; setQueueActionBusy(false) }
  }

  const retryTask = async () => {
    const sourceJob = jobRef.current
    if (!sourceJob || !['failed', 'interrupted', 'cancelled'].includes(sourceJob.status)) return
    setFormError(''); setPollError(''); setEnqueueBusy(true)
    try {
      const payload = { sourceType: sourceJob.sourceType, ...(sourceJob.sourceType === 'url' ? { url: sourceJob.source } : { path: sourceJob.source }), profile: sourceJob.profile, device: sourceJob.device, cpuThreads: sourceJob.cpuThreads, restorePunctuation: sourceJob.restorePunctuation, ...diarizationRequestFields({ diarizationEnabled: sourceJob.diarizationEnabled === true, diarizationDevice: sanitizeDiarizationPrefs(sourceJob).diarizationDevice }) }
      const created = await fetchJson<SimpleJob>('/api/simple/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      setTaskSummaries((current) => ({ ...current, [created.id]: created }))
      setOpenIds((ids) => [...new Set([...ids, created.id])])
      setOrdinals((current) => assignOrdinals(current, [created.id]))
      setQueue(await api.queue())
      await selectTask(created.id)
    } catch (error) { setFormError(`Не удалось добавить повторную попытку. ${safeError(errorText(error)) ?? errorText(error)}`) }
    finally { setEnqueueBusy(false) }
  }
  const submit = (event: React.FormEvent) => { event.preventDefault(); void enqueueCurrent() }

  const cancelJob = async () => {
    const current = jobRef.current
    if (!isActive(current) || cancelBusy) return
    const cancelGeneration = cancelGenerationRef.current
    const isStaleCancel = () => cancelGenerationRef.current !== cancelGeneration || jobRef.current?.id !== current.id
    setCancelBusy(true); setRetryableCancel(false); setCancelError('')
    try {
      const updated = await api.cancel(current.id)
      if (isStaleCancel()) return
      const previous = jobRef.current
      jobRef.current = updated
      setJob(updated)
      if (updated.status === 'completed' && previous?.status !== 'completed') putDraft(updated.text ?? '', updated.revision)
      if (updated.notice) setNotice(updated.notice)
      const retryable = isRetryableCancellation(updated)
      setCancelBusy(false)
      setRetryableCancel(retryable)
      setCancelError(retryable ? updated.error ?? 'Не удалось остановить обработку. Повторите отмену.' : '')
    } catch (error) {
      if (isStaleCancel()) return
      const code = errorStatus(error)
      let confirmedStatus: JobStatus | null = null
      let latestJob: SimpleJob | null = null
      try {
        const latest = await api.job(current.id)
        if (isStaleCancel()) return
        const previous = jobRef.current
        jobRef.current = latest; setJob(latest)
        if (latest.status === 'completed' && previous?.status !== 'completed') putDraft(latest.text ?? '', latest.revision)
        if (latest.notice) setNotice(latest.notice)
        if (!isActive(latest)) { setCancelBusy(false); setCancelError(''); setRetryableCancel(false); return }
        confirmedStatus = latest.status
        latestJob = latest
      } catch { /* Keep the last confirmed active state and the actionable cancellation error. */ }
      if (isStaleCancel()) return
      setCancelBusy(false)
      const retryable = code === 503 || isRetryableCancellation(latestJob)
      setRetryableCancel(retryable)
      const message = code === 404
        ? confirmedStatus ? `Отмена недоступна в текущем запуске. Текущее состояние задачи: ${statusLabels[confirmedStatus]}.` : 'Отмена недоступна в текущем запуске. Не удалось проверить состояние; показано последнее подтверждённое состояние задачи.'
        : retryable ? latestJob?.error ?? 'Не удалось остановить обработку. Повторите отмену.' : confirmedStatus ? `Не удалось отправить запрос на отмену. Текущее состояние задачи: ${statusLabels[confirmedStatus]}. Попробуйте ещё раз.` : 'Не удалось отправить запрос на отмену и проверить состояние. Показано последнее подтверждённое состояние; попробуйте ещё раз.'
      setCancelError(message)
    }
  }

  const copyText = async () => {
    const id = jobRef.current?.id
    if (!id) return
    try { const text = await flushAndRead(id, flushDraftById, (capturedId) => draftsRef.current.get(capturedId)?.text ?? ''); await navigator.clipboard.writeText(text); if (jobRef.current?.id === id) setNotice('Текст скопирован.') }
    catch { setFormError(saveState === 'failed' ? 'Сначала сохраните последние изменения. Черновик останется на экране.' : 'Не удалось скопировать текст. Проверьте разрешение браузера на доступ к буферу обмена.') }
  }
  const download = async (format: 'txt' | 'md') => {
    const task = jobRef.current
    if (!task) return
    const id = task.id
    try {
      await flushAndRead(id, flushDraftById, (capturedId) => capturedId)
      const response = await fetch(`${API_BASE}/api/simple/jobs/${encodeURIComponent(id)}/download?format=${format}`)
      if (!response.ok) throw new Error(response.status === 404 ? 'Файл больше недоступен. Создайте новую расшифровку.' : `Ошибка загрузки (${response.status}).`)
      const blob = await response.blob(); const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = task.files.find((item) => item.format === format)?.name ?? `transcript.${format}`; anchor.click(); URL.revokeObjectURL(url)
    } catch (error) { setFormError(saveState === 'failed' ? 'Не удалось сохранить последние изменения, поэтому файл не скачан. Черновик останется на экране.' : errorText(error)) }
  }
  return <main className="workspace">
    <aside className="settings-pane">
      <div className="settings-scroll">
      <div className="pane-heading"><div><p className="eyebrow">Новая задача</p><h1>Расшифровка видео</h1></div></div>
      <form id="new-task-form" onSubmit={submit}>
        <fieldset className="source-field" disabled={formLocked}>
          <legend>Источник видео</legend>
          <div className="source-tabs" role="radiogroup" aria-label="Тип источника">
            <label className={sourceType === 'url' ? 'selected' : ''}><input type="radio" name="source-type" value="url" checked={sourceType === 'url'} onChange={() => setSourceType('url')} /><span>По ссылке</span></label>
            <label className={sourceType === 'file' ? 'selected' : ''}><input type="radio" name="source-type" value="file" checked={sourceType === 'file'} onChange={() => setSourceType('file')} /><span>С компьютера</span></label>
          </div>
          {sourceType === 'url' ? <label className="field-label" htmlFor="source-url">Ссылка на видео<input id="source-url" type="url" placeholder="https://…" value={source} onChange={(event) => setSource(event.target.value)} /></label> : <div className="file-picker"><div
            className={`file-dropzone${dragActive ? ' drag-active' : ''}`}
            onDragEnter={(event) => {
              if (!Array.from(event.dataTransfer.types).includes('Files')) return
              event.preventDefault()
              if (!window.desktopApi || formLocked) { dragDepthRef.current = 0; setDragActive(false); return }
              dragDepthRef.current += 1
              setDragActive(true)
            }}
            onDragOver={(event) => {
              if (!Array.from(event.dataTransfer.types).includes('Files')) return
              event.preventDefault()
              event.dataTransfer.dropEffect = window.desktopApi && !formLocked ? 'copy' : 'none'
            }}
            onDragLeave={(event) => {
              if (!Array.from(event.dataTransfer.types).includes('Files')) return
              dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)
              if (dragDepthRef.current === 0) setDragActive(false)
            }}
            onDrop={(event) => {
              const isFileDrag = Array.from(event.dataTransfer.types).includes('Files')
              if (!isFileDrag) { dragDepthRef.current = 0; setDragActive(false); return }
              event.preventDefault()
              dragDepthRef.current = 0
              setDragActive(false)
              const files = Array.from(event.dataTransfer.files)
              if (!files.length) { setRejectedFiles([{ path: '', message: 'Папки нельзя добавить. Выберите отдельные файлы.' }]); return }
              if (formLocked) { setRejectedFiles([{ path: '', message: 'Добавление файлов сейчас недоступно. Повторите попытку через мгновение.' }]); return }
              if (!window.desktopApi) { setRejectedFiles([{ path: '', message: 'Перетаскивание доступно в настольном приложении. Используйте «Добавить файлы».' }]); return }
              void window.desktopApi.getDroppedFiles(files).then(appendSelection).catch((error: unknown) => setFormError(errorText(error)))
            }}
          ><button type="button" className="secondary-button" onClick={() => void pickFile()} disabled={pickerBusy || formLocked}>{pickerBusy ? <LoaderCircle size={16} className="spin" /> : <Plus size={16} />}Добавить файлы</button><span>{dragActive ? 'Отпустите файлы, чтобы добавить' : formLocked ? 'Добавление файлов сейчас недоступно' : window.desktopApi ? 'или перетащите файлы сюда' : 'Выберите несколько файлов на компьютере'}</span></div>{fileBatch.length > 0 && <ul className="file-batch-list" aria-label="Файлы для добавления">{fileBatch.map((file) => <li key={file.id} title={file.path}><span>{file.name}</span><button type="button" className="icon-button" aria-label={`Убрать ${file.name}`} onClick={() => setFileBatch((rows) => rows.filter((row) => row.id !== file.id))}><X size={15} /></button></li>)}</ul>}{rejectedFiles.length > 0 && <div className="inline-error" role="status">{rejectedFiles.map((file, index) => <div key={`${file.path}-${index}`}>{file.path ? `${file.path}: ` : ''}{file.message}</div>)}</div>}</div>}
        </fieldset>

        <fieldset className="profile-field" disabled={formLocked || !options}>
          <legend>Качество распознавания</legend>
          <div className="profile-options">{(['quality', 'balanced', 'economy'] as Profile[]).map((profile) => <label key={profile} className={`profile-option ${prefs.profile === profile ? 'chosen' : ''}`}><input type="radio" name="profile" value={profile} checked={prefs.profile === profile} onChange={() => setPrefs((value) => ({ ...value, profile }))} /><span className="radio-mark" /><span className="profile-text"><strong>{profile === 'quality' ? 'Качество' : profile === 'balanced' ? 'Баланс' : 'Экономный'}</strong><small>{profileCopy[profile]}</small></span></label>)}</div>
        </fieldset>

        <fieldset className="device-field" disabled={formLocked || !options}>
          <legend>Устройство</legend>
          <div className="segmented" role="radiogroup" aria-label="Устройство для распознавания">
            <button type="button" role="radio" aria-checked={prefs.device === 'cpu'} className={prefs.device === 'cpu' ? 'selected' : ''} onClick={() => setPrefs((value) => ({ ...value, device: 'cpu' }))}>Процессор</button>
            <button type="button" role="radio" aria-checked={prefs.device === 'gpu'} className={prefs.device === 'gpu' ? 'selected' : ''} disabled={!options?.devices.gpu.available} title={options?.devices.gpu.available ? undefined : 'Видеокарта недоступна. Можно продолжить на процессоре.'} onClick={() => setPrefs((value) => ({ ...value, device: 'gpu' }))}>Видеокарта{options && !options.devices.gpu.available ? <span className="unavailable">Недоступно</span> : null}</button>
          </div>
          {prefs.device === 'cpu' && options && <label className="thread-control" htmlFor="cpu-threads"><span>Потоки процессора</span><input id="cpu-threads" type="number" min={1} max={options.maxCpuThreads} value={prefs.cpuThreads} onChange={(event) => setPrefs((value) => ({ ...value, cpuThreads: clamp(Number(event.target.value) || 1, 1, options.maxCpuThreads) }))} /></label>}
        </fieldset>

        <fieldset className="punctuation-field" disabled={formLocked}>
          <label className="check-row"><input type="checkbox" checked={prefs.restorePunctuation} onChange={(event) => setPrefs((value) => ({ ...value, restorePunctuation: event.target.checked }))} /><span><strong>Восстанавливать пунктуацию</strong><small>Для русской речи</small></span></label>
        </fieldset>
        <fieldset className="diarization-field" disabled={formLocked}>
          <label className="check-row"><input type="checkbox" checked={prefs.diarizationEnabled} onChange={(event) => setPrefs((value) => ({ ...value, diarizationEnabled: event.target.checked }))} /><span><strong>Разделять по говорящим</strong><small>До 8 говорящих</small></span></label>
          {prefs.diarizationEnabled && <>
            <label className="diarization-device-label" htmlFor="diarization-device">Устройство для разделения говорящих</label>
            <select id="diarization-device" className="diarization-device-select" value={prefs.diarizationDevice} onChange={(event) => setPrefs((value) => ({ ...value, diarizationDevice: event.target.value }))}>
              {(() => {
                const devices = options?.diarization?.devices ?? []
                const withBaseOptions = [
                  ...devices,
                  ...(!devices.some((device) => device.id === 'auto') ? [{ id: 'auto', name: 'Automatic', backend: 'auto', available: false, reason: null }] : []),
                  ...(!devices.some((device) => device.id === 'cpu') ? [{ id: 'cpu', name: 'CPU', backend: 'cpu', available: false, reason: null }] : []),
                ]
                const chosenExists = withBaseOptions.some((device) => device.id === prefs.diarizationDevice)
                const visibleDevices = chosenExists ? withBaseOptions : [...withBaseOptions, { id: prefs.diarizationDevice, name: prefs.diarizationDevice, backend: 'unknown', available: false, reason: 'Сохранённое устройство больше не обнаружено.' }]
                return visibleDevices.map((device) => {
                  const isAuto = device.id === 'auto'
                  const isCpu = device.id === 'cpu'
                  const backend = device.backend === 'cuda' ? 'CUDA' : device.backend === 'vulkan' ? 'Vulkan' : ''
                  const label = isAuto ? 'Авто' : isCpu ? 'Процессор' : `${device.name}${backend ? ` (${backend})` : ''}`
                  return <option key={device.id} value={device.id}>{label}{device.available ? '' : ' — недоступно'}</option>
                })
              })()}
            </select>
            {(() => {
              const selected = options?.diarization?.devices.find((device) => device.id === prefs.diarizationDevice)
              const missingSelection = Boolean(options?.diarization && !selected)
              const unavailableReason = missingSelection
                ? 'Сохранённое устройство больше не обнаружено. Выберите доступное устройство вручную.'
                : selected && !selected.available
                  ? diarizationReason(selected.reason)
                  : options?.diarization && !options.diarization.ready
                    ? 'Модель или компонент разделения говорящих пока не готовы.'
                    : !options?.diarization
                      ? 'Сервер не сообщил состояние разделения говорящих; задача может быть отклонена.'
                      : ''
              return <>
                <p className={`diarization-readiness${unavailableReason ? ' unavailable' : ''}`} role="status">{unavailableReason ? `Недоступно: ${unavailableReason}` : `Готово · ${prefs.diarizationDevice === 'auto' ? 'Автоматический выбор' : diarizationDeviceLabel(prefs.diarizationDevice, options?.diarization?.devices)}`}</p>
                {unavailableReason && <p className="diarization-submit-note">Настройки сохранены. При запуске сервер может отклонить задачу.</p>}
              </>
            })()}
            {!options?.diarization && <p className="diarization-readiness">Состояние устройств пока неизвестно.</p>}
          </>}
        </fieldset>
        {!options && !optionsError && <p className="muted">Загружаем настройки…</p>}
        {optionsError && <div className="inline-error" role="alert">{optionsError}<button type="button" className="text-button" onClick={() => { setOptionsError(''); api.options().then(setOptions).catch((error) => setOptionsError(errorText(error))) }}>Повторить</button></div>}
      </form>
      </div>
      <div className="settings-actions"><button className="primary-button start-button" type="submit" form="new-task-form" disabled={!canSubmit}>{enqueueBusy ? <><LoaderCircle size={17} className="spin" />Добавляем…</> : 'Добавить в очередь'}</button><div className="settings-foot"><span className="privacy-dot" />Файлы и результаты остаются на этом компьютере</div></div>
    </aside>
    <section className="result-pane" aria-label="Результат расшифровки">
      <div className="queue-summary" aria-label="Состояние очереди" ref={queueFocusFallbackRef} tabIndex={-1}>
        <span>{queue.cancelling ? 'Останавливаем очередь…' : queueError ? 'Нет связи с очередью' : !queueLoaded ? 'Проверяем очередь…' : queue.paused ? 'Очередь на паузе' : queue.activeJob ? `Сейчас: ${sourceHeading(queue.activeJob as SimpleJob)}` : queue.waitingCount > 0 ? 'Задачи ожидают запуска' : 'Очередь свободна'}</span>
        <span>Ожидают: {queue.waitingCount}</span>
        {queueError && <span className="queue-error" role="alert">{queueError}</span>}
        {queue.paused && <><span className="queue-reason">{queue.pauseReason?.message || 'Очередь приостановлена'}</span><button className="text-button" type="button" disabled={queueActionBusy} onClick={() => void resumeQueue()}>{queueActionBusy ? 'Продолжаем…' : 'Продолжить очередь'}</button></>}
        {(queue.activeJob || queue.waitingCount > 0) && <button ref={cancelAllTriggerRef} className="text-button danger-text" type="button" disabled={queueActionBusy || queue.cancelling} onClick={() => { setCancelAllError(''); setCancelAllConfirm(true) }}>{queueActionBusy ? 'Отменяем…' : 'Отменить всё'}</button>}
      </div>
      {openIds.length > 0 && <div className="task-tabs" role="tablist" aria-label="Задачи в рабочем пространстве">{openIds.map((id, index) => { const selected = job?.id === id; const summary = taskSummaries[id] ?? [queue.activeJob, ...queue.waitingJobs].find((item) => item?.id === id); const label = summary ? sourceHeading(summary as SimpleJob) : id === job?.id ? sourceHeading(job) : `Задача ${ordinals[id] ?? ''}`; const status = summary?.status ?? (selected ? job?.status : undefined); const canClose = status !== undefined && !isActiveStatus(status); const isRovingTab = selected || (index === 0 && !openIds.some((openId) => openId === job?.id)); return <div className={`task-tab ${selected ? 'selected' : ''}`} role="presentation" key={id}><button ref={(element) => { if (element) tabButtonRefs.current.set(id, element); else tabButtonRefs.current.delete(id) }} type="button" role="tab" tabIndex={isRovingTab ? 0 : -1} aria-selected={selected} aria-busy={pendingTaskId === id || undefined} aria-controls="task-panel" id={`tab-${id}`} title={summary?.source ?? (selected ? job?.source : label)} onKeyDown={(event) => handleTaskTabKey(event, index)} onClick={() => void selectTask(id)}>{ordinals[id] ? `${ordinals[id]}. ` : ''}{label}<span className={`tab-status ${status ?? ''}`}>{pendingTaskId === id ? 'Открываем…' : status ? statusLabels[status] : ''}</span>{pendingTaskId === id && <span className="sr-only">Открываем задачу…</span>}</button><button type="button" className="tab-close" aria-label={`Закрыть ${label}`} disabled={!canClose} onClick={() => void closeTask(id)}><X size={14} /></button></div> })}</div>}
      {showInitialLoading ? <div className="initial-loading" role="status"><LoaderCircle size={18} className="spin" />Загружаем настройки…</div> : !job ? <Welcome /> : <>
        <div id="task-panel" role="tabpanel" aria-labelledby={`tab-${job.id}`} aria-busy={pendingTaskId !== null || undefined} className="task-panel"><div className="result-heading"><div className="result-heading-title"><p className="eyebrow">{active ? 'Обработка задачи' : 'Результат'}</p><h2 title={job.source}>{job.status === 'completed' ? 'Текст расшифровки' : sourceHeading(job)}</h2><p className="job-settings">{profileLabels[job.profile]} · {job.device === 'gpu' ? 'Видеокарта' : `Процессор, ${job.cpuThreads} поток.`} · {job.restorePunctuation ? 'Пунктуация включена' : 'Пунктуация выключена'} · {job.diarizationEnabled ? `Диаризация включена · ${diarizationDeviceLabel(job.diarizationDevice ?? job.diarization?.requestedDevice, options?.diarization?.devices)}` : 'Диаризация выключена'}{job.diarizationEnabled && job.status === 'completed' && job.diarization?.effectiveDevice && ` · Фактически: ${job.diarization.effectiveDevice.name}`}</p>{active && <p className="source-caption" title={sourceLabel(job)}>Источник: {sourceLabel(job)}</p>}</div>{job.status === 'completed' && <div className="result-actions"><span className={`save-indicator ${saveState}`} aria-live="polite">{saveState === 'saved' ? <><Check size={14} />Сохранено</> : saveState === 'saving' ? <><LoaderCircle size={14} className="spin" />Сохраняем…</> : 'Не сохранено'}</span><button type="button" className="secondary-button" onClick={() => void copyText()}><Clipboard size={16} />Копировать</button><button type="button" className="secondary-button" onClick={() => void download('txt')}>Скачать TXT</button><button type="button" className="secondary-button" onClick={() => void download('md')}>Скачать MD</button></div>}</div>
        {(() => {
          const checkpoints = checkpointsFor(job)
          const currentStageOrder = stageOrderFor(job.diarizationEnabled === true) as readonly Stage[]
          const currentStage = job.progress?.stage && currentStageOrder.includes(job.progress.stage) ? job.progress.stage : job.stage
          const currentCheckpoint = checkpoints.find((checkpoint) => checkpoint.stage === currentStage)
          const progress = job.progress
          const percent = typeof progress?.percent === 'number' && Number.isFinite(progress.percent) ? Math.max(0, Math.min(100, progress.percent)) : null
          const quiet = Boolean(active && progress?.updatedAt && ['preparation', 'source'].includes(currentStage) && elapsedSince(progress.updatedAt, clockNow) !== null && elapsedSince(progress.updatedAt, clockNow)! > 20)
          const stageAge = elapsedSince(currentCheckpoint?.startedAt ?? null, clockNow)
          const detail = stageProgressText(progress)
          const heading = progress?.message?.trim() || stageName(currentStage)
          const waitingSeconds = job.status === 'queued' ? elapsedSince(job.createdAt, clockNow) : job.queueWaitSeconds ?? null
          const hasSplitTiming = job.startedAt !== undefined || job.queueWaitSeconds !== undefined || job.processingDurationSeconds !== undefined
          const processingSeconds = job.status === 'running' || job.status === 'cancelling'
            ? elapsedSince(job.startedAt, clockNow)
            : job.processingDurationSeconds ?? (job.startedAt || job.finishedAt ? null : job.durationSeconds ?? null)
          const queuedPosition = queue.waitingJobs.find((item) => item.id === job.id)?.queuePosition ?? taskSummaries[job.id]?.queuePosition ?? null
          const retryableCancellation = retryableCancel || isRetryableCancellation(job)
          const cancellationPending = job.status === 'cancelling' && !retryableCancellation
          return <section className={`work-panel ${job.status}`} aria-label="Этапы расшифровки">
            {job.status === 'queued' && <div className="queue-position"><div><strong>Ожидает запуска</strong><p>Место в очереди: {queuedPosition ?? '—'}</p></div><span className="status-badge queued">Ожидает</span><button type="button" className="cancel-button" onClick={() => void cancelJob()} disabled={cancelBusy} aria-label={cancelBusy ? 'Отменяем задачу' : 'Отмена'}>{cancelBusy ? <LoaderCircle size={16} className="spin" /> : <X size={16} />}<span>{cancelBusy ? 'Отменяем…' : 'Отмена'}</span></button></div>}
            {active && job.status !== 'queued' && <div className="work-current"><div className="work-current-icon"><LoaderCircle size={19} className="spin" /></div><div className="work-current-copy"><strong>{heading}</strong><p>{quiet ? 'Ожидаем данные…' : stageName(currentStage)}{stageAge !== null && <span className="elapsed"> · этап {clockText(stageAge)}</span>}</p></div><span className={`status-badge ${job.status}`}>{statusLabels[job.status]}</span><button type="button" className="cancel-button" onClick={() => void cancelJob()} disabled={cancelBusy || cancellationPending} aria-label={cancelBusy || cancellationPending ? 'Отменяем задачу' : 'Отмена'}>{cancelBusy || cancellationPending ? <LoaderCircle size={16} className="spin" /> : <X size={16} />}<span>{cancelBusy || cancellationPending ? 'Отменяем…' : 'Отмена'}</span></button></div>}
            {!active && <div className={`work-terminal-heading ${job.status}`}><span>{job.status === 'completed' ? <Check size={18} /> : job.status === 'cancelled' ? <Minus size={18} /> : <AlertCircle size={18} />}</span><div><strong>{job.status === 'completed' ? 'Расшифровка готова' : job.status === 'cancelled' ? 'Отменена' : job.status === 'failed' ? 'Не удалось расшифровать видео' : 'Задача была прервана'}</strong>{job.status !== 'completed' && <p>{job.status === 'cancelled' ? 'Обработка остановлена. Можно запустить новую задачу.' : safeError(job.error) || 'Можно запустить новую попытку с текущими настройками.'}</p>}</div></div>}
            {hasSplitTiming ? <div className={job.status === 'completed' ? 'completed-timing' : undefined}><div className="overall-time"><span>Ожидание в очереди</span><strong>{waitingSeconds !== null ? clockText(waitingSeconds) : '—'}</strong></div><div className="overall-time"><span>Обработка</span><strong>{processingSeconds !== null ? clockText(processingSeconds) : job.status === 'interrupted' && !job.finishedAt ? 'Время неизвестно' : '—'}</strong></div></div> : <div className="overall-time"><span>Общее время</span><strong>{terminalDurationSeconds(job) !== null ? clockText(terminalDurationSeconds(job)!) : '—'}</strong></div>}
            {checkpoints.length > 0 && <ol className="checkpoint-list" aria-label="Этапы обработки">
              {checkpoints.map((checkpoint) => <li key={checkpoint.stage} className={`checkpoint ${checkpoint.status}${checkpoint.stage === currentStage && active ? ' current' : ''}`} aria-current={checkpoint.stage === currentStage && active ? 'step' : undefined}>
                <span className="checkpoint-icon" aria-hidden="true">{checkpoint.status === 'completed' ? <Check size={14} /> : checkpoint.status === 'active' && active ? <LoaderCircle size={14} className="spin" /> : checkpoint.status === 'failed' ? <AlertCircle size={14} /> : checkpoint.status === 'skipped' ? <span className="skip-mark">–</span> : checkpoint.status === 'cancelled' ? <Minus size={14} /> : <Circle size={12} />}</span>
                <span className="checkpoint-label">{stageName(checkpoint.stage)}{(checkpoint.status === 'skipped' || checkpoint.status === 'cancelled') && <small>{checkpoint.status === 'skipped' ? 'Пропущено' : 'Отменено'}</small>}</span>
              </li>)}
            </ol>}
            {active && job.status !== 'queued' && <div className="stage-progress" aria-label="Прогресс текущей операции">
              {percent !== null ? <><div className="progress-track" role="progressbar" aria-label="Прогресс текущей операции" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}><span style={{ width: `${percent}%` }} /></div><div className="progress-details"><span>{heading}</span><strong>{percent}%</strong></div></> : <><div className="progress-track indeterminate" role="progressbar" aria-label={`${heading}: выполняется`} aria-valuetext="Выполняется" /><div className="progress-details"><span>{heading}</span><strong>Выполняется…</strong></div></>}
              {detail && <p className="progress-data">{detail}</p>}
            </div>}
            {pollError && <div className="poll-warning" role="alert"><AlertCircle size={16} />{pollError}</div>}
            {cancelError && <div className="poll-warning" role="alert"><AlertCircle size={16} />{cancelError}</div>}
            {(job.status === 'failed' || job.status === 'interrupted' || job.status === 'cancelled') && <button type="button" className="primary-button retry-button" disabled={enqueueBusy || active} onClick={() => void retryTask()}>{enqueueBusy ? <><LoaderCircle size={16} className="spin" />Добавляем…</> : 'Повторить попытку'}</button>}
          </section>
        })()}
        {job.status === 'completed' ? <><textarea className="transcript-editor" aria-label="Текст расшифровки" value={draft} onChange={(event) => changeDraft(event.target.value)} spellCheck placeholder="Текст расшифровки появится здесь" />{job.diarizationEnabled && <p className="notice">Номера говорящих относятся только к этой записи. Неопределённость и наложения речи отмечены в тексте.</p>}{job.detectedLanguage && job.detectedLanguage.toLowerCase().startsWith('ru') === false && job.restorePunctuation && <p className="notice">Пунктуация восстанавливается только для русской речи.</p>}{notice && <p className="notice">{notice}</p>}</> : null}
      </div></>}
      {formError && <div className="workspace-alert" role="alert">{formError}<button className="alert-dismiss" aria-label="Закрыть сообщение" onClick={() => setFormError('')}>×</button></div>}
    </section>
    <ConfirmDialog open={cancelAllConfirm} title="Отменить всю очередь?" description="Текущая обработка и ожидающие задачи будут остановлены. Готовые результаты останутся в истории." confirmLabel="Отменить всё" cancelLabel="Оставить" busyLabel="Отменяем…" busy={queueActionBusy} error={cancelAllError} fallbackFocusRef={queueFocusFallbackRef} onConfirm={() => void stopQueue()} onClose={() => { setCancelAllConfirm(false); setCancelAllError('') }} />
  </main>
}

function Welcome() { return <div className="welcome-state"><div className="welcome-icon"><FileText size={23} /></div><h2>Здесь появится ваша расшифровка</h2><p>Добавьте ссылку на видео или выберите файл на компьютере. Настройки можно оставить по умолчанию.</p><div className="welcome-hint"><span>1</span><p>Выберите источник видео слева</p></div><div className="welcome-hint"><span>2</span><p>Нажмите «Добавить в очередь»</p></div></div> }

type ConfirmDialogProps = {
  open: boolean
  title: string
  description: string
  confirmLabel: string
  cancelLabel: string
  busyLabel: string
  busy: boolean
  error: string
  fallbackFocusRef: React.RefObject<HTMLElement | null>
  onConfirm: () => void
  onClose: () => void
}

function ConfirmDialog({ open, title, description, confirmLabel, cancelLabel, busyLabel, busy, error, fallbackFocusRef, onConfirm, onClose }: ConfirmDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null)
  const cancelRef = useRef<HTMLButtonElement>(null)
  const previousFocusRef = useRef<HTMLElement | null>(null)
  const dialogId = useId()

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    if (open && !dialog.open) {
      previousFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
      dialog.showModal()
      cancelRef.current?.focus({ preventScroll: true })
    } else if (!open && dialog.open) {
      dialog.close()
      const previous = previousFocusRef.current
      const fallback = fallbackFocusRef.current
      if (previous?.isConnected) previous.focus({ preventScroll: true })
      else fallback?.focus({ preventScroll: true })
      previousFocusRef.current = null
    }
  }, [open, fallbackFocusRef])

  return <dialog ref={dialogRef} className="confirm-dialog" role="alertdialog" aria-modal="true" aria-busy={busy || undefined} aria-labelledby={`${dialogId}-title`} aria-describedby={`${dialogId}-description`}
    onCancel={(event) => { event.preventDefault(); if (!busy) onClose() }}
    onClick={(event) => {
      if (event.target !== event.currentTarget) return
      const bounds = event.currentTarget.getBoundingClientRect()
      const outside = event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom
      if (outside && !busy) onClose()
    }}>
    <h2 id={`${dialogId}-title`}>{title}</h2>
    <p id={`${dialogId}-description`}>{description}</p>
    {error && <p className="dialog-error" role="alert">{error}</p>}
    <div className="dialog-actions">
      <button ref={cancelRef} type="button" className="secondary-button" onClick={onClose} disabled={busy}>{cancelLabel}</button>
      <button type="button" className="danger-button" onClick={onConfirm} disabled={busy}>{busy ? busyLabel : confirmLabel}</button>
    </div>
  </dialog>
}

function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<HistoryItem | null>(null)
  const [busy, setBusy] = useState(false)
  const [deleteError, setDeleteError] = useState('')
  const deleteInFlightRef = useRef(false)
  const historyFallbackRef = useRef<HTMLButtonElement>(null)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [latest, setLatest] = useState<{ id: string; createdAt: string } | null>(null)
  const navigate = useNavigate()
  const refresh = useCallback(async () => {
    try { const result = await api.history(); setItems(result.items); setError('') }
    catch (reason) { setError(`Не удалось загрузить историю. ${errorText(reason)}`) }
    finally { setLoading(false) }
  }, [])
  useEffect(() => {
    const handleHistoryChanged = () => { void refresh() }
    window.addEventListener('simple-history-changed', handleHistoryChanged)
    return () => window.removeEventListener('simple-history-changed', handleHistoryChanged)
  }, [refresh])
  useEffect(() => {
    let canceled = false
    let timer: ReturnType<typeof setTimeout>
    let priorStatus: JobStatus | null = null
    const pollSession = async () => {
      try {
        const { job } = await api.session()
        if (canceled) return
        setLatest(job ? { id: job.id, createdAt: job.createdAt } : null)
        setActiveId(isActive(job) ? job.id : null)
        if (priorStatus !== null && job && job.status !== priorStatus) void refresh()
        priorStatus = job?.status ?? null
        if (isActive(job)) timer = setTimeout(() => { void pollSession() }, 1000)
      } catch { if (!canceled) setActiveId(null) }
    }
    void Promise.resolve().then(refresh)
    void pollSession()
    return () => { canceled = true; clearTimeout(timer) }
  }, [refresh])
  const downloadHistory = async (item: HistoryItem, format: 'txt' | 'md') => {
    try {
      await (flushAllWorkspaceDrafts ?? flushCurrentDraft)?.()
      const downloadUrl = item.files.find((file) => file.format === format)?.downloadUrl ?? `/api/simple/jobs/${encodeURIComponent(item.id)}/download?format=${format}`
      const response = await fetch(`${API_BASE}${downloadUrl}`)
      if (!response.ok) throw new Error(response.status === 404 ? 'Файл больше недоступен.' : `Ошибка загрузки (${response.status}).`)
      const blob = await response.blob(); const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = item.files.find((file) => file.format === format)?.name ?? `transcript.${format}`; anchor.click(); URL.revokeObjectURL(url)
    } catch (reason) { setError(errorText(reason)) }
  }
  const remove = async () => {
    if (!deleteTarget || deleteInFlightRef.current) return
    const target = deleteTarget
    deleteInFlightRef.current = true; setBusy(true); setDeleteError('')
    try { await (flushAllWorkspaceDrafts ?? flushCurrentDraft)?.(); const response = await fetch(`${API_BASE}/api/simple/history/${encodeURIComponent(target.id)}`, { method: 'DELETE' }); if (!response.ok) throw new Error(response.status === 409 ? 'Нельзя удалить задачу, пока она выполняется.' : `Ошибка удаления (${response.status}).`); if (latest?.id === target.id || (!latest && items[0]?.id === target.id)) { clearedLatestJob = { id: target.id, createdAt: target.createdAt }; try { localStorage.setItem('video-transcribe-cleared-latest', JSON.stringify(clearedLatestJob)) } catch { /* optional marker */ } } clearWorkspaceJob?.(target.id); setItems((current) => current.filter((item) => item.id !== target.id)); setDeleteTarget(null) }
    catch (reason) { setDeleteError(`Не удалось удалить задачу. ${errorText(reason)}`) } finally { deleteInFlightRef.current = false; setBusy(false) }
  }
  return <main className="history-page">
    <div className="history-title"><div><p className="eyebrow">Ваши задачи</p><h1>История расшифровок</h1><p>Готовые тексты и файлы можно скачать в любое время.</p></div><button ref={historyFallbackRef} className="primary-button" onClick={() => navigate('/')}><Plus size={17} />Новая расшифровка</button></div>
    {error && <div className="inline-error" role="alert">{error}<button className="text-button" onClick={() => { setLoading(true); void refresh() }}>Повторить</button></div>}
    {loading ? <div className="history-empty">Загружаем историю…</div> : items.length === 0 ? <div className="history-empty"><div className="welcome-icon"><FileText size={22} /></div><h2>Пока нет расшифровок</h2><p>Созданные задачи появятся здесь.</p></div> : <div className="history-table-wrap"><table className="history-table"><thead><tr><th>Источник</th><th>Дата</th><th>Статус</th><th>Файлы</th><th><span className="sr-only">Действия</span></th></tr></thead><tbody>{items.map((item) => { const activeItem = isActiveStatus(item.status); const downloadable = item.status !== 'cancelled'; const duration = terminalDurationSeconds(item); return <tr key={item.id}><td className="history-source" title={sourceLabel(item)}>{sourceLabel(item)}</td><td className="history-date"><span>{formatDate(item.createdAt)}</span>{duration !== null && <small>Длительность · {clockText(duration)}</small>}</td><td><span className={`status-badge ${item.status}`}>{statusLabels[item.status]}</span></td><td><div className="file-links">{downloadable && item.files.some((file) => file.format === 'txt') && <button className="file-link" onClick={() => void downloadHistory(item, 'txt')}>TXT</button>}{downloadable && item.files.some((file) => file.format === 'md') && <button className="file-link" onClick={() => void downloadHistory(item, 'md')}>MD</button>}</div></td><td><div className="history-action-buttons"><button className="file-link" onClick={() => { openWorkspaceTask?.(item.id); navigate('/') }}>Открыть</button><button className="icon-button" aria-label={`Удалить ${sourceLabel(item)}`} title={activeItem || item.id === activeId ? 'Нельзя удалить выполняющуюся задачу' : 'Удалить из истории'} disabled={activeItem || item.id === activeId} onClick={() => { setDeleteError(''); setDeleteTarget(item) }}><Trash2 size={17} /></button></div></td></tr> })}</tbody></table></div>}
    <ConfirmDialog open={Boolean(deleteTarget)} title="Удалить расшифровку?" description={deleteTarget ? `${sourceLabel(deleteTarget)}. Это удалит задачу из истории и файлы результата.` : ''} confirmLabel="Удалить" cancelLabel="Отмена" busyLabel="Удаляем…" busy={busy} error={deleteError} fallbackFocusRef={historyFallbackRef} onConfirm={() => void remove()} onClose={() => { setDeleteTarget(null); setDeleteError('') }} />
  </main>
}

function isHttpUrl(value: string) { try { const url = new URL(value); return url.protocol === 'http:' || url.protocol === 'https:' } catch { return false } }
function clamp(value: number, min: number, max: number) { return Math.min(max, Math.max(min, value)) }
