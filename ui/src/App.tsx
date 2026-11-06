import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { AlertCircle, Check, Circle, Clipboard, FileText, LoaderCircle, Minus, Plus, Trash2, X } from 'lucide-react'
import { API_BASE, fetchJson } from './lib/api'

type Profile = 'quality' | 'balanced' | 'economy'
type Device = 'cpu' | 'gpu'
type JobStatus = 'queued' | 'running' | 'cancelling' | 'completed' | 'failed' | 'interrupted' | 'cancelled'
type Stage = 'preparation' | 'source' | 'audio' | 'recognition' | 'postprocess' | 'result'
type Checkpoint = { stage: Stage; status: 'pending' | 'active' | 'completed' | 'skipped' | 'failed' | 'cancelled'; startedAt: string | null; completedAt: string | null }
type JobProgress = { stage: Stage; percent: number | null; completedBytes: number | null; totalBytes: number | null; processedSeconds: number | null; totalSeconds: number | null; message: string | null; updatedAt: string }
type SimpleJob = {
  id: string; sourceType: 'url' | 'file'; source: string; profile: Profile; device: Device
  cpuThreads: number; restorePunctuation: boolean; status: JobStatus
  stage: Stage; progress?: JobProgress | null; checkpoints?: Checkpoint[]; createdAt: string
  finishedAt?: string | null; durationSeconds?: number | null
  updatedAt: string; detectedLanguage: string | null; error: string | null; notice: string | null
  revision: number; text: string | null; files: { format: 'txt' | 'md'; name: string; downloadUrl: string }[]
}
type HistoryItem = Omit<SimpleJob, 'text'>
type Options = {
  profiles: { id: Profile; label: string; model: string }[]
  devices: { cpu: { available: boolean; name: string }; gpu: { available: boolean; name: string; reason: string | null } }
  defaultDevice: Device; maxCpuThreads: number
}
type ApiError = Error & { status?: number }
type Prefs = { profile: Profile; device: Device; cpuThreads: number; restorePunctuation: boolean }

const preferenceKey = 'video-transcribe-simple-preferences'
let flushCurrentDraft: (() => Promise<void>) | null = null
let clearWorkspaceJob: ((jobId: string) => void) | null = null
let clearedLatestJob: { id: string; createdAt: string } | null = null
const defaults: Prefs = { profile: 'balanced', device: 'cpu', cpuThreads: 1, restorePunctuation: true }
const stageOrder: Stage[] = ['preparation', 'source', 'audio', 'recognition', 'postprocess', 'result']
const stages: Record<Stage, string> = {
  preparation: 'Подготовка модели', source: 'Получение записи', audio: 'Подготовка аудио',
  recognition: 'Распознавание речи', postprocess: 'Пунктуация', result: 'Готово',
}
const statusLabels: Record<JobStatus, string> = {
  queued: 'Ожидает', running: 'Обрабатывается', cancelling: 'Отменяется', completed: 'Готово', failed: 'Ошибка', interrupted: 'Прервана', cancelled: 'Отменена',
}
const profileCopy: Record<Profile, string> = {
  quality: 'Более точное распознавание', balanced: 'Для обычного использования', economy: 'Меньше нагрузка на память',
}
const api = {
  options: () => fetchJson<Options>('/api/simple/options'),
  session: () => fetchJson<{ job: SimpleJob | null }>('/api/simple/session'),
  history: () => fetchJson<{ items: HistoryItem[] }>('/api/simple/history'),
  pick: () => fetchJson<{ path: string | null; name: string | null }>('/api/simple/pick-file', { method: 'POST' }),
  job: (id: string, signal?: AbortSignal) => fetchJson<SimpleJob>(`/api/simple/jobs/${encodeURIComponent(id)}`, { signal }),
  cancel: (id: string) => fetchJson<SimpleJob>(`/api/simple/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
}
function errorText(error: unknown) { return error instanceof Error ? error.message : 'Не удалось выполнить запрос. Проверьте соединение и повторите попытку.' }
function errorStatus(error: unknown) { return typeof error === 'object' && error !== null && 'status' in error ? (error as ApiError).status : undefined }
function readPrefs(): Partial<Prefs> {
  try { return JSON.parse(localStorage.getItem(preferenceKey) ?? '{}') as Partial<Prefs> } catch { return {} }
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
function checkpointsFor(job: SimpleJob): Checkpoint[] {
  const known = Array.isArray(job.checkpoints) ? job.checkpoints : []
  if (known.length === 0 && job.status === 'completed') return []
  return stageOrder.map((stage) => {
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
    try { await flushCurrentDraft?.(); navigate(path) } catch { /* keep the unsaved draft visible */ }
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
  const storedPrefsRef = useRef<Partial<Prefs>>(readPrefs())
  const [options, setOptions] = useState<Options | null>(null)
  const [optionsError, setOptionsError] = useState('')
  const [preferencesReady, setPreferencesReady] = useState(false)
  const [job, setJob] = useState<SimpleJob | null>(null)
  const [prefs, setPrefs] = useState<Prefs>(() => ({ ...defaults, ...storedPrefsRef.current }))
  const [sourceType, setSourceType] = useState<'url' | 'file'>('url')
  const [source, setSource] = useState('')
  const [selectedPath, setSelectedPath] = useState('')
  const [pickerBusy, setPickerBusy] = useState(false)
  const [submitBusy, setSubmitBusy] = useState(false)
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
  const draftRef = useRef('')
  const revisionRef = useRef(0)
  const editVersionRef = useRef(0)
  const savedVersionRef = useRef(0)
  const saveQueueRef = useRef<Promise<void>>(Promise.resolve())
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pollingGenerationRef = useRef(0)
  const cancelGenerationRef = useRef(0)
  useEffect(() => { if (preferencesReady) { try { localStorage.setItem(preferenceKey, JSON.stringify(prefs)) } catch { /* preferences are optional */ } } }, [prefs, preferencesReady])
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
        if (!deletedLatest) {
          setClockNow(Date.now())
          jobRef.current = restored; setJob(restored)
          setSourceType(restored.sourceType); setSource(restored.sourceType === 'url' ? restored.source : ''); setSelectedPath(restored.sourceType === 'file' ? restored.source : '')
          nextPrefs = { ...nextPrefs, profile: restored.profile, device: restored.device, cpuThreads: restored.cpuThreads, restorePunctuation: restored.restorePunctuation }
          if (restored.status === 'completed') putDraft(restored.text ?? '', restored.revision)
          if (restored.notice) setNotice(restored.notice)
        }
      }
      setPrefs(nextPrefs)
      setPreferencesReady(true)
    })
    return () => { active = false }
  }, [])

  function putDraft(text: string, revision: number) {
    draftRef.current = text; revisionRef.current = revision; editVersionRef.current = 0; savedVersionRef.current = 0
    setDraft(text); setSaveState('saved')
  }

  const saveSnapshot = useCallback((id: string, text: string, version: number) => {
    const task = saveQueueRef.current.then(async () => {
      const currentJob = jobRef.current
      if (!currentJob || currentJob.id !== id || version <= savedVersionRef.current) return
      setSaveState('saving')
      const response = await fetchJson<{ text: string; revision: number }>(`/api/simple/jobs/${encodeURIComponent(id)}/text`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, baseRevision: revisionRef.current }),
      })
      if (jobRef.current?.id === id) {
        revisionRef.current = response.revision
        savedVersionRef.current = Math.max(savedVersionRef.current, version)
        setJob((existing) => existing?.id === id ? { ...existing, text: response.text, revision: response.revision } : existing)
        setSaveState(savedVersionRef.current >= editVersionRef.current ? 'saved' : 'saving')
      }
    }).catch((error: unknown) => {
      if (jobRef.current?.id === id) {
        if (errorStatus(error) === 409) setFormError('Текст изменился в другом окне. Ваш черновик сохранён на экране; скопируйте его перед обновлением страницы.')
        else setFormError(`Не удалось сохранить текст. Проверьте соединение и повторите попытку. ${errorText(error)}`)
        setSaveState('failed')
      }
      throw error
    })
    saveQueueRef.current = task.catch(() => undefined)
    return task
  }, [])

  const flushDraft = useCallback(async () => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = null
    const currentJob = jobRef.current
    if (!currentJob || currentJob.status !== 'completed' || savedVersionRef.current >= editVersionRef.current) {
      await saveQueueRef.current
      if (saveState === 'failed') throw new Error('Последние изменения не сохранены. Проверьте соединение; черновик останется в редакторе.')
      return
    }
    await saveSnapshot(currentJob.id, draftRef.current, editVersionRef.current)
    if (savedVersionRef.current < editVersionRef.current) await flushDraft()
  }, [saveSnapshot, saveState])

  useEffect(() => {
    flushCurrentDraft = flushDraft
    return () => { if (flushCurrentDraft === flushDraft) flushCurrentDraft = null }
  }, [flushDraft])

  useEffect(() => {
    clearWorkspaceJob = (jobId) => {
      if (jobRef.current?.id !== jobId) return
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
      jobRef.current = null
      draftRef.current = ''
      editVersionRef.current = 0
      savedVersionRef.current = 0
      setJob(null); setDraft(''); setSaveState('saved'); setNotice(''); setFormError('')
    }
    return () => { clearWorkspaceJob = null }
  }, [])

  const changeDraft = (value: string) => {
    draftRef.current = value; editVersionRef.current += 1
    setDraft(value); setSaveState('saving'); setFormError('')
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    const version = editVersionRef.current
    saveTimerRef.current = setTimeout(() => {
      const currentJob = jobRef.current
      if (currentJob) void saveSnapshot(currentJob.id, draftRef.current, version).catch(() => undefined)
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
      if (editVersionRef.current > savedVersionRef.current || saveState === 'saving' || saveState === 'failed') {
        event.preventDefault()
        event.returnValue = ''
      }
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [job, saveState])

  const active = job !== null && isActiveStatus(job.status)
  const hasSource = sourceType === 'url' ? isHttpUrl(source.trim()) : selectedPath.length > 0
  const formLocked = active || submitBusy
  const canSubmit = Boolean(hasSource && options && !formLocked)
  const showInitialLoading = !preferencesReady && !job

  const pickFile = async () => {
    setPickerBusy(true); setFormError('')
    try {
      const result = await api.pick()
      if (result.path) { setSelectedPath(result.path); setSourceType('file') }
    } catch (error) { setFormError(errorText(error)) } finally { setPickerBusy(false) }
  }

  const startJob = async () => {
    setFormError(''); setPollError('')
    if (!hasSource) { setFormError(sourceType === 'url' ? 'Введите корректную ссылку на видео.' : 'Выберите файл на компьютере.'); return }
    cancelGenerationRef.current += 1
    setCancelBusy(false); setRetryableCancel(false); setCancelError('')
    setSubmitBusy(true)
    try {
      await flushDraft()
      const payload = { sourceType, ...(sourceType === 'url' ? { url: source.trim() } : { path: selectedPath }), profile: prefs.profile, device: prefs.device, cpuThreads: prefs.cpuThreads, restorePunctuation: prefs.restorePunctuation }
      const created = await fetchJson<SimpleJob>('/api/simple/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      setClockNow(Date.now())
      jobRef.current = created; setJob(created); setNotice(created.notice ?? ''); putDraft('', created.revision)
    } catch (error) {
      const code = errorStatus(error)
      setFormError(code === 409 ? 'Сейчас уже выполняется задача. Дождитесь её завершения.' : code === 422 ? safeError(errorText(error)) ?? errorText(error) : `Не удалось запустить расшифровку. ${safeError(errorText(error)) ?? errorText(error)}`)
      if (code === 409) {
        try {
          const { job: existing } = await api.session()
          if (existing) { setClockNow(Date.now()); jobRef.current = existing; setJob(existing); setSourceType(existing.sourceType); setSource(existing.sourceType === 'url' ? existing.source : ''); setSelectedPath(existing.sourceType === 'file' ? existing.source : ''); if (existing.status === 'completed') putDraft(existing.text ?? '', existing.revision) }
        } catch { /* keep the actionable conflict message */ }
      }
    } finally { setSubmitBusy(false) }
  }
  const submit = (event: React.FormEvent) => { event.preventDefault(); void startJob() }

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
    try { await flushDraft(); await navigator.clipboard.writeText(draftRef.current); setNotice('Текст скопирован.') }
    catch { setFormError(saveState === 'failed' ? 'Сначала сохраните последние изменения. Черновик останется на экране.' : 'Не удалось скопировать текст. Проверьте разрешение браузера на доступ к буферу обмена.') }
  }
  const download = async (format: 'txt' | 'md') => {
    try {
      await flushDraft()
      const response = await fetch(`${API_BASE}/api/simple/jobs/${encodeURIComponent(job!.id)}/download?format=${format}`)
      if (!response.ok) throw new Error(response.status === 404 ? 'Файл больше недоступен. Создайте новую расшифровку.' : `Ошибка загрузки (${response.status}).`)
      const blob = await response.blob(); const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = job!.files.find((item) => item.format === format)?.name ?? `transcript.${format}`; anchor.click(); URL.revokeObjectURL(url)
    } catch (error) { setFormError(saveState === 'failed' ? 'Не удалось сохранить последние изменения, поэтому файл не скачан. Черновик останется на экране.' : errorText(error)) }
  }
  return <main className="workspace">
    <aside className="settings-pane">
      <div className="pane-heading"><div><p className="eyebrow">Новая задача</p><h1>Расшифровка видео</h1></div></div>
      <form onSubmit={submit}>
        <fieldset className="source-field" disabled={formLocked}>
          <legend>Источник видео</legend>
          <div className="source-tabs" role="radiogroup" aria-label="Тип источника">
            <label className={sourceType === 'url' ? 'selected' : ''}><input type="radio" name="source-type" value="url" checked={sourceType === 'url'} onChange={() => setSourceType('url')} /><span>По ссылке</span></label>
            <label className={sourceType === 'file' ? 'selected' : ''}><input type="radio" name="source-type" value="file" checked={sourceType === 'file'} onChange={() => setSourceType('file')} /><span>С компьютера</span></label>
          </div>
          {sourceType === 'url' ? <label className="field-label" htmlFor="source-url">Ссылка на видео<input id="source-url" type="url" placeholder="https://…" value={source} onChange={(event) => setSource(event.target.value)} /></label> : <div className="file-picker"><button type="button" className="secondary-button" onClick={() => void pickFile()} disabled={pickerBusy}>{pickerBusy ? <LoaderCircle size={16} className="spin" /> : <Plus size={16} />}Выбрать файл</button><div className="selected-path" title={selectedPath}>{selectedPath || 'Файл не выбран'}</div></div>}
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
        {!options && !optionsError && <p className="muted">Загружаем настройки…</p>}
        {optionsError && <div className="inline-error" role="alert">{optionsError}<button type="button" className="text-button" onClick={() => { setOptionsError(''); api.options().then(setOptions).catch((error) => setOptionsError(errorText(error))) }}>Повторить</button></div>}
        <button className="primary-button start-button" type="submit" disabled={!canSubmit}>{submitBusy ? <><LoaderCircle size={17} className="spin" />Запускаем…</> : 'Расшифровать'}</button>
      </form>
      <div className="settings-foot"><span className="privacy-dot" />Файлы и результаты остаются на этом компьютере</div>
    </aside>
    <section className="result-pane" aria-label="Результат расшифровки">
      {showInitialLoading ? <div className="initial-loading" role="status"><LoaderCircle size={18} className="spin" />Загружаем настройки…</div> : !job ? <Welcome /> : <>
        <div className="result-heading"><div className="result-heading-title"><p className="eyebrow">{active ? 'Обработка задачи' : 'Результат'}</p><h2 title={job.source}>{job.status === 'completed' ? 'Текст расшифровки' : sourceHeading(job)}</h2>{active && <p className="source-caption" title={sourceLabel(job)}>Источник: {sourceLabel(job)}</p>}</div>{job.status === 'completed' && <div className="result-actions"><span className={`save-indicator ${saveState}`} aria-live="polite">{saveState === 'saved' ? <><Check size={14} />Сохранено</> : saveState === 'saving' ? <><LoaderCircle size={14} className="spin" />Сохраняем…</> : 'Не сохранено'}</span><button type="button" className="secondary-button" onClick={() => void copyText()}><Clipboard size={16} />Копировать</button><button type="button" className="secondary-button" onClick={() => void download('txt')}>Скачать TXT</button><button type="button" className="secondary-button" onClick={() => void download('md')}>Скачать MD</button></div>}</div>
        {(() => {
          const checkpoints = checkpointsFor(job)
          const currentStage = job.progress?.stage && stageOrder.includes(job.progress.stage) ? job.progress.stage : job.stage
          const currentCheckpoint = checkpoints.find((checkpoint) => checkpoint.stage === currentStage)
          const progress = job.progress
          const percent = typeof progress?.percent === 'number' && Number.isFinite(progress.percent) ? Math.max(0, Math.min(100, progress.percent)) : null
          const quiet = Boolean(active && progress?.updatedAt && ['preparation', 'source'].includes(currentStage) && elapsedSince(progress.updatedAt, clockNow) !== null && elapsedSince(progress.updatedAt, clockNow)! > 20)
          const stageAge = elapsedSince(currentCheckpoint?.startedAt ?? null, clockNow)
          const detail = stageProgressText(progress)
          const heading = progress?.message?.trim() || stageName(currentStage)
          const overallSeconds = active ? elapsedSince(job.createdAt, clockNow) : terminalDurationSeconds(job)
          const retryableCancellation = retryableCancel || isRetryableCancellation(job)
          const cancellationPending = job.status === 'cancelling' && !retryableCancellation
          return <section className={`work-panel ${job.status}`} aria-label="Этапы расшифровки">
            {active && <div className="work-current"><div className="work-current-icon"><LoaderCircle size={19} className="spin" /></div><div className="work-current-copy"><strong>{heading}</strong><p>{quiet ? 'Ожидаем данные…' : stageName(currentStage)}{stageAge !== null && <span className="elapsed"> · этап {clockText(stageAge)}</span>}</p></div><span className={`status-badge ${job.status}`}>{statusLabels[job.status]}</span><button type="button" className="cancel-button" onClick={() => void cancelJob()} disabled={cancelBusy || cancellationPending} aria-label={cancelBusy || cancellationPending ? 'Отменяем задачу' : 'Отмена'}>{cancelBusy || cancellationPending ? <LoaderCircle size={16} className="spin" /> : <X size={16} />}<span>{cancelBusy || cancellationPending ? 'Отменяем…' : 'Отмена'}</span></button></div>}
            {!active && <div className={`work-terminal-heading ${job.status}`}><span>{job.status === 'completed' ? <Check size={18} /> : job.status === 'cancelled' ? <Minus size={18} /> : <AlertCircle size={18} />}</span><div><strong>{job.status === 'completed' ? 'Расшифровка готова' : job.status === 'cancelled' ? 'Отменена' : job.status === 'failed' ? 'Не удалось расшифровать видео' : 'Задача была прервана'}</strong><p>{job.status === 'completed' ? 'Все этапы завершены.' : job.status === 'cancelled' ? 'Обработка остановлена. Можно запустить новую задачу.' : safeError(job.error) || 'Можно запустить новую попытку с текущими настройками.'}</p></div></div>}
            <div className="overall-time"><span>Общее время</span><strong>{overallSeconds !== null ? clockText(overallSeconds) : active ? '—' : 'Время не сохранено'}</strong></div>
            {checkpoints.length > 0 && <ol className="checkpoint-list" aria-label="Этапы обработки">
              {checkpoints.map((checkpoint) => <li key={checkpoint.stage} className={`checkpoint ${checkpoint.status}${checkpoint.stage === currentStage && active ? ' current' : ''}`} aria-current={checkpoint.stage === currentStage && active ? 'step' : undefined}>
                <span className="checkpoint-icon" aria-hidden="true">{checkpoint.status === 'completed' ? <Check size={14} /> : checkpoint.status === 'active' && active ? <LoaderCircle size={14} className="spin" /> : checkpoint.status === 'failed' ? <AlertCircle size={14} /> : checkpoint.status === 'skipped' ? <span className="skip-mark">–</span> : checkpoint.status === 'cancelled' ? <Minus size={14} /> : <Circle size={12} />}</span>
                <span className="checkpoint-label">{stageName(checkpoint.stage)}{(checkpoint.status === 'skipped' || checkpoint.status === 'cancelled') && <small>{checkpoint.status === 'skipped' ? 'Пропущено' : 'Отменено'}</small>}</span>
              </li>)}
            </ol>}
            {active && <div className="stage-progress" aria-label="Прогресс текущей операции">
              {percent !== null ? <><div className="progress-track" role="progressbar" aria-label="Прогресс текущей операции" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}><span style={{ width: `${percent}%` }} /></div><div className="progress-details"><span>{heading}</span><strong>{percent}%</strong></div></> : <><div className="progress-track indeterminate" role="progressbar" aria-label={`${heading}: выполняется`} aria-valuetext="Выполняется" /><div className="progress-details"><span>{heading}</span><strong>Выполняется…</strong></div></>}
              {detail && <p className="progress-data">{detail}</p>}
            </div>}
            {pollError && <div className="poll-warning" role="alert"><AlertCircle size={16} />{pollError}</div>}
            {cancelError && <div className="poll-warning" role="alert"><AlertCircle size={16} />{cancelError}</div>}
            {(job.status === 'failed' || job.status === 'interrupted' || job.status === 'cancelled') && <button type="button" className="primary-button retry-button" disabled={submitBusy || active} onClick={() => void startJob()}>{submitBusy ? <><LoaderCircle size={16} className="spin" />Запускаем…</> : 'Повторить попытку'}</button>}
          </section>
        })()}
        {job.status === 'completed' ? <><textarea className="transcript-editor" aria-label="Текст расшифровки" value={draft} onChange={(event) => changeDraft(event.target.value)} spellCheck placeholder="Текст расшифровки появится здесь" />{job.detectedLanguage && job.detectedLanguage.toLowerCase().startsWith('ru') === false && prefs.restorePunctuation && <p className="notice">Пунктуация восстанавливается только для русской речи.</p>}{notice && <p className="notice">{notice}</p>}</> : null}
      </>}
      {formError && <div className="workspace-alert" role="alert">{formError}<button className="alert-dismiss" aria-label="Закрыть сообщение" onClick={() => setFormError('')}>×</button></div>}
    </section>
  </main>
}

function Welcome() { return <div className="welcome-state"><div className="welcome-icon"><FileText size={23} /></div><h2>Здесь появится ваша расшифровка</h2><p>Добавьте ссылку на видео или выберите файл на компьютере. Настройки можно оставить по умолчанию.</p><div className="welcome-hint"><span>1</span><p>Выберите источник видео слева</p></div><div className="welcome-hint"><span>2</span><p>Нажмите «Расшифровать»</p></div></div> }

function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<HistoryItem | null>(null)
  const [busy, setBusy] = useState(false)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [latest, setLatest] = useState<{ id: string; createdAt: string } | null>(null)
  const navigate = useNavigate()
  const refresh = useCallback(async () => {
    try { const result = await api.history(); setItems(result.items); setError('') }
    catch (reason) { setError(`Не удалось загрузить историю. ${errorText(reason)}`) }
    finally { setLoading(false) }
  }, [])
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
      await flushCurrentDraft?.()
      const downloadUrl = item.files.find((file) => file.format === format)?.downloadUrl ?? `/api/simple/jobs/${encodeURIComponent(item.id)}/download?format=${format}`
      const response = await fetch(`${API_BASE}${downloadUrl}`)
      if (!response.ok) throw new Error(response.status === 404 ? 'Файл больше недоступен.' : `Ошибка загрузки (${response.status}).`)
      const blob = await response.blob(); const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = item.files.find((file) => file.format === format)?.name ?? `transcript.${format}`; anchor.click(); URL.revokeObjectURL(url)
    } catch (reason) { setError(errorText(reason)) }
  }
  const remove = async () => {
    if (!deleteTarget) return
    setBusy(true)
    try { const response = await fetch(`${API_BASE}/api/simple/history/${encodeURIComponent(deleteTarget.id)}`, { method: 'DELETE' }); if (!response.ok) throw new Error(response.status === 409 ? 'Нельзя удалить задачу, пока она выполняется.' : `Ошибка удаления (${response.status}).`); if (latest?.id === deleteTarget.id || (!latest && items[0]?.id === deleteTarget.id)) { clearedLatestJob = { id: deleteTarget.id, createdAt: deleteTarget.createdAt }; try { localStorage.setItem('video-transcribe-cleared-latest', JSON.stringify(clearedLatestJob)) } catch { /* optional marker */ } } clearWorkspaceJob?.(deleteTarget.id); setItems((current) => current.filter((item) => item.id !== deleteTarget.id)); setDeleteTarget(null) }
    catch (reason) { setError(errorText(reason)) } finally { setBusy(false) }
  }
  return <main className="history-page">
    <div className="history-title"><div><p className="eyebrow">Ваши задачи</p><h1>История расшифровок</h1><p>Готовые тексты и файлы можно скачать в любое время.</p></div><button className="primary-button" onClick={() => navigate('/')}><Plus size={17} />Новая расшифровка</button></div>
    {error && <div className="inline-error" role="alert">{error}<button className="text-button" onClick={() => { setLoading(true); void refresh() }}>Повторить</button></div>}
    {loading ? <div className="history-empty">Загружаем историю…</div> : items.length === 0 ? <div className="history-empty"><div className="welcome-icon"><FileText size={22} /></div><h2>Пока нет расшифровок</h2><p>Созданные задачи появятся здесь.</p></div> : <div className="history-table-wrap"><table className="history-table"><thead><tr><th>Источник</th><th>Дата</th><th>Статус</th><th>Файлы</th><th><span className="sr-only">Действия</span></th></tr></thead><tbody>{items.map((item) => { const activeItem = isActiveStatus(item.status); const downloadable = item.status !== 'cancelled'; const duration = terminalDurationSeconds(item); return <tr key={item.id}><td className="history-source" title={sourceLabel(item)}>{sourceLabel(item)}</td><td className="history-date"><span>{formatDate(item.createdAt)}</span>{duration !== null && <small>Длительность · {clockText(duration)}</small>}</td><td><span className={`status-badge ${item.status}`}>{statusLabels[item.status]}</span></td><td><div className="file-links">{downloadable && item.files.some((file) => file.format === 'txt') && <button className="file-link" onClick={() => void downloadHistory(item, 'txt')}>TXT</button>}{downloadable && item.files.some((file) => file.format === 'md') && <button className="file-link" onClick={() => void downloadHistory(item, 'md')}>MD</button>}</div></td><td><button className="icon-button" aria-label={`Удалить ${sourceLabel(item)}`} title={activeItem || item.id === activeId ? 'Нельзя удалить выполняющуюся задачу' : 'Удалить из истории'} disabled={activeItem || item.id === activeId} onClick={() => setDeleteTarget(item)}><Trash2 size={17} /></button></td></tr> })}</tbody></table></div>}
    {deleteTarget && <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setDeleteTarget(null) }}><section className="confirm-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-title"><h2 id="delete-title">Удалить расшифровку?</h2><p>{sourceLabel(deleteTarget)}</p><p>Это удалит задачу из истории и файлы результата.</p><div className="dialog-actions"><button className="secondary-button" onClick={() => setDeleteTarget(null)} disabled={busy}>Отмена</button><button className="danger-button" onClick={() => void remove()} disabled={busy}>{busy ? 'Удаляем…' : 'Удалить'}</button></div></section></div>}
  </main>
}

function isHttpUrl(value: string) { try { const url = new URL(value); return url.protocol === 'http:' || url.protocol === 'https:' } catch { return false } }
function clamp(value: number, min: number, max: number) { return Math.min(max, Math.max(min, value)) }
