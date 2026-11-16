import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { AlertCircle, Check, Clock3, FilePlus2, FolderOpen, Image as ImageIcon, LoaderCircle, Mic, Monitor, Play, Plus, Radio, Search, Settings2, Square, Undo2, Volume2, X } from 'lucide-react'
import { registerPendingSave } from '../lib/pendingSaves'
import { canApplyMeetingEditResult, confirmedMeetingDate, isPendingMeetingValueCurrent, latestFailedMeetingJob, meetingId, meetingLabel, meetingsApi, parseMeetingCompositeKey, readMeetingSnapshot } from '../lib/meetingsApi'
import type { AssetPreparation, BackupJob, BackupStatus, CleanupPreview, ConnectorStatus, GlossaryEntry, HardwarePreparation, MeetingAssetStatus, MeetingDetail, MeetingHardware, MeetingJobSummary, MeetingManifest, MeetingQueueState, MeetingRecord, MeetingSettings, RecorderSelection, RecorderStatus, TimelineItem } from '../lib/meetingsApi'
import '../meetings.css'

type StageName = 'speech' | 'screen' | 'audio' | 'index'
type SegmentDraft = { text: string; savedText: string; version: number; dirty: boolean }
type MeetingError = { title: string; reason: string; recommendation: string } | null
type Toast = { message: string; onUndo?: () => void }

const stageLabels: Record<StageName, string> = { speech: 'Речь', screen: 'Экран', audio: 'Звук', index: 'Поиск' }
const queueStageLabels: Record<string, string> = {
  preparation: 'Подготовка',
  source: 'Проверка файла',
  audio: 'Звук',
  recognition: 'Распознавание речи',
  diarization: 'Участники',
  postprocess: 'Правка текста',
  result: 'Сохранение',
  speech: 'Речь',
  screen: 'Экран',
  index: 'Поиск',
}

function queueStageLabel(stage?: string | null): string {
  return stage ? queueStageLabels[stage] ?? 'Обработка' : 'Обработка'
}
const allStages: StageName[] = ['speech', 'screen', 'audio', 'index']

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Не удалось выполнить запрос. Проверьте соединение и повторите попытку.'
}

function errorStatus(error: unknown): number | undefined {
  return typeof error === 'object' && error !== null && 'status' in error && typeof error.status === 'number' ? error.status : undefined
}

function isMeetingJobSummary(value: unknown): value is MeetingJobSummary {
  return typeof value === 'object' && value !== null && 'id' in value && typeof value.id === 'string'
    && 'status' in value && typeof value.status === 'string'
}

function isCleanupPreview(value: unknown): value is CleanupPreview {
  return typeof value === 'object' && value !== null && 'items' in value && Array.isArray(value.items)
    && 'eligible_count' in value && typeof value.eligible_count === 'number'
}

function formatMeetingDate(value: string | null | undefined): { key: string; label: string } {
  if (!value) return { key: 'unknown', label: 'Дата встречи не подтверждена' }
  const date = /^\d{4}-\d{2}-\d{2}$/.test(value) ? new Date(`${value}T00:00:00`) : new Date(value)
  if (Number.isNaN(date.getTime())) return { key: 'unknown', label: 'Дата встречи не подтверждена' }
  return { key: `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`, label: new Intl.DateTimeFormat('ru-RU', { dateStyle: 'full' }).format(date) }
}

function formatTime(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0) return 'Время неизвестно'
  const total = Math.floor(seconds)
  return `${String(Math.floor(total / 3600)).padStart(2, '0')}:${String(Math.floor((total % 3600) / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`
}

function formatMeetingClock(value: string | null | undefined): string {
  if (!value) return 'Время неизвестно'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? 'Время неизвестно' : new Intl.DateTimeFormat('ru-RU', { timeStyle: 'short' }).format(date)
}

function backupOperationLabel(operation: string): string {
  const labels: Record<string, string> = {
    backup: 'Создаём резервную копию', verify: 'Проверяем резервную копию',
    restore_test: 'Выполняем тест восстановления', restore: 'Восстанавливаем архив',
    cleanup_preview: 'Проверяем условия очистки', cleanup: 'Удаляем проверенные исходники',
  }
  return labels[operation] ?? 'Операция с резервной копией'
}

function statusText(status: string | undefined, manifest?: MeetingManifest | null): string {
  const actual = String(manifest?.status?.overall ?? status ?? '').toLowerCase()
  const labels: Record<string, string> = {
    imported: 'Ожидает обработки', queued: 'В очереди', queued_for_processing: 'В очереди',
    speech_ready: 'Текст готов · экран ещё обрабатывается', screen_active: 'Читаем экран', processing: 'Обрабатывается',
    partial: 'Готова часть материалов', ready: 'Готово', completed: 'Готово', failed: 'Ошибка обработки',
    interrupted: 'Обработка прервана', cancelled: 'Обработка остановлена', unavailable: 'Недоступно',
  }
  return labels[actual] ?? (actual ? actual.replaceAll('_', ' ') : 'Статус неизвестен')
}

function statusClass(status: string | undefined, manifest?: MeetingManifest | null): string {
  const actual = String(manifest?.status?.overall ?? status ?? '').toLowerCase()
  if (['ready', 'completed'].includes(actual)) return 'ready'
  if (['failed', 'interrupted', 'cancelled'].includes(actual)) return 'failed'
  if (['processing', 'screen_active', 'queued', 'queued_for_processing'].includes(actual)) return 'active'
  return 'pending'
}

function mergeDirtyTimeline(items: TimelineItem[], drafts: Map<string, SegmentDraft>, id: string): TimelineItem[] {
  return items.map((item) => {
    if (item.kind !== 'transcript' || !item.id) return item
    const draft = drafts.get(`${id}:${item.id}`)
    return draft?.dirty ? { ...item, text: draft.text } : item
  })
}

export function MeetingRecordingBanner() {
  const [recorder, setRecorder] = useState<RecorderStatus | null>(null)
  useEffect(() => {
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const result = await meetingsApi.recorder()
        if (!stopped) setRecorder(result)
      } catch {
        if (!stopped) setRecorder(null)
      }
      if (!stopped) timer = setTimeout(() => void poll(), 2500)
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [])
  if (!recorder || recorder.state !== 'recording') return null
  return <div className="meeting-recording-banner" role="status"><span className="recording-pulse" aria-hidden="true" /><Radio size={17} /><strong>Идёт запись</strong><span>{recorder.title || 'Встреча'}</span><time>{formatTime(recorder.duration_seconds)}</time></div>
}

export function MeetingWorkspace() {
  const location = useLocation()
  const tab = location.pathname.startsWith('/meetings/glossary') ? 'glossary' : location.pathname.startsWith('/meetings/settings') ? 'settings' : 'meetings'
  const [meetings, setMeetings] = useState<MeetingRecord[]>([])
  const [listBusy, setListBusy] = useState(true)
  const [listError, setListError] = useState('')
  const [filter, setFilter] = useState('')
  const [activeId, setActiveId] = useState(() => { try { return localStorage.getItem('video-transcribe-active-meeting') ?? '' } catch { return '' } })
  const activeIdRef = useRef(activeId)
  useLayoutEffect(() => { activeIdRef.current = activeId }, [activeId])
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [detail, setDetail] = useState<MeetingDetail | null>(null)
  const [timeline, setTimeline] = useState<TimelineItem[]>([])
  const [detailBusy, setDetailBusy] = useState(false)
  const [detailError, setDetailError] = useState('')
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const draftMapRef = useRef(new Map<string, SegmentDraft>())
  const saveTimersRef = useRef(new Map<string, ReturnType<typeof setTimeout>>())
  const savingRef = useRef(new Map<string, Promise<void>>())
  const dirtySpeakerRef = useRef(new Map<string, string>())
  const [speakerDrafts, setSpeakerDrafts] = useState<Record<string, string>>({})
  const [retryStages, setRetryStages] = useState<Set<StageName>>(new Set(allStages))
  const [processBusy, setProcessBusy] = useState(false)
  const [importBusy, setImportBusy] = useState(false)
  const [selectedTime, setSelectedTime] = useState(0)
  const [viewerUrl, setViewerUrl] = useState('')
  const [selectedFrame, setSelectedFrame] = useState('')
  const [errorDialog, setErrorDialog] = useState<MeetingError>(null)
  const [toast, setToast] = useState<Toast | null>(null)
  const [settings, setSettings] = useState<MeetingSettings | null>(null)
  const [hardware, setHardware] = useState<MeetingHardware | null>(null)
  const [hardwarePreparation, setHardwarePreparation] = useState<HardwarePreparation | null>(null)
  const [hardwarePrepareBusy, setHardwarePrepareBusy] = useState(false)
  const [assets, setAssets] = useState<MeetingAssetStatus | null>(null)
  const [assetPreparation, setAssetPreparation] = useState<AssetPreparation | null>(null)
  const [assetsPrepareBusy, setAssetsPrepareBusy] = useState(false)
  const [backupStatus, setBackupStatus] = useState<BackupStatus | null>(null)
  const [backupJob, setBackupJob] = useState<BackupJob | null>(null)
  const [backupBusy, setBackupBusy] = useState(false)
  const [backupTarget, setBackupTarget] = useState('')
  const [cleanupPreview, setCleanupPreview] = useState<CleanupPreview | null>(null)
  const [backupConfirmation, setBackupConfirmation] = useState<{ operation: 'restore' | 'cleanup'; snapshotId?: string } | null>(null)
  const [settingsBusy, setSettingsBusy] = useState(false)
  const [settingsError, setSettingsError] = useState('')
  const [connector, setConnector] = useState<ConnectorStatus | null>(null)
  const [connectorBusy, setConnectorBusy] = useState(false)
  const [glossary, setGlossary] = useState<GlossaryEntry[]>([])
  const [glossaryBusy, setGlossaryBusy] = useState(false)
  const [glossaryQuery, setGlossaryQuery] = useState('')
  const [glossaryTerm, setGlossaryTerm] = useState('')
  const [glossaryDefinition, setGlossaryDefinition] = useState('')
  const [glossaryMeeting, setGlossaryMeeting] = useState('')
  const [recorder, setRecorder] = useState<RecorderStatus | null>(null)
  const [recorderError, setRecorderError] = useState('')
  const [queue, setQueue] = useState<MeetingQueueState | null>(null)
  const [queueBusy, setQueueBusy] = useState(false)
  const queueErrorShown = useRef(false)
  const reportedJobFailures = useRef(new Set<string>())
  const reportedHardwareFailures = useRef(new Set<string>())
  const hardwarePollErrorShown = useRef(false)
  const reportedAssetFailures = useRef(new Set<string>())
  const assetPollErrorShown = useRef(false)
  const backupPollErrorShown = useRef(false)
  const reportedBackupFailures = useRef(new Set<string>())
  const reportedBackupCompletions = useRef(new Set<string>())
  const [recorderPrepared, setRecorderPrepared] = useState(false)
  const [recorderBusy, setRecorderBusy] = useState(false)
  const [recorderTitle, setRecorderTitle] = useState('')
  const [recorderSelection, setRecorderSelection] = useState<RecorderSelection>({ monitor_id: null, browser_id: null, microphone_id: null })
  const [previewFailed, setPreviewFailed] = useState(false)
  const requestGeneration = useRef(0)
  const settingsLoadedRef = useRef(false)
  const connectorLoadedRef = useRef(false)

  const refreshMeetings = useCallback(async (quiet = false) => {
    try {
      const result = await meetingsApi.list()
      setMeetings(result.items)
      for (const item of result.items) {
        const id = meetingId(item)
        const failed = latestFailedMeetingJob((item.jobs ?? []).filter(isMeetingJobSummary))
        if (!failed) continue
        const key = `${id}:${failed.id}`
        if (reportedJobFailures.current.has(key)) continue
        reportedJobFailures.current.add(key)
        const stage = queueStageLabel(failed.stage)
        const reason = failed.error?.trim() || `Этап «${stage}» завершился с ошибкой.`
        setErrorDialog({ title: 'Обработка встречи не завершилась', reason, recommendation: 'Выберите встречу, отметьте нужные этапы и запустите их повторно.' })
        break
      }
      setListError('')
      setActiveId((current) => current && result.items.some((item) => meetingId(item) === current) ? current : meetingId(result.items[0] ?? {}))
    } catch (error) {
      setListError(errorStatus(error) === 404 ? 'Архив встреч пока не подключён.' : errorMessage(error))
    } finally {
      if (!quiet) setListBusy(false)
    }
  }, [])

  useEffect(() => {
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    timer = setTimeout(() => void refreshMeetings(), 0)
    const poll = async () => {
      if (!stopped) await refreshMeetings(true)
      if (!stopped) timer = setTimeout(() => void poll(), 4500)
    }
    timer = setTimeout(() => void poll(), 4500)
    return () => { stopped = true; clearTimeout(timer) }
  }, [refreshMeetings])

  useEffect(() => {
    if (!activeId || tab !== 'meetings') {
      return
    }
    const controller = new AbortController()
    const generation = ++requestGeneration.current
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    let loadedRevision: string | undefined
    const load = async (showBusy: boolean) => {
      if (showBusy) {
        setDetailBusy(true)
        setDetail(null)
        setTimeline([])
      }
      try {
        const snapshot = await readMeetingSnapshot(activeId, loadedRevision, controller.signal)
        if (stopped || generation !== requestGeneration.current) return
        if (snapshot) {
          loadedRevision = snapshot.revision
          setDetail(snapshot)
          const nextTimeline = mergeDirtyTimeline(snapshot.timeline, draftMapRef.current, activeId).map((item) => {
            if (!item.speaker_cluster) return item
            const confirmedName = dirtySpeakerRef.current.get(`${activeId}:${item.speaker_cluster}`)
            return confirmedName ? { ...item, confirmed_speaker: confirmedName } : item
          })
          setTimeline(nextTimeline)
          setDetailError('')
        }
      } catch (error) {
        if (stopped || controller.signal.aborted) return
        setDetailError(errorStatus(error) === 404 ? 'Материал встречи недоступен.' : errorMessage(error))
      } finally {
        if (!stopped && generation === requestGeneration.current) {
          if (showBusy) setDetailBusy(false)
          timer = setTimeout(() => void load(false), 6500)
        }
      }
    }
    void load(true)
    return () => { stopped = true; controller.abort(); clearTimeout(timer) }
  }, [activeId, tab])

  useEffect(() => {
    if (!activeId || tab !== 'meetings') return
    let stopped = false
    const load = async () => {
      try {
        const result = await meetingsApi.evidence(activeId, selectedTime)
        if (!stopped && result.url) setViewerUrl(result.url)
      } catch { if (!stopped) setViewerUrl('') }
    }
    void load()
    return () => { stopped = true }
  }, [activeId, selectedTime, tab])

  const persistSegment = useCallback(async (key: string): Promise<void> => {
    while (true) {
      const existing = savingRef.current.get(key)
      if (existing) { await existing; continue }
      const record = draftMapRef.current.get(key)
      if (!record?.dirty) return
      const parsedKey = parseMeetingCompositeKey(key)
      if (!parsedKey) return
      const [meeting, segmentId] = parsedKey
      const version = record.version
      const text = record.text
      const operation = meetingsApi.saveSegment(meeting, segmentId, text).then((result) => {
        const current = draftMapRef.current.get(key)
        if (current && current.version === version) {
          current.savedText = text
          current.dirty = false
        }
        if (canApplyMeetingEditResult(meeting, activeIdRef.current)) {
          setTimeline((items) => items.map((item) => item.id === segmentId && current?.version === version ? { ...item, text } : item))
        }
        if (result.remembered_term_undo_id) {
          if (canApplyMeetingEditResult(meeting, activeIdRef.current)) setToast({ message: 'Исправление сохранено; термин добавлен в словарь.', onUndo: () => {
            void meetingsApi.deleteGlossaryEntry(meeting, result.remembered_term_undo_id!).then(() => {
              setToast({ message: 'Запоминание отменено.' })
            }).catch((error) => setErrorDialog({ title: 'Не удалось отменить запоминание', reason: errorMessage(error), recommendation: 'Повторите отмену позже.' }))
          } })
        }
        setDrafts((values) => ({ ...values, [key]: current?.text ?? text }))
      }).catch((error: unknown) => {
        if (canApplyMeetingEditResult(meeting, activeIdRef.current)) setErrorDialog({ title: 'Не удалось сохранить исправление', reason: errorMessage(error), recommendation: 'Черновик остался на экране. Проверьте соединение и повторите сохранение.' })
        throw error
      }).finally(() => { savingRef.current.delete(key) })
      savingRef.current.set(key, operation)
      await operation
    }
  }, [])

  const flushMeetingEdits = useCallback(async () => {
    for (const timer of saveTimersRef.current.values()) clearTimeout(timer)
    saveTimersRef.current.clear()
    await Promise.all([...draftMapRef.current.entries()].filter(([, record]) => record.dirty).map(([key]) => persistSegment(key)))
    for (const [key, name] of dirtySpeakerRef.current.entries()) {
      const parsedKey = parseMeetingCompositeKey(key)
      if (!parsedKey) continue
      const [meeting, speakerId] = parsedKey
      await meetingsApi.saveSpeaker(meeting, speakerId, name)
      if (isPendingMeetingValueCurrent(dirtySpeakerRef.current.get(key), name)) dirtySpeakerRef.current.delete(key)
    }
  }, [persistSegment])

  useEffect(() => registerPendingSave('meetings', flushMeetingEdits), [flushMeetingEdits])

  useEffect(() => {
    if (tab !== 'settings' && (settingsLoadedRef.current && connectorLoadedRef.current)) return
    let stopped = false
    setSettingsBusy(true)
    void Promise.all([meetingsApi.settings(), meetingsApi.connectorStatus()]).then(([nextSettings, nextConnector]) => {
      if (stopped) return
      settingsLoadedRef.current = true
      connectorLoadedRef.current = true
      setSettings(nextSettings)
      setConnector(nextConnector)
      setSettingsError('')
    }).catch((error: unknown) => {
      if (!stopped) setSettingsError(errorStatus(error) === 404 ? 'Настройки архива ещё не доступны.' : errorMessage(error))
    }).finally(() => { if (!stopped) setSettingsBusy(false) })
    return () => { stopped = true }
  }, [tab])

  useEffect(() => {
    let stopped = false
    void meetingsApi.hardware().then((result) => { if (!stopped) setHardware(result) }).catch((error: unknown) => {
      if (!stopped) setErrorDialog({ title: 'Состояние устройств недоступно', reason: errorMessage(error), recommendation: 'Проверьте подключение и обновите сведения в настройках.' })
    })
    return () => { stopped = true }
  }, [])

  const hardwarePreparationId = hardwarePreparation?.id ?? hardwarePreparation?.job_id
  const hardwarePreparationStatus = hardwarePreparation?.status
  useEffect(() => {
    if (!hardwarePreparationId || hardwarePreparationStatus !== 'running') return
    let stopped = false
    const poll = async () => {
      try {
        const response = await meetingsApi.hardwarePreparation()
        if (stopped || !response.preparation) return
        hardwarePollErrorShown.current = false
        const next = response.preparation
        setHardwarePreparation(next)
        if (next.status === 'completed' || next.status === 'failed') {
          const updated = await meetingsApi.hardware()
          if (!stopped) setHardware(updated)
        }
        if (next.status === 'failed' || next.result?.ok === false) {
          const key = next.id ?? hardwarePreparationId
          if (!reportedHardwareFailures.current.has(key)) {
            reportedHardwareFailures.current.add(key)
            setErrorDialog({ title: 'Подготовка GPU не завершилась', reason: next.result?.reason ?? next.result?.error ?? 'Проверка DirectML завершилась с ошибкой.', recommendation: 'Проверьте подключение и требования среды, затем запустите подготовку ещё раз.' })
          }
        }
      } catch (error) {
        if (!stopped && !hardwarePollErrorShown.current) {
          hardwarePollErrorShown.current = true
          setErrorDialog({ title: 'Не удалось проверить подготовку GPU', reason: errorMessage(error), recommendation: 'Повторите проверку позже; подготовка не запускается повторно автоматически.' })
        }
      }
    }
    const timer = setInterval(() => void poll(), 1500)
    void poll()
    return () => { stopped = true; clearInterval(timer) }
  }, [hardwarePreparationId, hardwarePreparationStatus])

  useEffect(() => {
    let stopped = false
    void meetingsApi.assets().then((result) => { if (!stopped) setAssets(result.semantic_search) }).catch((error: unknown) => {
      if (!stopped) setErrorDialog({ title: 'Состояние семантического поиска недоступно', reason: errorMessage(error), recommendation: 'Проверьте соединение и обновите сведения в настройках.' })
    })
    return () => { stopped = true }
  }, [])

  useEffect(() => {
    let stopped = false
    void meetingsApi.backupStatus().then((result) => { if (!stopped) setBackupStatus(result) }).catch((error: unknown) => {
      if (!stopped) setErrorDialog({ title: 'Состояние резервных копий недоступно', reason: errorMessage(error), recommendation: 'Проверьте подключение и повторите запрос позже.' })
    })
    return () => { stopped = true }
  }, [])

  const assetPreparationId = assetPreparation?.id ?? assetPreparation?.job_id
  const assetPreparationStatus = assetPreparation?.status
  useEffect(() => {
    if (!assetPreparationId || assetPreparationStatus !== 'running') return
    let stopped = false
    const poll = async () => {
      try {
        const response = await meetingsApi.assetPreparation()
        if (stopped) return
        assetPollErrorShown.current = false
        setAssets(response.semantic_search)
        if (!response.preparation) return
        const next = response.preparation
        setAssetPreparation(next)
        if (next.status === 'failed' || next.error || next.result?.ok === false) {
          const key = next.id ?? assetPreparationId
          if (!reportedAssetFailures.current.has(key)) {
            reportedAssetFailures.current.add(key)
            setErrorDialog({ title: 'Подготовка семантического поиска не завершилась', reason: next.error ?? next.result?.reason ?? next.result?.error ?? 'Загрузка локальной модели завершилась с ошибкой.', recommendation: 'Проверьте подключение и место на диске, затем запустите подготовку ещё раз.' })
          }
        }
      } catch (error) {
        if (!stopped && !assetPollErrorShown.current) {
          assetPollErrorShown.current = true
          setErrorDialog({ title: 'Не удалось проверить подготовку поиска', reason: errorMessage(error), recommendation: 'Повторите проверку позже; загрузка не запускается повторно автоматически.' })
        }
      }
    }
    const timer = setInterval(() => void poll(), 1500)
    void poll()
    return () => { stopped = true; clearInterval(timer) }
  }, [assetPreparationId, assetPreparationStatus])

  const backupJobId = backupJob?.job_id
  const backupJobStatus = backupJob?.status
  useEffect(() => {
    if (!backupJobId || backupJobStatus !== 'running') return
    let stopped = false
    const poll = async () => {
      try {
        const result = await meetingsApi.backupStatus(backupJobId)
        if (stopped) return
        backupPollErrorShown.current = false
        setBackupStatus(result)
        if (!result.job) return
        const job = result.job
        setBackupJob(job)
        if (job.status === 'failed') {
          if (!reportedBackupFailures.current.has(job.job_id)) {
            reportedBackupFailures.current.add(job.job_id)
            setErrorDialog({ title: 'Операция с резервной копией не завершилась', reason: job.error?.message ?? 'Проверка резервного архива завершилась с ошибкой.', recommendation: 'Проверьте сообщение и повторите действие после устранения причины.' })
          }
          return
        }
        if (job.status !== 'completed' || reportedBackupCompletions.current.has(job.job_id)) return
        reportedBackupCompletions.current.add(job.job_id)
        if (job.operation !== 'cleanup_preview') setCleanupPreview(null)
        if (job.operation === 'cleanup_preview' && isCleanupPreview(job.result)) {
          setCleanupPreview(job.result)
          setToast({ message: `Проверка очистки завершена: можно удалить ${job.result.eligible_count} исходников.` })
        } else if (job.operation === 'cleanup') {
          setCleanupPreview(null)
          setToast({ message: 'Очистка завершена. Текст, звук и сохранённые кадры оставлены в архиве.' })
          await refreshMeetings(true)
        } else if (job.operation === 'backup') {
          setToast({ message: 'Резервная копия создана. Теперь проверьте её и выполните тест восстановления.' })
        } else if (job.operation === 'verify') {
          setToast({ message: 'Проверка резервной копии завершена.' })
        } else if (job.operation === 'restore_test') {
          setToast({ message: 'Тест восстановления завершён.' })
        } else if (job.operation === 'restore') {
          setToast({ message: 'Архив восстановлен в выбранную папку.' })
        }
      } catch (error) {
        if (!stopped && !backupPollErrorShown.current) {
          backupPollErrorShown.current = true
          setErrorDialog({ title: 'Не удалось проверить резервную операцию', reason: errorMessage(error), recommendation: 'Повторите проверку позже; операция не запускается заново автоматически.' })
        }
      }
    }
    const timer = setInterval(() => void poll(), 1500)
    void poll()
    return () => { stopped = true; clearInterval(timer) }
  }, [backupJobId, backupJobStatus, refreshMeetings])

  useEffect(() => {
    if (tab !== 'glossary') return
    let stopped = false
    void Promise.resolve().then(() => { setGlossaryBusy(true); return meetingsApi.glossary() }).then((result) => { if (!stopped) setGlossary(result.items) }).catch((error: unknown) => {
      if (!stopped) setErrorDialog({ title: 'Словарь недоступен', reason: errorStatus(error) === 404 ? 'Служба архива пока не готова.' : errorMessage(error), recommendation: 'Проверьте подключение к архиву и обновите страницу.' })
    }).finally(() => { if (!stopped) setGlossaryBusy(false) })
    return () => { stopped = true }
  }, [tab])

  useEffect(() => {
    let stopped = false
    const poll = async () => {
      try {
        const nextQueue = await meetingsApi.queue()
        if (!stopped) {
          setQueue(nextQueue)
          queueErrorShown.current = false
        }
      } catch (error) {
        if (!stopped && !queueErrorShown.current) {
          queueErrorShown.current = true
          setErrorDialog({ title: 'Очередь обработки недоступна', reason: errorStatus(error) === 404 ? 'Служба очереди пока не подключена.' : errorMessage(error), recommendation: 'Проверьте соединение с приложением и обновите страницу.' })
        }
      }
      try {
        const next = await meetingsApi.recorder()
        if (stopped) return
        setRecorder(next)
        setRecorderError('')
        setRecorderSelection((current) => ({
          monitor_id: current.monitor_id ?? next.selection.monitor_id,
          browser_id: current.browser_id ?? next.selection.browser_id,
          microphone_id: current.microphone_id ?? next.selection.microphone_id,
        }))
      } catch (error) {
        if (stopped) return
        setRecorder(null)
        setRecorderError(errorStatus(error) === 404 ? 'Запись пока не настроена на этом компьютере.' : errorMessage(error))
      }
    }
    void poll()
    const timer = setInterval(() => void poll(), 2500)
    return () => { stopped = true; clearInterval(timer) }
  }, [])

  const resumeProcessingQueue = async () => {
    setQueueBusy(true)
    try {
      setQueue(await meetingsApi.resumeQueue())
      queueErrorShown.current = false
    } catch (error) {
      setErrorDialog({ title: 'Не удалось продолжить очередь', reason: errorMessage(error), recommendation: 'Проверьте состояние обработки и повторите действие позже.' })
    } finally { setQueueBusy(false) }
  }

  const activeMeeting = useMemo(() => meetings.find((item) => meetingId(item) === activeId) ?? detail?.meeting ?? null, [meetings, activeId, detail])
  const searchableMeetings = useMemo(() => {
    const needle = filter.trim().toLocaleLowerCase('ru')
    return meetings.filter((item) => !needle || `${meetingLabel(item)} ${item.source_filename ?? ''} ${meetingId(item)}`.toLocaleLowerCase('ru').includes(needle))
  }, [meetings, filter])
  const groupedMeetings = useMemo(() => {
    const groups = new Map<string, { label: string; items: MeetingRecord[] }>()
    for (const item of searchableMeetings) {
      const meetingDate = confirmedMeetingDate(item)
      const date = formatMeetingDate(meetingDate?.value)
      const group = groups.get(date.key) ?? { label: date.label, items: [] }
      group.items.push(item)
      groups.set(date.key, group)
    }
    return [...groups.values()]
  }, [searchableMeetings])

  const openMeeting = (id: string) => {
    setActiveId(id)
    setSelectedTime(0)
    try { localStorage.setItem('video-transcribe-active-meeting', id) } catch { /* optional selection */ }
  }

  const toggleSelected = (id: string) => setSelectedIds((current) => {
    const next = new Set(current)
    if (next.has(id)) next.delete(id); else next.add(id)
    return next
  })

  const selectAllVisible = () => setSelectedIds((current) => {
    const next = new Set(current)
    if (searchableMeetings.every((item) => next.has(meetingId(item)))) searchableMeetings.forEach((item) => next.delete(meetingId(item)))
    else searchableMeetings.forEach((item) => next.add(meetingId(item)))
    return next
  })

  const importMeetings = async () => {
    setImportBusy(true)
    try {
      const selection = await meetingsApi.pickFiles()
      const paths = selection.files.map((file) => file.path).filter(Boolean)
      if (paths.length === 0) {
        if (selection.rejected.length) setErrorDialog({ title: 'Не удалось добавить файлы', reason: selection.rejected.map((item) => item.message).join(' '), recommendation: 'Выберите видео- или аудиофайлы из доступного места.' })
        return
      }
      const result = await meetingsApi.importFiles(paths)
      await refreshMeetings()
      const first = result.items[0]
      if (first) openMeeting(meetingId(first))
      if (result.rejected.length) setErrorDialog({ title: 'Часть файлов не добавлена', reason: result.rejected.map((item) => item.reason).join(' '), recommendation: 'Проверьте форматы неподдержанных файлов.' })
      else setToast({ message: `Добавлено встреч: ${result.items.length}. Обработка не запускалась.` })
    } catch (error) {
      setErrorDialog({ title: 'Не удалось добавить встречу', reason: errorMessage(error), recommendation: 'Проверьте выбранные файлы и повторите попытку.' })
    } finally { setImportBusy(false) }
  }

  const processMeetings = async (ids: string[], stages: StageName[] = allStages) => {
    if (!settings || ids.length === 0 || stages.length === 0) return
    setProcessBusy(true)
    try {
      await flushMeetingEdits()
      await meetingsApi.process(ids, settings, stages)
      setSelectedIds(new Set())
      await refreshMeetings()
      setToast({ message: ids.length === 1 ? 'Обработка встречи добавлена в очередь.' : `В очередь добавлено встреч: ${ids.length}.` })
    } catch (error) {
      setErrorDialog({ title: 'Не удалось начать обработку', reason: errorMessage(error), recommendation: settings.execution_mode === 'manual' && settings.device === 'gpu' ? 'Проверьте выбранное устройство и повторите обработку после исправления настройки.' : 'Проверьте свободное место и повторите обработку позже.' })
    } finally { setProcessBusy(false) }
  }

  const changeSegment = (item: TimelineItem, value: string) => {
    const id = meetingId(detail?.meeting ?? activeMeeting ?? {})
    if (!id || !item.id) return
    const key = `${id}:${item.id}`
    const prior = draftMapRef.current.get(key)
    draftMapRef.current.set(key, { text: value, savedText: prior?.savedText ?? String(item.text ?? ''), version: (prior?.version ?? 0) + 1, dirty: true })
    setDrafts((current) => ({ ...current, [key]: value }))
    setTimeline((current) => current.map((row) => row.id === item.id ? { ...row, text: value } : row))
    const previousTimer = saveTimersRef.current.get(key)
    if (previousTimer) clearTimeout(previousTimer)
    saveTimersRef.current.set(key, setTimeout(() => {
      saveTimersRef.current.delete(key)
      void persistSegment(key).catch(() => undefined)
    }, 700))
  }

  const changeSpeaker = (speakerId: string, value: string) => {
    const id = meetingId(detail?.meeting ?? activeMeeting ?? {})
    const key = `${id}:${speakerId}`
    setSpeakerDrafts((current) => ({ ...current, [key]: value }))
    dirtySpeakerRef.current.set(key, value)
  }

  const saveSpeaker = async (speakerId: string) => {
    const id = meetingId(detail?.meeting ?? activeMeeting ?? {})
    const key = `${id}:${speakerId}`
    const pendingName = dirtySpeakerRef.current.get(key)
    if (pendingName === undefined || !pendingName.trim()) return
    const name = pendingName.trim()
    try {
      await meetingsApi.saveSpeaker(id, speakerId, name)
      if (isPendingMeetingValueCurrent(dirtySpeakerRef.current.get(key), pendingName)) dirtySpeakerRef.current.delete(key)
      if (canApplyMeetingEditResult(id, activeIdRef.current)) {
        setTimeline((current) => current.map((item) => item.speaker_cluster === speakerId ? { ...item, confirmed_speaker: name } : item))
        setToast({ message: `Имя говорящего сохранено: ${name}.` })
      }
    } catch (error) {
      if (canApplyMeetingEditResult(id, activeIdRef.current)) setErrorDialog({ title: 'Не удалось сохранить имя', reason: errorMessage(error), recommendation: 'Имя осталось в поле. Проверьте соединение и повторите сохранение.' })
      throw error
    }
  }

  const showEvidence = async (item: TimelineItem) => {
    if (item.start !== null && Number.isFinite(item.start)) setSelectedTime(item.start)
    const screen = item.kind === 'screen' ? item : timeline.find((candidate) => candidate.kind === 'screen' && candidate.frame && item.start !== null && candidate.start !== null && candidate.start <= item.start && (candidate.end === null || candidate.end === undefined || item.start < candidate.end))
    const evidenceId = screen?.evidence_id ?? screen?.id
    const id = meetingId(detail?.meeting ?? activeMeeting ?? {})
    if (screen && evidenceId && id) {
      setSelectedFrame(meetingsApi.frameUrl(id, evidenceId))
    } else {
      setSelectedFrame('')
    }
  }

  const updateSettings = <K extends keyof MeetingSettings>(key: K, value: MeetingSettings[K]) => setSettings((current) => current ? { ...current, [key]: value } : current)
  const chooseDirectory = async (key: 'archive_root' | 'backup_root') => {
    try {
      const result = await meetingsApi.pickDirectory()
      if (settings && result.path) setSettings({ ...settings, [key]: result.path })
    } catch (error) {
      setErrorDialog({ title: 'Не удалось выбрать папку', reason: errorMessage(error), recommendation: 'Проверьте доступ к системному окну выбора папки.' })
    }
  }

  const saveSettings = async () => {
    if (!settings) return
    setSettingsBusy(true)
    try {
      const values = { ...settings }
      delete values.disk
      const saved = await meetingsApi.saveSettings(values)
      setSettings(saved)
      setBackupStatus(await meetingsApi.backupStatus())
      setCleanupPreview(null)
      setSettingsError('')
      setToast({ message: 'Настройки сохранены.' })
    } catch (error) {
      setErrorDialog({ title: 'Не удалось сохранить настройки', reason: errorMessage(error), recommendation: 'Проверьте доступ к папке архива и повторите сохранение.' })
    } finally { setSettingsBusy(false) }
  }

  const prepareGpu = async () => {
    setHardwarePrepareBusy(true)
    try {
      const result = await meetingsApi.prepareHardware()
      setHardwarePreparation({ job_id: result.job_id, status: result.status, estimated_download_bytes: result.estimated_download_bytes })
    } catch (error) {
      setErrorDialog({ title: 'Не удалось подготовить GPU', reason: errorMessage(error), recommendation: 'Проверьте подключение и повторите подготовку вручную.' })
    } finally { setHardwarePrepareBusy(false) }
  }

  const prepareSemanticSearch = async () => {
    setAssetsPrepareBusy(true)
    try {
      const result = await meetingsApi.prepareAssets()
      setAssets(result.semantic_search)
      setAssetPreparation({ job_id: result.job_id, status: result.status, estimated_download_bytes: result.estimated_download_bytes })
      if (result.status === 'completed') setToast({ message: 'Семантический поиск уже готов.' })
    } catch (error) {
      setErrorDialog({ title: 'Не удалось подготовить семантический поиск', reason: errorMessage(error), recommendation: 'Проверьте подключение и повторите подготовку вручную.' })
    } finally { setAssetsPrepareBusy(false) }
  }

  const runBackupOperation = async (operation: 'backup' | 'verify' | 'restore_test' | 'restore' | 'cleanup_preview' | 'cleanup', snapshotId?: string) => {
    setBackupBusy(true)
    try {
      const result = operation === 'backup' ? await meetingsApi.createBackup()
        : operation === 'verify' && snapshotId ? await meetingsApi.verifyBackup(snapshotId)
          : operation === 'restore_test' && snapshotId ? await meetingsApi.restoreTest(snapshotId)
            : operation === 'restore' && snapshotId ? await meetingsApi.restoreBackup(snapshotId, backupTarget)
              : operation === 'cleanup_preview' ? await meetingsApi.cleanupPreview()
                : operation === 'cleanup' ? await meetingsApi.runCleanup() : null
      if (!result) return
      if (result.status === 'failed' || result.error) {
        setErrorDialog({ title: 'Не удалось выполнить операцию', reason: result.error?.message ?? 'Служба резервного копирования вернула ошибку.', recommendation: 'Проверьте настройки папки и условия операции, затем повторите действие.' })
        const refreshed = await meetingsApi.backupStatus()
        setBackupStatus(refreshed)
        return
      }
      if (result.job_id) {
        setBackupJob({ job_id: result.job_id, operation: result.operation ?? operation, status: result.status })
      } else if (result.status === 'completed') {
        const refreshed = await meetingsApi.backupStatus()
        setBackupStatus(refreshed)
        setToast({ message: 'Операция завершена.' })
      }
    } catch (error) {
      setErrorDialog({ title: 'Не удалось выполнить операцию', reason: errorMessage(error), recommendation: 'Проверьте настройки папок и повторите действие позже.' })
    } finally { setBackupBusy(false) }
  }

  const chooseBackupRestoreTarget = async () => {
    try {
      const result = await meetingsApi.pickDirectory()
      if (result.path) setBackupTarget(result.path)
    } catch (error) {
      setErrorDialog({ title: 'Не удалось выбрать папку восстановления', reason: errorMessage(error), recommendation: 'Выберите новую или пустую папку, которая не совпадает с архивом и резервной копией.' })
    }
  }

  const confirmBackupAction = () => {
    const confirmation = backupConfirmation
    setBackupConfirmation(null)
    if (!confirmation) return
    void runBackupOperation(confirmation.operation === 'restore' ? 'restore' : 'cleanup', confirmation.snapshotId)
  }

  const runConnector = async (action: 'connect' | 'check' | 'disconnect') => {
    setConnectorBusy(true)
    try {
      const result = action === 'connect' ? await meetingsApi.connectCodex() : action === 'check' ? await meetingsApi.checkCodex() : await meetingsApi.disconnectCodex()
      setConnector(result)
      if (result.client_verified) setToast({ message: 'Codex проверен реальным вызовом инструмента.' })
      else if (action === 'connect') setToast({ message: 'Сервер зарегистрирован. Перезапустите Codex и проверьте реальный доступ.' })
      else if (action === 'check') setToast({ message: result.protocol_checked ? 'Протокол проверен. Реальный вызов в Codex ещё не подтверждён.' : 'Проверка подключения не завершена.' })
    } catch (error) {
      setErrorDialog({ title: 'Не удалось обновить подключение Codex', reason: errorMessage(error), recommendation: 'Проверьте, что Codex установлен и открыт, затем повторите действие.' })
    } finally { setConnectorBusy(false) }
  }

  const addGlossary = async () => {
    const target = glossaryMeeting || activeId || meetingId(meetings[0] ?? {})
    if (!target || !glossaryTerm.trim()) return
    setGlossaryBusy(true)
    try {
      await meetingsApi.addGlossaryEntry(target, glossaryTerm.trim(), glossaryDefinition.trim())
      setGlossaryTerm(''); setGlossaryDefinition(''); setGlossaryMeeting(target)
      const result = await meetingsApi.glossary()
      setGlossary(result.items)
    } catch (error) {
      setErrorDialog({ title: 'Не удалось добавить слово', reason: errorMessage(error), recommendation: 'Проверьте, что встреча доступна, и повторите попытку.' })
    } finally { setGlossaryBusy(false) }
  }

  const deleteTerm = async (entry: GlossaryEntry) => {
    if (!entry.meeting_id) return
    try {
      const result = await meetingsApi.deleteGlossaryEntry(entry.meeting_id, entry.id)
      setGlossary((current) => current.filter((item) => item.id !== entry.id || item.meeting_id !== entry.meeting_id))
      if (result.undo_id) setToast({ message: `Удалено: ${entry.term}.`, onUndo: () => {
        void meetingsApi.undoGlossaryEntry(entry.meeting_id!, result.undo_id!).then(async () => {
          const refreshed = await meetingsApi.glossary()
          setGlossary(refreshed.items)
          setToast({ message: 'Слово восстановлено.' })
        }).catch((error) => setErrorDialog({ title: 'Не удалось восстановить слово', reason: errorMessage(error), recommendation: 'Обновите словарь и проверьте запись.' }))
      } })
    } catch (error) {
      setErrorDialog({ title: 'Не удалось удалить слово', reason: errorMessage(error), recommendation: 'Обновите словарь и повторите попытку.' })
    }
  }

  const loadRecorder = async (operation: () => Promise<RecorderStatus>, success?: string) => {
    setRecorderBusy(true)
    try {
      const next = await operation()
      setRecorder(next); setRecorderError(''); setRecorderSelection(next.selection)
      if (success) setToast({ message: success })
      return next
    } catch (error) {
      setErrorDialog({ title: 'Запись недоступна', reason: errorStatus(error) === 404 ? 'Модуль записи ещё не настроен.' : errorMessage(error), recommendation: 'Настройте запись или обратитесь к администратору приложения.' })
      return null
    } finally { setRecorderBusy(false) }
  }

  const prepareRecording = () => void loadRecorder(() => meetingsApi.prepareRecorder(), 'Источник записи подготовлен. Проверьте изображение и уровни звука.')
  const configureRecording = () => void loadRecorder(() => meetingsApi.configureRecorder(recorderSelection), 'Источники записи сохранены.')
  const startRecording = async () => {
    if (!recorderSelection.monitor_id || !recorderSelection.browser_id || !recorderSelection.microphone_id) {
      setErrorDialog({ title: 'Выберите источники записи', reason: 'Для записи нужны экран, звук созвона и микрофон.', recommendation: 'Выберите доступные источники и настройте их перед началом.' })
      return
    }
    setRecorderBusy(true)
    try {
      const result = await meetingsApi.startRecording(recorderTitle.trim() || 'Новая встреча')
      setRecorder(result); setRecorderError(''); setToast({ message: 'Запись началась.' })
      await refreshMeetings()
    } catch (error) {
      setErrorDialog({ title: 'Не удалось начать запись', reason: errorMessage(error), recommendation: 'Проверьте доступ к экрану и источникам звука.' })
    } finally { setRecorderBusy(false) }
  }

  const stopRecording = async () => {
    setRecorderBusy(true)
    try {
      const result = await meetingsApi.stopRecording()
      setRecorder(result.recorder)
      openMeeting(meetingId(result.meeting))
      await refreshMeetings()
      setToast({ message: 'Запись сохранена. Встреча ожидает обработки.' })
    } catch (error) {
      setErrorDialog({ title: 'Не удалось завершить запись', reason: errorMessage(error), recommendation: 'Проверьте состояние записи и повторите остановку.' })
    } finally { setRecorderBusy(false) }
  }

  const visibleGlossary = useMemo(() => {
    const needle = glossaryQuery.trim().toLocaleLowerCase('ru')
    return glossary.filter((item) => !needle || `${item.term} ${item.definition ?? ''}`.toLocaleLowerCase('ru').includes(needle))
  }, [glossary, glossaryQuery])
  const speakerIds = [...new Set(timeline.map((item) => item.speaker_cluster).filter((value): value is string => Boolean(value)))].sort((a, b) => a.localeCompare(b, 'ru', { numeric: true }))
  const screenGpu = hardware?.stages.screen?.gpu
  const latestHardwareProgress = hardwarePreparation?.progress?.at(-1)
  const latestAssetProgress = assetPreparation?.progress?.at(-1)
  const gpuPrepared = Boolean(screenGpu?.prepared || hardware?.gpu_runtime.prepared)
  const gpuVerified = Boolean(screenGpu?.verified)
  const backupRootSaved = Boolean(settings?.backup_root && backupStatus?.storage.backup_root
    && settings.backup_root.trim().toLocaleLowerCase() === backupStatus.storage.backup_root.trim().toLocaleLowerCase())
  const activeStatus = statusText(detail?.meeting.status ?? activeMeeting?.status, detail?.manifest)
  const retryable = ['failed', 'interrupted', 'partial'].some((value) => String(detail?.meeting.status ?? detail?.manifest?.status?.overall ?? '').toLowerCase().includes(value))

  return <main className="meeting-workspace">
    <div className="meeting-subnav" aria-label="Разделы архива встреч" role="navigation">
      <Link to="/meetings" className={tab === 'meetings' ? 'selected' : ''}>Встречи</Link>
      <Link to="/meetings/glossary" className={tab === 'glossary' ? 'selected' : ''}>Словарь</Link>
      <Link to="/meetings/settings" className={tab === 'settings' ? 'selected' : ''}>Настройки</Link>
    </div>

    {tab === 'meetings' && <>
      <section className="meetings-heading">
        <div><p className="eyebrow">Архив разговоров</p><h1>Встречи</h1><p>Записи, материалы и сохранённые доказательства.</p></div>
        <div className="meeting-heading-actions">
          <button className="secondary-button" onClick={() => void importMeetings()} disabled={importBusy}>{importBusy ? <LoaderCircle size={16} className="spin" /> : <FilePlus2 size={16} />}Добавить файлы</button>
          <button className="primary-button" onClick={() => setRecorderPrepared(false)}><Radio size={16} />Начать запись</button>
          <button className="primary-button" onClick={() => void processMeetings([...selectedIds])} disabled={processBusy || settingsBusy || !settings || selectedIds.size === 0}>{processBusy ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />}Обработать выбранные{selectedIds.size ? ` · ${selectedIds.size}` : ''}</button>
        </div>
      </section>
      {queue && (queue.paused || queue.activeJob || queue.waitingCount > 0 || (recorder?.state === 'recording' && queue.waitingCount > 0)) && <aside className="meeting-queue-banner" role="status">
        <div><strong>{recorder?.state === 'recording' && queue.waitingCount > 0 ? 'Очередь ждёт окончания записи' : queue.paused ? 'Очередь обработки приостановлена' : queue.activeJob ? 'Встречи обрабатываются в общей очереди' : 'Встречи ожидают обработки'}</strong><p>{recorder?.state === 'recording' && queue.waitingCount > 0 ? 'Ожидающие задачи продолжатся после сохранения встречи.' : queue.pauseReason?.message || (queue.activeJob ? `Сейчас: ${queueStageLabel(queue.activeJob.stage)}` : 'Задачи запускаются по очереди.')}</p></div>
        <span>Ожидают: {queue.waitingCount}</span>
        {queue.paused && <button className="secondary-button" type="button" onClick={() => void resumeProcessingQueue()} disabled={queueBusy}>{queueBusy ? 'Продолжаем…' : 'Продолжить очередь'}</button>}
      </aside>}
      <RecorderPanel recorder={recorder} error={recorderError} busy={recorderBusy} prepared={recorderPrepared} title={recorderTitle} selection={recorderSelection} previewFailed={previewFailed}
        onTitle={setRecorderTitle} onSelection={setRecorderSelection} onPrepare={() => { setRecorderPrepared(true); prepareRecording() }} onConfigure={configureRecording} onStart={() => void startRecording()} onStop={() => void stopRecording()} onPreviewError={() => setPreviewFailed(true)} />
      <div className="meetings-grid">
        <section className="meeting-list-panel" aria-label="Список встреч">
          <div className="meeting-list-tools">
            <label className="meeting-search"><Search size={16} /><span className="sr-only">Найти встречу</span><input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Найти встречу" /></label>
            <button className="text-action" onClick={selectAllVisible}>{searchableMeetings.length > 0 && searchableMeetings.every((item) => selectedIds.has(meetingId(item))) ? 'Снять выбор' : 'Выбрать все'}</button>
          </div>
          {listError && <div className="meeting-inline-error" role="alert"><AlertCircle size={16} />{listError}<button className="text-action" onClick={() => { setListBusy(true); void refreshMeetings() }}>Повторить</button></div>}
          {listBusy && meetings.length === 0 ? <div className="meeting-empty">Загружаем встречи…</div> : searchableMeetings.length === 0 ? <div className="meeting-empty"><div className="meeting-empty-icon"><FolderOpen size={20} /></div><h2>{filter ? 'Ничего не найдено' : 'Здесь будут ваши встречи'}</h2><p>{filter ? 'Измените запрос поиска.' : 'Запишите встречу или добавьте файл. Обработка начнётся только по вашему действию.'}</p><button className="secondary-button" onClick={() => void importMeetings()}><FilePlus2 size={16} />Добавить файлы</button></div> : <div className="meeting-list-scroll">
            {groupedMeetings.map((group) => <section className="meeting-date-group" key={group.label}><h2>{group.label}</h2>{group.items.map((item) => {
              const id = meetingId(item)
              return <article className={`meeting-list-item ${activeId === id ? 'active' : ''}`} key={id}>
                <input type="checkbox" aria-label={`Выбрать ${meetingLabel(item)}`} checked={selectedIds.has(id)} onChange={() => toggleSelected(id)} />
                <button className="meeting-list-open" onClick={() => openMeeting(id)}>
                  <strong>{meetingLabel(item)}</strong>
                  <span>{confirmedMeetingDate(item)?.hasTime ? `${confirmedMeetingDate(item)?.source} · ${formatMeetingClock(item.started_at)}` : confirmedMeetingDate(item) ? `Дата подтверждена: ${confirmedMeetingDate(item)?.source}` : 'Дата встречи не подтверждена'}</span>
                  <span className={`meeting-status ${statusClass(item.status)}`}>{statusText(item.status)}</span>
                </button>
              </article>
            })}</section>)}
          </div>}
        </section>
        <section className="meeting-detail-panel" aria-label="Содержимое встречи">
          {!activeId ? <div className="meeting-empty meeting-empty-detail"><h2>Выберите встречу</h2><p>Откройте запись слева, чтобы прочитать текст и проверить материалы.</p></div> : detailBusy && !detail ? <div className="meeting-empty">Загружаем материалы встречи…</div> : detailError && !detail ? <div className="meeting-inline-error" role="alert"><AlertCircle size={16} />{detailError}</div> : detail && <>
            <div className="meeting-detail-heading">
              <div><p className="eyebrow">{formatMeetingDate(confirmedMeetingDate(detail.meeting)?.value).label}</p><h2>{meetingLabel(detail.meeting)}</h2>{confirmedMeetingDate(detail.meeting)?.source && <small className="meeting-date-provenance">Источник даты: {confirmedMeetingDate(detail.meeting)?.source}</small>}<span className={`meeting-status ${statusClass(detail.meeting.status, detail.manifest)}`}>{activeStatus}</span></div>
              <div className="meeting-detail-actions"><button className="secondary-button" onClick={() => { void flushMeetingEdits().then(() => meetingsApi.evidence(activeId, selectedTime)).then((result) => window.open(result.url, '_blank', 'noopener,noreferrer')).catch((error: unknown) => setErrorDialog({ title: 'Не удалось открыть подтверждение', reason: errorMessage(error), recommendation: 'Проверьте, что встреча и локальный просмотрщик доступны.' })) }}><ImageIcon size={15} />Открыть подтверждение</button>{retryable && <button className="primary-button" disabled={processBusy || retryStages.size === 0} onClick={() => void processMeetings([activeId], [...retryStages])}>Повторить выбранные этапы</button>}</div>
            </div>
            {retryable && <div className="meeting-retry-stages" aria-label="Выберите этапы для повтора">{allStages.map((stage) => <label key={stage}><input type="checkbox" checked={retryStages.has(stage)} onChange={() => setRetryStages((current) => { const next = new Set(current); if (next.has(stage)) next.delete(stage); else next.add(stage); return next })} />{stageLabels[stage]}</label>)}</div>}
            {detail.manifest?.source_availability?.source === false && <p className="meeting-retention-note">Полное видео больше не хранится. Доступны сохранённые кадры и звук.</p>}
            <div className="meeting-evidence-layout">
              <section className="meeting-transcript" aria-label="Текст и события встречи">
                {speakerIds.length > 0 && <div className="meeting-speakers"><h3>Участники</h3>{speakerIds.map((speakerId) => {
                  const item = timeline.find((row) => row.speaker_cluster === speakerId)
                  const key = `${activeId}:${speakerId}`
                  const initial = speakerDrafts[key] ?? item?.confirmed_speaker ?? ''
                  return <label key={speakerId}><span>{item?.confirmed_speaker ? 'Имя участника' : `Говорящий ${speakerId}`}</span><input value={initial} onChange={(event) => changeSpeaker(speakerId, event.target.value)} onBlur={() => { void saveSpeaker(speakerId).catch(() => undefined) }} onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); void saveSpeaker(speakerId).catch(() => undefined) } }} placeholder="Добавить имя" /></label>
                })}</div>}
                {detailBusy && timeline.length === 0 ? <div className="meeting-empty">Загружаем временную шкалу…</div> : timeline.length === 0 ? <div className="meeting-empty">В этой встрече пока нет текста и сохранённых кадров.</div> : <div className="meeting-timeline">{timeline.map((item, index) => {
                  const segmentKey = item.id ? `${activeId}:${item.id}` : ''
                  const value = item.id ? drafts[segmentKey] ?? String(item.text ?? '') : String(item.text ?? '')
                  return <article className={`timeline-item timeline-${item.kind}`} key={`${item.kind}-${item.id ?? item.evidence_id ?? index}`}>
                    <button className="timeline-time" onClick={() => void showEvidence(item)} aria-label={`Открыть материал в ${formatTime(item.start)}`}><Clock3 size={14} />{formatTime(item.start)}</button>
                    <div className="timeline-content">
                      <div className="timeline-source-label">{item.kind === 'screen' ? 'Видно на экране' : item.kind === 'orphan_edit' ? 'Исправление без исходного фрагмента' : 'Сказано на встрече'}</div>
                      {item.kind === 'transcript' && item.id ? <textarea aria-label={`Исправить фрагмент ${index + 1}`} value={value} onChange={(event) => changeSegment(item, event.target.value)} onBlur={() => { if (segmentKey) void persistSegment(segmentKey).catch(() => undefined) }} rows={Math.max(2, Math.min(5, Math.ceil(value.length / 100)))} /> : <p>{item.text || 'Текст не распознан'}</p>}
                      {item.original_text && item.original_text !== item.text && <details><summary>Исходный вариант</summary><p>{item.original_text}</p></details>}
                      {item.kind === 'screen' && item.frame && <button className="text-action" onClick={() => void showEvidence(item)}>Посмотреть сохранённый кадр</button>}
                    </div>
                  </article>
                })}</div>}
              </section>
              <aside className="meeting-evidence-viewer" aria-label="Подтверждение по времени">
                <h3>Материалы в {formatTime(selectedTime)}</h3>
                {selectedFrame ? <img src={selectedFrame} alt="Сохранённый кадр экрана" onError={() => setSelectedFrame('')} /> : <div className="meeting-frame-placeholder"><ImageIcon size={20} /><span>Выберите фрагмент или кадр, чтобы увидеть изображение.</span></div>}
                {viewerUrl && <details className="human-evidence-link"><summary>Видео или сохранённый звук</summary><iframe title="Видео и звук встречи" src={viewerUrl} loading="lazy" /> <a href={viewerUrl} target="_blank" rel="noreferrer">Открыть в отдельном окне</a></details>}
              </aside>
            </div>
          </>}
        </section>
      </div>
    </>}

    {tab === 'glossary' && <section className="meeting-page-card glossary-page">
      <div className="meeting-page-heading"><div><p className="eyebrow">Запоминаемые термины</p><h1>Словарь</h1><p>Исправленные слова сохраняются здесь. Их можно удалить и восстановить отменой.</p></div></div>
      <div className="glossary-add-form">
        <label>Слово или фраза<input value={glossaryTerm} onChange={(event) => setGlossaryTerm(event.target.value)} placeholder="Например, DBeaver" /></label>
        <label>Пояснение <span>необязательно</span><input value={glossaryDefinition} onChange={(event) => setGlossaryDefinition(event.target.value)} placeholder="Короткое описание" /></label>
        <label>Встреча<select value={glossaryMeeting || activeId || meetingId(meetings[0] ?? {})} onChange={(event) => setGlossaryMeeting(event.target.value)}><option value="">Выберите встречу</option>{meetings.map((item) => <option key={meetingId(item)} value={meetingId(item)}>{meetingLabel(item)}</option>)}</select></label>
        <button className="primary-button" onClick={() => void addGlossary()} disabled={glossaryBusy || !glossaryTerm.trim() || !meetings.length}><Plus size={16} />Добавить слово</button>
      </div>
      <div className="glossary-toolbar"><label className="meeting-search"><Search size={16} /><span className="sr-only">Найти слово</span><input value={glossaryQuery} onChange={(event) => setGlossaryQuery(event.target.value)} placeholder="Найти слово" /></label><span>{glossary.length} слов</span></div>
      {glossaryBusy ? <div className="meeting-empty">Загружаем словарь…</div> : visibleGlossary.length === 0 ? <div className="meeting-empty"><h2>Словарь пока пуст</h2><p>Исправления встреч будут пополнять его автоматически.</p></div> : <div className="glossary-list">{visibleGlossary.map((entry) => <article className="glossary-entry" key={`${entry.meeting_id ?? ''}:${entry.id}`}><div><strong>{entry.term}</strong>{entry.definition && <p>{entry.definition}</p>}<small>{entry.meeting_id ? meetingLabel(meetings.find((item) => meetingId(item) === entry.meeting_id) ?? { meeting_id: entry.meeting_id }) : 'Общий термин'}</small></div><button className="icon-button" aria-label={`Удалить ${entry.term}`} onClick={() => void deleteTerm(entry)}><X size={16} /></button></article>)}</div>}
    </section>}

    {tab === 'settings' && <section className="meeting-page-card settings-page">
      <div className="meeting-page-heading"><div><p className="eyebrow">Хранение и подключения</p><h1>Настройки встреч</h1><p>Изменения пути применяются к следующим действиям архива.</p></div></div>
      {settingsError && <div className="meeting-inline-error" role="alert"><AlertCircle size={16} />{settingsError}</div>}
      {settingsBusy && !settings ? <div className="meeting-empty">Загружаем настройки…</div> : settings && <>
        <section className="settings-group"><h2>Папки</h2><FolderField label="Папка архива" value={settings.archive_root} onChoose={() => void chooseDirectory('archive_root')} />{settings.backup_root !== undefined && <FolderField label="Резервная папка" value={settings.backup_root ?? ''} onChoose={() => void chooseDirectory('backup_root')} clearable onClear={() => updateSettings('backup_root', null)} />}
          {settings.disk && <p className="disk-space">Свободно: {typeof settings.disk.free_bytes === 'number' ? formatBytes(settings.disk.free_bytes) : 'нет данных'}</p>}
        </section>
        <section className="settings-group"><h2>Обработка</h2><label className="settings-field">Профиль распознавания<select value={settings.profile} onChange={(event) => updateSettings('profile', event.target.value as MeetingSettings['profile'])}><option value="balanced">Обычный · Parakeet</option><option value="quality">Точнее · Qwen</option><option value="economy">Экономнее · Whisper</option></select><small>Профиль применяется только после нажатия «Обработать».</small></label><fieldset className="settings-choice"><legend>Режим устройства</legend><label><input type="radio" checked={settings.execution_mode === 'auto'} onChange={() => updateSettings('execution_mode', 'auto')} />Авто <small>Приложение выбирает проверенный способ для каждого этапа.</small></label><label><input type="radio" checked={settings.execution_mode === 'manual'} onChange={() => updateSettings('execution_mode', 'manual')} />Вручную <small>Выбранное устройство не заменяется незаметно.</small></label></fieldset>{settings.execution_mode === 'manual' && <label className="settings-field">Устройство<select value={settings.device} onChange={(event) => updateSettings('device', event.target.value as 'cpu' | 'gpu')}><option value="cpu">Процессор</option><option value="gpu">Видеокарта</option></select>{settings.device === 'gpu' && screenGpu?.prepared && !screenGpu.verified && <small>Ручной запуск экранного OCR будет экспериментальным до проверки качества и скорости.</small>}</label>}<label className="settings-field">Срок хранения полного видео<input type="number" min={1} max={365} value={settings.original_retention_days} onChange={(event) => updateSettings('original_retention_days', Math.max(1, Number(event.target.value) || 30))} /><small>Текст, звук и сохранённые кадры остаются в архиве.</small></label><p className="settings-note">{settings.auto_cleanup ? 'Автоматическая очистка включена: оригиналы старше выбранного срока удаляются только при готовом архиве и проверенной резервной копии.' : 'Автоматическая очистка выключена.'}</p></section>
        <section className="settings-group hardware-settings"><h2>Устройства и подготовка GPU</h2>
          {!hardware ? <p className="settings-note">Проверяем доступные устройства…</p> : <>
            <p className="hardware-detected"><strong>Видеокарта:</strong> {hardware.hardware.gpu?.name || 'Совместимая видеокарта не обнаружена'}</p>
            {hardware.hardware.gpu?.driver && <p className="settings-note">Версия драйвера: {hardware.hardware.gpu.driver}</p>}
            <ul className="hardware-stage-list">
              <li><strong>Экран · Авто:</strong> {screenGpu?.verified ? 'GPU DirectML, проверено' : 'CPU'}<small>{screenGpu?.verified ? 'Есть актуальная проверка качества и скорости.' : screenGpu?.reason ?? 'Автоматический режим использует проверенный CPU.'}</small></li>
              <li><strong>Сжатие звука · Авто:</strong> {hardware.stages.audio?.auto_device === 'cpu' ? 'CPU' : hardware.stages.audio?.auto_device ?? 'не определено'}<small>{hardware.stages.audio?.reason ?? 'Используется проверенный путь.'}</small></li>
              <li><strong>Говорящие · Авто:</strong> {hardware.stages.diarization?.auto_device === 'cpu' ? 'CPU' : hardware.stages.diarization?.auto_device ?? 'CPU'}<small>{hardware.stages.diarization?.reason ?? 'CPU · нет сопоставимого замера качества и скорости'}</small></li>
              <li><strong>Поиск · Авто:</strong> {hardware.stages.index?.auto_device === 'cpu' ? 'CPU' : hardware.stages.index?.auto_device ?? 'не определено'}<small>{hardware.stages.index?.query_reason ?? hardware.stages.index?.bulk_reason ?? 'GPU для поиска не выбран.'}</small></li>
            </ul>
            <p className="hardware-runtime-status">GPU среда: {gpuVerified ? 'подготовлена и проверена' : gpuPrepared ? 'подготовлена, но ещё не проверена; ручной запуск будет экспериментальным' : 'не подготовлена'}</p>
            {hardwarePreparation?.status === 'running' && <div className="hardware-preparation-progress" role="status"><LoaderCircle size={16} className="spin" /><div><strong>Подготавливаем среду</strong><p>{latestHardwareProgress?.stage ?? hardwarePreparation.stage ?? 'Подготовка'}{latestHardwareProgress?.detail ? ` · ${latestHardwareProgress.detail}` : ''}</p></div></div>}
            {screenGpu?.can_prepare && <div className="hardware-prepare-action"><p>Оценка загрузки: {formatBytes(hardware.gpu_runtime.estimated_download_bytes ?? hardwarePreparation?.estimated_download_bytes ?? 0)}. Подготовка запускается только после вашего действия.</p><button className="secondary-button" type="button" onClick={() => void prepareGpu()} disabled={hardwarePrepareBusy || hardwarePreparation?.status === 'running'}>{hardwarePrepareBusy ? 'Запускаем…' : gpuPrepared ? 'Проверить GPU' : 'Подготовить среду GPU'}</button></div>}
            {!screenGpu?.can_prepare && !gpuPrepared && <p className="settings-note">Подготовка DirectML сейчас недоступна для обнаруженного оборудования.</p>}
          </>}
        </section>
        <section className="settings-group semantic-assets-settings"><h2>Семантический поиск</h2>
          {!assets ? <p className="settings-note">Проверяем локальную модель…</p> : assets.available ? <>
            <p className="asset-ready-status"><Check size={16} />Локальный семантический поиск готов{assets.version ? ` · ${assets.version}` : ''}</p>
            {assets.files?.length ? <ul className="asset-file-list">{assets.files.map((file) => <li key={file.name}>{file.name} · {file.available ? 'проверен' : 'не найден'}</li>)}</ul> : null}
          </> : <>
            <p className="settings-note">Модель не загружается при запуске. После подготовки запросы смогут находить похожие по смыслу фрагменты.</p>
            <p className="asset-download-estimate">Оценка загрузки: {formatBytes(assets.download_bytes ?? assetPreparation?.estimated_download_bytes ?? 0)}.</p>
            {assetPreparation?.status === 'running' && <div className="hardware-preparation-progress" role="status"><LoaderCircle size={16} className="spin" /><div><strong>Подготавливаем семантический поиск</strong><p>{latestAssetProgress?.file ?? assetPreparation.stage ?? 'Загрузка модели'}{typeof latestAssetProgress?.completed_bytes === 'number' && typeof latestAssetProgress.total_bytes === 'number' ? ` · ${formatBytes(latestAssetProgress.completed_bytes)} из ${formatBytes(latestAssetProgress.total_bytes)}` : typeof latestAssetProgress?.percent === 'number' ? ` · ${latestAssetProgress.percent}%` : ''}</p></div></div>}
            <button className="secondary-button" type="button" onClick={() => void prepareSemanticSearch()} disabled={assetsPrepareBusy || assetPreparation?.status === 'running'}>{assetsPrepareBusy ? 'Запускаем…' : 'Подготовить семантический поиск'}</button>
          </>}
        </section>
        <section className="settings-group backup-settings"><h2>Резервные копии и очистка</h2>
          <label className="settings-choice backup-cleanup-choice"><input type="checkbox" checked={settings.auto_cleanup} onChange={(event) => updateSettings('auto_cleanup', event.target.checked)} /><span><strong>Разрешить очистку старых исходников</strong><small>После включения приложение периодически проверяет срок хранения и условия удаления. Нужны проверенная копия и успешный тест восстановления.</small></span></label>
          <p className="settings-note">Срок хранения исходников: {settings.original_retention_days} дн. Папка резервных копий задаётся выше и сохраняется перед запуском операций.</p>
          {!backupStatus ? <p className="settings-note">Проверяем резервные копии…</p> : <>
            {backupStatus.storage.error && <p className="settings-note">{backupStatus.storage.error.message}</p>}
            {backupStatus.storage.configured ? <p className="backup-storage-state">Папка подключена: {backupStatus.storage.backup_root}</p> : <p className="backup-storage-state">Папка резервных копий не настроена. Выберите её выше и сохраните настройки.</p>}
            {backupJob?.status === 'running' && <div className="hardware-preparation-progress" role="status"><LoaderCircle size={16} className="spin" /><div><strong>{backupOperationLabel(backupJob.operation)}</strong><p>{backupJob.stage ?? 'Выполняется'}{backupJob.progress?.at(-1)?.path ? ` · ${backupJob.progress.at(-1)?.path}` : ''}{typeof backupJob.progress?.at(-1)?.completed === 'number' && typeof backupJob.progress?.at(-1)?.total === 'number' ? ` · ${backupJob.progress.at(-1)?.completed} из ${backupJob.progress.at(-1)?.total}` : ''}</p></div></div>}
            {settings.backup_root && !backupRootSaved && <p className="settings-note">Сохраните выбранную папку резервной копии перед запуском операций.</p>}
            <div className="backup-actions"><button className="primary-button" type="button" onClick={() => void runBackupOperation('backup')} disabled={!backupStatus.storage.configured || !backupRootSaved || backupBusy || backupJob?.status === 'running'}>Создать резервную копию</button><button className="secondary-button" type="button" onClick={() => void runBackupOperation('cleanup_preview')} disabled={backupBusy || backupJob?.status === 'running'}>Проверить условия очистки</button></div>
            {backupStatus.storage.snapshots.length === 0 ? <p className="settings-note">Снимков пока нет.</p> : <div className="backup-snapshot-list">{backupStatus.storage.snapshots.map((snapshot) => <article className="backup-snapshot" key={snapshot.snapshot_id}>
              <div className="backup-snapshot-summary"><strong>{snapshot.created_at ? new Intl.DateTimeFormat('ru-RU', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(snapshot.created_at)) : snapshot.snapshot_id}</strong><small>{snapshot.file_count} файлов · {formatBytes(snapshot.total_bytes)}</small><small>{snapshot.verified_receipt ? 'Копия проверена' : 'Копия ещё не проверена'} · {snapshot.restore_test_passed ? 'Тест восстановления пройден' : 'Тест восстановления не пройден'}</small></div>
              <div className="backup-snapshot-actions"><button className="text-action" type="button" onClick={() => void runBackupOperation('verify', snapshot.snapshot_id)} disabled={backupBusy || backupJob?.status === 'running'}>Проверить</button><button className="text-action" type="button" onClick={() => void runBackupOperation('restore_test', snapshot.snapshot_id)} disabled={backupBusy || backupJob?.status === 'running'}>Тест восстановления</button></div>
              <div className="backup-restore-controls"><label className="settings-field">Папка для восстановления<input value={backupTarget} readOnly placeholder="Выберите новую или пустую папку" /></label><button className="secondary-button" type="button" onClick={() => void chooseBackupRestoreTarget()} disabled={backupBusy || backupJob?.status === 'running'}>Выбрать папку</button><button className="text-action" type="button" onClick={() => setBackupConfirmation({ operation: 'restore', snapshotId: snapshot.snapshot_id })} disabled={!backupTarget || backupBusy || backupJob?.status === 'running'}>Восстановить в папку</button></div>
            </article>)}</div>}
            {cleanupPreview && <div className="cleanup-preview"><strong>Проверка очистки: {cleanupPreview.eligible_count} можно удалить, {cleanupPreview.blocked_count} заблокировано.</strong>{cleanupPreview.items.slice(0, 5).map((item) => <article key={item.meeting_id}><span>{item.filename}{typeof item.age_days === 'number' ? ` · ${item.age_days} дн.` : ''}</span><small>{item.eligible ? 'Есть проверенная копия и исходник готов к очистке.' : item.blocked_reasons.join(' ')}</small></article>)}{cleanupPreview.items.length > 5 && <small>Показаны первые пять файлов из {cleanupPreview.items.length}.</small>}{!settings.auto_cleanup || !backupStatus.storage.auto_cleanup_enabled ? <small>Сохраните настройку, разрешающую очистку, чтобы выполнить удаление.</small> : null}<button className="danger-button" type="button" onClick={() => setBackupConfirmation({ operation: 'cleanup' })} disabled={!settings.auto_cleanup || !backupStatus.storage.auto_cleanup_enabled || cleanupPreview.eligible_count === 0 || backupBusy || backupJob?.status === 'running'}>Удалить проверенные старые исходники</button></div>}
          </>}
        </section>
        <section className="settings-group"><h2>Codex</h2><ConnectorPanel status={connector} busy={connectorBusy} onAction={(action) => void runConnector(action)} /></section>
        <div className="settings-save-row"><button className="primary-button" onClick={() => void saveSettings()} disabled={settingsBusy}>{settingsBusy ? <LoaderCircle size={16} className="spin" /> : <Check size={16} />}Сохранить настройки</button></div>
      </>}
    </section>}

    {toast && <div className="meeting-toast" role="status"><span>{toast.message}</span>{toast.onUndo && <button onClick={toast.onUndo}><Undo2 size={14} />Отменить</button>}<button aria-label="Закрыть сообщение" onClick={() => setToast(null)}><X size={14} /></button></div>}
    {errorDialog && <div className="meeting-dialog-backdrop"><section className="meeting-error-dialog" role="alertdialog" aria-modal="true" aria-labelledby="meeting-error-title" aria-describedby="meeting-error-reason"><AlertCircle size={22} /><h2 id="meeting-error-title">{errorDialog.title}</h2><p id="meeting-error-reason">{errorDialog.reason}</p><p>{errorDialog.recommendation}</p><button className="primary-button" autoFocus onClick={() => setErrorDialog(null)}>ОК</button></section></div>}
    {backupConfirmation && <div className="meeting-dialog-backdrop"><section className="meeting-error-dialog" role="alertdialog" aria-modal="true" aria-labelledby="backup-confirm-title"><AlertCircle size={22} /><h2 id="backup-confirm-title">{backupConfirmation.operation === 'cleanup' ? 'Удалить исходники?' : 'Восстановить резервную копию?'}</h2><p>{backupConfirmation.operation === 'cleanup' ? 'Будут удалены только исходные файлы, которые прошли свежую проверку контрольных сумм, срока хранения и теста восстановления. Текст, звук и сохранённые кадры останутся.' : 'Копия будет восстановлена в выбранную новую или пустую папку. Существующие файлы в целевой папке не перезаписываются.'}</p><div className="backup-confirm-actions"><button className="secondary-button" type="button" onClick={() => setBackupConfirmation(null)}>Отмена</button><button className={backupConfirmation.operation === 'cleanup' ? 'danger-button' : 'primary-button'} type="button" autoFocus onClick={confirmBackupAction}>Продолжить</button></div></section></div>}
  </main>
}

function RecorderPanel(props: {
  recorder: RecorderStatus | null; error: string; busy: boolean; prepared: boolean; title: string; selection: RecorderSelection; previewFailed: boolean;
  onTitle: (value: string) => void; onSelection: (value: RecorderSelection) => void; onPrepare: () => void; onConfigure: () => void;
  onStart: () => void; onStop: () => void; onPreviewError: () => void;
}) {
  const recording = props.recorder?.state === 'recording'
  const available = Boolean(props.recorder?.available && props.recorder.connected)
  const selectValue = (key: keyof RecorderSelection, value: string) => props.onSelection({ ...props.selection, [key]: value || null })
  return <section className={`recorder-panel ${recording ? 'is-recording' : ''}`} aria-label="Запись встречи">
    <div className="recorder-heading"><div className="recorder-icon"><Radio size={18} /></div><div><h2>{recording ? 'Идёт запись' : 'Запись встречи'}</h2><p>{recording ? `${props.recorder?.title || 'Встреча'} · ${formatTime(props.recorder?.duration_seconds)}` : 'Перед началом проверьте изображение и источники звука.'}</p></div>{recording ? <button className="danger-button" onClick={props.onStop} disabled={props.busy}><Square size={15} />Закончить запись</button> : <button className="secondary-button" onClick={props.onPrepare} disabled={props.busy}><Settings2 size={15} />Подготовить</button>}</div>
    {!recording && <div className="recorder-controls">
      {props.error && <p className="recorder-unavailable" role="status"><AlertCircle size={15} />{props.error}</p>}
      {props.recorder && <>
        <label className="recorder-select"><span><Monitor size={14} />Экран</span><select value={props.selection.monitor_id ?? ''} onChange={(event) => selectValue('monitor_id', event.target.value)} disabled={!available}><option value="">Выберите экран</option>{props.recorder.monitors.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
        <label className="recorder-select"><span><Volume2 size={14} />Звук созвона</span><select value={props.selection.browser_id ?? ''} onChange={(event) => selectValue('browser_id', event.target.value)} disabled={!available}><option value="">Выберите источник звука</option>{props.recorder.browsers.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
        <label className="recorder-select"><span><Mic size={14} />Микрофон</span><select value={props.selection.microphone_id ?? ''} onChange={(event) => selectValue('microphone_id', event.target.value)} disabled={!available}><option value="">Выберите микрофон</option>{props.recorder.microphones.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
        <div className="recorder-preview">{props.recorder.preview_url && !props.previewFailed ? <img src={meetingsApi.previewUrl(props.recorder.preview_url)} alt="Предварительный просмотр выбранного экрана" onError={props.onPreviewError} /> : <div><ImageIcon size={22} /><span>{props.previewFailed ? 'Предварительный просмотр недоступен.' : 'Предварительный просмотр появится после подготовки.'}</span></div>}</div>
        <div className="audio-levels"><AudioLevel label="Звук созвона" value={props.recorder.levels.browser} stale={props.recorder.levels.stale} /><AudioLevel label="Микрофон" value={props.recorder.levels.microphone} stale={props.recorder.levels.stale} /></div>
        <label className="recorder-title">Название встречи<input value={props.title} onChange={(event) => props.onTitle(event.target.value)} placeholder="Например, Планирование на сегодня" /></label>
        <div className="recorder-actions"><button className="secondary-button" disabled={!available || props.busy || !props.prepared} onClick={props.onConfigure}>Сохранить источники</button><button className="primary-button" disabled={!available || props.busy || !props.prepared} onClick={props.onStart}><Radio size={15} />Начать запись</button></div>
      </>}
    </div>}
  </section>
}

function AudioLevel({ label, value, stale }: { label: string; value: number | null; stale: boolean }) {
  const usable = typeof value === 'number' && Number.isFinite(value) && !stale
  const level = usable ? Math.max(0, Math.min(100, value)) : 0
  return <div className="audio-level"><span>{label}</span>{usable ? <div className="level-track" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={level}><i style={{ width: `${level}%` }} /></div> : <small>Уровень пока недоступен</small>}</div>
}

function FolderField({ label, value, onChoose, clearable = false, onClear }: { label: string; value: string; onChoose: () => void; clearable?: boolean; onClear?: () => void }) {
  return <div className="folder-field"><label>{label}<input value={value} readOnly placeholder="Выберите папку" /></label><button className="secondary-button" onClick={onChoose}><FolderOpen size={15} />Выбрать</button>{clearable && <button className="text-action" onClick={onClear}>Очистить</button>}</div>
}

function ConnectorPanel({ status, busy, onAction }: { status: ConnectorStatus | null; busy: boolean; onAction: (action: 'connect' | 'check' | 'disconnect') => void }) {
  const label = status?.client_verified ? 'Доступ проверен реальным вызовом' : status?.protocol_checked ? 'Протокол проверен; клиент Codex ещё не подтверждён' : status?.registered ? 'Сервер зарегистрирован; обновите Codex и проверьте доступ' : 'Codex не подключён'
  const uriStatus = status?.evidence_protocol
  const canAddUriHandler = Boolean(status?.registered && uriStatus?.supported && !uriStatus.owned && !uriStatus.conflict)
  const connectLabel = canAddUriHandler ? 'Настроить ссылки для встреч' : 'Подключить'
  return <div className="connector-panel"><div className={`connector-status ${status?.client_verified ? 'verified' : ''}`}><span className="connector-dot" /><p>{label}</p></div><p className="settings-note">Подключение даёт Codex доступ к встречам только для чтения.</p>{uriStatus?.owned && <p className="settings-note">Ссылки video-transcribe:// можно открывать повторно после закрытия Codex.</p>}{uriStatus?.conflict && <p className="settings-note">Схема ссылок уже занята другим приложением; существующая регистрация сохранена.</p>}<div className="connector-actions"><button className="primary-button" onClick={() => onAction('connect')} disabled={busy || (Boolean(status?.registered) && !canAddUriHandler)}>{busy ? 'Подключаем…' : connectLabel}</button><button className="secondary-button" onClick={() => onAction('check')} disabled={busy}>Проверить подключение</button><button className="secondary-button" onClick={() => onAction('disconnect')} disabled={busy || !status?.registered}>Отключить</button></div></div>
}

function formatBytes(value: number): string {
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(0)} МБ`
  return `${(value / 1024 ** 3).toFixed(1)} ГБ`
}

