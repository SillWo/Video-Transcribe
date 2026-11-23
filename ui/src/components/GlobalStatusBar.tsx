import { useEffect, useRef, useState, type MouseEvent } from 'react'
import { AlertCircle, Check, ChevronUp, LoaderCircle, Play, Radio, Square, X } from 'lucide-react'
import { meetingsApi, type MeetingQueueState, type RecorderStatus } from '../lib/meetingsApi'
import { fetchJson } from '../lib/api'
import { recentFailedConversions, type ConversionHistoryItem } from '../lib/conversionHistory'
import { normalizeTaskStage, taskStageLabel } from '../lib/taskProgress'

type QueueSummary = { id: string; status?: string; stage?: string; source?: string; title?: string }
type QueueView = Omit<MeetingQueueState, 'waitingJobs' | 'activeJob'> & { waitingJobs?: QueueSummary[]; activeJob?: QueueSummary | null }
const duration = (seconds: number) => { const n = Math.max(0, Math.floor(seconds)); return `${String(Math.floor(n / 3600)).padStart(2, '0')}:${String(Math.floor(n % 3600 / 60)).padStart(2, '0')}:${String(n % 60).padStart(2, '0')}` }
const stageLabel = (stage?: string) => {
  const normalized = normalizeTaskStage(stage)
  return normalized ? taskStageLabel(normalized) : 'Обработка'
}

export default function GlobalStatusBar() {
  const [recorder, setRecorder] = useState<RecorderStatus | null>(null)
  const [recorderStale, setRecorderStale] = useState(false)
  const [queue, setQueue] = useState<QueueView | null>(null)
  const [failedConversions, setFailedConversions] = useState<ConversionHistoryItem[]>([])
  const [queueStale, setQueueStale] = useState(false)
  const [open, setOpen] = useState(false)
  const [popoverMounted, setPopoverMounted] = useState(false)
  const [openSource, setOpenSource] = useState<'pointer' | 'keyboard'>('keyboard')
  const [busy, setBusy] = useState(false)
  const [recordingTransition, setRecordingTransition] = useState<'starting' | 'stopping' | null>(null)
  const [confirmCancel, setConfirmCancel] = useState(false)
  const [error, setError] = useState('')
  const [recorderError, setRecorderError] = useState('')
  const [recorderNotice, setRecorderNotice] = useState('')
  const cancelDialog = useRef<HTMLDialogElement>(null)
  const cancelBackRef = useRef<HTMLButtonElement>(null)
  const queueTriggerRef = useRef<HTMLButtonElement>(null)
  const closePopoverTimer = useRef<number | null>(null)
  const queueSignature = useRef('')
  const historyPolledAt = useRef(0)
  const historySignature = useRef('')
  useEffect(() => {
    let live = true, timer: number
    const poll = async () => {
      const now = Date.now()
      const shouldRefreshHistory = now - historyPolledAt.current >= 5000
      if (shouldRefreshHistory) historyPolledAt.current = now
      const [recording, queued, history] = await Promise.allSettled([
        meetingsApi.recorder(),
        meetingsApi.queue(),
        shouldRefreshHistory ? fetchJson<{ items: ConversionHistoryItem[] }>('/api/simple/history') : Promise.resolve(null),
      ])
      if (!live) return
      if (recording.status === 'fulfilled') { setRecorder(recording.value); setRecorderStale(false); setRecorderError('') }
      else { setRecorderStale(true); setRecorderError('Не удалось проверить состояние записи') }
      if (queued.status === 'fulfilled') {
        const next = queued.value
        const signature = JSON.stringify({ paused: next.paused, active: next.activeJob && [next.activeJob.id, next.activeJob.status, next.activeJob.stage], waiting: next.waitingJobs?.map((job) => [job.id, job.status, job.stage]) ?? [] })
        if (queueSignature.current && queueSignature.current !== signature) window.dispatchEvent(new Event('archive-changed'))
        queueSignature.current = signature
        setQueue(next); setQueueStale(false)
      }
      else setQueueStale(true)
      if (history.status === 'fulfilled' && history.value) {
        setFailedConversions(recentFailedConversions(history.value.items))
        const signature = JSON.stringify(history.value.items.map((item) => [item.id, item.status]))
        if (signature !== historySignature.current) {
          historySignature.current = signature
          window.dispatchEvent(new Event('archive-changed'))
          window.dispatchEvent(new Event('simple-history-changed'))
        }
      }
      timer = window.setTimeout(() => void poll(), 1800)
    }
    void poll()
    return () => { live = false; window.clearTimeout(timer) }
  }, [])
  useEffect(() => () => { if (closePopoverTimer.current !== null) window.clearTimeout(closePopoverTimer.current) }, [])
  useEffect(() => {
    const onTransition = (event: Event) => {
      const phase = (event as CustomEvent<'starting' | 'stopping' | 'settled'>).detail
      setRecordingTransition(phase === 'settled' ? null : phase)
      if (phase === 'starting') setRecorderNotice('')
    }
    window.addEventListener('recorder-transition', onTransition)
    return () => window.removeEventListener('recorder-transition', onTransition)
  }, [])
  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        setOpen(false); setPopoverMounted(false)
        queueTriggerRef.current?.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [open])
  function toggleQueue(event: MouseEvent<HTMLButtonElement>) {
    const keyboard = event.detail === 0
    setOpenSource(keyboard ? 'keyboard' : 'pointer')
    if (closePopoverTimer.current !== null) window.clearTimeout(closePopoverTimer.current)
    if (open) {
      setOpen(false)
      if (keyboard) setPopoverMounted(false)
      else closePopoverTimer.current = window.setTimeout(() => setPopoverMounted(false), 180)
    } else { setPopoverMounted(true); setOpen(true) }
  }
  useEffect(() => {
    const dialog = cancelDialog.current
    if (!dialog) return
    if (confirmCancel && !dialog.open) { dialog.showModal(); cancelBackRef.current?.focus() }
    else if (!confirmCancel && dialog.open) dialog.close()
  }, [confirmCancel])
  async function stopRecording() {
    if (recordingTransition) return
    setRecordingTransition('stopping'); setError(''); setRecorderError('')
    try { const result = await meetingsApi.stopRecording(); setRecorder(result.recorder); setRecorderStale(false); setRecorderError(''); setRecorderNotice('Запись сохранена и добавлена в очередь обработки'); window.dispatchEvent(new Event('archive-changed')); window.dispatchEvent(new Event('simple-history-changed')) }
    catch (reason) { setRecorderError(reason instanceof Error ? reason.message : 'Не удалось остановить запись') }
    finally { setRecordingTransition(null) }
  }
  async function resume() {
    setBusy(true); setError('')
    try { setQueue(await meetingsApi.resumeQueue()) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Не удалось продолжить очередь') }
    finally { setBusy(false) }
  }
  async function cancelWaiting() {
    setBusy(true); setError('')
    try { setQueue(await fetchJson<QueueView>('/api/simple/queue/cancel', { method: 'POST' })); setConfirmCancel(false) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Не удалось отменить задачи') }
    finally { setBusy(false) }
  }
  const activeCount = queue?.waitingCount ?? 0
  const hasQueue = queueStale || failedConversions.length > 0 || Boolean(queue && (queue.activeJob || activeCount || queue.paused))
  const recorderState = recorder?.state?.toLowerCase()
  const starting = recordingTransition === 'starting' || recorderState === 'starting' || recorderState === 'start_pending'
  const stopping = recordingTransition === 'stopping' || ['stopping', 'saving', 'finalizing'].includes(recorderState ?? '')
  const isRecording = recorderState === 'recording'
  const recorderUnknown = !starting && !stopping && (recorder === null || recorderStale || recorderState === 'unavailable' || !recorder.connected)
  const queueLabel = queueStale ? 'Очередь недоступна' : queue?.paused ? 'Очередь приостановлена' : queue?.activeJob ? `Обработка · ${stageLabel(queue.activeJob.stage)}` : activeCount ? 'Задачи ожидают' : failedConversions.length ? `Ошибки обработки · ${failedConversions.length}` : 'Очередь свободна'
  return <footer className="global-statusbar">
    <div className="global-status-left">
      {stopping ? <><span className="status-dot paused" /><LoaderCircle size={14} className="spin" /><strong>Останавливаем и сохраняем…</strong>{recorder && <time>{duration(recorder.duration_seconds ?? 0)}</time>}<span className="status-title">{recorder?.title || 'Встреча'}</span></>
        : starting ? <><span className="status-dot working" /><LoaderCircle size={14} className="spin" /><strong>Запускаем запись…</strong><span className="status-title">{recorder?.title || 'Встреча'}</span></>
          : isRecording ? <><span className="status-dot live" /><Radio size={15} /><strong>Идёт запись</strong><time>{duration(recorder?.duration_seconds ?? 0)}</time><span className="status-title">{recorder?.title || 'Встреча'}</span>{recorderError && <span className="status-recorder-error" role="alert">{recorderError}</span>}<button type="button" className="status-stop" onClick={() => void stopRecording()} disabled={recordingTransition !== null}><Square size={13} />Стоп</button></>
            : recorderUnknown ? <><span className="status-dot paused" /><AlertCircle size={14} /><span>{recorderError || 'Состояние записи неизвестно'}</span></> : <><span className="status-dot ready" /><Check size={14} /><span>{recorderNotice || recorderError || 'Запись не идёт'}</span></>}
    </div>
    <div className="global-status-right">
      <button ref={queueTriggerRef} type="button" className={`queue-status ${hasQueue ? 'has-queue' : ''}`} aria-expanded={open} aria-controls="global-queue-popover" onClick={toggleQueue}><span className={`status-dot ${queueStale || queue?.paused || failedConversions.length ? 'paused' : queue?.activeJob ? 'working' : ''}`} /><span>{queueLabel}</span><span className="queue-count">{activeCount || failedConversions.length}</span><ChevronUp size={14} className={open ? 'turned' : ''} /></button>
      {popoverMounted && <section id="global-queue-popover" className={`queue-popover${open ? ' open' : ' closing'}${openSource === 'keyboard' ? ' instant' : ''}`} aria-label="Очередь обработки" aria-hidden={!open} inert={!open}>
        <header><strong>Обработка</strong><span>{activeCount} в очереди</span></header>
        {(error || queueStale) && <p className="queue-error" role="alert"><AlertCircle size={14} />{error || 'Не удалось проверить состояние очереди'}</p>}
        {queue?.activeJob && <article className="queue-item active"><LoaderCircle size={15} className="spin" /><div><strong>{queue.activeJob.title || queue.activeJob.source || 'Текущая задача'}</strong><small>{stageLabel(queue.activeJob.stage)}</small></div></article>}
        {queue?.waitingJobs?.map((job) => <article className="queue-item" key={job.id}><span className="queue-index">•</span><div><strong>{job.title || job.source || 'Задача'}</strong><small>Ожидает</small></div><button type="button" className="icon-button" aria-label={`Отменить ${job.title || job.source || 'задачу'}`} disabled={busy} onClick={async () => { try { await fetchJson(`/api/simple/jobs/${encodeURIComponent(job.id)}/cancel`, { method: 'POST' }); setQueue(await meetingsApi.queue()) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Не удалось отменить задачу') } }}><X size={15} /></button></article>)}
        {failedConversions.map((item) => <article className="queue-item failed" key={item.id}><AlertCircle size={15} /><div><strong>{item.title || item.source || 'Импортированный материал'}</strong><small>{item.status === 'interrupted' ? 'Обработка прервана' : 'Не удалось обработать'}{item.error ? ` · ${item.error}` : ''}</small></div><button type="button" className="quiet-button" onClick={() => window.dispatchEvent(new CustomEvent('transcribe-open-task', { detail: { id: item.id, archiveRoot: item.archiveRoot } }))}>Открыть и повторить</button></article>)}
        {!queue?.activeJob && !queue?.waitingJobs?.length && !failedConversions.length && <p className="queue-empty">Активных задач нет</p>}
        <div className="queue-actions">{queue?.paused && <button type="button" className="primary-button" onClick={() => void resume()} disabled={busy}><Play size={14} />Продолжить</button>}{(activeCount > 0 || queue?.activeJob) && <button type="button" className="quiet-button" onClick={() => setConfirmCancel(true)} disabled={busy}>Отменить задачи</button>}</div>
      </section>}
    </div>
    <dialog ref={cancelDialog} className="queue-confirm-dialog" onCancel={(event) => { event.preventDefault(); if (!busy) setConfirmCancel(false) }}><strong>Отменить ожидающие задачи?</strong><p>Текущая задача остановится, ожидающие будут сняты с очереди.</p><div><button ref={cancelBackRef} type="button" className="quiet-button" onClick={() => setConfirmCancel(false)} disabled={busy}>Назад</button><button type="button" className="danger-button" onClick={() => void cancelWaiting()} disabled={busy}>Отменить</button></div></dialog>
  </footer>
}
