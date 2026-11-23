import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { AlertCircle, Clipboard, FilePlus2, FileText, Image as ImageIcon, LoaderCircle, Play, Search, Trash2 } from 'lucide-react'
import { API_BASE } from '../lib/api'
import { archiveApi, archiveSelectionKey, filterArchiveRows, hasCompletedTranscript, type ArchiveKind, type ArchiveRow } from '../lib/archiveApi'
import { sharedDraftManager, transcriptionDraftKey } from '../lib/draftManager'
import { registerPendingSave } from '../lib/pendingSaves'
import { meetingsApi, readMeetingTimeline } from '../lib/meetingsApi'
import type { MeetingDetail, TimelineItem } from '../lib/meetingsApi'
import { flushPendingSaves } from '../lib/pendingSaves'
import ToastNotice from '../components/ToastNotice'
import MeasuredDisclosure from '../components/MeasuredDisclosure'
import { notifyLearnedTerms } from '../lib/learnedTermsNotice'

const pageSize = 50
type MeetingStage = 'speech' | 'screen' | 'audio' | 'index'
const meetingStages: MeetingStage[] = ['speech', 'screen', 'audio', 'index']
const meetingStageNames: Record<MeetingStage, string> = { speech: 'Речь', screen: 'Экран', audio: 'Звук', index: 'Поиск' }
const kindName = (kind: ArchiveKind) => kind === 'video' ? 'Видео' : 'Аудио'
const statusName = (status: string) => ({ ready: 'Готово', completed: 'Готово', failed: 'Ошибка', interrupted: 'Прервана', cancelled: 'Отменена', cancelling: 'Останавливается', queued: 'В очереди', queued_for_processing: 'Ожидает обработки', imported: 'Ожидает обработки', running: 'Обрабатывается', processing: 'Обрабатывается', partial: 'Частично готово' }[status.toLowerCase()] ?? 'Состояние обновляется')
const dateLabel = (value?: string | null) => { if (!value) return 'Дата не указана'; const date = new Date(value); return Number.isNaN(date.valueOf()) ? 'Дата не указана' : new Intl.DateTimeFormat('ru-RU', { dateStyle: 'medium', timeStyle: 'short' }).format(date) }
const archiveDateLabel = (row: ArchiveRow) => row.kind === 'video' ? row.started_at ? dateLabel(row.started_at) : 'Дата встречи не подтверждена' : dateLabel(row.created_at)

export default function ArchivePage() {
  const [params, setParams] = useSearchParams()
  const [filter, setFilter] = useState<'all' | ArchiveKind>(() => ['audio', 'transcription'].includes(params.get('kind') ?? '') ? 'audio' : ['video', 'meeting'].includes(params.get('kind') ?? '') ? 'video' : 'all')
  const [query, setQuery] = useState('')
  const [items, setItems] = useState<ArchiveRow[]>([])
  const [nextOffset, setNextOffset] = useState<number | null>(null)
  const [total, setTotal] = useState(0)
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<ArchiveRow | null>(null)
  const [text, setText] = useState('')
  const [detailBusy, setDetailBusy] = useState(false)
  const [transcriptionAvailable, setTranscriptionAvailable] = useState(false)
  const [saveState, setSaveState] = useState<'saved' | 'saving' | 'failed'>('saved')
  const [detailError, setDetailError] = useState('')
  const [timeline, setTimeline] = useState<TimelineItem[]>([])
  const [meeting, setMeeting] = useState<MeetingDetail | null>(null)
  const [evidence, setEvidence] = useState('')
  const [playerUrl, setPlayerUrl] = useState('')
  const [compactEvidenceOpen, setCompactEvidenceOpen] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const deleteDialogRef = useRef<HTMLDialogElement>(null)
  const deleteCancelRef = useRef<HTMLButtonElement>(null)
  const [selectedMeetings, setSelectedMeetings] = useState<Set<string>>(new Set())
  const [selectedStages, setSelectedStages] = useState<Set<MeetingStage>>(new Set(meetingStages))
  const [processBusy, setProcessBusy] = useState(false)
  const [importBusy, setImportBusy] = useState(false)
  const [segmentDrafts, setSegmentDrafts] = useState<Record<string, string>>({})
  const segmentDraftsRef = useRef(new Map<string, { text: string; dirty: boolean; saving?: Promise<void> }>())
  const [speakerDrafts, setSpeakerDrafts] = useState<Record<string, string>>({})
  const speakerDraftsRef = useRef(new Map<string, { meetingId: string; name: string; dirty: boolean; saving?: Promise<void> }>())
  const [meetingToast, setMeetingToast] = useState('')
  const meetingRevisionRef = useRef('')
  const [listWidth, setListWidth] = useState(() => { try { return Number(localStorage.getItem('video-transcribe-archive-width')) || 340 } catch { return 340 } })
  const [evidenceWidth, setEvidenceWidth] = useState(() => { try { return Number(localStorage.getItem('video-transcribe-evidence-width')) || 360 } catch { return 360 } })
  const editTimerRef = useRef<number | null>(null)
  const listGenerationRef = useRef(0)
  const selectedKey = selected ? archiveSelectionKey(selected.kind, selected.id, selected.archive_root) : ''
  const selectedRef = useRef(selectedKey)
  selectedRef.current = selectedKey
  const resolvedDeepLinkRef = useRef('')
  const visibleItems = useMemo(() => filterArchiveRows(items, filter, query), [items, filter, query])

  const flushMeetingEdits = useCallback(async () => {
    const tasks: Promise<void>[] = []
    for (const [key, draft] of segmentDraftsRef.current) {
      if (!draft.dirty) continue
      const [meetingId, segmentId] = key.split('\u0000')
      const persist = async () => {
        while (true) {
          const current = segmentDraftsRef.current.get(key)
          if (!current?.dirty) return
          const text = current.text
          const result = await meetingsApi.saveSegment(meetingId, segmentId, text)
          const latest = segmentDraftsRef.current.get(key)
          if (latest?.text === text) latest.dirty = false
          const learned = result.remembered_terms ?? []
          notifyLearnedTerms(learned)
          if (selectedRef.current === archiveSelectionKey('video', meetingId)) setMeetingToast('Изменения встречи сохранены')
        }
      }
      if (!draft.saving) draft.saving = persist().finally(() => { const latest = segmentDraftsRef.current.get(key); if (latest) latest.saving = undefined })
      tasks.push(draft.saving)
    }
    for (const [speakerKey, draft] of speakerDraftsRef.current) {
      if (!draft.dirty) continue
      const [, speakerId] = speakerKey.split('\u0000')
      const persistSpeaker = async () => {
        while (true) {
          const latest = speakerDraftsRef.current.get(speakerKey)
          if (!latest?.dirty) return
          const name = latest.name
          await meetingsApi.saveSpeaker(draft.meetingId, speakerId, name)
          const current = speakerDraftsRef.current.get(speakerKey)
          if (current?.name === name) current.dirty = false
        }
      }
      if (!draft.saving) draft.saving = persistSpeaker().finally(() => { const latest = speakerDraftsRef.current.get(speakerKey); if (latest) latest.saving = undefined })
      tasks.push(draft.saving)
    }
    await Promise.all(tasks)
  }, [])

  useEffect(() => registerPendingSave('archive-meeting-edits', flushMeetingEdits), [flushMeetingEdits])

  const loadList = useCallback(async (append = false, next = 0, requestedGeneration?: number) => {
    const generation = requestedGeneration ?? ++listGenerationRef.current
    setBusy(true); setError('')
    try {
      const result = await archiveApi.list(filter, query.trim(), next, pageSize)
      if (listGenerationRef.current !== generation) return
      setItems((current) => append ? [...current, ...result.items.filter((row) => !current.some((prior) => archiveSelectionKey(prior.kind, prior.id, prior.archive_root) === archiveSelectionKey(row.kind, row.id, row.archive_root)))] : result.items)
      setNextOffset(result.next_offset); setTotal(result.total)
    } catch (reason) {
      if (listGenerationRef.current === generation) setError(reason instanceof Error ? reason.message : 'Не удалось загрузить архив')
    } finally { if (listGenerationRef.current === generation) setBusy(false) }
  }, [filter, query])

  useEffect(() => {
    const refreshOnArchiveChange = () => void loadList()
    window.addEventListener('archive-changed', refreshOnArchiveChange)
    window.addEventListener('simple-history-changed', refreshOnArchiveChange)
    return () => { window.removeEventListener('archive-changed', refreshOnArchiveChange); window.removeEventListener('simple-history-changed', refreshOnArchiveChange) }
  }, [loadList])

  useEffect(() => {
    const generation = ++listGenerationRef.current
    const timer = window.setTimeout(() => void loadList(false, 0, generation), 180)
    return () => {
      window.clearTimeout(timer)
      if (listGenerationRef.current === generation) listGenerationRef.current += 1
    }
  }, [loadList])
  useEffect(() => {
    const rawKind = params.get('kind'), id = params.get('id')
    if (!id) return
    const kind = rawKind === 'meeting' ? 'video' : rawKind === 'transcription' ? 'audio' : rawKind
    if (kind !== 'video' && kind !== 'audio') return
    const match = items.find((row) => row.kind === kind && row.id === id)
    if (match && archiveSelectionKey(match.kind, match.id, match.archive_root) !== selectedKey) void selectItem(match, false)
    if (!match && !busy && resolvedDeepLinkRef.current !== `${kind}:${id}`) {
      const key = `${kind}:${id}`
      const priorSelection = selectedRef.current
      resolvedDeepLinkRef.current = key
      void (async () => {
        try {
          if (kind === 'audio') {
            const detail = await archiveApi.transcription(id)
            if (selectedRef.current === priorSelection || selectedRef.current === key) await selectItem(detail.item, false)
          } else {
            const detail = await meetingsApi.detail(id, undefined, false)
            if (selectedRef.current !== priorSelection && selectedRef.current !== key) return
            const row: ArchiveRow = { id: detail.meeting.meeting_id || id, meeting_id: id, kind: 'video', title: detail.meeting.title || 'Встреча', created_at: detail.meeting.created_at || new Date().toISOString(), started_at: detail.meeting.started_at, status: String(detail.meeting.status ?? 'ready') }
            await selectItem(row, false)
          }
        } catch (reason) { if (selectedRef.current === priorSelection || selectedRef.current === key) setError(reason instanceof Error ? reason.message : 'Не удалось открыть запись по ссылке') }
      })()
    }
  // URL selection is resolved after the first archive page arrives.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, params])
  useEffect(() => () => { if (editTimerRef.current !== null) window.clearTimeout(editTimerRef.current) }, [])
  useEffect(() => {
    const dialog = deleteDialogRef.current
    if (!dialog) return
    if (confirmDelete && !dialog.open) { dialog.showModal(); deleteCancelRef.current?.focus() }
    else if (!confirmDelete && dialog.open) dialog.close()
  }, [confirmDelete])

  useEffect(() => {
    if (!selected || selected.kind !== 'video' || detailBusy) return
    const id = meetingIdFor(selected), key = archiveSelectionKey(selected.kind, selected.id, selected.archive_root)
    let stopped = false, timer: number
    const pollRevision = async () => {
      try {
        const result = await meetingsApi.revision(id)
        if (stopped || selectedRef.current !== key || !result.revision) return
        if (meetingRevisionRef.current && meetingRevisionRef.current !== result.revision) {
          const [detail, rows] = await Promise.all([meetingsApi.detail(id, undefined, false), readMeetingTimeline(id)])
          if (stopped || selectedRef.current !== key) return
          const merged = rows.map((row) => {
            if (row.kind !== 'transcript' || !row.id) return row
            const draft = segmentDraftsRef.current.get(`${id}\u0000${row.id}`)
            return draft?.dirty ? { ...row, text: draft.text } : row
          })
          setMeeting(detail); setTimeline(merged)
        }
        meetingRevisionRef.current = result.revision
      } catch { /* Keep the last complete timeline until the next revision check. */ }
      if (!stopped) timer = window.setTimeout(() => void pollRevision(), 4500)
    }
    void pollRevision()
    return () => { stopped = true; window.clearTimeout(timer) }
  }, [selected, selectedKey, detailBusy])

  const saveText = useCallback(async () => {
    if (!selected || selected.kind !== 'audio') return
    const key = transcriptionDraftKey(selected.id, selected.archive_root)
    await sharedDraftManager.flush(key)
    if (selectedRef.current === archiveSelectionKey(selected.kind, selected.id, selected.archive_root)) setSaveState('saved')
  }, [selected])

  const registerArchiveDraft = (item: ArchiveRow, value: string, revision: number) => {
    const key = transcriptionDraftKey(item.id, item.archive_root)
    sharedDraftManager.hydrate(key, value, revision)
    sharedDraftManager.registerPersister(key, async (draftText, baseRevision) => {
      if (selectedRef.current === archiveSelectionKey(item.kind, item.id, item.archive_root)) setSaveState('saving')
      const result = await archiveApi.saveText(item.id, draftText, baseRevision, item.archive_root)
      if (selectedRef.current === archiveSelectionKey(item.kind, item.id, item.archive_root)) setSaveState('saved')
      const terms = result.remembered_terms ?? []
      notifyLearnedTerms(terms)
      return { text: result.text ?? draftText, revision: result.revision }
    })
    return key
  }

  const meetingIdFor = (item: ArchiveRow) => item.meeting_id || item.id
  const visibleMeetingIds = visibleItems.filter((row) => row.kind === 'video').map(meetingIdFor)
  const toggleMeeting = (id: string) => setSelectedMeetings((current) => { const next = new Set(current); if (next.has(id)) next.delete(id); else next.add(id); return next })
  const selectAllMeetings = () => setSelectedMeetings((current) => visibleMeetingIds.length > 0 && visibleMeetingIds.every((id) => current.has(id)) ? new Set([...current].filter((id) => !visibleMeetingIds.includes(id))) : new Set([...current, ...visibleMeetingIds]))

  async function changeFilter(value: 'all' | ArchiveKind) {
    try { await flushPendingSaves() } catch (reason) { setError(reason instanceof Error ? reason.message : 'Не удалось сохранить изменения'); return }
    if (editTimerRef.current !== null) { window.clearTimeout(editTimerRef.current); editTimerRef.current = null }
    selectedRef.current = ''; setSelected(null); setFilter(value); setParams(value === 'all' ? {} : { kind: value })
  }

  async function importMeetings() {
    setImportBusy(true); setError('')
    try {
      const selection = window.desktopApi ? await window.desktopApi.pickFiles() : await meetingsApi.pickFiles()
      const paths = selection.files.map((file) => file.path).filter(Boolean)
      if (!paths.length) { if (selection.rejected.length) setError(selection.rejected.map((item) => item.message).join(' ')); return }
      const result = await meetingsApi.importFiles(paths)
      window.dispatchEvent(new Event('archive-changed'))
      window.dispatchEvent(new Event('simple-history-changed'))
      await loadList()
      if (result.jobs.length) setMeetingToast(`Добавлено в очередь обработки: ${result.jobs.length}`)
      if (result.rejected.length) setError(result.rejected.map((item) => item.reason).join(' '))
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Не удалось добавить файлы') }
    finally { setImportBusy(false) }
  }

  async function processMeetings(ids: string[], stages: MeetingStage[] = meetingStages) {
    if (!ids.length || !stages.length) return
    setProcessBusy(true); setError('')
    try {
      await flushMeetingEdits(); await saveText()
      const settings = await meetingsApi.settings()
      await meetingsApi.process(ids, settings, stages)
      setSelectedMeetings(new Set()); setMeetingToast(ids.length === 1 ? 'Обработка встречи добавлена в очередь' : `В очередь добавлено встреч: ${ids.length}`)
      window.dispatchEvent(new Event('simple-history-changed'))
      await loadList()
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Не удалось запустить обработку') }
    finally { setProcessBusy(false) }
  }

  function editSegment(item: TimelineItem, value: string) {
    if (!selected || selected.kind !== 'video' || !item.id) return
    const key = `${meetingIdFor(selected)}\u0000${item.id}`
    const draft = segmentDraftsRef.current.get(key) ?? { text: String(item.text ?? ''), dirty: false }
    draft.text = value; draft.dirty = true; segmentDraftsRef.current.set(key, draft)
    setSegmentDrafts((current) => ({ ...current, [key]: value }))
  }

  function editSpeaker(id: string, name: string) {
    if (!selected) return
    const meetingId = meetingIdFor(selected)
    const key = `${meetingId}\u0000${id}`
    const draft = speakerDraftsRef.current.get(key) ?? { meetingId, name, dirty: false }
    draft.name = name; draft.dirty = true; speakerDraftsRef.current.set(key, draft)
    setSpeakerDrafts((current) => ({ ...current, [key]: name }))
  }

  useEffect(() => {
    if (!selected || selected.kind !== 'audio') return
    return registerPendingSave(`archive-transcription:${selected.id}`, saveText)
  }, [selected, saveText])

  async function selectItem(item: ArchiveRow, updateUrl = true) {
    const nextKey = archiveSelectionKey(item.kind, item.id, item.archive_root)
    if (selected && selectedKey !== nextKey) {
      try { await flushPendingSaves() } catch (reason) { setError(reason instanceof Error ? reason.message : 'Не удалось сохранить изменения'); return }
    }
    if (editTimerRef.current !== null) { window.clearTimeout(editTimerRef.current); editTimerRef.current = null }
    if (selected && selectedKey !== nextKey && selected.kind === 'audio') {
      try { await saveText() } catch { setSaveState('failed'); return }
    }
    if (selected && selectedKey !== nextKey && selected.kind === 'video') {
      try { await flushMeetingEdits() } catch (reason) { setError(reason instanceof Error ? reason.message : 'Не удалось сохранить правки встречи'); return }
    }
    selectedRef.current = nextKey
    meetingRevisionRef.current = ''
    setSelected(item); setDetailBusy(true); setTranscriptionAvailable(false); setDetailError(''); setTimeline([]); setMeeting(null); setEvidence(''); setPlayerUrl('')
    if (updateUrl) setParams({ kind: item.kind, id: item.id })
    const itemKey = archiveSelectionKey(item.kind, item.id, item.archive_root)
    try {
      if (item.kind === 'video') {
        const id = item.meeting_id || item.id
        const detail = await meetingsApi.detail(id, undefined, false)
        if (selectedRef.current !== itemKey) return
        setMeeting(detail)
        const [pages, revision] = await Promise.all([readMeetingTimeline(id), meetingsApi.revision(id).catch(() => ({ revision: '' }))])
        if (selectedRef.current !== itemKey) return
        setTimeline(pages)
        meetingRevisionRef.current = revision.revision
        const firstFrame = pages.find((row) => row.kind === 'screen' && row.evidence_id)
        if (firstFrame?.evidence_id) setEvidence(meetingsApi.frameUrl(id, firstFrame.evidence_id))
      } else {
        const detail = await archiveApi.transcription(item.id)
        if (selectedRef.current && selectedRef.current !== archiveSelectionKey(item.kind, item.id, item.archive_root)) return
        const hasText = hasCompletedTranscript(detail.item.status, detail.text)
        setTranscriptionAvailable(hasText)
        const resolvedItem = { ...item, ...detail.item, archive_root: detail.archive_root ?? detail.item.archive_root ?? item.archive_root }
        const resolvedKey = archiveSelectionKey(resolvedItem.kind, resolvedItem.id, resolvedItem.archive_root)
        if (selectedRef.current && selectedRef.current !== resolvedKey) return
        selectedRef.current = resolvedKey
        setSelected(resolvedItem)
        const key = registerArchiveDraft(resolvedItem, detail.text ?? '', detail.revision)
        setText(sharedDraftManager.get(key)?.text ?? detail.text ?? '')
      }
    } catch (reason) { if (selectedRef.current === itemKey) setDetailError(reason instanceof Error ? reason.message : 'Не удалось загрузить запись') }
    finally { if (selectedRef.current === itemKey) setDetailBusy(false) }
  }

  function editText(value: string) {
    if (!selected || selected.kind !== 'audio') return
    sharedDraftManager.edit(transcriptionDraftKey(selected.id, selected.archive_root), value); setText(value); setSaveState('saving')
    if (editTimerRef.current !== null) window.clearTimeout(editTimerRef.current)
    editTimerRef.current = window.setTimeout(() => { void saveText().catch(() => setSaveState('failed')) }, 700)
  }

  async function download(format: 'txt' | 'md') {
    if (!selected || selected.kind !== 'audio') return
    try {
      await saveText()
      const response = await fetch(`${API_BASE}${archiveApi.download(selected.id, format)}`)
      if (!response.ok) throw new Error(`Ошибка загрузки (${response.status})`)
      const url = URL.createObjectURL(await response.blob()); const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${selected.title || 'transcript'}.${format}`; anchor.click(); URL.revokeObjectURL(url)
    } catch (reason) { setDetailError(reason instanceof Error ? reason.message : 'Не удалось скачать файл') }
  }

  async function copyText() {
    if (!selected || selected.kind !== 'audio') return
    try { await navigator.clipboard.writeText(text) } catch (reason) { setDetailError(reason instanceof Error ? reason.message : 'Не удалось скопировать текст') }
  }

  async function openTranscriptionWorkspace() {
    if (!selected || selected.kind !== 'audio') return
    try { await flushPendingSaves(); window.dispatchEvent(new CustomEvent('transcribe-open-task', { detail: { id: selected.id, archiveRoot: selected.archive_root, onUnavailable: () => setDetailError('Эта копия доступна в архиве; исходная задача относится к другой папке.') } })) }
    catch (reason) { setDetailError(reason instanceof Error ? reason.message : 'Не удалось открыть расшифровку') }
  }

  async function removeSelected() {
    if (!selected || selected.kind !== 'audio') return
    try { await saveText(); await archiveApi.remove(selected.id); sharedDraftManager.remove(transcriptionDraftKey(selected.id, selected.archive_root)); setItems((current) => current.filter((row) => row.id !== selected.id || row.kind !== selected.kind)); setSelected(null); selectedRef.current = ''; setParams({}); setConfirmDelete(false) }
    catch (reason) { setDetailError(reason instanceof Error ? reason.message : 'Не удалось удалить запись'); setConfirmDelete(false) }
  }

  async function showMeetingEvidence(row: TimelineItem) {
    if (!selected || selected.kind !== 'video') return
    const id = meetingIdFor(selected), key = archiveSelectionKey(selected.kind, selected.id, selected.archive_root)
    if (row.evidence_id) setEvidence(meetingsApi.frameUrl(id, row.evidence_id))
    if (row.start == null) return
    try { const result = await meetingsApi.evidence(id, row.start); if (selectedRef.current === key) setPlayerUrl(meetingsApi.previewUrl(result.url)) } catch { /* saved frame remains available */ }
  }

  function resize(which: 'list' | 'evidence', event: React.PointerEvent) {
    const startX = event.clientX, initial = which === 'list' ? listWidth : evidenceWidth
    const handle = event.currentTarget as HTMLElement
    handle.setPointerCapture(event.pointerId)
    const move = (pointer: PointerEvent) => {
      const next = Math.max(which === 'list' ? 260 : 260, Math.min(which === 'list' ? 520 : 520, initial + (which === 'list' ? pointer.clientX - startX : startX - pointer.clientX)))
      if (which === 'list') { setListWidth(next); try { localStorage.setItem('video-transcribe-archive-width', String(next)) } catch { /* optional preference */ } }
      else { setEvidenceWidth(next); try { localStorage.setItem('video-transcribe-evidence-width', String(next)) } catch { /* optional preference */ } }
    }
    const done = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', done) }
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', done)
  }

  const activeFrame = useMemo(() => timeline.find((row) => row.kind === 'screen' && row.frame), [timeline])
  const meetingTitle = meeting?.meeting.title || selected?.title || ''
  const speakerIds = [...new Set(timeline.map((row) => row.speaker_cluster).filter((value): value is string => Boolean(value)))].sort((a, b) => Number(a) - Number(b))
  const retryableMeeting = ['failed', 'interrupted', 'partial'].some((status) => String(meeting?.meeting.status ?? meeting?.manifest?.status?.overall ?? '').toLowerCase().includes(status))
  const renderEvidenceMaterials = () => <><div className="archive-evidence-heading">Материалы</div>{evidence ? <img className="archive-frame" src={evidence} alt="Сохранённый кадр встречи" onError={() => setEvidence('')} /> : activeFrame?.evidence_id ? <img className="archive-frame" src={meetingsApi.frameUrl(selected?.meeting_id || selected?.id || '', activeFrame.evidence_id)} alt="Сохранённый кадр встречи" /> : <div className="archive-evidence-empty"><ImageIcon size={20} /><span>Сохранённых кадров для этой встречи пока нет</span></div>}{playerUrl && <details className="archive-player"><summary>Открыть видео или звук</summary><iframe title="Запись встречи" src={playerUrl} /> <a href={playerUrl} target="_blank" rel="noreferrer">Открыть отдельно</a></details>}{meeting?.manifest?.source_availability?.source !== false && selected?.source && <p className="archive-evidence-source-note">Источник: {selected.source}</p>}</>
  return <main className={`archive-page${selected?.kind === 'video' ? ' with-evidence' : ''}`} style={{ '--archive-list-width': `${listWidth}px`, '--archive-evidence-width': `${evidenceWidth}px` } as React.CSSProperties}>
    <section className="archive-list-pane" aria-label="Архив">
      <header className="archive-list-heading"><div className="archive-heading-row"><div><h1>Архив</h1><span>{total}</span></div><button type="button" className="icon-button" aria-label="Импортировать аудио или видео" title="Импортировать аудио или видео" disabled={importBusy} onClick={() => void importMeetings()}><FilePlus2 size={16} /></button></div>
        {visibleMeetingIds.length > 0 && <div className="archive-bulk-actions"><button type="button" className="text-action" onClick={selectAllMeetings}>{visibleMeetingIds.every((id) => selectedMeetings.has(id)) ? 'Снять выбор' : 'Выбрать встречи'}</button>{selectedMeetings.size > 0 && <button type="button" className="primary-button" disabled={processBusy} onClick={() => void processMeetings([...selectedMeetings])}>{processBusy ? <LoaderCircle size={14} className="spin" /> : <Play size={14} />}Обработать · {selectedMeetings.size}</button>}</div>}
        <label className="archive-search"><Search size={16} /><span className="sr-only">Поиск в архиве</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Поиск по названию или источнику" /></label>
        <div className="archive-filters" role="group" aria-label="Тип записи">{(['all', 'audio', 'video'] as const).map((value) => <button key={value} type="button" className={filter === value ? 'selected' : ''} aria-pressed={filter === value} onClick={() => void changeFilter(value)}>{value === 'all' ? 'Все' : kindName(value)}</button>)}</div>
      </header>
      <ToastNotice show={Boolean(meetingToast)} className="archive-inline-status"><span>{meetingToast}</span><button type="button" aria-label="Закрыть сообщение" onClick={() => setMeetingToast('')}>×</button></ToastNotice>
      {error && <div className="archive-error" role="alert"><AlertCircle size={16} />{error}<button type="button" onClick={() => void loadList()}>Повторить</button></div>}
      <div className="archive-list-scroll" aria-live="polite">
        {busy && items.length === 0 ? <div className="archive-empty"><LoaderCircle size={18} className="spin" />Загружаем архив</div> : visibleItems.length === 0 ? <div className="archive-empty"><FileText size={20} /><strong>{query ? 'Ничего не найдено' : 'Записей пока нет'}</strong><span>{query ? 'Попробуйте изменить поиск' : 'Аудио и видео появятся здесь после публикации результата'}</span></div> : visibleItems.map((item) => <article key={archiveSelectionKey(item.kind, item.id, item.archive_root)} className={`archive-row-shell ${selectedKey === archiveSelectionKey(item.kind, item.id, item.archive_root) ? 'selected' : ''}`}>
          {item.kind === 'video' && <input type="checkbox" aria-label={`Выбрать ${item.title || 'встречу'}`} checked={selectedMeetings.has(meetingIdFor(item))} onChange={() => toggleMeeting(meetingIdFor(item))} />}
          <button type="button" className="archive-row" aria-current={selectedKey === archiveSelectionKey(item.kind, item.id, item.archive_root) ? 'page' : undefined} onClick={() => void selectItem(item)}>
            <span className={`archive-kind ${item.kind}`}>{kindName(item.kind)}</span><strong>{item.title || 'Без названия'}</strong><small>{archiveDateLabel(item)}</small><span className={`archive-status ${item.status}`}>{statusName(item.status)}</span>
          </button>
        </article>)}
        {nextOffset !== null && <button className="archive-more" type="button" disabled={busy} onClick={() => void loadList(true, nextOffset)}>Показать ещё</button>}
      </div>
    </section>
    <div className="resize-handle list-resize" role="separator" tabIndex={0} aria-orientation="vertical" aria-label="Изменить ширину списка" aria-valuemin={260} aria-valuemax={520} aria-valuenow={listWidth} onPointerDown={(event) => resize('list', event)} onKeyDown={(event) => { if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); const next = Math.max(260, Math.min(520, listWidth + (event.key === 'ArrowRight' ? 16 : -16))); setListWidth(next); try { localStorage.setItem('video-transcribe-archive-width', String(next)) } catch { /* optional preference */ } } }} />
    <section className="archive-detail-pane" aria-label="Содержимое записи">
      {!selected ? <div className="archive-placeholder"><FileText size={22} /><p>Выберите запись, чтобы прочитать текст и проверить материалы</p></div> : <>
        <header className="archive-detail-heading"><div><span className={`archive-kind ${selected.kind}`}>{kindName(selected.kind)}</span><h2>{selected.kind === 'video' ? meetingTitle : selected.title}</h2><p>{archiveDateLabel(selected)}{selected.source ? ` · ${selected.source}` : ''}</p></div>
          {selected.kind === 'audio' && <div className="archive-actions">{transcriptionAvailable && <><span className={`archive-save-state ${saveState}`}>{saveState === 'saved' ? 'Сохранено' : saveState === 'saving' ? 'Сохраняем…' : 'Ошибка сохранения'}</span><button type="button" className="quiet-button" aria-label="Скопировать текст" title="Скопировать текст" onClick={() => void copyText()}><Clipboard size={15} /></button><button type="button" className="quiet-button" onClick={() => void download('txt')}>TXT</button><button type="button" className="quiet-button" onClick={() => void download('md')}>MD</button></>}<button type="button" className="quiet-button archive-open-task" onClick={() => void openTranscriptionWorkspace()}>{['failed', 'interrupted', 'cancelled'].includes(selected.status.toLowerCase()) ? 'Повторить' : 'Открыть задачу'}</button>{!['queued', 'running', 'cancelling', 'processing'].includes(selected.status.toLowerCase()) && <button type="button" className="icon-button" aria-label="Удалить расшифровку" title="Удалить" onClick={() => setConfirmDelete(true)}><Trash2 size={16} /></button>}</div>}
          {selected.kind === 'video' && retryableMeeting && <button type="button" className="primary-button archive-retry" disabled={processBusy || selectedStages.size === 0} onClick={() => void processMeetings([meetingIdFor(selected)], [...selectedStages])}>{processBusy ? <LoaderCircle size={14} className="spin" /> : <Play size={14} />}Повторить этапы</button>}
          {selected.kind === 'video' && <button type="button" className="quiet-button archive-evidence-toggle" aria-expanded={compactEvidenceOpen} onClick={() => setCompactEvidenceOpen((open) => !open)}><ImageIcon size={15} />Материалы</button>}
        </header>
        {selected.kind === 'video' && retryableMeeting && <div className="archive-retry-stages" aria-label="Этапы для повтора">{meetingStages.map((stage) => <label key={stage}><input type="checkbox" checked={selectedStages.has(stage)} onChange={() => setSelectedStages((current) => { const next = new Set(current); if (next.has(stage)) next.delete(stage); else next.add(stage); return next })} />{meetingStageNames[stage]}</label>)}</div>}
        {detailError && <p className="archive-error" role="alert">{detailError}</p>}
        {detailBusy ? <div className="archive-loading"><LoaderCircle size={18} className="spin" />Загружаем запись</div> : selected.kind === 'audio' ? transcriptionAvailable ? <>{selected.speech_availability === 'no_speech' && !text.trim() && <p className="retention-note">Речь не обнаружена</p>}<textarea className="archive-text-editor" aria-label="Текст расшифровки" value={text} onChange={(event) => editText(event.target.value)} placeholder="Текст расшифровки" /></> : <div className="archive-empty archive-transcription-state"><strong>{selected.speech_availability === 'no_speech' ? 'Речь не обнаружена' : statusName(selected.status)}</strong><span>{selected.speech_availability === 'no_speech' ? 'В импортированном аудио не удалось распознать речь.' : ['queued', 'running', 'cancelling', 'processing'].includes(selected.status.toLowerCase()) ? 'Задача обрабатывается. Текст появится после завершения.' : 'Для этой задачи пока нет готового текста.'}</span></div> : <div className="archive-meeting-content">
          {meeting?.meeting.speechAvailability === 'no_speech' && <p className="retention-note">Речь не обнаружена</p>}
          {meeting?.meeting.speechAvailability === 'unavailable' && <p className="retention-note">Звук недоступен, сохранены кадры</p>}
          {meeting?.manifest?.source_availability?.source === false && <p className="retention-note">Полное видео больше не хранится. Текст и сохранённые кадры доступны.</p>}
          <div className="archive-transcript-scroll">
            {speakerIds.length > 0 && <section className="archive-speakers"><h3>Участники</h3>{speakerIds.map((speakerId) => { const key = `${meetingIdFor(selected)}\u0000${speakerId}`, initial = speakerDrafts[key] ?? timeline.find((row) => row.speaker_cluster === speakerId)?.confirmed_speaker ?? ''; return <label key={speakerId}><span>{initial ? 'Имя участника' : `Говорящий ${speakerId}`}</span><input value={initial} onChange={(event) => editSpeaker(speakerId, event.target.value)} onBlur={() => void flushMeetingEdits().catch((reason) => setDetailError(reason instanceof Error ? reason.message : 'Не удалось сохранить имя участника'))} onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); void flushMeetingEdits().catch(() => undefined) } }} placeholder="Добавить имя" /></label>})}</section>}
            {timeline.map((row, index) => {
              const segmentKey = row.id ? `${meetingIdFor(selected)}\u0000${row.id}` : ''
              const value = segmentKey ? segmentDrafts[segmentKey] ?? String(row.text ?? '') : String(row.text ?? '')
              const speakerKey = `${meetingIdFor(selected)}\u0000${row.speaker_cluster ?? ''}`
              const label = row.kind === 'screen' ? 'Видно на экране' : row.kind === 'orphan_edit' ? 'Исправление без исходного фрагмента' : speakerDrafts[speakerKey] || row.confirmed_speaker || ''
              return <article className={`archive-transcript-line timeline-${row.kind}`} key={`${row.kind}-${row.id ?? row.evidence_id ?? index}`}><button type="button" className="archive-timecode" onClick={() => void showMeetingEvidence(row)} aria-label={row.start == null ? 'Открыть материал без времени' : `Открыть материал в ${new Date((row.start ?? 0) * 1000).toISOString().slice(11, 19)}`}>{row.start == null ? '—:—' : new Date((row.start ?? 0) * 1000).toISOString().slice(11, 19)}</button><div className="archive-line-content">{label && <small>{label}</small>}{row.kind === 'transcript' && row.id ? <textarea aria-label={`Исправить фрагмент ${index + 1}`} value={value} onChange={(event) => editSegment(row, event.target.value)} onBlur={() => void flushMeetingEdits().catch((reason) => setDetailError(reason instanceof Error ? reason.message : 'Не удалось сохранить правку'))} rows={Math.max(2, Math.min(5, Math.ceil(value.length / 100)))} /> : <p>{row.text || 'Текст не распознан'}</p>}{row.original_text && row.original_text !== row.text && <details><summary>Исходный вариант</summary><p>{row.original_text}</p></details>}{row.kind === 'screen' && row.evidence_id && <button type="button" className="text-action" onClick={() => void showMeetingEvidence(row)}>Посмотреть сохранённый кадр</button>}</div></article>
            })}
            {timeline.length === 0 && <div className="archive-evidence-empty">В этой встрече пока нет текста и сохранённых кадров</div>}
          </div>
        </div>}
        {selected.kind === 'video' && <MeasuredDisclosure open={compactEvidenceOpen} className="archive-compact-evidence-wrap"><section className="archive-compact-evidence" aria-label="Видео и кадры встречи">{renderEvidenceMaterials()}</section></MeasuredDisclosure>}
      </>}
    </section>
    {selected?.kind === 'video' && <div className="resize-handle evidence-resize" role="separator" tabIndex={0} aria-orientation="vertical" aria-label="Изменить ширину подтверждения" aria-valuemin={260} aria-valuemax={520} aria-valuenow={evidenceWidth} onPointerDown={(event) => resize('evidence', event)} onKeyDown={(event) => { if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); const next = Math.max(260, Math.min(520, evidenceWidth + (event.key === 'ArrowLeft' ? 16 : -16))); setEvidenceWidth(next); try { localStorage.setItem('video-transcribe-evidence-width', String(next)) } catch { /* optional preference */ } } }} />}
    {selected?.kind === 'video' && <aside className="archive-evidence-pane" aria-label="Видео и кадры">{renderEvidenceMaterials()}</aside>}
    <dialog ref={deleteDialogRef} className="archive-delete-dialog" role="alertdialog" aria-modal="true" aria-labelledby="archive-delete-title" onCancel={(event) => { event.preventDefault(); setConfirmDelete(false) }}><h2 id="archive-delete-title">Удалить расшифровку?</h2><p>Запись и её текст будут удалены из архива.</p><div><button ref={deleteCancelRef} type="button" className="quiet-button" onClick={() => setConfirmDelete(false)}>Отмена</button><button type="button" className="danger-button" onClick={() => void removeSelected()}>Удалить</button></div></dialog>
  </main>
}
