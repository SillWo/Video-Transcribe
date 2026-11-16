import { API_BASE, fetchJson, type FileSelection } from './api.ts'

export type MeetingStatus = 'imported' | 'queued' | 'speech_ready' | 'screen_active' | 'partial' | 'ready' | 'failed' | 'interrupted' | 'processing' | string
export type MeetingJobSummary = { id: string; status: string; stage?: string; error?: string; errorCode?: string; createdAt?: string; updatedAt?: string }

function jobCreatedAt(job: MeetingJobSummary): number {
  if (!job.createdAt) return Number.NEGATIVE_INFINITY
  const timestamp = Date.parse(job.createdAt)
  return Number.isNaN(timestamp) ? Number.NEGATIVE_INFINITY : timestamp
}

export function latestFailedMeetingJob(jobs: MeetingJobSummary[] | undefined): MeetingJobSummary | undefined {
  if (!jobs?.length) return undefined
  const latest = [...jobs].sort((a, b) => {
    const timestampA = jobCreatedAt(a)
    const timestampB = jobCreatedAt(b)
    if (timestampA !== timestampB) return timestampA < timestampB ? 1 : -1
    return b.id.localeCompare(a.id)
  })[0]
  return latest.status.toLowerCase() === 'failed' ? latest : undefined
}
export type MeetingRecord = {
  meeting_id?: string
  id?: string
  title?: string | null
  started_at?: string | null
  created_at?: string | null
  source_filename?: string | null
  source_available?: boolean
  status?: MeetingStatus
  readiness?: Record<string, string>
  availability?: Record<string, boolean>
  jobs?: MeetingJobSummary[]
  [key: string]: unknown
}
export type TimelineItem = {
  id?: string
  evidence_id?: string
  kind: 'transcript' | 'screen' | 'orphan_edit' | string
  start: number | null
  end?: number | null
  text?: string | null
  original_text?: string | null
  frame?: string | null
  speaker_cluster?: string | null
  confirmed_speaker?: string | null
  regions?: unknown[]
  tables?: unknown[]
}
export type MeetingManifest = {
  status?: Record<string, string>
  readiness?: Record<string, string>
  source_availability?: Record<string, boolean>
  limitations?: string[]
  [key: string]: unknown
}
export type MeetingDetail = { meeting: MeetingRecord; timeline?: TimelineItem[]; manifest?: MeetingManifest | null; jobs?: unknown[] }
export type MeetingSnapshot = MeetingDetail & { timeline: TimelineItem[]; revision: string }
export type TimelinePage = { items: TimelineItem[]; total: number; next_offset: number | null; complete: boolean; screen_status?: string }
export type MeetingSettings = {
  archive_root: string
  backup_root?: string | null
  auto_cleanup: boolean
  profile: 'balanced' | 'quality' | 'economy'
  execution_mode: 'auto' | 'manual'
  device: 'cpu' | 'gpu'
  original_retention_days: number
  cpu_threads?: number
  disk?: { free_bytes?: number; total_bytes?: number; used_bytes?: number; [key: string]: unknown }
}
export type GlossaryEntry = { id: string; meeting_id?: string; term: string; definition?: string; aliases?: string[]; provenance?: Record<string, unknown> }
export type ConnectorStatus = {
  registered?: boolean
  protocol_checked?: boolean
  client_verified?: boolean
  evidence_protocol?: { supported?: boolean; registered?: boolean; owned?: boolean; conflict?: boolean }
  [key: string]: unknown
}
export type RecorderSelection = { monitor_id: string | null; browser_id: string | null; microphone_id: string | null }
export type MeetingQueueState = {
  paused: boolean
  pauseReason: { code?: string; message?: string } | null
  cancelling?: boolean
  activeJob: { id: string; status: string; stage?: string } | null
  waitingJobs?: { id: string; status: string; stage?: string }[]
  waitingCount: number
}
export type HardwareStage = {
  available?: boolean
  verified?: boolean
  prepared?: boolean
  backend?: string
  reason?: string | null
  can_prepare?: boolean
  auto_device?: string
  query_reason?: string
  bulk_reason?: string
}
export type HardwarePreparation = {
  id?: string
  job_id?: string
  status: string
  stage?: string
  progress?: { stage?: string; status?: string; detail?: string }[]
  result?: { ok?: boolean; prepared?: boolean; verified?: boolean; error?: string; reason?: string; log_tail?: string }
  estimated_download_bytes?: number
}
export type MeetingHardware = {
  execution_mode: 'auto' | 'manual'
  device: 'cpu' | 'gpu'
  hardware: { cpu?: string; gpu?: { name?: string | null; vendor_id?: number | null; driver?: string | null } | null; directml_registered?: boolean; intel_arc_verified?: boolean }
  stages: { screen?: { cpu?: HardwareStage; gpu?: HardwareStage }; audio?: HardwareStage; diarization?: HardwareStage; index?: HardwareStage }
  gpu_runtime: { python?: string; requirements_bytes?: number; prepared?: boolean; estimated_download_bytes?: number; receipt?: string | null }
  preparation?: HardwarePreparation | null
}
export type MeetingAssetStatus = { available: boolean; version?: string; download_bytes?: number; files?: { name: string; available: boolean; expected_bytes: number; downloaded_bytes: number }[]; path?: string | null }
export type AssetPreparation = {
  id?: string | null
  job_id?: string | null
  status: string
  stage?: string
  progress?: { file?: string; stage?: string; completed_bytes?: number; total_bytes?: number; percent?: number; ready?: boolean }[]
  error?: string
  result?: { ok?: boolean; status?: string; error?: string; reason?: string }
  estimated_download_bytes?: number
}
export type BackupSnapshot = {
  snapshot_id: string
  created_at?: string
  file_count: number
  total_bytes: number
  verified_receipt: boolean
  restore_test_passed: boolean
}
export type CleanupPreview = {
  retention_days: number
  cutoff: string
  auto_cleanup_enabled: boolean
  items: { meeting_id: string; filename: string; modified_at?: string; age_days?: number; eligible: boolean; snapshot_id?: string | null; blocked_reasons: string[] }[]
  eligible_count: number
  blocked_count: number
}
export type BackupStorage = {
  configured: boolean
  archive_root?: string
  backup_root?: string | null
  auto_cleanup_enabled?: boolean
  retention_days?: number
  snapshots: BackupSnapshot[]
  error?: { code?: string; message?: string }
}
export type BackupJob = {
  job_id: string
  operation: string
  status: string
  stage?: string
  progress?: { stage?: string; completed?: number; total?: number; path?: string }[]
  result?: Record<string, unknown> | CleanupPreview | null
  error?: { code?: string; message?: string } | null
}
export type BackupStatus = { storage: BackupStorage; job: BackupJob | null; jobs: BackupJob[] }
export type RecorderStatus = {
  available: boolean
  connected: boolean
  state: 'idle' | 'recording' | 'unavailable' | string
  duration_seconds: number
  monitors: { id: string; name: string }[]
  browsers: { id: string; name: string }[]
  microphones: { id: string; name: string }[]
  selection: RecorderSelection
  preview_url: string | null
  levels: { browser: number | null; microphone: number | null; stale: boolean }
  error: string | null
  title?: string | null
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  return fetchJson<T>(path, {
    method: 'POST',
    ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  })
}

async function put<T>(path: string, body: unknown): Promise<T> {
  return fetchJson<T>(path, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
}

const idPath = (meetingId: string) => `/api/meetings/${encodeURIComponent(meetingId)}`

export const meetingsApi = {
  list: () => fetchJson<{ items: MeetingRecord[] }>('/api/meetings'),
  revision: (meetingId: string, signal?: AbortSignal) => fetchJson<{ revision: string }>(`${idPath(meetingId)}/revision`, { signal }),
  detail: (meetingId: string, signal?: AbortSignal, includeTimeline = true) => fetchJson<MeetingDetail>(
    `${idPath(meetingId)}${includeTimeline ? '' : '?include_timeline=false'}`, { signal }),
  timeline: (meetingId: string, start = 0, end?: number, limit = 200, offset = 0, signal?: AbortSignal, includeGeometry = true) => {
    const params = new URLSearchParams({ start: String(start), limit: String(limit), offset: String(offset) })
    if (end !== undefined) params.set('end', String(end))
    if (!includeGeometry) params.set('include_geometry', 'false')
    return fetchJson<TimelinePage>(`${idPath(meetingId)}/timeline?${params}`, { signal })
  },
  pickFiles: () => post<FileSelection>('/api/simple/pick-files'),
  importFiles: (paths: string[]) => post<{ items: MeetingRecord[]; rejected: { path: string; reason: string }[] }>('/api/meetings/import', { paths }),
  process: (meetingIds: string[], settings: Pick<MeetingSettings, 'execution_mode' | 'device' | 'cpu_threads' | 'profile'>, stages: ('speech' | 'screen' | 'audio' | 'index')[] = ['speech', 'screen', 'audio', 'index']) => post<{ jobs: unknown[] }>('/api/meetings/process', {
    meeting_ids: meetingIds,
    profile: settings.profile,
    device: settings.device,
    cpu_threads: settings.cpu_threads ?? 8,
    diarization: true,
    stages,
    execution_mode: settings.execution_mode,
  }),
  saveSegment: (meetingId: string, segmentId: string, text: string) => put<{ segment: TimelineItem; remembered_term_undo_id?: string | null }>(`${idPath(meetingId)}/segments/${encodeURIComponent(segmentId)}`, { text, remember: true }),
  saveSpeaker: (meetingId: string, speakerId: string, name: string) => put<{ speaker_id?: string; name: string }>(`${idPath(meetingId)}/speakers/${encodeURIComponent(speakerId)}`, { name }),
  evidence: (meetingId: string, time: number) => fetchJson<{ url: string }>(`${idPath(meetingId)}/evidence?t=${encodeURIComponent(String(time))}`),
  frameUrl: (meetingId: string, evidenceId: string) => `${API_BASE}${idPath(meetingId)}/frames/${encodeURIComponent(evidenceId)}`,
  settings: async () => (await fetchJson<{ settings: MeetingSettings }>('/api/meetings/settings')).settings,
  saveSettings: async (settings: MeetingSettings) => (await put<{ settings: MeetingSettings }>('/api/meetings/settings', settings)).settings,
  pickDirectory: () => post<{ path: string }>('/api/meetings/pick-directory'),
  glossary: () => fetchJson<{ items: GlossaryEntry[] }>('/api/meetings/glossary'),
  addGlossaryEntry: (meetingId: string, term: string, definition: string) => post<{ item?: GlossaryEntry; id?: string }>('/api/meetings/glossary', { meeting_id: meetingId, term, definition }),
  deleteGlossaryEntry: (meetingId: string, entryId: string) => fetchJson<{ undo_id?: string }>(`/api/meetings/glossary/${encodeURIComponent(meetingId)}/${encodeURIComponent(entryId)}`, { method: 'DELETE' }),
  undoGlossaryEntry: (meetingId: string, undoId: string) => post<{ restored?: boolean }>(`/api/meetings/glossary/${encodeURIComponent(meetingId)}/${encodeURIComponent(undoId)}/undo`),
  connectorStatus: () => fetchJson<ConnectorStatus>('/api/meetings/connector/status'),
  connectCodex: () => post<ConnectorStatus>('/api/meetings/connector/connect'),
  checkCodex: () => post<ConnectorStatus>('/api/meetings/connector/check'),
  disconnectCodex: () => post<ConnectorStatus>('/api/meetings/connector/disconnect'),
  recorder: (signal?: AbortSignal) => fetchJson<RecorderStatus>('/api/meetings/recorder', { signal }),
  queue: () => fetchJson<MeetingQueueState>('/api/simple/queue'),
  resumeQueue: () => post<MeetingQueueState>('/api/simple/queue/resume'),
  hardware: () => fetchJson<MeetingHardware>('/api/meetings/hardware'),
  prepareHardware: () => post<{ job_id: string; status: string; estimated_download_bytes: number }>('/api/meetings/hardware/prepare'),
  hardwarePreparation: () => fetchJson<{ preparation: HardwarePreparation | null }>('/api/meetings/hardware/prepare'),
  assets: () => fetchJson<{ semantic_search: MeetingAssetStatus }>('/api/meetings/assets'),
  prepareAssets: () => post<{ job_id: string | null; status: string; estimated_download_bytes: number; semantic_search: MeetingAssetStatus }>('/api/meetings/assets/prepare', { component: 'semantic_search' }),
  assetPreparation: () => fetchJson<{ preparation: AssetPreparation | null; semantic_search: MeetingAssetStatus }>('/api/meetings/assets/prepare'),
  backupStatus: (jobId?: string) => fetchJson<BackupStatus>(`/api/meetings/backup/status${jobId ? `?job_id=${encodeURIComponent(jobId)}` : ''}`),
  createBackup: () => post<{ job_id?: string; operation?: string; status: string; error?: { code?: string; message?: string } }>('/api/meetings/backup/backup'),
  verifyBackup: (snapshotId: string) => post<{ job_id?: string; operation?: string; status: string; error?: { code?: string; message?: string } }>('/api/meetings/backup/verify', { snapshot_id: snapshotId }),
  restoreTest: (snapshotId: string) => post<{ job_id?: string; operation?: string; status: string; error?: { code?: string; message?: string } }>('/api/meetings/backup/restore-test', { snapshot_id: snapshotId }),
  restoreBackup: (snapshotId: string, target: string) => post<{ job_id?: string; operation?: string; status: string; error?: { code?: string; message?: string } }>('/api/meetings/backup/restore', { snapshot_id: snapshotId, target }),
  cleanupPreview: () => post<{ job_id?: string; operation?: string; status: string; error?: { code?: string; message?: string } }>('/api/meetings/backup/cleanup/preview'),
  runCleanup: () => post<{ job_id?: string; operation?: string; status: string; error?: { code?: string; message?: string } }>('/api/meetings/backup/cleanup/run'),
  prepareRecorder: () => post<RecorderStatus>('/api/meetings/recorder/prepare'),
  configureRecorder: (selection: RecorderSelection) => post<RecorderStatus>('/api/meetings/recorder/configure', selection),
  startRecording: (title: string) => post<RecorderStatus>('/api/meetings/recorder/start', { title }),
  stopRecording: () => post<{ meeting: MeetingRecord; recorder: RecorderStatus }>('/api/meetings/recorder/stop'),
  previewUrl: (url: string) => `${API_BASE}${url}`,
}

export async function readMeetingTimeline(meetingIdValue: string, signal?: AbortSignal, pageSize = 200,
                                          includeGeometry = true): Promise<TimelineItem[]> {
  const items: TimelineItem[] = []
  let offset = 0
  for (let pageCount = 0; pageCount < 10000; pageCount += 1) {
    const page = await meetingsApi.timeline(meetingIdValue, 0, undefined, pageSize, offset, signal, includeGeometry)
    items.push(...page.items)
    if (page.complete) return items
    if (page.next_offset === null || page.next_offset <= offset) {
      throw new Error('Архив вернул неполную временную шкалу без следующей страницы. Данные не считаются загруженными.')
    }
    offset = page.next_offset
  }
  throw new Error('Встреча слишком большая для загрузки одной страницей. Попробуйте открыть её позже.')
}

export async function readMeetingSnapshot(meetingIdValue: string, knownRevision?: string,
                                          signal?: AbortSignal, pageSize = 200): Promise<MeetingSnapshot | null> {
  const { revision } = await meetingsApi.revision(meetingIdValue, signal)
  if (signal?.aborted) throw new DOMException('The operation was aborted.', 'AbortError')
  if (knownRevision !== undefined && revision === knownRevision) return null
  const [detail, timeline] = await Promise.all([
    meetingsApi.detail(meetingIdValue, signal, false),
    readMeetingTimeline(meetingIdValue, signal, pageSize, false),
  ])
  return { ...detail, timeline, revision }
}

export function meetingId(meeting: MeetingRecord): string {
  return String(meeting.meeting_id ?? meeting.id ?? '')
}

export function parseMeetingCompositeKey(key: string): [meetingId: string, entityId: string] | null {
  const separator = key.indexOf(':')
  if (separator <= 0 || separator === key.length - 1) return null
  return [key.slice(0, separator), key.slice(separator + 1)]
}

export function canApplyMeetingEditResult(requestMeetingId: string, activeMeetingId: string): boolean {
  return Boolean(requestMeetingId) && requestMeetingId === activeMeetingId
}

export function isPendingMeetingValueCurrent(currentValue: string | undefined, savedValue: string): boolean {
  return currentValue === savedValue
}

export function meetingLabel(meeting: MeetingRecord): string {
  const label = meeting.title ?? meeting.source_filename ?? meetingId(meeting)
  return label ? String(label) : 'Встреча'
}

export function confirmedMeetingDate(meeting: MeetingRecord): { value: string; source: string; hasTime: boolean } | null {
  const startedAt = meeting.started_at
  if (typeof startedAt === 'string' && startedAt.trim() && !Number.isNaN(new Date(startedAt).getTime())) {
    return { value: startedAt, source: 'Начало встречи', hasTime: true }
  }
  const dateOnly = meeting.meeting_date
  const source = meeting.meeting_date_source
  if (typeof dateOnly === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(dateOnly)
    && meeting.meeting_date_confirmed === true && typeof source === 'string' && source.trim()) {
    const parsed = new Date(`${dateOnly}T00:00:00`)
    if (!Number.isNaN(parsed.getTime())) return { value: dateOnly, source: source.trim(), hasTime: false }
  }
  return null
}

