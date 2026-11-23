import assert from 'node:assert/strict'
import test from 'node:test'
import { canApplyMeetingEditResult, confirmedMeetingDate, isPendingMeetingValueCurrent, latestFailedMeetingJob, meetingsApi, parseMeetingCompositeKey, preserveMeetingSettingsDraft, readMeetingSnapshot, readMeetingTimeline } from '../src/lib/meetingsApi.ts'

function withFetch(handler: (url: string, init?: RequestInit) => Response | Promise<Response>) {
  const original = globalThis.fetch
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => handler(String(input), init)) as typeof fetch
  return () => { globalThis.fetch = original }
}

test('settings refresh keeps unsaved profile, device, folder, and retention drafts', () => {
  const draft = { archive_root: 'D:/local-draft', backup_root: 'D:/backup-draft', auto_cleanup: false, profile: 'quality' as const, execution_mode: 'manual' as const, device: 'gpu' as const, original_retention_days: 91 }
  const server = { ...draft, archive_root: 'D:/server', profile: 'balanced' as const, execution_mode: 'auto' as const, device: 'cpu' as const, original_retention_days: 30 }
  assert.equal(preserveMeetingSettingsDraft(draft, server), draft)
  assert.equal(preserveMeetingSettingsDraft(null, server), server)
})

test('timeline reader follows every offset page and keeps the complete tail', async () => {
  const requests: string[] = []
  const restore = withFetch((url) => {
    requests.push(url)
    const offset = Number(new URL(url, 'http://localhost').searchParams.get('offset'))
    const complete = offset === 2
    return Response.json({
      items: [{ kind: 'transcript', id: `segment-${offset}`, start: offset, text: 'строка' }],
      total: 3,
      next_offset: complete ? null : offset + 1,
      complete,
    })
  })
  try {
    const items = await readMeetingTimeline('meeting/1', undefined, 1)
    assert.deepEqual(items.map((item) => item.id), ['segment-0', 'segment-1', 'segment-2'])
    assert.equal(requests.length, 3)
    assert.ok(requests.every((url) => url.includes('meeting%2F1')))
  } finally { restore() }
})

test('timeline reader rejects an incomplete page without forward progress', async () => {
  const restore = withFetch(() => Response.json({ items: [], total: 2, next_offset: 0, complete: false }))
  try {
    await assert.rejects(readMeetingTimeline('meeting-1'), /без следующей страницы/)
  } finally { restore() }
})

test('unchanged meeting snapshot polls only its lightweight revision', async () => {
  const requests: string[] = []
  const restore = withFetch((url) => {
    requests.push(url)
    return Response.json({ revision: 'same-revision' })
  })
  try {
    const snapshot = await readMeetingSnapshot('meeting-1', 'same-revision')
    assert.equal(snapshot, null)
    assert.deepEqual(requests, ['/api/meetings/meeting-1/revision'])
  } finally { restore() }
})

test('changed snapshot fetches detail and all geometry-free timeline pages', async () => {
  const requests: string[] = []
  const restore = withFetch((url) => {
    requests.push(url)
    if (url.endsWith('/revision')) return Response.json({ revision: 'revision-2' })
    if (url.includes('?include_timeline=false')) return Response.json({ meeting: { meeting_id: 'meeting-2' }, manifest: null, jobs: [] })
    const parsed = new URL(url, 'http://localhost')
    const offset = Number(parsed.searchParams.get('offset'))
    return Response.json({
      items: [{ kind: 'screen', id: `screen-${offset}`, start: offset, text: 'slide', width: 10 }],
      total: 2, next_offset: offset === 0 ? 1 : null, complete: offset === 1,
    })
  })
  try {
    const snapshot = await readMeetingSnapshot('meeting-2', 'revision-1', undefined, 1)
    assert.equal(snapshot?.revision, 'revision-2')
    assert.deepEqual(snapshot?.timeline.map((item) => item.id), ['screen-0', 'screen-1'])
    assert.equal(requests.length, 4)
    assert.ok(requests[1].includes('include_timeline=false'))
    assert.ok(requests.slice(2).every((url) => url.includes('include_geometry=false')))
    assert.ok(requests.every((url) => url.includes('meeting-2')))
  } finally { restore() }
})

test('aborted snapshot does not continue into timeline fetches', async () => {
  const controller = new AbortController()
  const requests: string[] = []
  const restore = withFetch((url, init) => {
    requests.push(url)
    assert.equal(init?.signal, controller.signal)
    if (url.endsWith('/revision')) {
      controller.abort()
      return Response.json({ revision: 'revision-2' })
    }
    throw new DOMException('Aborted', 'AbortError')
  })
  try {
    await assert.rejects(readMeetingSnapshot('meeting-abort', undefined, controller.signal))
    assert.deepEqual(requests, ['/api/meetings/meeting-abort/revision'])
  } finally { restore() }
})

test('composite draft keys preserve complete colon, slash and unicode segment IDs', () => {
  const meetingA = '12345678-1234-1234-1234-123456789abc'
  const meetingB = 'abcdefab-cdef-abcd-efab-cdefabcdefab'
  const segmentId = 'speech:5955a32f8bb19d868a6f3880:chunk/会议/δ'
  const first = parseMeetingCompositeKey(`${meetingA}:${segmentId}`)
  const second = parseMeetingCompositeKey(`${meetingB}:${segmentId}`)
  assert.deepEqual(first, [meetingA, segmentId])
  assert.deepEqual(second, [meetingB, segmentId])
  assert.notEqual(`${meetingA}:${segmentId}`, `${meetingB}:${segmentId}`)
  assert.equal(parseMeetingCompositeKey('missing-separator'), null)
})

test('saving parsed composite key URL-encodes the complete segment identifier', async () => {
  const meetingId = '12345678-1234-1234-1234-123456789abc'
  const segmentId = 'speech:5955a32f8bb19d868a6f3880:part/会议/δ'
  const requests: { url: string; body: unknown }[] = []
  const restore = withFetch((url, init) => {
    requests.push({ url, body: JSON.parse(String(init?.body)) })
    return Response.json({ segment: { id: segmentId, kind: 'transcript', start: 1, text: 'corrected' }, remembered_term_undo_id: 'undo-1' })
  })
  try {
    const parsed = parseMeetingCompositeKey(`${meetingId}:${segmentId}`)
    assert.ok(parsed)
    await meetingsApi.saveSegment(parsed[0], parsed[1], 'corrected')
    assert.equal(requests[0].url, `/api/meetings/${meetingId}/segments/${encodeURIComponent(segmentId)}`)
    assert.deepEqual(requests[0].body, { text: 'corrected', remember: true })
  } finally { restore() }
})

test('late save response for meeting A cannot update meeting B with a matching segment ID', async () => {
  const segmentId = 'speech:5955a32f8bb19d868a6f3880'
  const requests: string[] = []
  let resolveResponse!: (response: Response) => void
  const restore = withFetch((url) => {
    requests.push(url)
    return new Promise<Response>((resolve) => { resolveResponse = resolve })
  })
  try {
    const pending = meetingsApi.saveSegment('meeting-A', segmentId, 'saved for A')
    const activeMeeting = 'meeting-B'
    resolveResponse(Response.json({ segment: { id: segmentId, kind: 'transcript', start: 0, text: 'saved for A' } }))
    const result = await pending
    const appliedToCurrentTimeline = canApplyMeetingEditResult('meeting-A', activeMeeting)
    assert.equal(result.segment.text, 'saved for A')
    assert.equal(appliedToCurrentTimeline, false)
    assert.notEqual(`${activeMeeting}:${segmentId}`, `meeting-A:${segmentId}`)
    assert.equal(requests[0], `/api/meetings/meeting-A/segments/${encodeURIComponent(segmentId)}`)
  } finally { restore() }
})

test('late speaker save only clears the name that was actually sent', () => {
  const savedName = 'Alicia'
  const newlyTypedName = 'Alice'
  assert.equal(isPendingMeetingValueCurrent(savedName, savedName), true)
  assert.equal(isPendingMeetingValueCurrent(newlyTypedName, savedName), false)
})

test('processing request carries explicit stages and manual hardware selection', async () => {
  let body: Record<string, unknown> | null = null
  const restore = withFetch((_url, init) => {
    body = JSON.parse(String(init?.body)) as Record<string, unknown>
    return Response.json({ jobs: [] })
  })
  try {
    await meetingsApi.process(['meeting-1'], { execution_mode: 'manual', device: 'gpu', cpu_threads: 6, profile: 'quality' }, ['speech', 'screen'])
    assert.deepEqual(body, {
      meeting_ids: ['meeting-1'], profile: 'quality', device: 'gpu', cpu_threads: 6,
      diarization: true, stages: ['speech', 'screen'], execution_mode: 'manual',
    })
  } finally { restore() }
})

test('settings, import rejection and recorder start match backend response envelopes', async () => {
  const settings = {
    archive_root: 'C:/meetings', backup_root: null, auto_cleanup: false,
    original_retention_days: 30, execution_mode: 'auto', device: 'cpu', cpu_threads: 8, profile: 'balanced',
  } as const
  const recorder = {
    available: true, connected: true, state: 'recording', duration_seconds: 12,
    monitors: [], browsers: [], microphones: [],
    selection: { monitor_id: null, browser_id: null, microphone_id: null },
    preview_url: '/api/meetings/recorder/preview',
    levels: { browser: null, microphone: null, stale: true }, error: null,
  } as const
  const calls: { url: string; init?: RequestInit }[] = []
  const restore = withFetch((url, init) => {
    calls.push({ url, init })
    if (url.endsWith('/settings')) return Response.json({ settings })
    if (url.endsWith('/import')) return Response.json({ jobs: [{ id: 'queued-import', status: 'queued' }], rejected: [{ path: 'bad.bin', reason: 'unsupported_format' }] })
    if (url.endsWith('/recorder/start')) return Response.json(recorder)
    throw new Error(`Unexpected API request: ${url}`)
  })
  try {
    assert.deepEqual(await meetingsApi.settings(), settings)
    assert.deepEqual(await meetingsApi.saveSettings(settings), settings)
    const imported = await meetingsApi.importFiles(['bad.bin'])
    assert.deepEqual(imported.jobs, [{ id: 'queued-import', status: 'queued' }])
    assert.deepEqual(imported.rejected, [{ path: 'bad.bin', reason: 'unsupported_format' }])
    assert.deepEqual(await meetingsApi.startRecording('Planning call'), recorder)
    assert.equal(calls.length, 4)
    assert.deepEqual(JSON.parse(String(calls[1].init?.body)), settings)
    assert.deepEqual(JSON.parse(String(calls[3].init?.body)), { title: 'Planning call' })
  } finally { restore() }
})

test('shared queue GET and resume preserve the backend pause state', async () => {
  const state = { paused: true, pauseReason: { code: 'manual', message: 'Ожидает подтверждения' }, cancelling: false, activeJob: null, waitingJobs: [], waitingCount: 0 }
  const calls: { url: string; method: string }[] = []
  const restore = withFetch((url, init) => {
    calls.push({ url, method: String(init?.method ?? 'GET') })
    return Response.json(state)
  })
  try {
    assert.deepEqual(await meetingsApi.queue(), state)
    assert.deepEqual(await meetingsApi.resumeQueue(), state)
    assert.deepEqual(calls, [
      { url: '/api/simple/queue', method: 'GET' },
      { url: '/api/simple/queue/resume', method: 'POST' },
    ])
  } finally { restore() }
})

test('meeting date groups never fall back to imported or created timestamps', () => {
  assert.equal(confirmedMeetingDate({ meeting_id: 'one', created_at: '2026-11-14T14:00:00Z' }), null)
  assert.deepEqual(confirmedMeetingDate({ meeting_id: 'two', started_at: '2026-11-10T10:30:00Z', created_at: '2026-11-14T14:00:00Z' }), {
    value: '2026-11-10T10:30:00Z', source: 'Начало встречи', hasTime: true,
  })
  assert.equal(confirmedMeetingDate({ meeting_id: 'three', meeting_date: '2026-11-11' }), null)
  assert.deepEqual(confirmedMeetingDate({ meeting_id: 'four', meeting_date: '2026-11-11', meeting_date_confirmed: true, meeting_date_source: 'user-confirmed' }), {
    value: '2026-11-11', source: 'user-confirmed', hasTime: false,
  })
})

test('latest queued or completed job suppresses an older failure', () => {
  const oldFailure = { id: 'old-failure', status: 'failed', createdAt: '2026-11-14T09:00:00Z', error: 'старый сбой' }
  assert.equal(latestFailedMeetingJob([oldFailure, { id: 'retry', status: 'queued', createdAt: '2026-11-15T09:00:00Z' }]), undefined)
  assert.equal(latestFailedMeetingJob([oldFailure, { id: 'retry', status: 'completed', createdAt: '2026-11-15T09:00:00Z' }]), undefined)
})

test('jobs with missing or invalid dates sort older than jobs with valid dates', () => {
  const validJob = { id: 'aa-valid', status: 'completed', createdAt: '2026-11-15T09:00:00Z' }
  const olderJobs = [
    { id: 'zz-invalid', status: 'failed', createdAt: 'bad' },
    { id: 'zz-missing', status: 'failed' },
  ]
  for (const oldJob of olderJobs) {
    assert.equal(latestFailedMeetingJob([oldJob, validJob]), undefined)
    assert.equal(latestFailedMeetingJob([validJob, oldJob]), undefined)
  }
})

test('shuffled history selects only its latest failed job and preserves its reason', () => {
  const jobs = [
    { id: 'failed-new', status: 'failed', stage: 'screen', error: 'Короткая причина', errorCode: 'screen_failed', createdAt: '2026-11-15T10:00:00Z' },
    { id: 'older', status: 'completed', createdAt: '2026-11-14T10:00:00Z' },
    { id: 'failed-old', status: 'failed', createdAt: '2026-11-13T10:00:00Z', error: 'Старая причина' },
  ]
  assert.equal(latestFailedMeetingJob(jobs), jobs[0])
  assert.equal(latestFailedMeetingJob([...jobs].reverse()), jobs[0])
  assert.equal(jobs[0].error, 'Короткая причина')
  assert.equal(jobs[0].errorCode, 'screen_failed')
})

test('latest failure ties use a deterministic job id and a new failure has a new key', () => {
  const tie = [
    { id: 'job-a', status: 'failed', createdAt: '2026-11-15T10:00:00Z' },
    { id: 'job-z', status: 'failed', createdAt: '2026-11-15T10:00:00Z' },
  ]
  assert.equal(latestFailedMeetingJob(tie)?.id, 'job-z')
  const reported = new Set<string>()
  const firstId = latestFailedMeetingJob(tie)?.id
  assert.equal(reported.has(firstId ?? ''), false)
  if (firstId) reported.add(firstId)
  assert.equal(reported.has(latestFailedMeetingJob(tie)?.id ?? ''), true)
  const nextFailure = [...tie, { id: 'job-next', status: 'failed', createdAt: '2026-11-15T11:00:00Z', errorCode: 'new_failure' }]
  assert.equal(latestFailedMeetingJob(nextFailure)?.id, 'job-next')
  assert.equal(reported.has(latestFailedMeetingJob(nextFailure)?.id ?? ''), false)
})

test('hardware status, explicit preparation and progress use separate API routes', async () => {
  const hardware = {
    execution_mode: 'auto', device: 'cpu', hardware: { cpu: 'CPU', gpu: { name: 'GPU X' }, directml_registered: true, intel_arc_verified: false },
    stages: { screen: { cpu: { available: true, verified: true }, gpu: { available: false, prepared: false, verified: false, can_prepare: true, reason: 'unprepared' } }, audio: { auto_device: 'cpu', reason: 'CPU path' }, index: { auto_device: 'cpu', query_reason: 'fast' } },
    gpu_runtime: { python: 'runtime/python.exe', requirements_bytes: 12, estimated_download_bytes: 1024, prepared: false, receipt: null }, preparation: null,
  }
  const calls: { url: string; method: string }[] = []
  const restore = withFetch((url, init) => {
    calls.push({ url, method: String(init?.method ?? 'GET') })
    if (url === '/api/meetings/hardware') return Response.json(hardware)
    if (url === '/api/meetings/hardware/prepare' && init?.method === 'POST') return Response.json({ job_id: 'job-1', status: 'running', estimated_download_bytes: 1024 })
    if (url === '/api/meetings/hardware/prepare') return Response.json({ preparation: { id: 'job-1', status: 'running', stage: 'packages', progress: [{ stage: 'packages', status: 'installing', detail: 'DirectML' }] } })
    throw new Error(`Unexpected API request: ${url}`)
  })
  try {
    assert.deepEqual(await meetingsApi.hardware(), hardware)
    assert.deepEqual(await meetingsApi.prepareHardware(), { job_id: 'job-1', status: 'running', estimated_download_bytes: 1024 })
    assert.deepEqual(await meetingsApi.hardwarePreparation(), { preparation: { id: 'job-1', status: 'running', stage: 'packages', progress: [{ stage: 'packages', status: 'installing', detail: 'DirectML' }] } })
    assert.deepEqual(calls, [
      { url: '/api/meetings/hardware', method: 'GET' },
      { url: '/api/meetings/hardware/prepare', method: 'POST' },
      { url: '/api/meetings/hardware/prepare', method: 'GET' },
    ])
  } finally { restore() }
})

test('optional semantic assets are read lazily and prepared only by explicit request', async () => {
  const status = { available: false, version: 'revision-1', download_bytes: 487351893, files: [{ name: 'model.onnx', available: false, expected_bytes: 470268510, downloaded_bytes: 0 }], path: 'runtime/models/onnx' }
  const progress = { preparation: { id: 'asset-1', status: 'running', stage: 'model.onnx', progress: [{ file: 'model.onnx', completed_bytes: 1024, total_bytes: 487351893, percent: 1 }] }, semantic_search: status }
  const calls: { url: string; method: string; body?: string }[] = []
  const restore = withFetch((url, init) => {
    calls.push({ url, method: String(init?.method ?? 'GET'), ...(init?.body ? { body: String(init.body) } : {}) })
    if (url === '/api/meetings/assets') return Response.json({ semantic_search: status })
    if (url === '/api/meetings/assets/prepare' && init?.method === 'POST') return Response.json({ job_id: 'asset-1', status: 'running', estimated_download_bytes: status.download_bytes, semantic_search: status })
    if (url === '/api/meetings/assets/prepare') return Response.json(progress)
    throw new Error(`Unexpected API request: ${url}`)
  })
  try {
    assert.deepEqual(await meetingsApi.assets(), { semantic_search: status })
    assert.deepEqual(await meetingsApi.prepareAssets(), { job_id: 'asset-1', status: 'running', estimated_download_bytes: 487351893, semantic_search: status })
    assert.deepEqual(await meetingsApi.assetPreparation(), progress)
    assert.deepEqual(calls, [
      { url: '/api/meetings/assets', method: 'GET' },
      { url: '/api/meetings/assets/prepare', method: 'POST', body: '{"component":"semantic_search"}' },
      { url: '/api/meetings/assets/prepare', method: 'GET' },
    ])
  } finally { restore() }
})

test('backup actions send the exact snapshot, target and preview request bodies', async () => {
  const storage = { configured: true, backup_root: 'C:/backups', auto_cleanup_enabled: true, retention_days: 30, snapshots: [] }
  const status = { storage, job: null, jobs: [] }
  const calls: { url: string; method: string; body?: string }[] = []
  const restore = withFetch((url, init) => {
    calls.push({ url, method: String(init?.method ?? 'GET'), ...(init?.body ? { body: String(init.body) } : {}) })
    if (url.startsWith('/api/meetings/backup/status')) return Response.json(status)
    return Response.json({ job_id: 'backup-job', operation: 'running', status: 'running' })
  })
  try {
    assert.deepEqual(await meetingsApi.backupStatus(), status)
    await meetingsApi.createBackup()
    await meetingsApi.verifyBackup('snapshot-1')
    await meetingsApi.restoreTest('snapshot-1')
    await meetingsApi.restoreBackup('snapshot-1', 'C:/restore')
    await meetingsApi.cleanupPreview()
    await meetingsApi.runCleanup()
    await meetingsApi.backupStatus('job / 1')
    assert.deepEqual(calls, [
      { url: '/api/meetings/backup/status', method: 'GET' },
      { url: '/api/meetings/backup/backup', method: 'POST' },
      { url: '/api/meetings/backup/verify', method: 'POST', body: '{"snapshot_id":"snapshot-1"}' },
      { url: '/api/meetings/backup/restore-test', method: 'POST', body: '{"snapshot_id":"snapshot-1"}' },
      { url: '/api/meetings/backup/restore', method: 'POST', body: '{"snapshot_id":"snapshot-1","target":"C:/restore"}' },
      { url: '/api/meetings/backup/cleanup/preview', method: 'POST' },
      { url: '/api/meetings/backup/cleanup/run', method: 'POST' },
      { url: '/api/meetings/backup/status?job_id=job%20%2F%201', method: 'GET' },
    ])
  } finally { restore() }
})
