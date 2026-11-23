import assert from 'node:assert/strict'
import test from 'node:test'
import { archiveApi, archiveSelectionKey, filterArchiveRows, hasCompletedTranscript, type ArchiveRow } from '../src/lib/archiveApi.ts'
import { dictionaryApi, upsertDictionaryEntry } from '../src/lib/dictionaryApi.ts'
import { notifyLearnedTerms } from '../src/lib/learnedTermsNotice.ts'

const rows: ArchiveRow[] = [
  { id: 'same', kind: 'video', title: 'Планирование', created_at: '2026-01-01', status: 'ready', source: 'Zoom' },
  { id: 'same', kind: 'audio', title: 'Заметки по проекту', created_at: '2026-01-02', status: 'completed', source: 'local.wav' },
]

test('archive selection separates kinds that share an id and filters by kind/title/source', () => {
  assert.notEqual(archiveSelectionKey('video', 'same'), archiveSelectionKey('audio', 'same'))
  assert.equal(archiveSelectionKey('audio', 'same', 'D:\\Archive\\A'), archiveSelectionKey('audio', 'same', 'd:/archive/a/'))
  assert.notEqual(archiveSelectionKey('audio', 'same', 'D:\\Archive\\A'), archiveSelectionKey('audio', 'same', 'E:\\Archive\\B'))
  assert.deepEqual(filterArchiveRows(rows, 'video', 'zoom').map((row) => row.title), ['Планирование'])
  assert.deepEqual(filterArchiveRows(rows, 'all', 'проект').map((row) => row.kind), ['audio'])
  assert.deepEqual(filterArchiveRows(rows, 'audio', '').map((row) => row.id), ['same'])
})

test('archive audio detail and edits use canonical API paths', async () => {
  const priorFetch = globalThis.fetch
  const paths: string[] = []
  globalThis.fetch = async (input) => {
    paths.push(new URL(String(input), 'http://local').pathname)
    return new Response(JSON.stringify({ item: {}, record: {}, text: '', revision: 1 }), { status: 200, headers: { 'Content-Type': 'application/json' } })
  }
  try {
    await archiveApi.transcription('audio-id')
    await archiveApi.saveText('audio-id', 'text', 0)
  } finally { globalThis.fetch = priorFetch }
  assert.deepEqual(paths, ['/api/archive/audio/audio-id', '/api/archive/audio/audio-id/text'])
})

test('archive transcript actions stay unavailable until completed text exists', () => {
  assert.equal(hasCompletedTranscript('queued', null), false)
  assert.equal(hasCompletedTranscript('running', 'partial output'), false)
  assert.equal(hasCompletedTranscript('completed', null), false)
  assert.equal(hasCompletedTranscript('completed', ''), true)
})

test('archive text saves pin the canonical root to prevent editing a same-id item elsewhere', async () => {
  const priorFetch = globalThis.fetch
  let requestBody: Record<string, unknown> | undefined
  globalThis.fetch = async (_input, init) => {
    requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>
    return new Response(JSON.stringify({ item: {}, record: {}, text: requestBody.text, revision: 3 }), { status: 200, headers: { 'Content-Type': 'application/json' } })
  }
  try { await archiveApi.saveText('same-id', 'B edit', 2, 'E:\\Archive\\B') }
  finally { globalThis.fetch = priorFetch }
  assert.deepEqual(requestBody, { text: 'B edit', baseRevision: 2, expectedArchiveRoot: 'E:\\Archive\\B' })
})

test('dictionary API keeps personal CRUD, undo, and auto-learn routes distinct', async () => {
  const calls: { url: string; method: string; body?: unknown }[] = []
  const priorFetch = globalThis.fetch
  globalThis.fetch = async (input, init) => {
    const url = String(input)
    const method = init?.method ?? 'GET'
    calls.push({ url, method, body: init?.body ? JSON.parse(String(init.body)) : undefined })
    const payload = url.includes('/settings') ? { auto_learn: true } : url.endsWith('/undo/token') ? { item: { id: 'term-1', term: 'Whisper' } } : url.endsWith('/term-1') && method === 'DELETE' ? { undo_id: 'token' } : { item: { id: 'term-1', term: 'Whisper' }, items: [] }
    return new Response(JSON.stringify(payload), { status: 200, headers: { 'Content-Type': 'application/json' } })
  }
  try {
    await dictionaryApi.list('Whisper')
    await dictionaryApi.add('Whisper', '')
    await dictionaryApi.update('term-1', 'Whisper', 'Speech recognition')
    await dictionaryApi.remove('term-1')
    await dictionaryApi.undo('token')
    await dictionaryApi.getSettings()
    await dictionaryApi.saveSettings(false)
  } finally { globalThis.fetch = priorFetch }
  assert.deepEqual(calls.map(({ method, url }) => [method, new URL(url, 'http://local').pathname]), [
    ['GET', '/api/dictionary'], ['POST', '/api/dictionary'], ['PUT', '/api/dictionary/term-1'],
    ['DELETE', '/api/dictionary/term-1'], ['POST', '/api/dictionary/undo/token'],
    ['GET', '/api/dictionary/settings'], ['PUT', '/api/dictionary/settings'],
  ])
  assert.deepEqual(calls[1]?.body, { term: 'Whisper' })
  assert.deepEqual(calls[6]?.body, { auto_learn: false })
})

test('duplicate normalized terms and undo results do not duplicate stable entry ids', () => {
  const first = { id: 'stable-id', term: 'Whisper' }
  const updated = { id: 'stable-id', term: 'WHISPER' }
  const result = upsertDictionaryEntry([first], updated)
  assert.equal(result.length, 1)
  assert.equal(result[0]?.term, 'WHISPER')
})

test('learned-term notification only publishes undoable terms for the root toast', () => {
  const priorWindow = globalThis.window
  const events: CustomEvent[] = []
  globalThis.window = { dispatchEvent: (event: Event) => { events.push(event as CustomEvent); return true } } as unknown as Window & typeof globalThis
  try { notifyLearnedTerms([{ term: 'Qwen', undo_id: 'undo-1' }, { term: 'fixed' }]) }
  finally { globalThis.window = priorWindow }
  assert.equal(events.length, 1)
  assert.deepEqual(events[0]?.detail, [{ term: 'Qwen', undo_id: 'undo-1' }])
})
