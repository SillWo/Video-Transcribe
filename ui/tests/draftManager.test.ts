import assert from 'node:assert/strict'
import test from 'node:test'
import { DraftManager, SelectionGate, flushAndRead, sharedDraftManager, transcriptionDraftKey } from '../src/lib/draftManager.ts'

test('keeps drafts and revisions isolated by job id while saves are delayed', async () => {
  const drafts = new DraftManager()
  drafts.open('A', 'A0', 3)
  drafts.open('B', 'B0', 8)
  drafts.edit('A', 'A1')
  drafts.edit('B', 'B1')
  const persisted: string[] = []
  const persist = async (id: string, text: string, revision: number) => {
    await new Promise((resolve) => setTimeout(resolve, id === 'A' ? 15 : 1))
    persisted.push(`${id}:${text}@${revision}`)
    return { text, revision: revision + 1 }
  }
  await drafts.flushAll(persist)
  assert.deepEqual(persisted.sort(), ['A:A1@3', 'B:B1@8'])
  assert.equal(drafts.get('A')?.revision, 4)
  assert.equal(drafts.get('B')?.revision, 9)
})

test('serializes rapid edits and flushes the newest version', async () => {
  const drafts = new DraftManager()
  drafts.open('A', 'initial', 1)
  drafts.edit('A', 'first')
  let release!: () => void
  const blocked = new Promise<void>((resolve) => { release = resolve })
  const versions: string[] = []
  const persist = async (text: string, revision: number) => {
    if (revision === 1) await blocked
    versions.push(text)
    return { text, revision: revision + 1 }
  }
  const first = drafts.save('A', persist, 1)
  drafts.edit('A', 'latest')
  const last = drafts.flush('A', persist)
  release()
  await Promise.all([first, last])
  assert.deepEqual(versions, ['first', 'latest'])
  assert.equal(drafts.get('A')?.savedVersion, 2)
  assert.equal(drafts.get('A')?.text, 'latest')
})

test('does not overwrite a dirty draft when a completion response hydrates it', () => {
  const drafts = new DraftManager()
  drafts.open('A', '', 0)
  drafts.edit('A', 'local unsaved text')
  drafts.hydrate('A', 'server result', 4)
  assert.equal(drafts.get('A')?.text, 'local unsaved text')
  assert.equal(drafts.get('A')?.revision, 0)
})

test('a stale clean GET cannot roll back text or revision after a successful save', async () => {
  const drafts = new DraftManager()
  drafts.open('A', 'old text', 1)
  drafts.edit('A', 'saved text')
  await drafts.flush('A', async (text, revision) => ({ text, revision: revision + 1 }))
  drafts.hydrate('A', 'older server text', 1)
  assert.equal(drafts.get('A')?.text, 'saved text')
  assert.equal(drafts.get('A')?.revision, 2)
})

test('failed flush retains the editable draft and can be retried', async () => {
  const drafts = new DraftManager()
  drafts.open('A', 'source', 2)
  drafts.edit('A', 'edited')
  await assert.rejects(drafts.flush('A', async () => { throw new Error('offline') }), /offline/)
  assert.equal(drafts.get('A')?.text, 'edited')
  await drafts.flush('A', async (text, revision) => ({ text, revision: revision + 1 }))
  assert.equal(drafts.get('A')?.savedVersion, 1)
})

test('ignores a late task response after the selection changes', async () => {
  const gate = new SelectionGate()
  const first = gate.begin()
  const delayedA = new Promise<string>((resolve) => setTimeout(() => resolve('A'), 10))
  const second = gate.begin()
  const responseB = Promise.resolve('B')
  const applied: string[] = []
  const [a, b] = await Promise.all([delayedA, responseB])
  if (gate.isCurrent(first)) applied.push(a)
  if (gate.isCurrent(second)) applied.push(b)
  assert.deepEqual(applied, ['B'])
})

test('hydrates a completed nonempty result after a queued task had no draft record', () => {
  const drafts = new DraftManager()
  assert.equal(drafts.get('queued'), undefined)
  drafts.hydrate('queued', 'completed text', 6)
  assert.equal(drafts.get('queued')?.text, 'completed text')
  assert.equal(drafts.get('queued')?.revision, 6)
})

test('copy/export keeps the captured task id while another task becomes selected', async () => {
  const drafts = new DraftManager()
  drafts.open('A', 'edited A', 2)
  drafts.open('B', 'draft B', 4)
  let selectedId = 'A'
  const exported = flushAndRead('A', async (id) => {
    await new Promise((resolve) => setTimeout(resolve, 10))
    selectedId = 'B'
    await drafts.flush(id, async (text, revision) => ({ text, revision: revision + 1 }))
  }, (capturedId) => drafts.get(capturedId)?.text)
  assert.equal(await exported, 'edited A')
  assert.equal(selectedId, 'B')
})

test('archive and transcription views observe the same draft and late saves preserve the newest edit', async () => {
  const id = `shared-${Date.now()}`
  sharedDraftManager.open(id, 'server text', 2)
  const observed: string[] = []
  const unsubscribe = sharedDraftManager.subscribe(id, (record) => observed.push(record.text))
  sharedDraftManager.edit(id, 'archive edit')
  let release!: () => void
  const blocked = new Promise<void>((resolve) => { release = resolve })
  const persisted: string[] = []
  const save = sharedDraftManager.save(id, async (text, revision) => {
    if (text === 'archive edit') await blocked
    persisted.push(text)
    return { text, revision: revision + 1 }
  }, 1)
  sharedDraftManager.edit(id, 'transcription edit')
  const flush = sharedDraftManager.flush(id, async (text, revision) => { persisted.push(text); return { text, revision: revision + 1 } })
  release()
  await Promise.all([save, flush])
  assert.equal(sharedDraftManager.get(id)?.text, 'transcription edit')
  assert.equal(sharedDraftManager.get(id)?.savedVersion, 2)
  assert.deepEqual(persisted, ['archive edit', 'transcription edit'])
  assert.ok(observed.includes('archive edit'))
  assert.ok(observed.includes('transcription edit'))
  unsubscribe()
  sharedDraftManager.remove(id)
})

test('same job id in different archive roots has isolated drafts and keeps each root-bound save adapter', async () => {
  const drafts = new DraftManager()
  const firstKey = transcriptionDraftKey('same-id', 'D:\\Archive\\A')
  const secondKey = transcriptionDraftKey('same-id', 'E:/Archive/B/')
  assert.equal(transcriptionDraftKey('same-id', 'd:/archive/a/'), firstKey)
  assert.notEqual(firstKey, secondKey)
  drafts.open(firstKey, 'A source', 2)
  drafts.open(secondKey, 'B source', 9)
  const saved: string[] = []
  drafts.registerPersister(firstKey, async (text, revision) => { saved.push(`A:${text}@${revision}`); return { text, revision: revision + 1 } })
  drafts.registerPersister(secondKey, async (text, revision) => { saved.push(`B:${text}@${revision}`); return { text, revision: revision + 1 } })
  drafts.edit(firstKey, 'A edit')
  drafts.edit(secondKey, 'B edit')
  await drafts.flushAll(async () => { throw new Error('bound adapter must take precedence') })
  assert.deepEqual(saved.sort(), ['A:A edit@2', 'B:B edit@9'])
  assert.equal(drafts.get(firstKey)?.text, 'A edit')
  assert.equal(drafts.get(firstKey)?.revision, 3)
  assert.equal(drafts.get(secondKey)?.text, 'B edit')
  assert.equal(drafts.get(secondKey)?.revision, 10)
})
