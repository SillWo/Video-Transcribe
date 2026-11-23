export type DraftPersist = (text: string, revision: number) => Promise<{ text: string; revision: number }>
export type DraftRecord = { text: string; revision: number; version: number; savedVersion: number; queue: Promise<void>; error: unknown | null; persist?: DraftPersist }

export function transcriptionDraftKey(id: string, archiveRoot?: string | null) {
  const rawRoot = (archiveRoot ?? '').trim().replace(/[\\/]+/g, '\\').toLocaleLowerCase('en-US')
  const root = /^[a-z]:\\$/.test(rawRoot) ? rawRoot : rawRoot.replace(/\\+$/, '')
  return JSON.stringify([root, id])
}

export class SelectionGate {
  private generation = 0
  begin() { this.generation += 1; return this.generation }
  isCurrent(generation: number) { return generation === this.generation }
}

export async function flushAndRead<T>(id: string, flush: (id: string) => Promise<void>, read: (id: string) => T): Promise<T> {
  await flush(id)
  return read(id)
}

export class DraftManager {
  private records = new Map<string, DraftRecord>()
  private listeners = new Map<string, Set<(record: DraftRecord) => void>>()

  private notify(id: string) {
    const record = this.records.get(id)
    if (!record) return
    for (const listener of this.listeners.get(id) ?? []) listener(record)
    if (typeof window !== 'undefined') window.dispatchEvent(new CustomEvent('shared-draft-changed', { detail: { id, text: record.text, revision: record.revision, version: record.version, savedVersion: record.savedVersion } }))
  }

  subscribe(id: string, listener: (record: DraftRecord) => void) {
    const listeners = this.listeners.get(id) ?? new Set<(record: DraftRecord) => void>()
    listeners.add(listener); this.listeners.set(id, listeners)
    return () => { listeners.delete(listener); if (listeners.size === 0) this.listeners.delete(id) }
  }

  open(id: string, text: string, revision: number) {
    const current = this.records.get(id)
    if (current) return current
    const record = { text, revision, version: 0, savedVersion: 0, queue: Promise.resolve(), error: null }
    this.records.set(id, record)
    this.notify(id)
    return record
  }

  hydrate(id: string, text: string, revision: number) {
    const record = this.records.get(id)
    if (!record) return this.open(id, text, revision)
    if (revision < record.revision) return record
    if (record.version === record.savedVersion) {
      record.text = text
      record.revision = revision
      record.error = null
    }
    this.notify(id)
    return record
  }

  get(id: string) { return this.records.get(id) }
  registerPersister(id: string, persist: DraftPersist) {
    const record = this.records.get(id)
    if (record) record.persist = persist
  }
  edit(id: string, text: string) {
    const record = this.records.get(id)
    if (!record) throw new Error(`Draft ${id} is not open`)
    record.text = text
    record.version += 1
    record.error = null
    this.notify(id)
    return record.version
  }

  async save(id: string, persist: DraftPersist | undefined, version?: number) {
    const record = this.records.get(id)
    if (!record) return
    const write = persist ?? record.persist
    if (!write) throw new Error(`Draft ${id} has no persistence adapter`)
    const targetVersion = version ?? record.version
    const text = record.text
    const task = record.queue.then(async () => {
      if (targetVersion <= record.savedVersion) return
      const result = await write(text, record.revision)
      record.revision = result.revision
      record.savedVersion = Math.max(record.savedVersion, targetVersion)
      record.error = null
      this.notify(id)
    }).catch((error: unknown) => { record.error = error; throw error })
    record.queue = task.catch(() => undefined)
    return task
  }

  async flush(id: string, persist?: DraftPersist) {
    const record = this.records.get(id)
    if (!record) return
    await record.queue
    while (record.savedVersion < record.version) await this.save(id, persist, record.version)
    if (record.error) throw record.error
  }

  async flushAll(persist?: (id: string, text: string, revision: number) => Promise<{ text: string; revision: number }>) {
    const tasks = [...this.records.entries()].map(([id, record]) => {
      const fallback = persist ? (text: string, revision: number) => persist(id, text, revision) : undefined
      return this.flush(id, record.persist ?? fallback)
    })
    await Promise.all(tasks)
  }

  remove(id: string) { this.records.delete(id) }
}

// The simple transcription workspace and the archive editor share this store so
// opening the same result from either screen never creates competing drafts.
export const sharedDraftManager = new DraftManager()
