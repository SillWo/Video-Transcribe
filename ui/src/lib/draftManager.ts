export type DraftRecord = { text: string; revision: number; version: number; savedVersion: number; queue: Promise<void>; error: unknown | null }

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

  open(id: string, text: string, revision: number) {
    const current = this.records.get(id)
    if (current) return current
    const record = { text, revision, version: 0, savedVersion: 0, queue: Promise.resolve(), error: null }
    this.records.set(id, record)
    return record
  }

  hydrate(id: string, text: string, revision: number) {
    const record = this.records.get(id)
    if (!record) return this.open(id, text, revision)
    if (record.version === record.savedVersion) {
      record.text = text
      record.revision = revision
      record.error = null
    }
    return record
  }

  get(id: string) { return this.records.get(id) }
  edit(id: string, text: string) {
    const record = this.records.get(id)
    if (!record) throw new Error(`Draft ${id} is not open`)
    record.text = text
    record.version += 1
    record.error = null
    return record.version
  }

  async save(id: string, persist: (text: string, revision: number) => Promise<{ text: string; revision: number }>, version?: number) {
    const record = this.records.get(id)
    if (!record) return
    const targetVersion = version ?? record.version
    const text = record.text
    const task = record.queue.then(async () => {
      if (targetVersion <= record.savedVersion) return
      const result = await persist(text, record.revision)
      record.revision = result.revision
      record.savedVersion = Math.max(record.savedVersion, targetVersion)
      record.error = null
    }).catch((error: unknown) => { record.error = error; throw error })
    record.queue = task.catch(() => undefined)
    return task
  }

  async flush(id: string, persist: (text: string, revision: number) => Promise<{ text: string; revision: number }>) {
    const record = this.records.get(id)
    if (!record) return
    await record.queue
    while (record.savedVersion < record.version) await this.save(id, persist, record.version)
    if (record.error) throw record.error
  }

  async flushAll(persist: (id: string, text: string, revision: number) => Promise<{ text: string; revision: number }>) {
    await Promise.all([...this.records.keys()].map((id) => this.flush(id, (text, revision) => persist(id, text, revision))))
  }

  remove(id: string) { this.records.delete(id) }
}
