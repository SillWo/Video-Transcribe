type Flush = () => Promise<void>

const pendingSaves = new Map<string, Flush>()

export function registerPendingSave(key: string, flush: Flush): () => void {
  pendingSaves.set(key, flush)
  return () => {
    if (pendingSaves.get(key) === flush) pendingSaves.delete(key)
  }
}

export async function flushPendingSaves(): Promise<void> {
  const callbacks = [...pendingSaves.values()]
  const failures: unknown[] = []
  for (const flush of callbacks) {
    try {
      await flush()
    } catch (error) {
      failures.push(error)
    }
  }
  if (failures.length > 0) throw failures[0]
}

