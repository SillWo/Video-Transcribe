import { fetchJson } from './api.ts'

export type ArchiveKind = 'audio' | 'video'
export type ArchiveRow = {
  id: string
  kind: ArchiveKind
  title: string
  created_at: string
  started_at?: string | null
  status: string
  source?: string | null
  job_id?: string | null
  meeting_id?: string | null
  speech_availability?: string | null
  archive_root?: string | null
}
export type ArchiveList = { items: ArchiveRow[]; total: number; next_offset: number | null }
export type TranscriptionArchiveDetail = { item: ArchiveRow; record: Record<string, unknown>; text: string | null; revision: number; remembered_terms?: { id?: string; term?: string; undo_id?: string }[]; archive_root?: string | null }

export function archiveSelectionKey(kind: ArchiveKind, id: string, archiveRoot?: string | null) {
  const rawRoot = (archiveRoot ?? '').trim().replace(/[\\/]+/g, '\\').toLocaleLowerCase('en-US')
  const root = /^[a-z]:\\$/.test(rawRoot) ? rawRoot : rawRoot.replace(/\\+$/, '')
  return JSON.stringify([kind, id, root])
}
export function hasCompletedTranscript(status: string, text: string | null | undefined) { return status.toLowerCase() === 'completed' && text !== null && text !== undefined }
export function filterArchiveRows(items: ArchiveRow[], kind: 'all' | ArchiveKind, query: string) {
  const needle = query.trim().toLocaleLowerCase('ru')
  return items.filter((item) => (kind === 'all' || item.kind === kind) && (!needle || `${item.title} ${item.source ?? ''}`.toLocaleLowerCase('ru').includes(needle)))
}

export const archiveApi = {
  list: (kind: 'all' | ArchiveKind, query = '', offset = 0, limit = 50) => {
    const params = new URLSearchParams({ kind, query, offset: String(offset), limit: String(limit) })
    return fetchJson<ArchiveList>(`/api/archive?${params}`)
  },
  transcription: (id: string, signal?: AbortSignal) => fetchJson<TranscriptionArchiveDetail>(`/api/archive/audio/${encodeURIComponent(id)}`, { signal }),
  saveText: (id: string, text: string, baseRevision: number, expectedArchiveRoot?: string | null) => fetchJson<TranscriptionArchiveDetail>(`/api/archive/audio/${encodeURIComponent(id)}/text`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, baseRevision, ...(expectedArchiveRoot ? { expectedArchiveRoot } : {}) }),
  }),
  download: (id: string, format: 'txt' | 'md') => `/api/archive/audio/${encodeURIComponent(id)}/download?format=${format}`,
  remove: (id: string) => fetchJson<{ deleted?: boolean }>(`/api/archive/audio/${encodeURIComponent(id)}`, { method: 'DELETE' }),
}
