import { fetchJson } from './api.ts'

export type DictionaryProvenance = { meeting_id?: string; meeting_title?: string; source?: string; created_at?: string; [key: string]: unknown }
export type DictionaryEntry = { id: string; term: string; definition?: string | null; provenance?: DictionaryProvenance[] }
export type DictionaryResponse = { items: DictionaryEntry[]; version?: string; auto_learn?: boolean }
export function upsertDictionaryEntry(items: DictionaryEntry[], item: DictionaryEntry) { return [item, ...items.filter((existing) => existing.id !== item.id)] }

export const dictionaryApi = {
  list: (query = '') => fetchJson<DictionaryResponse>(`/api/dictionary?query=${encodeURIComponent(query)}`),
  add: (term: string, definition: string) => fetchJson<{ item: DictionaryEntry }>('/api/dictionary', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ term, ...(definition ? { definition } : {}) }) }),
  update: (id: string, term: string, definition: string) => fetchJson<{ item: DictionaryEntry }>(`/api/dictionary/${encodeURIComponent(id)}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ term, ...(definition ? { definition } : {}) }) }),
  remove: (id: string) => fetchJson<{ undo_id: string }>(`/api/dictionary/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  undo: (id: string) => fetchJson<{ item: DictionaryEntry }>(`/api/dictionary/undo/${encodeURIComponent(id)}`, { method: 'POST' }),
  getSettings: () => fetchJson<{ auto_learn: boolean }>('/api/dictionary/settings'),
  saveSettings: (auto_learn: boolean) => fetchJson<{ auto_learn: boolean }>('/api/dictionary/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ auto_learn }) }),
}
