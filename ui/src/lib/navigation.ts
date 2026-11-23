export type RailDestination = { path: '/recording' | '/dictionary' | '/transcribe' | '/settings' | '/archive'; label: string; group: 'top' | 'bottom' }
export const railDestinations: RailDestination[] = [
  { path: '/recording', label: 'Запись', group: 'top' },
  { path: '/dictionary', label: 'Словарь', group: 'top' },
  { path: '/transcribe', label: 'Расшифровка', group: 'top' },
  { path: '/settings', label: 'Настройки', group: 'bottom' },
  { path: '/archive', label: 'Архив', group: 'bottom' },
]

export const legacyRedirects: Record<string, string> = {
  '/': '/archive',
  '/meetings': '/archive',
  '/meetings/glossary': '/dictionary',
  '/meetings/settings': '/settings',
  '/history': '/archive?kind=transcription',
}
