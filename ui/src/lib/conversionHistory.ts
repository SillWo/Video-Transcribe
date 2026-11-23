export type ConversionHistoryItem = {
  id: string
  status: string
  source?: string
  title?: string
  error?: string | null
  archiveRoot?: string | null
  conversionOrigin?: string | null
  createdAt?: string
}

export function recentFailedConversions(items: ConversionHistoryItem[], limit = 5) {
  return items
    .filter((item) => ['failed', 'interrupted'].includes(item.status) && ['archive_import', 'recorder', 'mcp'].includes(item.conversionOrigin ?? ''))
    .sort((a, b) => Date.parse(b.createdAt ?? '') - Date.parse(a.createdAt ?? ''))
    .slice(0, Math.max(0, limit))
}
