import assert from 'node:assert/strict'
import test from 'node:test'
import { recentFailedConversions } from '../src/lib/conversionHistory.ts'

test('recent failed conversions include durable imports and recorder jobs but omit ordinary tasks', () => {
  const rows = [
    { id: 'old-import', status: 'failed', conversionOrigin: 'archive_import', createdAt: '2026-01-01' },
    { id: 'ordinary', status: 'failed', createdAt: '2026-01-03' },
    { id: 'recorder', status: 'interrupted', conversionOrigin: 'recorder', createdAt: '2026-01-02' },
    { id: 'completed-import', status: 'completed', conversionOrigin: 'archive_import', createdAt: '2026-01-04' },
    { id: 'mcp', status: 'failed', conversionOrigin: 'mcp', createdAt: '2026-01-05' },
  ]
  assert.deepEqual(recentFailedConversions(rows).map((row) => row.id), ['mcp', 'recorder', 'old-import'])
})
