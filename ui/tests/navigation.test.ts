import assert from 'node:assert/strict'
import test from 'node:test'
import { readFile } from 'node:fs/promises'
import { legacyRedirects, railDestinations } from '../src/lib/navigation.ts'

test('rail destinations keep the approved order and groups', () => {
  assert.deepEqual(railDestinations.map((item) => `${item.group}:${item.path}`), [
    'top:/recording', 'top:/dictionary', 'top:/transcribe', 'bottom:/settings', 'bottom:/archive',
  ])
})

test('legacy locations resolve to the replacement screens', () => {
  assert.deepEqual(legacyRedirects, {
    '/': '/archive', '/meetings': '/archive', '/meetings/glossary': '/dictionary',
    '/meetings/settings': '/settings', '/history': '/archive?kind=transcription',
  })
})

test('every approved destination has an explicit route, including the persistent recorder screen', async () => {
  const app = await readFile(new URL('../src/App.tsx', import.meta.url), 'utf8')
  for (const route of ['/archive', '/recording', '/dictionary', '/transcribe', '/settings']) {
    assert.match(app, new RegExp(`<Route path="${route}" element=`))
  }
})

test('failed route flush keeps the current draft visible and offers an explicit retry', async () => {
  const app = await readFile(new URL('../src/App.tsx', import.meta.url), 'utf8')
  assert.match(app, /const navigateAfterSave = async \(path: string\)[\s\S]*await flushAllRootDrafts\(\)[\s\S]*navigate\(path\)[\s\S]*catch \(reason\)[\s\S]*Черновик остаётся здесь/)
  assert.match(app, /onClick=\{\(\) => void navigateAfterSave\(navSaveTarget\)\}>Повторить/)
  assert.match(app, /className="nav-save-notice" role="alert"/)
})
