import assert from 'node:assert/strict'
import test from 'node:test'
import { defaultDiarizationPrefs, diarizationRequestFields, sanitizeDiarizationPrefs, sanitizeStoredPrefs, stageOrderFor } from '../src/lib/diarization.ts'

const safeStoredDefaults = { profile: 'balanced', restorePunctuation: true, ...defaultDiarizationPrefs }

test('empty and malformed stored preferences keep the complete safe defaults', () => {
  assert.deepEqual(sanitizeStoredPrefs({}), safeStoredDefaults)
  assert.deepEqual(sanitizeStoredPrefs(null), safeStoredDefaults)
  assert.deepEqual(sanitizeStoredPrefs([]), safeStoredDefaults)
  assert.deepEqual(sanitizeStoredPrefs({ profile: 'broken', restorePunctuation: 1, diarizationEnabled: 'yes', diarizationDevice: '  ' }), safeStoredDefaults)
})

test('absent CPU preferences stay absent so API defaults can be applied', () => {
  assert.deepEqual(sanitizeStoredPrefs({}), safeStoredDefaults)
  assert.deepEqual(sanitizeStoredPrefs({ device: 'gpu', cpuThreads: 8 }).device, 'gpu')
  assert.equal(sanitizeStoredPrefs({ device: 'gpu', cpuThreads: 8 }).cpuThreads, 8)
})

test('diarization preferences default safely for old and malformed records', () => {
  assert.deepEqual(sanitizeDiarizationPrefs({}), defaultDiarizationPrefs)
  assert.deepEqual(sanitizeDiarizationPrefs({ diarizationEnabled: 'yes', diarizationDevice: 4 }), defaultDiarizationPrefs)
  assert.deepEqual(sanitizeDiarizationPrefs({ diarizationEnabled: true, diarizationDevice: '\u0000bad' }), { diarizationEnabled: true, diarizationDevice: 'auto' })
  assert.deepEqual(sanitizeDiarizationPrefs({ diarizationDevice: '   ' }), defaultDiarizationPrefs)
  assert.deepEqual(sanitizeDiarizationPrefs({ diarizationDevice: 'gpu\u0085id' }), defaultDiarizationPrefs)
})

test('request snapshot carries the selected diarization settings', () => {
  assert.deepEqual(diarizationRequestFields({ diarizationEnabled: true, diarizationDevice: 'gpu:intel:1' }), {
    diarizationEnabled: true,
    diarizationDevice: 'gpu:intel:1',
  })
})

test('diarization checkpoint appears only when the task requested it', () => {
  assert.equal(stageOrderFor(false).includes('diarization' as never), false)
  assert.equal(stageOrderFor(true).indexOf('diarization'), 4)
  assert.equal(stageOrderFor(true)[5], 'postprocess')
})
