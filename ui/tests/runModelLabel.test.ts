import assert from 'node:assert/strict'
import test from 'node:test'
import { runModelDisplayLabel } from '../src/lib/runModelLabel.ts'

test('completed historical Parakeet jobs use the persisted model name', () => {
  assert.equal(runModelDisplayLabel({ status: 'completed', profile: 'balanced', model: 'parakeet-tdt-0.6b-v3' }), 'Parakeet TDT 0.6B v3')
})

test('large-v3-turbo and exact large-v3 variants stay distinct', () => {
  assert.equal(runModelDisplayLabel({ status: 'completed', profile: 'balanced', model: 'large-v3-turbo' }), 'Whisper large-v3-turbo')
  assert.equal(runModelDisplayLabel({ status: 'completed', profile: 'balanced', model: 'large-v3' }), 'Whisper large-v3')
})

test('effective model takes precedence over requested model', () => {
  assert.equal(runModelDisplayLabel({ status: 'completed', profile: 'balanced', model: 'large-v3', effectiveModel: 'large-v3-turbo' }), 'Whisper large-v3-turbo')
})

test('completed jobs without actual model data do not guess from current profile', () => {
  assert.equal(runModelDisplayLabel({ status: 'completed', profile: 'balanced' }), 'Модель не указана')
})

test('queued jobs without actual model data can show the selected profile', () => {
  assert.equal(runModelDisplayLabel({ status: 'queued', profile: 'balanced' }), 'Whisper large-v3')
})
