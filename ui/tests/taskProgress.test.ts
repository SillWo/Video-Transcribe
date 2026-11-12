import assert from 'node:assert/strict'
import test from 'node:test'
import { resolveTaskProgressView, type TaskProgress, type TaskStage } from '../src/lib/taskProgress.ts'

const stageOrder: readonly TaskStage[] = ['preparation', 'source', 'audio', 'recognition', 'diarization', 'postprocess', 'result']

test('alignment uses its own label and phase start while public stage stays recognition', () => {
  const progress: TaskProgress = {
    stage: 'recognition', phase: 'alignment', phaseStartedAt: '2026-11-12T10:00:00Z',
    percent: 64, message: 'Распознавание речи',
  }

  const view = resolveTaskProgressView('recognition', progress, stageOrder)

  assert.equal(view.currentStage, 'recognition')
  assert.equal(view.aligningWords, true)
  assert.equal(view.label, 'Выравнивание слов')
  assert.equal(view.startedAt, '2026-11-12T10:00:00Z')
  assert.equal(view.progress, progress)
})

test('the job stage wins when progress is stale from recognition during punctuation', () => {
  const staleProgress: TaskProgress = {
    stage: 'recognition', phase: 'alignment', phaseStartedAt: '2026-11-12T10:00:00Z',
    percent: 64, completedBytes: 1234, message: 'Распознавание речи',
  }

  const view = resolveTaskProgressView('postprocess', staleProgress, stageOrder)

  assert.equal(view.currentStage, 'postprocess')
  assert.equal(view.label, null)
  assert.equal(view.aligningWords, false)
  assert.equal(view.startedAt, null)
  assert.equal(view.progress, null)
})

test('legacy recognition progress without a phase keeps the regular stage presentation', () => {
  const oldProgress: TaskProgress = { stage: 'recognition', percent: 42, message: 'Распознавание речи' }

  const view = resolveTaskProgressView('recognition', oldProgress, stageOrder)

  assert.equal(view.currentStage, 'recognition')
  assert.equal(view.label, null)
  assert.equal(view.aligningWords, false)
  assert.equal(view.startedAt, null)
  assert.equal(view.progress, oldProgress)
})
