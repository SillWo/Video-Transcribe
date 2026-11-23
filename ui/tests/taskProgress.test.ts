import assert from 'node:assert/strict'
import test from 'node:test'
import { resolveTaskProgressView, taskCheckpointsFor, taskStageLabel, type TaskCheckpoint, type TaskProgress, type TaskStage } from '../src/lib/taskProgress.ts'
import { stageOrderFor } from '../src/lib/diarization.ts'

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

test('meeting video pipeline uses the chronological visible stage order', () => {
  const order = stageOrderFor(false, true)
  assert.deepEqual(order, ['source_check', 'preparation', 'source', 'audio', 'recognition', 'postprocess', 'screen', 'archive_audio', 'index', 'result'])
  assert.equal(taskStageLabel('source_check'), 'Проверка записи')
  assert.equal(taskStageLabel('archive_audio'), 'Сохранение аудио')
  assert.equal(taskStageLabel('speech'), 'Распознавание речи')
  assert.equal(taskStageLabel('screen'), 'Извлечение кадров')
  assert.equal(taskStageLabel('index'), 'Обновление индекса архива')
})

test('meeting screen and index progress resolve to known stages instead of unknown labels', () => {
  const order = stageOrderFor(false, true)
  const screenView = resolveTaskProgressView('screen', { stage: 'screen', percent: 42, message: 'screen' }, order)
  const indexView = resolveTaskProgressView('index', { stage: 'index', percent: 75, message: 'index' }, order)
  assert.equal(screenView.currentStage, 'screen')
  assert.equal(taskStageLabel(screenView.currentStage), 'Извлечение кадров')
  assert.equal(screenView.progress?.percent, 42)
  assert.equal(indexView.currentStage, 'index')
  assert.equal(taskStageLabel(indexView.currentStage), 'Обновление индекса архива')
  assert.equal(indexView.progress?.percent, 75)
})

test('unrecognized job and progress stages fall back to a localized stage', () => {
  const order = stageOrderFor(false, true)
  const view = resolveTaskProgressView('future-stage', { stage: 'other-future-stage' }, order)
  assert.equal(view.currentStage, 'source_check')
  assert.equal(taskStageLabel(view.currentStage), 'Проверка записи')
})

test('legacy speech progress resolves to the single recognition row', () => {
  const order: readonly TaskStage[] = ['source_check', 'preparation', 'source', 'audio', 'recognition', 'postprocess', 'result']
  const progress: TaskProgress = { stage: 'speech', percent: 61 }
  const view = resolveTaskProgressView('speech', progress, order)
  assert.equal(view.currentStage, 'recognition')
  assert.equal(view.progress, progress)
  assert.equal(taskCheckpointsFor([{ stage: 'speech', status: 'completed', startedAt: '2026-11-12T10:00:00Z', completedAt: '2026-11-12T10:01:00Z' }], order, 'recognition', 'running').filter((item) => item.stage === 'recognition').length, 1)
})

test('canonical recognition checkpoint wins over a legacy speech duplicate', () => {
  const order: readonly TaskStage[] = ['source_check', 'recognition', 'postprocess', 'result']
  const view = taskCheckpointsFor([
    { stage: 'speech', status: 'active', startedAt: '2026-11-12T10:00:00Z', completedAt: null },
    { stage: 'recognition', status: 'completed', startedAt: '2026-11-12T10:00:00Z', completedAt: '2026-11-12T10:01:00Z' },
  ], order, 'postprocess', 'running')
  assert.deepEqual(view.filter(({ stage }) => stage === 'recognition'), [{
    stage: 'recognition', status: 'completed', startedAt: '2026-11-12T10:00:00Z', completedAt: '2026-11-12T10:01:00Z',
  }])
})

test('recorded legacy activity wins over a synthetic pending canonical duplicate', () => {
  const order: readonly TaskStage[] = ['recognition', 'postprocess', 'result']
  const view = taskCheckpointsFor([
    { stage: 'speech', status: 'completed', startedAt: '2026-11-12T10:00:00Z', completedAt: '2026-11-12T10:01:00Z' },
    { stage: 'recognition', status: 'pending', startedAt: null, completedAt: null },
  ], order, 'postprocess', 'completed')
  assert.deepEqual(view.filter(({ stage }) => stage === 'recognition'), [{
    stage: 'recognition', status: 'completed', startedAt: '2026-11-12T10:00:00Z', completedAt: '2026-11-12T10:01:00Z',
  }])
})

test('a pending current checkpoint becomes active, but completed state stays completed', () => {
  const order: readonly TaskStage[] = ['recognition', 'postprocess', 'result']
  const active = taskCheckpointsFor([
    { stage: 'recognition', status: 'pending', startedAt: null, completedAt: null },
  ], order, 'recognition', 'running')
  assert.equal(active.find(({ stage }) => stage === 'recognition')?.status, 'active')

  const completed = taskCheckpointsFor([
    { stage: 'recognition', status: 'completed', startedAt: '2026-11-12T10:00:00Z', completedAt: '2026-11-12T10:01:00Z' },
  ], order, 'recognition', 'running')
  assert.equal(completed.find(({ stage }) => stage === 'recognition')?.status, 'completed')
})

test('active checkpoints follow recorded chronology, then pending pipeline order, with result last', () => {
  const order: readonly TaskStage[] = ['source_check', 'preparation', 'source', 'audio', 'recognition', 'postprocess', 'screen', 'archive_audio', 'index', 'result']
  const known: TaskCheckpoint[] = [
    { stage: 'postprocess', status: 'completed', startedAt: '2026-11-12T10:05:00Z', completedAt: '2026-11-12T10:06:00Z' },
    { stage: 'speech', status: 'completed', startedAt: '2026-11-12T10:03:00Z', completedAt: '2026-11-12T10:04:00Z' },
    { stage: 'result', status: 'completed', startedAt: '2026-11-12T10:02:00Z', completedAt: '2026-11-12T10:02:30Z' },
  ]
  const view = taskCheckpointsFor(known, order, 'screen', 'running')
  assert.deepEqual(view.map(({ stage }) => stage), ['recognition', 'postprocess', 'screen', 'source_check', 'preparation', 'source', 'audio', 'archive_audio', 'index', 'result'])
  assert.equal(view.find(({ stage }) => stage === 'screen')?.status, 'active')
  assert.equal(view.find(({ stage }) => stage === 'result')?.status, 'completed')
})

test('terminal jobs without checkpoint history do not invent completed work', () => {
  const order: readonly TaskStage[] = ['source_check', 'preparation', 'source', 'audio', 'recognition', 'postprocess', 'result']
  const view = taskCheckpointsFor(undefined, order, 'result', 'completed')
  assert.deepEqual(view, [])
})

test('terminal jobs with partial history do not invent missing pipeline steps', () => {
  const order: readonly TaskStage[] = ['source_check', 'preparation', 'recognition', 'postprocess', 'result']
  const view = taskCheckpointsFor([
    { stage: 'recognition', status: 'completed', startedAt: '2026-11-12T10:00:00Z', completedAt: '2026-11-12T10:01:00Z' },
  ], order, 'result', 'completed')
  assert.deepEqual(view.map(({ stage }) => stage), ['recognition'])
})

test('current progress never reuses a percentage from a different stage', () => {
  const progress: TaskProgress = { stage: 'speech', phase: 'alignment', percent: 64, phaseStartedAt: '2026-11-12T10:00:00Z' }
  const order: readonly TaskStage[] = ['source_check', 'preparation', 'source', 'audio', 'recognition', 'postprocess', 'result']
  const view = resolveTaskProgressView('postprocess', progress, order)
  assert.equal(view.currentStage, 'postprocess')
  assert.equal(view.progress, null)
  assert.equal(view.aligningWords, false)
  assert.equal(view.startedAt, null)
})
