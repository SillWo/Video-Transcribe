export type TaskStage = 'source_check' | 'preparation' | 'source' | 'audio' | 'recognition' | 'speech' | 'diarization' | 'postprocess' | 'screen' | 'archive_audio' | 'index' | 'result'

export type TaskCheckpoint = {
  stage: TaskStage
  status: 'pending' | 'active' | 'completed' | 'skipped' | 'failed' | 'cancelled'
  startedAt: string | null
  completedAt: string | null
}

const stageLabels: Record<TaskStage, string> = {
  source_check: 'Проверка записи',
  preparation: 'Подготовка модели',
  source: 'Получение записи',
  audio: 'Подготовка аудио',
  recognition: 'Распознавание речи',
  speech: 'Распознавание речи',
  diarization: 'Разделение по говорящим',
  postprocess: 'Пунктуация',
  screen: 'Извлечение кадров',
  archive_audio: 'Сохранение аудио',
  index: 'Обновление индекса архива',
  result: 'Готово',
}

export type TaskProgress = {
  stage: string
  phase?: string | null
  phaseStartedAt?: string | null
  percent?: number | null
  completedBytes?: number | null
  totalBytes?: number | null
  processedSeconds?: number | null
  totalSeconds?: number | null
  message?: string | null
  updatedAt?: string
}

export function normalizeTaskStage(value: string | null | undefined): TaskStage | null {
  const normalized = value?.trim().toLocaleLowerCase('en-US')
  return normalized && normalized in stageLabels ? normalized as TaskStage : null
}

export function taskStageLabel(stage: TaskStage) {
  return stageLabels[stage]
}

function canonicalStage(stage: TaskStage): TaskStage {
  return stage === 'speech' ? 'recognition' : stage
}

/**
 * Builds the visible stage list from the current pipeline and any worker history.
 * Worker timestamps are useful for recovering chronology when old workers emitted
 * checkpoints in a different order than the current pipeline.
 */
export function taskCheckpointsFor(
  known: readonly TaskCheckpoint[] | null | undefined,
  stageOrder: readonly TaskStage[],
  currentStage: TaskStage,
  status: 'queued' | 'running' | 'cancelling' | 'completed' | 'failed' | 'interrupted' | 'cancelled',
): TaskCheckpoint[] {
  const order: TaskStage[] = []
  for (const stage of stageOrder) {
    const canonical = canonicalStage(stage)
    if (!order.includes(canonical)) order.push(canonical)
  }
  const current = canonicalStage(currentStage)
  if (!order.includes(current) && (status === 'running' || status === 'cancelling')) order.push(current)

  const knownByStage = new Map<TaskStage, { checkpoint: TaskCheckpoint; canonical: boolean }>()
  for (const checkpoint of known ?? []) {
    if (!checkpoint || !checkpoint.stage) continue
    const stage = canonicalStage(checkpoint.stage)
    if (!order.includes(stage)) continue
    const existing = knownByStage.get(stage)
    // Prefer a canonical record over its legacy alias when both were emitted.
    const isCanonical = checkpoint.stage === stage
    const isSyntheticPending = checkpoint.status === 'pending'
      && checkpoint.startedAt === null && checkpoint.completedAt === null
    const existingIsSyntheticPending = existing?.checkpoint.status === 'pending'
      && existing.checkpoint.startedAt === null && existing.checkpoint.completedAt === null
    const shouldReplace = !existing
      || (isCanonical && !existing.canonical && !isSyntheticPending)
      || (!isCanonical && existing.canonical && existingIsSyntheticPending)
    if (shouldReplace) {
      knownByStage.set(stage, { checkpoint: { ...checkpoint, stage }, canonical: isCanonical })
    }
  }

  const active = status === 'running' || status === 'cancelling'
  const canSynthesize = status === 'queued' || active
  const pipelineIndex = new Map(order.map((stage, index) => [stage, index]))
  const recordedCheckpoints = [...knownByStage.values()].map(({ checkpoint }) => checkpoint)
  // A terminal job may predate checkpoint persistence, or have only a partial
  // history. Do not manufacture a complete pending pipeline for that job. An
  // all-pending record is equivalent to no history (the old initial snapshot).
  const terminalHasHistory = recordedCheckpoints.some((checkpoint) => checkpoint.status !== 'pending'
    || checkpoint.startedAt !== null || checkpoint.completedAt !== null)
  const checkpoints = order.flatMap((stage): TaskCheckpoint[] => {
    const recorded = (!active && status !== 'queued' && !terminalHasHistory)
      ? undefined
      : knownByStage.get(stage)?.checkpoint
    if (recorded) {
      // A worker can persist the pending snapshot before the active event. The
      // active job stage is enough to expose the truthful active state. Never
      // overwrite a completed/failed/etc. checkpoint from a stale job stage.
      if (active && stage === current && recorded.status === 'pending') {
        return [{ ...recorded, status: 'active' }]
      }
      return [recorded]
    }
    if (!canSynthesize || (!active && status !== 'queued' && !terminalHasHistory)) return []
    return [{
      stage,
      status: active && stage === current ? 'active' : 'pending',
      startedAt: null,
      completedAt: null,
    }]
  })

  const timeValue = (value: string | null) => {
    if (!value) return null
    const timestamp = Date.parse(value)
    return Number.isFinite(timestamp) ? timestamp : null
  }
  const hasHistory = (checkpoint: TaskCheckpoint) => checkpoint.status !== 'pending'
  const eventTime = (checkpoint: TaskCheckpoint) => timeValue(checkpoint.startedAt) ?? timeValue(checkpoint.completedAt)

  // Sort historical/started stages by worker timestamps. Untimed active or
  // terminal records keep their pipeline position ahead of pending stages.
  const history = checkpoints.filter(hasHistory)
  const pending = checkpoints.filter((checkpoint) => !hasHistory(checkpoint))
  history.sort((a, b) => {
    const aTime = eventTime(a)
    const bTime = eventTime(b)
    if (aTime !== null && bTime !== null && aTime !== bTime) return aTime - bTime
    if (aTime !== null && bTime === null) return -1
    if (aTime === null && bTime !== null) return 1
    return (pipelineIndex.get(a.stage) ?? 0) - (pipelineIndex.get(b.stage) ?? 0)
  })

  // Result is the conceptual terminal stage, even in older worker histories
  // where it was recorded before punctuation or another later stage.
  const result = checkpoints.filter((checkpoint) => checkpoint.stage === 'result')
  return [...history.filter((checkpoint) => checkpoint.stage !== 'result'), ...pending.filter((checkpoint) => checkpoint.stage !== 'result'), ...result]
}

export function resolveTaskProgressView<T extends TaskProgress>(jobStage: string, progress: T | null | undefined, stageOrder: readonly TaskStage[]) {
  const normalizedJobStage = normalizeTaskStage(jobStage)
  const normalizedProgressStage = normalizeTaskStage(progress?.stage)
  const plannedStages = stageOrder.map(canonicalStage)
  const currentStage = normalizedJobStage && plannedStages.includes(canonicalStage(normalizedJobStage))
    ? canonicalStage(normalizedJobStage)
    : normalizedProgressStage && plannedStages.includes(canonicalStage(normalizedProgressStage))
      ? canonicalStage(normalizedProgressStage)
      : plannedStages[0] ?? 'preparation'
  const currentProgress = normalizedProgressStage && canonicalStage(normalizedProgressStage) === currentStage ? progress : null
  const aligningWords = currentStage === 'recognition' && currentProgress?.phase === 'alignment'

  return {
    currentStage,
    progress: currentProgress,
    aligningWords,
    label: aligningWords ? 'Выравнивание слов' : null,
    startedAt: aligningWords ? currentProgress?.phaseStartedAt ?? null : null,
  }
}
