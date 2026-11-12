export type TaskStage = 'preparation' | 'source' | 'audio' | 'recognition' | 'diarization' | 'postprocess' | 'result'

export type TaskProgress = {
  stage: TaskStage
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

export function resolveTaskProgressView<T extends TaskProgress>(jobStage: string, progress: T | null | undefined, stageOrder: readonly TaskStage[]) {
  const validJobStage = stageOrder.includes(jobStage as TaskStage)
  const currentStage = validJobStage
    ? jobStage as TaskStage
    : progress && stageOrder.includes(progress.stage) ? progress.stage : jobStage as TaskStage
  const currentProgress = progress?.stage === currentStage ? progress : null
  const aligningWords = currentStage === 'recognition' && currentProgress?.phase === 'alignment'

  return {
    currentStage,
    progress: currentProgress,
    aligningWords,
    label: aligningWords ? 'Выравнивание слов' : null,
    startedAt: aligningWords ? currentProgress?.phaseStartedAt ?? null : null,
  }
}
