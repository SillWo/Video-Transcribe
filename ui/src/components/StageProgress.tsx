import { CheckCircle2, CircleDashed, LoaderCircle, XCircle } from 'lucide-react'

import { cn } from '../lib/utils'
import { useI18n } from '../i18n/useI18n'

type StageId = 'source' | 'audio' | 'recognition' | 'result'

type StageProgressProps = {
  currentStage: StageId
  status: 'queued' | 'running' | 'completed' | 'failed'
}

export function StageProgress({ currentStage, status }: StageProgressProps) {
  const { t } = useI18n()
  const stages: Array<{ id: StageId; label: string; description: string }> = [
    {
      id: 'source',
      label: t('stages.source.label'),
      description: t('stages.source.description'),
    },
    {
      id: 'audio',
      label: t('stages.audio.label'),
      description: t('stages.audio.description'),
    },
    {
      id: 'recognition',
      label: t('stages.recognition.label'),
      description: t('stages.recognition.description'),
    },
    {
      id: 'result',
      label: t('stages.result.label'),
      description: t('stages.result.description'),
    },
  ]
  const currentIndex = stages.findIndex((stage) => stage.id === currentStage)

  return (
    <div className="grid gap-3 rounded-[28px] border border-stone-200 bg-white/85 p-4 shadow-[0_12px_40px_-24px_rgba(41,37,36,0.35)] backdrop-blur md:grid-cols-4">
      {stages.map((stage, index) => {
        const isDone = status === 'completed' || index < currentIndex
        const isActive = status === 'running' && index === currentIndex
        const isFailed = status === 'failed' && index === currentIndex

        return (
          <div
            key={stage.id}
            className={cn(
              'rounded-3xl border p-4 transition',
              isDone && 'border-emerald-200 bg-emerald-50',
              isActive && 'border-orange-200 bg-orange-50',
              isFailed && 'border-rose-200 bg-rose-50',
              !isDone && !isActive && !isFailed && 'border-stone-200 bg-stone-50',
            )}
          >
            <div className="mb-3 flex items-center gap-2">
              {isDone ? (
                <CheckCircle2 className="h-5 w-5 text-emerald-600" />
              ) : isActive ? (
                <LoaderCircle className="h-5 w-5 animate-spin text-orange-600" />
              ) : isFailed ? (
                <XCircle className="h-5 w-5 text-rose-600" />
              ) : (
                <CircleDashed className="h-5 w-5 text-stone-400" />
              )}
              <span className="text-sm font-semibold text-stone-900">{stage.label}</span>
            </div>
            <p className="text-sm text-stone-500">{stage.description}</p>
          </div>
        )
      })}
    </div>
  )
}
