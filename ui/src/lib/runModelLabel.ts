export type RunModelProfile = 'quality' | 'balanced' | 'economy'
export type RunModelStatus = 'queued' | 'running' | 'cancelling' | 'completed' | 'failed' | 'interrupted' | 'cancelled'

const currentProfileLabels: Record<RunModelProfile, string> = {
  quality: 'Qwen3-ASR',
  balanced: 'Whisper large-v3',
  economy: 'Whisper small',
}

const knownModelLabels: Record<string, string> = {
  'qwen3-asr-1.7b': 'Qwen3-ASR 1.7B',
  'parakeet-tdt-0.6b-v3': 'Parakeet TDT 0.6B v3',
  'nemo-parakeet-tdt-0.6b-v3': 'Parakeet TDT 0.6B v3',
  'large-v3-turbo': 'Whisper large-v3-turbo',
  'large-v3': 'Whisper large-v3',
  small: 'Whisper small',
}

export function runModelDisplayLabel(input: {
  status: RunModelStatus
  profile: RunModelProfile
  model?: string | null
  effectiveModel?: string | null
}) {
  const actualModel = input.effectiveModel?.trim() || input.model?.trim()
  if (actualModel) return knownModelLabels[actualModel.toLowerCase()] ?? actualModel
  return input.status === 'completed' ? 'Модель не указана' : currentProfileLabels[input.profile]
}
