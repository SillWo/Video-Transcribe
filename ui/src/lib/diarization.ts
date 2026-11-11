export type DiarizationPrefs = { diarizationEnabled: boolean; diarizationDevice: string }
export type StoredPrefs = Partial<{ profile: 'quality' | 'balanced' | 'economy'; device: 'cpu' | 'gpu'; cpuThreads: number; restorePunctuation: boolean }> & DiarizationPrefs

export const defaultDiarizationPrefs: DiarizationPrefs = {
  diarizationEnabled: false,
  diarizationDevice: 'auto',
}

const MAX_DEVICE_ID_LENGTH = 512

function hasControlCharacter(value: string) {
  return Array.from(value).some((character) => {
    const code = character.charCodeAt(0)
    return code < 32 || (code >= 127 && code <= 159)
  })
}

export function sanitizeDiarizationPrefs(value: unknown): DiarizationPrefs {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return { ...defaultDiarizationPrefs }
  const stored = value as Record<string, unknown>
  const device = stored.diarizationDevice
  return {
    diarizationEnabled: typeof stored.diarizationEnabled === 'boolean' ? stored.diarizationEnabled : false,
    diarizationDevice: typeof device === 'string' && device.trim().length > 0 && device.length <= MAX_DEVICE_ID_LENGTH && !hasControlCharacter(device)
      ? device
      : 'auto',
  }
}

export function sanitizeStoredPrefs(value: unknown): StoredPrefs {
  const stored = typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
  const result: StoredPrefs = {
    profile: stored.profile === 'quality' || stored.profile === 'balanced' || stored.profile === 'economy' ? stored.profile : 'balanced',
    restorePunctuation: typeof stored.restorePunctuation === 'boolean' ? stored.restorePunctuation : true,
    ...sanitizeDiarizationPrefs(stored),
  }
  if (stored.device === 'cpu' || stored.device === 'gpu') result.device = stored.device
  if (typeof stored.cpuThreads === 'number' && Number.isFinite(stored.cpuThreads)) result.cpuThreads = stored.cpuThreads
  return result
}

export function diarizationRequestFields(prefs: DiarizationPrefs) {
  return {
    diarizationEnabled: prefs.diarizationEnabled,
    diarizationDevice: prefs.diarizationDevice,
  }
}

export function stageOrderFor(enabled: boolean) {
  return enabled
    ? ['preparation', 'source', 'audio', 'recognition', 'diarization', 'postprocess', 'result'] as const
    : ['preparation', 'source', 'audio', 'recognition', 'postprocess', 'result'] as const
}
