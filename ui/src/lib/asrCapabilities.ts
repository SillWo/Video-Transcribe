export type AsrDevice = 'cpu' | 'gpu'
export type AsrDeviceCapability = {
  available: boolean
  name: string
  reason?: string | null
  backend?: string | null
  state?: 'needs_preparation' | 'checking' | 'ready' | 'unavailable' | string
  canPrepare?: boolean
}
export type AsrProfileCapability = {
  id: string
  label: string
  model: string
  devices?: Partial<Record<AsrDevice, AsrDeviceCapability>>
  defaultDevice?: AsrDevice
}
export type AsrOptions = {
  profiles: AsrProfileCapability[]
  devices: Record<AsrDevice, AsrDeviceCapability>
  defaultDevice: AsrDevice
}

export function profileDeviceCapability(options: AsrOptions | null | undefined, profile: string, device: AsrDevice) {
  const profileOptions = options?.profiles.find((item) => item.id === profile)
  return profileOptions?.devices?.[device] ?? options?.devices[device] ?? null
}

export function isDeviceSelectable(capability: AsrDeviceCapability | null | undefined) {
  return Boolean(capability && (capability.available || capability.canPrepare))
}

export function defaultDeviceForProfile(options: AsrOptions | null | undefined, profile: string): AsrDevice {
  const profileOptions = options?.profiles.find((item) => item.id === profile)
  if (profileOptions?.defaultDevice) return profileOptions.defaultDevice
  const gpu = profileDeviceCapability(options, profile, 'gpu')
  return isDeviceSelectable(gpu) ? 'gpu' : 'cpu'
}

export function canSubmitOnDevice(options: AsrOptions | null | undefined, profile: string, device: AsrDevice) {
  if (device === 'cpu') return true
  return isDeviceSelectable(profileDeviceCapability(options, profile, 'gpu'))
}

export function gpuReadinessLabel(capability: AsrDeviceCapability | null | undefined) {
  if (!capability || capability.state === 'unavailable') return 'Недоступно'
  if (capability.state === 'checking') return 'Проверяется'
  if (capability.state === 'needs_preparation') return 'Подготовится автоматически'
  return capability.available ? 'Готово' : 'Недоступно'
}
