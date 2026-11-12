import assert from 'node:assert/strict'
import test from 'node:test'
import { canSubmitOnDevice, defaultDeviceForProfile, gpuReadinessLabel, isDeviceSelectable, profileDeviceCapability } from '../src/lib/asrCapabilities.ts'

const options = {
  defaultDevice: 'cpu' as const,
  devices: {
    cpu: { available: true, name: 'CPU' },
    gpu: { available: false, name: 'Legacy GPU', canPrepare: false, state: 'unavailable' },
  },
  profiles: [
    { id: 'quality', label: 'Качество', model: 'qwen3-asr-1.7b', defaultDevice: 'gpu' as const,
      devices: { cpu: { available: true, name: 'CPU' }, gpu: { available: false, name: 'Radeon', canPrepare: true, state: 'needs_preparation', backend: 'vulkan', reason: 'Подготовится автоматически' } } },
    { id: 'balanced', label: 'Баланс', model: 'parakeet-tdt-0.6b-v3', defaultDevice: 'cpu' as const,
      devices: { cpu: { available: true, name: 'CPU' }, gpu: { available: true, name: 'Radeon', canPrepare: true, state: 'ready', backend: 'directml' } } },
    { id: 'economy', label: 'Экономный', model: 'small', defaultDevice: 'gpu' as const,
      devices: { cpu: { available: true, name: 'CPU' }, gpu: { available: false, name: 'GPU', canPrepare: false, state: 'unavailable', reason: 'Vulkan unavailable' } } },
  ],
}

test('profile switching reads that profile GPU capability and preserves an explicit device choice', () => {
  const explicitlySelected: 'cpu' | 'gpu' = 'cpu'
  assert.equal(profileDeviceCapability(options, 'quality', 'gpu')?.backend, 'vulkan')
  assert.equal(profileDeviceCapability(options, 'balanced', 'gpu')?.backend, 'directml')
  assert.equal(defaultDeviceForProfile(options, 'quality'), 'gpu')
  assert.equal(defaultDeviceForProfile(options, 'balanced'), 'cpu')
  // Profile changes update the profile only; the user-selected device remains selected.
  const nextPrefs = { profile: 'economy', device: explicitlySelected }
  assert.equal(nextPrefs.device, 'cpu')
  assert.equal(canSubmitOnDevice(options, nextPrefs.profile, nextPrefs.device), true)
})

test('GPU is selectable while it can prepare, and unavailable saved GPU blocks submit', () => {
  const preparable = profileDeviceCapability(options, 'quality', 'gpu')
  const unavailable = profileDeviceCapability(options, 'economy', 'gpu')
  assert.equal(isDeviceSelectable(preparable), true)
  assert.equal(gpuReadinessLabel(preparable), 'Подготовится автоматически')
  assert.equal(isDeviceSelectable(unavailable), false)
  assert.equal(canSubmitOnDevice(options, 'economy', 'gpu'), false)
})

test('legacy options without per-profile capabilities keep top-level compatibility', () => {
  const legacy = { defaultDevice: 'gpu' as const, devices: options.devices, profiles: [{ id: 'balanced', label: 'Баланс', model: 'small' }] }
  assert.equal(profileDeviceCapability(legacy, 'balanced', 'gpu')?.name, 'Legacy GPU')
  assert.equal(defaultDeviceForProfile(legacy, 'balanced'), 'cpu')
})
