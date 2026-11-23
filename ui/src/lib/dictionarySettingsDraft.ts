import { dictionaryApi } from './dictionaryApi'

export type DictionarySettingsDraft = { value: boolean; savedValue: boolean; loaded: boolean; busy: boolean; error: string }
let state: DictionarySettingsDraft = { value: true, savedValue: true, loaded: false, busy: false, error: '' }
let loadTask: Promise<void> | null = null
const listeners = new Set<() => void>()

function publish(next: DictionarySettingsDraft) { state = next; for (const listener of listeners) listener() }
export function getDictionarySettingsDraft() { return state }
export function subscribeDictionarySettingsDraft(listener: () => void) { listeners.add(listener); return () => listeners.delete(listener) }

export function loadDictionarySettingsDraft() {
  if (state.loaded) return Promise.resolve()
  if (loadTask) return loadTask
  loadTask = dictionaryApi.getSettings().then((result) => {
    publish({ value: result.auto_learn, savedValue: result.auto_learn, loaded: true, busy: false, error: '' })
  }).catch((reason: unknown) => {
    publish({ ...state, loaded: true, busy: false, error: reason instanceof Error ? reason.message : 'Не удалось загрузить настройку' })
  }).finally(() => { loadTask = null })
  return loadTask
}

export function setDictionaryAutoLearn(value: boolean) { publish({ ...state, value, error: '' }) }

export async function commitDictionarySettingsDraft() {
  await loadDictionarySettingsDraft()
  if (!state.loaded || state.error && !state.savedValue && !state.value) throw new Error(state.error || 'Настройка словаря не загружена')
  if (state.value === state.savedValue) return
  const target = state.value
  publish({ ...state, busy: true, error: '' })
  try {
    const result = await dictionaryApi.saveSettings(target)
    publish({ ...state, value: state.value, savedValue: result.auto_learn, busy: false, error: '' })
  } catch (reason) {
    const message = reason instanceof Error ? reason.message : 'Не удалось сохранить настройку словаря'
    publish({ ...state, busy: false, error: message })
    throw new Error(message)
  }
}
