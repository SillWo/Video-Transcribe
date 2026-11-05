import { getLanguageName, type UiLocale } from './translations'

type LocalizedLabelMap = Record<string, Record<UiLocale, string>>

export const languageDisplayLabels: LocalizedLabelMap = {
  auto: {
    ru: 'Автоопределение',
    en: 'Auto detect',
  },
  ru: {
    ru: 'Русский',
    en: 'Russian',
  },
  en: {
    ru: 'English',
    en: 'English',
  },
}

export const deviceDisplayLabels: LocalizedLabelMap = {
  auto: { ru: 'Автоматически', en: 'Automatic' },
  vulkan: { ru: 'Видеокарта (Vulkan)', en: 'GPU (Vulkan)' },
  cpu: {
    ru: 'CPU',
    en: 'CPU',
  },
  cuda: {
    ru: 'CUDA (NVIDIA)',
    en: 'CUDA (NVIDIA)',
  },
}

export const outputFormatDisplayLabels: LocalizedLabelMap = {
  txt: {
    ru: 'Текстовый файл (TXT)',
    en: 'Text file (TXT)',
  },
  srt: {
    ru: 'Файл субтитров (SRT)',
    en: 'Subtitle file (SRT)',
  },
  json: {
    ru: 'Структурированные данные (JSON)',
    en: 'Structured data (JSON)',
  },
}

function getMappedLabel(
  locale: UiLocale,
  value: string,
  labels: LocalizedLabelMap,
  fallback: (value: string) => string,
) {
  return labels[value]?.[locale] ?? fallback(value)
}

export function getLanguageDisplayLabel(locale: UiLocale, value: string) {
  return getMappedLabel(locale, value, languageDisplayLabels, (code) =>
    getLanguageName(locale, code),
  )
}

export function getDeviceDisplayLabel(locale: UiLocale, value: string) {
  return getMappedLabel(locale, value, deviceDisplayLabels, (code) => code.toUpperCase())
}

export function getOutputFormatDisplayLabel(locale: UiLocale, value: string) {
  return getMappedLabel(locale, value, outputFormatDisplayLabels, (code) => code.toUpperCase())
}

export function getModelDisplayLabel(locale: UiLocale, value: string) {
  return value === 'large-v3-turbo'
    ? locale === 'ru'
      ? 'Large v3 Turbo — быстрее'
      : 'Large v3 Turbo — faster'
    : value
}
