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
  cpu: {
    ru: 'CPU',
    en: 'CPU',
  },
  cuda: {
    ru: 'Видеокарта',
    en: 'GPU',
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
