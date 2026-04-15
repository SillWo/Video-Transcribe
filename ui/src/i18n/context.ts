import { createContext } from 'react'

import type { TranslationParams, UiLocale } from './translations'

export type I18nContextValue = {
  locale: UiLocale
  setLocale: (locale: UiLocale) => void
  t: (key: string, params?: TranslationParams, fallback?: string) => string
}

export const I18nContext = createContext<I18nContextValue | null>(null)
