import {
  useEffect,
  useMemo,
  useState,
  type PropsWithChildren,
} from 'react'

import { I18nContext, type I18nContextValue } from './context'
import {
  DEFAULT_UI_LOCALE,
  UI_LOCALE_STORAGE_KEY,
  isUiLocale,
  translate,
  type UiLocale,
} from './translations'

function readInitialLocale(): UiLocale {
  if (typeof window === 'undefined') {
    return DEFAULT_UI_LOCALE
  }

  const storedLocale = window.localStorage.getItem(UI_LOCALE_STORAGE_KEY)
  return storedLocale && isUiLocale(storedLocale) ? storedLocale : DEFAULT_UI_LOCALE
}

export function I18nProvider({ children }: PropsWithChildren) {
  const [locale, setLocale] = useState<UiLocale>(readInitialLocale)

  useEffect(() => {
    window.localStorage.setItem(UI_LOCALE_STORAGE_KEY, locale)
    document.documentElement.lang = locale
  }, [locale])

  const value = useMemo<I18nContextValue>(
    () => ({
      locale,
      setLocale,
      t: (key, params, fallback) => translate(locale, key, params, fallback),
    }),
    [locale],
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}
