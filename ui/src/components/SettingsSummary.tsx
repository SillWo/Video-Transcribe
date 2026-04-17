import {
  getDeviceDisplayLabel,
  getLanguageDisplayLabel,
  getOutputFormatDisplayLabel,
} from '../i18n/displayLabels'
import { supportedUiLocales } from '../i18n/translations'
import { useI18n } from '../i18n/useI18n'

type SettingsSummaryProps = {
  sourceType: 'url' | 'file'
  language: string
  model: string
  device: string
  cpuThreads: number | null
  outputFormat: 'txt' | 'srt' | 'json'
  saveAudio: boolean
  useTimestamps: boolean
  restorePunctuation: boolean
}

const itemClass =
  'rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-xs font-medium uppercase tracking-[0.18em] text-stone-600'

export function SettingsSummary(props: SettingsSummaryProps) {
  const { locale, setLocale, t } = useI18n()

  return (
    <div className="rounded-[28px] border border-stone-200 bg-white/85 p-4 shadow-[0_12px_40px_-24px_rgba(41,37,36,0.35)] backdrop-blur">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-orange-600">
            {t('summary.eyebrow')}
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-stone-900">
            {t('summary.title')}
          </h1>
        </div>
        <div className="flex flex-col items-end gap-3 sm:flex-row sm:items-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-stone-200 bg-stone-50 px-2 py-1">
            <span className="px-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500">
              {t('languageSwitcher.label')}
            </span>
            <div className="flex rounded-full bg-white p-1 shadow-sm">
              {supportedUiLocales.map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setLocale(item)}
                  aria-pressed={locale === item}
                  className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                    locale === item
                      ? 'bg-stone-900 text-white'
                      : 'text-stone-600 hover:bg-stone-100'
                  }`}
                >
                  {t(`languageSwitcher.${item}`)}
                </button>
              ))}
            </div>
          </div>
          <div className="rounded-full border border-orange-200 bg-orange-50 px-3 py-1 text-xs font-semibold text-orange-700">
            {t('summary.badge')}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <span className={itemClass}>
          {t('summary.fields.source')}: {t(`summary.values.sourceType.${props.sourceType}`)}
        </span>
        <span className={itemClass}>
          {t('summary.fields.lang')}: {getLanguageDisplayLabel(locale, props.language)}
        </span>
        <span className={itemClass}>{t('summary.fields.model')}: {props.model}</span>
        <span className={itemClass}>
          {t('summary.fields.device')}: {getDeviceDisplayLabel(locale, props.device)}
        </span>
        {props.device === 'cpu' && props.cpuThreads ? (
          <span className={itemClass}>
            {t('summary.fields.cores')}: {props.cpuThreads}
          </span>
        ) : null}
        <span className={itemClass}>
          {t('summary.fields.format')}: {getOutputFormatDisplayLabel(locale, props.outputFormat)}
        </span>
        <span className={itemClass}>
          {t('summary.fields.timestamps')}:{' '}
          {props.outputFormat === 'srt'
            ? t('summary.values.timestamps.forced')
            : props.useTimestamps
              ? t('summary.values.timestamps.on')
              : t('summary.values.timestamps.off')}
        </span>
        <span className={itemClass}>
          {t('summary.fields.keepAudio')}:{' '}
          {props.saveAudio ? t('summary.values.toggles.on') : t('summary.values.toggles.off')}
        </span>
        <span className={itemClass}>
          {t('summary.fields.punctuation')}:{' '}
          {props.language === 'ru'
            ? props.restorePunctuation
              ? t('summary.values.punctuation.on')
              : t('summary.values.punctuation.off')
            : t('summary.values.punctuation.unavailable')}
        </span>
      </div>
    </div>
  )
}
