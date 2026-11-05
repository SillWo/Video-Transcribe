import { useEffect, useMemo } from 'react'
import {
  Cpu,
  FileAudio,
  Globe,
  Languages,
  LoaderCircle,
  MicVocal,
  Save,
  Sparkles,
} from 'lucide-react'

import { LogPanel } from '../components/LogPanel'
import { ModelSummaryPanel } from '../components/ModelSummaryPanel'
import { ResultPanel } from '../components/ResultPanel'
import { SectionCard } from '../components/SectionCard'
import { SettingsSummary } from '../components/SettingsSummary'
import { StageProgress } from '../components/StageProgress'
import {
  getDeviceDisplayLabel,
  getLanguageDisplayLabel,
  getModelDisplayLabel,
  getOutputFormatDisplayLabel,
} from '../i18n/displayLabels'
import { useI18n } from '../i18n/useI18n'
import type { OutputFormat, TranscriberController } from '../hooks/useTranscriberController'

export function TranscriberPage(props: { controller: TranscriberController }) {
  const { locale, t } = useI18n()
  const controller = props.controller

  useEffect(() => {
    document.title = t('summary.title')
  }, [t])

  const statusLabel = useMemo(() => {
    if (!controller.job) {
      return t('status.idle')
    }
    return `${t(`status.job.${controller.job.status}`)} / ${t(`stages.${controller.job.stage}.label`)}`
  }, [controller.job, t])

  const errorMessage = useMemo(() => {
    if (!controller.error) {
      return null
    }

    return controller.error.kind === 'message' ? controller.error.message : t(controller.error.key)
  }, [controller.error, t])

  const sourceCheckStatusLabel = useMemo(() => {
    if (controller.isCheckingSource) {
      return t('controls.sourceCheck.status.checking')
    }
    if (!controller.sourceCheck) {
      return t('controls.sourceCheck.status.idle')
    }
    return controller.sourceCheck.ok
      ? t('controls.sourceCheck.status.ready')
      : t('controls.sourceCheck.status.failed')
  }, [controller.isCheckingSource, controller.sourceCheck, t])

  const sourceCheckToneClass = useMemo(() => {
    if (controller.isCheckingSource) {
      return 'border-sky-200 bg-sky-50 text-sky-900'
    }
    if (!controller.sourceCheck) {
      return 'border-stone-200 bg-stone-50 text-stone-700'
    }
    return controller.sourceCheck.ok
      ? 'border-emerald-200 bg-emerald-50 text-emerald-900'
      : 'border-rose-200 bg-rose-50 text-rose-900'
  }, [controller.isCheckingSource, controller.sourceCheck])

  function deviceReason(reason: string | null) {
    if (!reason) return ''
    const normalized = reason.toLowerCase()
    if (normalized.includes('runtime is missing')) return locale === 'ru' ? 'Среда Vulkan не установлена' : 'Vulkan runtime is missing'
    if (normalized.includes('no non-cpu vulkan gpu')) return locale === 'ru' ? 'Видеокарта Vulkan не найдена' : 'No Vulkan GPU detected'
    if (normalized.includes('cuda libraries are missing')) return locale === 'ru' ? 'Библиотеки CUDA не установлены — используйте Vulkan' : reason
    if (normalized.includes('no cuda device')) return locale === 'ru' ? 'Устройство CUDA не найдено' : 'No CUDA device detected'
    if (normalized.includes('always available')) return locale === 'ru' ? 'Всегда доступен' : 'Always available'
    return reason
  }

  const pairedFieldWrapperClass = 'grid min-w-0 gap-2'
  const pairedFieldLabelClass = 'flex items-center gap-2 text-sm font-medium text-stone-700'
  const pairedFieldControlClass =
    'h-[56px] w-full min-w-0 appearance-auto rounded-xl border border-stone-200 bg-white px-4 py-3 text-stone-900 [color-scheme:light] outline-none transition-[border-color,background-color] focus:border-orange-400'

  return (
    <>
      <SettingsSummary
        sourceType={controller.sourceType}
        language={controller.language}
        model={controller.selectedModel}
        device={controller.device}
        cpuThreads={controller.device !== 'cuda' ? controller.effectiveCpuThreads : null}
        outputFormat={controller.outputFormat}
        saveAudio={controller.saveAudio}
        useTimestamps={controller.effectiveTimestamps}
        restorePunctuation={controller.restorePunctuation}
      />

      {controller.optionsError ? (
        <div role="status" className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <span>{locale === 'ru' ? 'Не удалось загрузить параметры устройств.' : 'Could not load device options.'}</span>
          <button type="button" onClick={controller.retryOptions} className="rounded-xl border border-amber-300 bg-white px-4 py-2 font-semibold">
            {locale === 'ru' ? 'Повторить' : 'Try again'}
          </button>
        </div>
      ) : null}

      <StageProgress
        currentStage={controller.job?.stage ?? 'source'}
        status={controller.job?.status ?? 'idle'}
      />

        <div className="grid min-w-0 gap-6 xl:grid-cols-[minmax(420px,540px)_minmax(0,1fr)]">
        <SectionCard
          title={t('controls.title')}
          description={t('controls.description')}
          className="backdrop-blur-none"
          action={
            <div role="status" aria-live="polite" className="max-w-full break-words rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] text-stone-600">
              {statusLabel}
            </div>
          }
        >
          <form className="grid gap-5" onSubmit={controller.handleSubmit}>
            <div className="grid gap-3">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                {t('controls.sourceLabel')}
              </p>
              <div role="group" aria-label={t('controls.sourceLabel')} className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => controller.setSourceType('url')}
                  aria-pressed={controller.sourceType === 'url'}
                  className={`rounded-3xl border px-4 py-3 text-left transition ${controller.sourceType === 'url' ? 'border-orange-300 bg-orange-50 text-orange-800' : 'border-stone-200 bg-stone-50 text-stone-700 hover:bg-stone-100'}`}
                >
                  <Globe className="mb-2 h-5 w-5" />
                  <div className="font-medium">{t('controls.sourceOptions.url.title')}</div>
                  <div className="text-sm text-stone-500">
                    {t('controls.sourceOptions.url.description')}
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => controller.setSourceType('file')}
                  aria-pressed={controller.sourceType === 'file'}
                  className={`rounded-3xl border px-4 py-3 text-left transition ${controller.sourceType === 'file' ? 'border-orange-300 bg-orange-50 text-orange-800' : 'border-stone-200 bg-stone-50 text-stone-700 hover:bg-stone-100'}`}
                >
                  <FileAudio className="mb-2 h-5 w-5" />
                  <div className="font-medium">{t('controls.sourceOptions.file.title')}</div>
                  <div className="text-sm text-stone-500">
                    {t('controls.sourceOptions.file.description')}
                  </div>
                </button>
              </div>

              {controller.sourceType === 'url' ? (
                <div className="grid gap-3">
                  <label className="grid gap-2">
                    <span className="text-sm font-medium text-stone-700">
                      {t('controls.fields.url')}
                    </span>
                    <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_220px]">
                      <input
                        value={controller.url}
                        onChange={(event) => controller.setUrl(event.target.value)}
                        placeholder={t('controls.placeholders.url')}
                        className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 outline-none ring-0 transition placeholder:text-stone-400 focus:border-orange-300 focus:bg-white"
                      />
                      <button
                        type="button"
                        onClick={() => {
                          void controller.handleCheckSource(
                            t('errors.enterVideoUrl'),
                            t('controls.sourceCheck.messages.requestFailed'),
                          )
                        }}
                        disabled={!controller.normalizedUrl || controller.isCheckingSource}
                        className="inline-flex items-center justify-center gap-2 rounded-2xl border border-stone-200 bg-white px-4 py-3 text-sm font-semibold text-stone-800 transition hover:bg-stone-50 disabled:cursor-not-allowed disabled:border-stone-200 disabled:bg-stone-100 disabled:text-stone-400"
                      >
                        {controller.isCheckingSource ? (
                          <>
                            <LoaderCircle className="h-4 w-4 animate-spin" />
                            {t('controls.buttons.checkingSource')}
                          </>
                        ) : (
                          <>
                            <Globe className="h-4 w-4" />
                            {t('controls.buttons.checkSource')}
                          </>
                        )}
                      </button>
                    </div>
                  </label>

                  <div className={`rounded-3xl border px-4 py-4 ${sourceCheckToneClass}`}>
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <div className="text-xs font-semibold uppercase tracking-[0.18em]">
                          {t('controls.sourceCheck.title')}
                        </div>
                        <p className="mt-1 text-sm text-current/80">
                          {t('controls.sourceCheck.description')}
                        </p>
                      </div>
                      <div className="rounded-full border border-current/15 bg-white/70 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-current">
                        {sourceCheckStatusLabel}
                      </div>
                    </div>

                    {controller.sourceCheck && !controller.isCheckingSource ? (
                      <div className="mt-4">
                        <button
                          type="button"
                          onClick={() => controller.setIsSourceCheckExpanded((value) => !value)}
                          aria-expanded={controller.isSourceCheckExpanded}
                          className="inline-flex items-center justify-center rounded-full border border-current/15 bg-white/70 px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-current transition hover:bg-white"
                        >
                          {controller.isSourceCheckExpanded
                            ? t('controls.sourceCheck.actions.hideDetails')
                            : t('controls.sourceCheck.actions.showDetails')}
                        </button>
                      </div>
                    ) : null}

                    {controller.sourceCheck && !controller.isCheckingSource ? (
                      <p className="mt-3 text-sm font-medium" role="status" aria-live="polite">
                        {controller.sourceCheck.message}
                      </p>
                    ) : null}

                    {controller.sourceCheck && !controller.isCheckingSource && controller.isSourceCheckExpanded ? (
                      <div className="mt-4 grid gap-4 border-t border-current/10 pt-4">
                        <div className="text-sm font-medium">
                          {controller.sourceCheck.message}
                        </div>

                        <p className="text-xs text-current/75">
                          {t('controls.sourceCheck.hint')}
                        </p>

                        {controller.sourceCheck.sourceInfo ? (
                          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                            <div className="rounded-2xl border border-current/10 bg-white/70 px-3 py-3">
                              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-current/70">
                                {t('controls.sourceCheck.fields.title')}
                              </div>
                              <div className="mt-1 break-words text-sm font-medium text-current">
                                {controller.sourceCheck.sourceInfo.title ?? t('controls.sourceCheck.values.unavailable')}
                              </div>
                            </div>
                            <div className="rounded-2xl border border-current/10 bg-white/70 px-3 py-3">
                              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-current/70">
                                {t('controls.sourceCheck.fields.extractor')}
                              </div>
                              <div className="mt-1 break-words text-sm font-medium text-current">
                                {controller.sourceCheck.sourceInfo.site ??
                                  controller.sourceCheck.extractor ??
                                  t('controls.sourceCheck.values.unavailable')}
                              </div>
                            </div>
                            <div className="rounded-2xl border border-current/10 bg-white/70 px-3 py-3">
                              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-current/70">
                                {t('controls.sourceCheck.fields.id')}
                              </div>
                              <div className="mt-1 break-all text-sm font-medium text-current">
                                {controller.sourceCheck.sourceInfo.id ?? t('controls.sourceCheck.values.unavailable')}
                              </div>
                            </div>
                            <div className="rounded-2xl border border-current/10 bg-white/70 px-3 py-3">
                              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-current/70">
                                {t('controls.sourceCheck.fields.formats')}
                              </div>
                              <div className="mt-1 text-sm font-medium text-current">
                                {controller.sourceCheck.formatsAvailable
                                  ? t('controls.sourceCheck.values.formatsReady', {
                                      count: controller.sourceCheck.sourceInfo.formatsCount,
                                    })
                                  : t('controls.sourceCheck.values.noFormats')}
                              </div>
                            </div>
                            <div className="rounded-2xl border border-current/10 bg-white/70 px-3 py-3">
                              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-current/70">
                                {t('controls.sourceCheck.fields.audio')}
                              </div>
                              <div className="mt-1 text-sm font-medium text-current">
                                {controller.sourceCheck.audioExtractable
                                  ? t('controls.sourceCheck.values.audioReady')
                                  : t('controls.sourceCheck.values.audioMissing')}
                              </div>
                            </div>
                            <div className="rounded-2xl border border-current/10 bg-white/70 px-3 py-3">
                              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-current/70">
                                {t('controls.sourceCheck.fields.kind')}
                              </div>
                              <div className="mt-1 text-sm font-medium text-current">
                                {controller.sourceCheck.sourceInfo.isPlaylist
                                  ? t('controls.sourceCheck.values.playlist', {
                                      count: controller.sourceCheck.sourceInfo.checkedEntries ?? 0,
                                    })
                                  : t('controls.sourceCheck.values.single')}
                              </div>
                            </div>
                          </div>
                        ) : null}

                        {controller.sourceCheck.sourceInfo?.formatSample?.length ? (
                          <div className="grid gap-2">
                            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-current/70">
                              {t('controls.sourceCheck.fields.formatSample')}
                            </div>
                            <div className="flex flex-wrap gap-2">
                              {controller.sourceCheck.sourceInfo.formatSample.map((item) => (
                                <span
                                  key={item}
                                  className="rounded-full border border-current/10 bg-white/80 px-3 py-1 text-xs text-current"
                                >
                                  {item}
                                </span>
                              ))}
                            </div>
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : (
                <label className="grid gap-2">
                  <span className="text-sm font-medium text-stone-700">
                    {t('controls.fields.mediaFile')}
                  </span>
                  <input
                    type="file"
                    accept="audio/*,video/*"
                    aria-label={t('controls.fields.mediaFile')}
                    aria-describedby="media-file-caption"
                    onChange={(event) => controller.setFile(event.target.files?.[0] ?? null)}
                    className="rounded-2xl border border-dashed border-stone-300 bg-stone-50 px-4 py-3 text-sm text-transparent file:mr-4 file:rounded-full file:border-0 file:bg-stone-900 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white"
                  />
                  <span id="media-file-caption" className="text-xs text-stone-500">
                    {controller.file
                      ? t('controls.file.selected', { filename: controller.file.name })
                      : t('controls.file.empty')}
                  </span>
                </label>
              )}
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <label className={pairedFieldWrapperClass}>
                <span className={pairedFieldLabelClass}>
                  <Languages className="h-4 w-4" />
                  {t('controls.fields.language')}
                </span>
                <select
                  value={controller.language}
                  onChange={(event) => controller.setLanguage(event.target.value)}
                  className={pairedFieldControlClass}
                >
                  {controller.options.languages.map((item) => (
                    <option key={item} value={item}>
                      {getLanguageDisplayLabel(locale, item)}
                    </option>
                  ))}
                </select>
              </label>

              <label className={pairedFieldWrapperClass}>
                <span className={pairedFieldLabelClass}>
                  <MicVocal className="h-4 w-4" />
                  {t('controls.fields.model')}
                </span>
                <select
                  value={controller.selectedModel}
                  onChange={(event) => controller.setModel(event.target.value)}
                  className={pairedFieldControlClass}
                >
                  {controller.availableModels.map((item) => (
                    <option key={item} value={item}>
                      {getModelDisplayLabel(locale, item)}
                    </option>
                  ))}
                </select>
              </label>

              <label className={pairedFieldWrapperClass}>
                <span className={pairedFieldLabelClass}>
                  <Cpu className="h-4 w-4" />
                  {t('controls.fields.device')}
                </span>
                <select
                  value={controller.device}
                  onChange={(event) => controller.setDevice(event.target.value)}
                  className={pairedFieldControlClass}
                >
                  {controller.options.deviceDetails.map((item) => (
                    <option key={item.id} value={item.id} disabled={!item.available}>
                      {getDeviceDisplayLabel(locale, item.id)}
                      {item.id === 'vulkan' && item.available && item.name ? ` — ${item.name}` : ''}
                      {!item.available && item.reason ? ` — ${deviceReason(item.reason)}` : ''}
                    </option>
                  ))}
                </select>
                {controller.options.deviceDetails.some((item) => item.id === 'vulkan' && item.available) &&
                (controller.device === 'vulkan' || controller.device === 'auto') ? (
                  <span className="text-xs text-stone-600">
                    {locale === 'ru'
                      ? `Ускорение Vulkan доступно для видеокарты ${controller.options.deviceDetails.find((item) => item.id === 'vulkan')?.name ?? ''}. Поддерживаются NVIDIA и AMD, включая встроенную графику.`
                      : `Vulkan acceleration is available for ${controller.options.deviceDetails.find((item) => item.id === 'vulkan')?.name ?? 'this GPU'}. NVIDIA and AMD GPUs, including integrated graphics, are supported.`}
                  </span>
                ) : null}
              </label>

              <label className={pairedFieldWrapperClass}>
                <span className={pairedFieldLabelClass}>
                  <Cpu className="h-4 w-4" />
                  {t('controls.fields.cpuThreads')}
                </span>
                <input
                  type="number"
                  min={1}
                  max={controller.maxCpuThreads}
                  value={controller.effectiveCpuThreads}
                  disabled={controller.device === 'cuda'}
                  onChange={(event) => {
                    const nextValue = Number.parseInt(event.target.value, 10)
                    controller.setNproc(Number.isNaN(nextValue) ? 1 : nextValue)
                  }}
                  className={`${pairedFieldControlClass} disabled:cursor-not-allowed disabled:bg-stone-100 disabled:text-stone-400`}
                />
              </label>

              <label className="grid min-w-0 gap-2 md:col-span-2">
                <span className={pairedFieldLabelClass}>
                  <Sparkles className="h-4 w-4" />
                  {t('controls.fields.output')}
                </span>
                <select
                  value={controller.outputFormat}
                  onChange={(event) => {
                    const nextValue = event.target.value as OutputFormat
                    controller.setOutputFormat(nextValue)
                    if (nextValue === 'srt') {
                      controller.setUseTimestamps(true)
                    }
                  }}
                  className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 outline-none transition focus:border-orange-300 focus:bg-white"
                >
                  {controller.options.outputFormats.map((item) => (
                    <option key={item} value={item}>
                      {getOutputFormatDisplayLabel(locale, item)}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <p className="break-all rounded-xl bg-stone-50 px-3 py-2 text-xs text-stone-600">
              {locale === 'ru' ? 'Модели хранятся в папке проекта' : 'Models are stored in the project folder'}
              {controller.options.modelsDirectory ? `: ${controller.options.modelsDirectory}` : '.'}
              {(controller.device === 'vulkan' || (controller.device === 'auto' && controller.options.deviceDetails.some((item) => item.id === 'vulkan' && item.available))) && (
                <> {locale === 'ru'
                  ? 'Модель Vulkan загрузится автоматически при первой расшифровке.'
                  : 'The Vulkan model downloads automatically during the first transcription.'}</>
              )}
            </p>

            <div className="grid gap-3">
              <label className="flex items-center justify-between rounded-3xl border border-stone-200 bg-stone-50 px-4 py-3">
                <div>
                  <div className="font-medium text-stone-800">
                    {t('controls.toggles.saveAudio.title')}
                  </div>
                  <div className="text-sm text-stone-500">
                    {t('controls.toggles.saveAudio.description')}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => controller.setSaveAudio((value) => !value)}
                  role="switch"
                  aria-checked={controller.saveAudio}
                  aria-label={t('controls.toggles.saveAudio.title')}
                  className={`relative inline-flex min-h-0 h-7 w-12 shrink-0 rounded-full p-1 transition-[background-color] ${controller.saveAudio ? 'bg-stone-900' : 'bg-stone-300'}`}
                >
                  <span
                    className={`absolute left-1 top-1 h-5 w-5 rounded-full bg-white transition-transform ${controller.saveAudio ? 'translate-x-5' : 'translate-x-0'}`}
                  />
                </button>
              </label>

              {controller.language === 'ru' ? (
                <label className="flex items-center justify-between rounded-3xl border border-stone-200 bg-stone-50 px-4 py-3">
                  <div>
                    <div className="font-medium text-stone-800">
                      {t('controls.toggles.restorePunctuation.title')}
                    </div>
                    <div className="text-sm text-stone-500">
                      {t('controls.toggles.restorePunctuation.description')}
                    </div>
                    {!controller.options.punctuationAvailable ? (
                      <div className="text-xs text-stone-500">{t('controls.toggles.restorePunctuation.unavailable')}</div>
                    ) : null}
                  </div>
                  <button
                    type="button"
                    onClick={() => controller.setRestorePunctuation((value) => !value)}
                    role="switch"
                    aria-checked={controller.restorePunctuation}
                    aria-label={t('controls.toggles.restorePunctuation.title')}
                    disabled={!controller.options.punctuationAvailable}
                  className={`relative inline-flex min-h-0 h-7 w-12 shrink-0 rounded-full p-1 transition-[background-color] disabled:cursor-not-allowed disabled:opacity-60 ${controller.restorePunctuation ? 'bg-stone-900' : 'bg-stone-300'}`}
                  >
                    <span
                      className={`absolute left-1 top-1 h-5 w-5 rounded-full bg-white transition-transform ${controller.restorePunctuation ? 'translate-x-5' : 'translate-x-0'}`}
                    />
                  </button>
                </label>
              ) : null}

              <label className="flex items-center justify-between rounded-3xl border border-stone-200 bg-stone-50 px-4 py-3">
                <div>
                  <div className="font-medium text-stone-800">
                    {t('controls.toggles.timestamps.title')}
                  </div>
                  <div className="text-sm text-stone-500">
                    {controller.outputFormat === 'srt'
                      ? t('controls.toggles.timestamps.srtLocked')
                      : t('controls.toggles.timestamps.enabled')}
                  </div>
                </div>
                <button
                  type="button"
                  disabled={controller.outputFormat === 'srt'}
                  onClick={() => {
                    if (controller.outputFormat !== 'srt') {
                      controller.setUseTimestamps((value) => !value)
                    }
                  }}
                  role="switch"
                  aria-checked={controller.effectiveTimestamps}
                  aria-label={t('controls.toggles.timestamps.title')}
                  className={`relative inline-flex min-h-0 h-7 w-12 shrink-0 rounded-full p-1 transition-[background-color] disabled:cursor-not-allowed disabled:opacity-60 ${controller.effectiveTimestamps ? 'bg-stone-900' : 'bg-stone-300'} ${controller.outputFormat === 'srt' ? 'cursor-not-allowed opacity-70' : ''}`}
                >
                  <span
                    className={`absolute left-1 top-1 h-5 w-5 rounded-full bg-white transition-transform ${controller.effectiveTimestamps ? 'translate-x-5' : 'translate-x-0'}`}
                  />
                </button>
              </label>
            </div>

            {errorMessage ? (
              <div role="alert" className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                {errorMessage}
              </div>
            ) : null}

            <button
              type="submit"
              disabled={controller.isSubmitting || controller.job?.status === 'queued' || controller.job?.status === 'running'}
              className="inline-flex items-center justify-center gap-2 rounded-full bg-stone-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-stone-800 disabled:cursor-not-allowed disabled:bg-stone-400"
            >
              {controller.isSubmitting ? (
                <>
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                  {t('controls.buttons.starting')}
                </>
              ) : (
                <>
                  <Save className="h-4 w-4" />
                  {t('controls.buttons.start')}
                </>
              )}
            </button>
          </form>
        </SectionCard>

        <div className="grid min-w-0 self-start content-start gap-6">
          <ResultPanel
            text={controller.deferredTranscript}
            rawPreview={controller.deferredPreview}
            canDownload={Boolean(controller.job?.downloadUrl)}
            copyStatus={controller.copyStatus}
            device={controller.job?.effectiveDevice ?? controller.job?.effective_device ?? null}
            engine={controller.job?.effectiveBackend ?? controller.job?.effective_backend ?? controller.job?.backend ?? null}
            onCopy={() => {
              void controller.handleCopy()
            }}
            onDownload={controller.handleDownload}
          />
          <LogPanel logText={controller.deferredLogText} />
        </div>
      </div>

      <ModelSummaryPanel />
    </>
  )
}
