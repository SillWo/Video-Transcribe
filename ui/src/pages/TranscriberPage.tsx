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

  const pairedFieldWrapperClass = 'grid min-w-0 gap-2'
  const pairedFieldLabelClass = 'flex items-center gap-2 text-sm font-medium text-stone-700'
  const pairedFieldControlClass =
    'h-[56px] w-full rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 outline-none transition focus:border-orange-300 focus:bg-white'

  return (
    <>
      <SettingsSummary
        sourceType={controller.sourceType}
        language={controller.language}
        model={controller.model}
        device={controller.device}
        cpuThreads={controller.device === 'cpu' ? controller.effectiveCpuThreads : null}
        outputFormat={controller.outputFormat}
        saveAudio={controller.saveAudio}
        useTimestamps={controller.effectiveTimestamps}
      />

      <StageProgress
        currentStage={controller.job?.stage ?? 'source'}
        status={controller.job?.status ?? 'queued'}
      />

      <div className="grid gap-6 xl:grid-cols-[540px_minmax(0,1fr)]">
        <SectionCard
          title={t('controls.title')}
          description={t('controls.description')}
          action={
            <div className="rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-stone-600">
              {statusLabel}
            </div>
          }
        >
          <form className="grid gap-5" onSubmit={controller.handleSubmit}>
            <div className="grid gap-3">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                {t('controls.sourceLabel')}
              </p>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => controller.setSourceType('url')}
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
                          className="inline-flex items-center justify-center rounded-full border border-current/15 bg-white/70 px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-current transition hover:bg-white"
                        >
                          {controller.isSourceCheckExpanded
                            ? t('controls.sourceCheck.actions.hideDetails')
                            : t('controls.sourceCheck.actions.showDetails')}
                        </button>
                      </div>
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
                    onChange={(event) => controller.setFile(event.target.files?.[0] ?? null)}
                    className="rounded-2xl border border-dashed border-stone-300 bg-stone-50 px-4 py-3 text-sm text-stone-600 file:mr-4 file:rounded-full file:border-0 file:bg-stone-900 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white"
                  />
                  <span className="text-xs text-stone-500">
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
                  value={controller.model}
                  onChange={(event) => controller.setModel(event.target.value)}
                  className={pairedFieldControlClass}
                >
                  {controller.options.models.map((item) => (
                    <option key={item} value={item}>
                      {item}
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
                  {controller.options.devices.map((item) => (
                    <option key={item} value={item}>
                      {getDeviceDisplayLabel(locale, item)}
                    </option>
                  ))}
                </select>
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
                  disabled={controller.device !== 'cpu'}
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
                  className={`relative inline-flex h-7 w-12 shrink-0 rounded-full p-1 transition ${controller.saveAudio ? 'bg-stone-900' : 'bg-stone-300'}`}
                >
                  <span
                    className={`absolute left-1 top-1 h-5 w-5 rounded-full bg-white transition-transform ${controller.saveAudio ? 'translate-x-5' : 'translate-x-0'}`}
                  />
                </button>
              </label>

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
                  onClick={() => {
                    if (controller.outputFormat !== 'srt') {
                      controller.setUseTimestamps((value) => !value)
                    }
                  }}
                  className={`relative inline-flex h-7 w-12 shrink-0 rounded-full p-1 transition ${controller.effectiveTimestamps ? 'bg-stone-900' : 'bg-stone-300'} ${controller.outputFormat === 'srt' ? 'cursor-not-allowed opacity-70' : ''}`}
                >
                  <span
                    className={`absolute left-1 top-1 h-5 w-5 rounded-full bg-white transition-transform ${controller.effectiveTimestamps ? 'translate-x-5' : 'translate-x-0'}`}
                  />
                </button>
              </label>
            </div>

            {errorMessage ? (
              <div className="rounded-3xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                {errorMessage}
              </div>
            ) : null}

            <button
              type="submit"
              disabled={controller.isSubmitting}
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

        <div className="grid gap-6">
          <LogPanel logText={controller.deferredLogText} />
          <ResultPanel
            text={controller.deferredTranscript}
            rawPreview={controller.deferredPreview}
            canDownload={Boolean(controller.job?.downloadUrl)}
            onCopy={() => {
              void controller.handleCopy()
            }}
            onDownload={controller.handleDownload}
          />
        </div>
      </div>

      <ModelSummaryPanel />
    </>
  )
}
