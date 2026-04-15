import { startTransition, useDeferredValue, useEffect, useMemo, useState } from 'react'
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

import { LogPanel } from './components/LogPanel'
import { ResultPanel } from './components/ResultPanel'
import { SectionCard } from './components/SectionCard'
import { SettingsSummary } from './components/SettingsSummary'
import { StageProgress } from './components/StageProgress'
import {
  getDeviceDisplayLabel,
  getLanguageDisplayLabel,
  getOutputFormatDisplayLabel,
} from './i18n/displayLabels'
import { useI18n } from './i18n/useI18n'

type SourceType = 'url' | 'file'
type OutputFormat = 'txt' | 'srt' | 'json'
type JobStatus = 'queued' | 'running' | 'completed' | 'failed'
type StageId = 'source' | 'audio' | 'recognition' | 'result'

type JobResponse = {
  id: string
  settings: {
    sourceType: SourceType
    url: string
    language: string
    model: string
    device: string
    nproc?: number
    outputFormat: OutputFormat
    saveAudio: boolean
    useTimestamps: boolean
  }
  status: JobStatus
  stage: StageId
  logs: string[]
  output_path: string | null
  output_format: string | null
  result_text: string | null
  rendered_result: string | null
  detected_language: string | null
  error: string | null
  downloadUrl: string | null
}

type OptionsResponse = {
  models: string[]
  devices: string[]
  outputFormats: OutputFormat[]
  languages: string[]
  maxCpuThreads: number
}

type UiError =
  | {
      kind: 'message'
      message: string
    }
  | {
      kind: 'translation'
      key: string
    }

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''
const FALLBACK_CPU_THREADS =
  typeof navigator !== 'undefined' ? Math.max(navigator.hardwareConcurrency || 1, 1) : 1
const DEFAULT_LANGUAGE_OPTIONS = ['ru', 'auto', 'en', 'de', 'fr', 'es', 'it', 'pt', 'uk', 'ja', 'ko', 'zh']
const DEFAULT_OPTIONS: OptionsResponse = {
  models: ['small', 'medium', 'large-v3'],
  devices: ['cpu', 'cuda'],
  outputFormats: ['txt', 'srt', 'json'],
  languages: DEFAULT_LANGUAGE_OPTIONS,
  maxCpuThreads: FALLBACK_CPU_THREADS,
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    throw new Error(await response.text())
  }
  return response.json() as Promise<T>
}

function mergeOrderedValues(preferred: string[], incoming?: string[]) {
  const seen = new Set<string>()
  const result: string[] = []

  for (const value of [...preferred, ...(incoming ?? [])]) {
    if (!value || seen.has(value)) {
      continue
    }

    seen.add(value)
    result.push(value)
  }

  return result
}

function normalizeOptionsResponse(payload: Partial<OptionsResponse> | null | undefined): OptionsResponse {
  return {
    models: payload?.models?.length ? payload.models : DEFAULT_OPTIONS.models,
    devices: mergeOrderedValues(DEFAULT_OPTIONS.devices, payload?.devices),
    outputFormats: mergeOrderedValues(DEFAULT_OPTIONS.outputFormats, payload?.outputFormats) as OutputFormat[],
    languages: mergeOrderedValues(DEFAULT_LANGUAGE_OPTIONS, payload?.languages),
    maxCpuThreads: Math.max(payload?.maxCpuThreads ?? 0, FALLBACK_CPU_THREADS, 1),
  }
}

export default function App() {
  const { locale, t } = useI18n()
  const [sourceType, setSourceType] = useState<SourceType>('url')
  const [url, setUrl] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [language, setLanguage] = useState('ru')
  const [model, setModel] = useState('small')
  const [device, setDevice] = useState('cpu')
  const [nproc, setNproc] = useState(FALLBACK_CPU_THREADS)
  const [outputFormat, setOutputFormat] = useState<OutputFormat>('srt')
  const [saveAudio, setSaveAudio] = useState(false)
  const [useTimestamps, setUseTimestamps] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<UiError | null>(null)
  const [job, setJob] = useState<JobResponse | null>(null)
  const [options, setOptions] = useState<OptionsResponse>(DEFAULT_OPTIONS)

  const deferredLogText = useDeferredValue(job?.logs.join('\n') ?? '')
  const deferredTranscript = useDeferredValue(job?.result_text ?? '')
  const deferredPreview = useDeferredValue(job?.rendered_result ?? '')
  const currentJobId = job?.id
  const currentJobStatus = job?.status

  useEffect(() => {
    document.title = t('summary.title')
  }, [t])

  useEffect(() => {
    let active = true

    async function loadOptions() {
      try {
        const payload = await fetchJson<Partial<OptionsResponse>>('/api/options')
        if (!active) {
          return
        }

        startTransition(() => {
          const normalizedOptions = normalizeOptionsResponse(payload)
          setOptions(normalizedOptions)
          setModel((currentModel) =>
            normalizedOptions.models.length > 0 && !normalizedOptions.models.includes(currentModel)
              ? normalizedOptions.models[0]
              : currentModel,
          )
          setNproc((currentThreads) =>
            Math.min(Math.max(currentThreads, 1), normalizedOptions.maxCpuThreads),
          )
        })
      } catch {
        // Keep the built-in fallback options for local dev if backend is not ready yet.
      }
    }

    void loadOptions()
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!currentJobId || (currentJobStatus !== 'queued' && currentJobStatus !== 'running')) {
      return
    }

    const timer = window.setInterval(async () => {
      try {
        const nextJob = await fetchJson<JobResponse>(`/api/jobs/${currentJobId}`)
        startTransition(() => {
          setJob(nextJob)
        })

        if (nextJob.status === 'failed') {
          setError(
            nextJob.error
              ? { kind: 'message', message: nextJob.error }
              : { kind: 'translation', key: 'errors.jobFailed' },
          )
        }
      } catch (fetchError) {
        setError(
          fetchError instanceof Error
            ? { kind: 'message', message: fetchError.message }
            : { kind: 'translation', key: 'errors.unableRefreshJob' },
        )
      }
    }, 1000)

    return () => window.clearInterval(timer)
  }, [currentJobId, currentJobStatus])

  const effectiveTimestamps = outputFormat === 'srt' ? true : useTimestamps
  const maxCpuThreads = Math.max(options.maxCpuThreads || 1, 1)
  const effectiveCpuThreads = Math.min(Math.max(nproc, 1), maxCpuThreads)
  const pairedFieldWrapperClass = 'grid min-w-0 gap-2'
  const pairedFieldLabelClass = 'flex items-center gap-2 text-sm font-medium text-stone-700'
  const pairedFieldControlClass =
    'h-[56px] w-full rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 outline-none transition focus:border-orange-300 focus:bg-white'

  const statusLabel = useMemo(() => {
    if (!job) {
      return t('status.idle')
    }
    return `${t(`status.job.${job.status}`)} / ${t(`stages.${job.stage}.label`)}`
  }, [job, t])

  const errorMessage = useMemo(() => {
    if (!error) {
      return null
    }

    return error.kind === 'message' ? error.message : t(error.key)
  }, [error, t])

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)

    if (sourceType === 'url' && !url.trim()) {
      setError({ kind: 'translation', key: 'errors.enterVideoUrl' })
      return
    }

    if (sourceType === 'file' && !file) {
      setError({ kind: 'translation', key: 'errors.chooseLocalFile' })
      return
    }

    const formData = new FormData()
    formData.append('sourceType', sourceType)
    formData.append('url', url.trim())
    formData.append('language', language)
    formData.append('model', model)
    formData.append('device', device)
    if (device === 'cpu') {
      formData.append('nproc', String(effectiveCpuThreads))
    }
    formData.append('outputFormat', outputFormat)
    formData.append('saveAudio', String(saveAudio))
    formData.append('useTimestamps', String(effectiveTimestamps))

    if (sourceType === 'file' && file) {
      formData.append('file', file)
    }

    try {
      setIsSubmitting(true)
      const createdJob = await fetchJson<JobResponse>('/api/transcriptions', {
        method: 'POST',
        body: formData,
      })
      setJob(createdJob)
    } catch (submitError) {
      setError(
        submitError instanceof Error
          ? { kind: 'message', message: submitError.message }
          : { kind: 'translation', key: 'errors.unableStart' },
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  async function handleCopy() {
    if (!deferredTranscript) {
      return
    }
    await navigator.clipboard.writeText(deferredTranscript)
  }

  function handleDownload() {
    if (!job?.downloadUrl) {
      return
    }
    window.open(`${API_BASE}${job.downloadUrl}`, '_blank', 'noopener,noreferrer')
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(251,146,60,0.20),_transparent_30%),radial-gradient(circle_at_top_right,_rgba(56,189,248,0.15),_transparent_28%),linear-gradient(180deg,_#fffaf5_0%,_#f5f5f4_48%,_#fafaf9_100%)] px-4 py-6 text-stone-800 md:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <SettingsSummary
          sourceType={sourceType}
          language={language}
          model={model}
          device={device}
          cpuThreads={device === 'cpu' ? effectiveCpuThreads : null}
          outputFormat={outputFormat}
          saveAudio={saveAudio}
          useTimestamps={effectiveTimestamps}
        />

        <StageProgress
          currentStage={job?.stage ?? 'source'}
          status={job?.status ?? 'queued'}
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
            <form className="grid gap-5" onSubmit={handleSubmit}>
              <div className="grid gap-3">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                  {t('controls.sourceLabel')}
                </p>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setSourceType('url')}
                    className={`rounded-3xl border px-4 py-3 text-left transition ${sourceType === 'url' ? 'border-orange-300 bg-orange-50 text-orange-800' : 'border-stone-200 bg-stone-50 text-stone-700 hover:bg-stone-100'}`}
                  >
                    <Globe className="mb-2 h-5 w-5" />
                    <div className="font-medium">{t('controls.sourceOptions.url.title')}</div>
                    <div className="text-sm text-stone-500">
                      {t('controls.sourceOptions.url.description')}
                    </div>
                  </button>
                  <button
                    type="button"
                    onClick={() => setSourceType('file')}
                    className={`rounded-3xl border px-4 py-3 text-left transition ${sourceType === 'file' ? 'border-orange-300 bg-orange-50 text-orange-800' : 'border-stone-200 bg-stone-50 text-stone-700 hover:bg-stone-100'}`}
                  >
                    <FileAudio className="mb-2 h-5 w-5" />
                    <div className="font-medium">{t('controls.sourceOptions.file.title')}</div>
                    <div className="text-sm text-stone-500">
                      {t('controls.sourceOptions.file.description')}
                    </div>
                  </button>
                </div>

                {sourceType === 'url' ? (
                  <label className="grid gap-2">
                    <span className="text-sm font-medium text-stone-700">
                      {t('controls.fields.url')}
                    </span>
                    <input
                      value={url}
                      onChange={(event) => setUrl(event.target.value)}
                      placeholder={t('controls.placeholders.url')}
                      className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 outline-none ring-0 transition placeholder:text-stone-400 focus:border-orange-300 focus:bg-white"
                    />
                  </label>
                ) : (
                  <label className="grid gap-2">
                    <span className="text-sm font-medium text-stone-700">
                      {t('controls.fields.mediaFile')}
                    </span>
                    <input
                      type="file"
                      accept="audio/*,video/*"
                      onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                      className="rounded-2xl border border-dashed border-stone-300 bg-stone-50 px-4 py-3 text-sm text-stone-600 file:mr-4 file:rounded-full file:border-0 file:bg-stone-900 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white"
                    />
                    <span className="text-xs text-stone-500">
                      {file
                        ? t('controls.file.selected', { filename: file.name })
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
                    value={language}
                    onChange={(event) => setLanguage(event.target.value)}
                    className={pairedFieldControlClass}
                  >
                    {options.languages.map((item) => (
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
                    value={model}
                    onChange={(event) => setModel(event.target.value)}
                    className={pairedFieldControlClass}
                  >
                    {options.models.map((item) => (
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
                    value={device}
                    onChange={(event) => setDevice(event.target.value)}
                    className={pairedFieldControlClass}
                  >
                    {options.devices.map((item) => (
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
                    max={maxCpuThreads}
                    value={effectiveCpuThreads}
                    disabled={device !== 'cpu'}
                    onChange={(event) => {
                      const nextValue = Number.parseInt(event.target.value, 10)
                      setNproc(Number.isNaN(nextValue) ? 1 : nextValue)
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
                    value={outputFormat}
                    onChange={(event) => {
                      const nextValue = event.target.value as OutputFormat
                      setOutputFormat(nextValue)
                      if (nextValue === 'srt') {
                        setUseTimestamps(true)
                      }
                    }}
                    className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 outline-none transition focus:border-orange-300 focus:bg-white"
                  >
                    {options.outputFormats.map((item) => (
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
                    onClick={() => setSaveAudio((value) => !value)}
                    className={`relative inline-flex h-7 w-12 shrink-0 rounded-full p-1 transition ${saveAudio ? 'bg-stone-900' : 'bg-stone-300'}`}
                  >
                    <span
                      className={`absolute left-1 top-1 h-5 w-5 rounded-full bg-white transition-transform ${saveAudio ? 'translate-x-5' : 'translate-x-0'}`}
                    />
                  </button>
                </label>

                <label className="flex items-center justify-between rounded-3xl border border-stone-200 bg-stone-50 px-4 py-3">
                  <div>
                    <div className="font-medium text-stone-800">
                      {t('controls.toggles.timestamps.title')}
                    </div>
                    <div className="text-sm text-stone-500">
                      {outputFormat === 'srt'
                        ? t('controls.toggles.timestamps.srtLocked')
                        : t('controls.toggles.timestamps.enabled')}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      if (outputFormat !== 'srt') {
                        setUseTimestamps((value) => !value)
                      }
                    }}
                    className={`relative inline-flex h-7 w-12 shrink-0 rounded-full p-1 transition ${effectiveTimestamps ? 'bg-stone-900' : 'bg-stone-300'} ${outputFormat === 'srt' ? 'cursor-not-allowed opacity-70' : ''}`}
                  >
                    <span
                      className={`absolute left-1 top-1 h-5 w-5 rounded-full bg-white transition-transform ${effectiveTimestamps ? 'translate-x-5' : 'translate-x-0'}`}
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
                disabled={isSubmitting}
                className="inline-flex items-center justify-center gap-2 rounded-full bg-stone-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-stone-800 disabled:cursor-not-allowed disabled:bg-stone-400"
              >
                {isSubmitting ? (
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
            <LogPanel logText={deferredLogText} />
            <ResultPanel
              text={deferredTranscript}
              rawPreview={deferredPreview}
              canDownload={Boolean(job?.downloadUrl)}
              onCopy={() => {
                void handleCopy()
              }}
              onDownload={handleDownload}
            />
          </div>
        </div>
      </div>
    </main>
  )
}
