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
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    throw new Error(await response.text())
  }
  return response.json() as Promise<T>
}

export default function App() {
  const [sourceType, setSourceType] = useState<SourceType>('url')
  const [url, setUrl] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [language, setLanguage] = useState('ru')
  const [model, setModel] = useState('small')
  const [device, setDevice] = useState('cpu')
  const [outputFormat, setOutputFormat] = useState<OutputFormat>('srt')
  const [saveAudio, setSaveAudio] = useState(false)
  const [useTimestamps, setUseTimestamps] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useState<JobResponse | null>(null)
  const [options, setOptions] = useState<OptionsResponse>({
    models: ['small', 'medium', 'large-v3'],
    devices: ['cpu', 'cuda'],
    outputFormats: ['txt', 'srt', 'json'],
    languages: ['ru', 'en', 'de', 'fr', 'es', 'it', 'pt', 'uk'],
  })

  const deferredLogText = useDeferredValue(job?.logs.join('\n') ?? '')
  const deferredTranscript = useDeferredValue(job?.result_text ?? '')
  const deferredPreview = useDeferredValue(job?.rendered_result ?? '')
  const currentJobId = job?.id
  const currentJobStatus = job?.status

  useEffect(() => {
    let active = true

    async function loadOptions() {
      try {
        const payload = await fetchJson<OptionsResponse>('/api/options')
        if (!active) {
          return
        }

        startTransition(() => {
          setOptions(payload)
          setModel((currentModel) =>
            payload.models.length > 0 && !payload.models.includes(currentModel)
              ? payload.models[0]
              : currentModel,
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
          setError(nextJob.error ?? 'Job failed')
        }
      } catch (fetchError) {
        setError(fetchError instanceof Error ? fetchError.message : 'Unable to refresh job status')
      }
    }, 1000)

    return () => window.clearInterval(timer)
  }, [currentJobId, currentJobStatus])

  const effectiveTimestamps = outputFormat === 'srt' ? true : useTimestamps

  const statusLabel = useMemo(() => {
    if (!job) {
      return 'idle'
    }
    return `${job.status} / ${job.stage}`
  }, [job])

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)

    if (sourceType === 'url' && !url.trim()) {
      setError('Enter a video URL.')
      return
    }

    if (sourceType === 'file' && !file) {
      setError('Choose a local media file.')
      return
    }

    const formData = new FormData()
    formData.append('sourceType', sourceType)
    formData.append('url', url.trim())
    formData.append('language', language)
    formData.append('model', model)
    formData.append('device', device)
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
      setError(submitError instanceof Error ? submitError.message : 'Unable to start transcription')
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
          outputFormat={outputFormat}
          saveAudio={saveAudio}
          useTimestamps={effectiveTimestamps}
        />

        <StageProgress
          currentStage={job?.stage ?? 'source'}
          status={job?.status ?? 'queued'}
        />

        <div className="grid gap-6 xl:grid-cols-[420px_minmax(0,1fr)]">
          <SectionCard
            title="Controls"
            description="Pick a source, set Whisper options, then start the local pipeline."
            action={
              <div className="rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-stone-600">
                {statusLabel}
              </div>
            }
          >
            <form className="grid gap-5" onSubmit={handleSubmit}>
              <div className="grid gap-3">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                  Source
                </p>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setSourceType('url')}
                    className={`rounded-3xl border px-4 py-3 text-left transition ${sourceType === 'url' ? 'border-orange-300 bg-orange-50 text-orange-800' : 'border-stone-200 bg-stone-50 text-stone-700 hover:bg-stone-100'}`}
                  >
                    <Globe className="mb-2 h-5 w-5" />
                    <div className="font-medium">Video link</div>
                    <div className="text-sm text-stone-500">YouTube, Rutube and other sources supported by yt-dlp</div>
                  </button>
                  <button
                    type="button"
                    onClick={() => setSourceType('file')}
                    className={`rounded-3xl border px-4 py-3 text-left transition ${sourceType === 'file' ? 'border-orange-300 bg-orange-50 text-orange-800' : 'border-stone-200 bg-stone-50 text-stone-700 hover:bg-stone-100'}`}
                  >
                    <FileAudio className="mb-2 h-5 w-5" />
                    <div className="font-medium">Local file</div>
                    <div className="text-sm text-stone-500">Upload a media file from this machine</div>
                  </button>
                </div>

                {sourceType === 'url' ? (
                  <label className="grid gap-2">
                    <span className="text-sm font-medium text-stone-700">URL</span>
                    <input
                      value={url}
                      onChange={(event) => setUrl(event.target.value)}
                      placeholder="https://www.youtube.com/watch?v=..."
                      className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 outline-none ring-0 transition placeholder:text-stone-400 focus:border-orange-300 focus:bg-white"
                    />
                  </label>
                ) : (
                  <label className="grid gap-2">
                    <span className="text-sm font-medium text-stone-700">Media file</span>
                    <input
                      type="file"
                      accept="audio/*,video/*"
                      onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                      className="rounded-2xl border border-dashed border-stone-300 bg-stone-50 px-4 py-3 text-sm text-stone-600 file:mr-4 file:rounded-full file:border-0 file:bg-stone-900 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white"
                    />
                    <span className="text-xs text-stone-500">
                      {file ? `Selected: ${file.name}` : 'No file selected'}
                    </span>
                  </label>
                )}
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <label className="grid gap-2">
                  <span className="flex items-center gap-2 text-sm font-medium text-stone-700">
                    <Languages className="h-4 w-4" />
                    Language
                  </span>
                  <select
                    value={language}
                    onChange={(event) => setLanguage(event.target.value)}
                    className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 outline-none transition focus:border-orange-300 focus:bg-white"
                  >
                    {options.languages.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="grid gap-2">
                  <span className="flex items-center gap-2 text-sm font-medium text-stone-700">
                    <MicVocal className="h-4 w-4" />
                    Model
                  </span>
                  <select
                    value={model}
                    onChange={(event) => setModel(event.target.value)}
                    className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 outline-none transition focus:border-orange-300 focus:bg-white"
                  >
                    {options.models.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="grid gap-2">
                  <span className="flex items-center gap-2 text-sm font-medium text-stone-700">
                    <Cpu className="h-4 w-4" />
                    Device
                  </span>
                  <select
                    value={device}
                    onChange={(event) => setDevice(event.target.value)}
                    className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 outline-none transition focus:border-orange-300 focus:bg-white"
                  >
                    {options.devices.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="grid gap-2">
                  <span className="flex items-center gap-2 text-sm font-medium text-stone-700">
                    <Sparkles className="h-4 w-4" />
                    Output
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
                        {item}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="grid gap-3">
                <label className="flex items-center justify-between rounded-3xl border border-stone-200 bg-stone-50 px-4 py-3">
                  <div>
                    <div className="font-medium text-stone-800">Save intermediate audio</div>
                    <div className="text-sm text-stone-500">Keep extracted MP3 in the output folder</div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setSaveAudio((value) => !value)}
                    className={`flex h-7 w-12 items-center rounded-full p-1 transition ${saveAudio ? 'bg-stone-900' : 'bg-stone-300'}`}
                  >
                    <span
                      className={`h-5 w-5 rounded-full bg-white transition ${saveAudio ? 'translate-x-5' : 'translate-x-0'}`}
                    />
                  </button>
                </label>

                <label className="flex items-center justify-between rounded-3xl border border-stone-200 bg-stone-50 px-4 py-3">
                  <div>
                    <div className="font-medium text-stone-800">Use timestamps</div>
                    <div className="text-sm text-stone-500">
                      {outputFormat === 'srt'
                        ? 'SRT always includes timestamps.'
                        : 'Apply timestamps to txt/json preview and saved file.'}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      if (outputFormat !== 'srt') {
                        setUseTimestamps((value) => !value)
                      }
                    }}
                    className={`flex h-7 w-12 items-center rounded-full p-1 transition ${effectiveTimestamps ? 'bg-stone-900' : 'bg-stone-300'} ${outputFormat === 'srt' ? 'cursor-not-allowed opacity-70' : ''}`}
                  >
                    <span
                      className={`h-5 w-5 rounded-full bg-white transition ${effectiveTimestamps ? 'translate-x-5' : 'translate-x-0'}`}
                    />
                  </button>
                </label>
              </div>

              {error ? (
                <div className="rounded-3xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                  {error}
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
                    Starting
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4" />
                    Start transcription
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
