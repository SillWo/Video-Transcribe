import {
  startTransition,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from 'react'

import { API_BASE, fetchJson } from '../lib/api'

export type SourceType = 'url' | 'file'
export type OutputFormat = 'txt' | 'srt' | 'json'
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed'
export type StageId = 'source' | 'audio' | 'recognition' | 'result'

export type JobResponse = {
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

export type OptionsResponse = {
  models: string[]
  devices: string[]
  outputFormats: OutputFormat[]
  languages: string[]
  maxCpuThreads: number
}

export type SourceCheckResponse = {
  ok: boolean
  canProcess: boolean
  message: string
  details: string[]
  sourceInfo: {
    title: string | null
    id: string | null
    extractor: string | null
    extractorKey: string | null
    site: string | null
    webpageUrl: string | null
    kind: string | null
    isPlaylist: boolean
    playlistTitle: string | null
    entryCount: number | null
    checkedEntries: number | null
    formatsCount: number
    audioFormatsCount: number
    audioExtractable: boolean
    availability: string | null
    formatSample: string[]
  } | null
  formatsAvailable: boolean
  audioExtractable: boolean
  extractor: string | null
  title: string | null
  id: string | null
  diagnosticCode: string
}

export type UiError =
  | {
      kind: 'message'
      message: string
    }
  | {
      kind: 'translation'
      key: string
    }

const FALLBACK_CPU_THREADS =
  typeof navigator !== 'undefined' ? Math.max(navigator.hardwareConcurrency || 1, 1) : 1
const DEFAULT_LANGUAGE_OPTIONS = ['ru', 'auto', 'en', 'de', 'fr', 'es', 'it', 'pt', 'uk', 'ja', 'ko', 'zh']
const DEFAULT_OPTIONS: OptionsResponse = {
  models: [],
  devices: ['cpu', 'cuda'],
  outputFormats: ['txt', 'srt', 'json'],
  languages: DEFAULT_LANGUAGE_OPTIONS,
  maxCpuThreads: FALLBACK_CPU_THREADS,
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

export function useTranscriberController(shouldLoadOptions: boolean) {
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
  const [sourceCheck, setSourceCheck] = useState<SourceCheckResponse | null>(null)
  const [isCheckingSource, setIsCheckingSource] = useState(false)
  const [isSourceCheckExpanded, setIsSourceCheckExpanded] = useState(false)
  const sourceCheckRequestRef = useRef(0)
  const optionsLoadedRef = useRef(false)

  const deferredLogText = useDeferredValue(job?.logs.join('\n') ?? '')
  const deferredTranscript = useDeferredValue(job?.result_text ?? '')
  const deferredPreview = useDeferredValue(job?.rendered_result ?? '')
  const currentJobId = job?.id
  const currentJobStatus = job?.status
  const normalizedUrl = url.trim()

  useEffect(() => {
    if (!shouldLoadOptions || optionsLoadedRef.current) {
      return
    }

    optionsLoadedRef.current = true
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
        // Keep built-in fallback options when backend options are not available.
      }
    }

    void loadOptions()
    return () => {
      active = false
    }
  }, [shouldLoadOptions])

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

  useEffect(() => {
    sourceCheckRequestRef.current += 1
    setSourceCheck(null)
    setIsCheckingSource(false)
    setIsSourceCheckExpanded(false)
  }, [sourceType, url])

  const effectiveTimestamps = outputFormat === 'srt' ? true : useTimestamps
  const maxCpuThreads = Math.max(options.maxCpuThreads || 1, 1)
  const effectiveCpuThreads = Math.min(Math.max(nproc, 1), maxCpuThreads)

  const errorMessage = useMemo(() => {
    if (!error) {
      return null
    }

    return error.kind === 'message' ? error.message : error.key
  }, [error])

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
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

  async function handleCheckSource(emptyUrlMessage: string, requestFailedMessage: string) {
    if (sourceType !== 'url') {
      return
    }

    if (!normalizedUrl) {
      setSourceCheck({
        ok: false,
        canProcess: false,
        message: emptyUrlMessage,
        details: [],
        sourceInfo: null,
        formatsAvailable: false,
        audioExtractable: false,
        extractor: null,
        title: null,
        id: null,
        diagnosticCode: 'empty_url',
      })
      return
    }

    const requestId = sourceCheckRequestRef.current + 1
    sourceCheckRequestRef.current = requestId

    try {
      setIsCheckingSource(true)
      const result = await fetchJson<SourceCheckResponse>('/api/check-source', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: normalizedUrl }),
      })

      if (sourceCheckRequestRef.current !== requestId) {
        return
      }

      startTransition(() => {
        setSourceCheck(result)
        setIsSourceCheckExpanded(false)
      })
    } catch (checkError) {
      if (sourceCheckRequestRef.current !== requestId) {
        return
      }

      setSourceCheck({
        ok: false,
        canProcess: false,
        message: checkError instanceof Error ? checkError.message : requestFailedMessage,
        details: [],
        sourceInfo: null,
        formatsAvailable: false,
        audioExtractable: false,
        extractor: null,
        title: null,
        id: null,
        diagnosticCode: 'request_failed',
      })
      setIsSourceCheckExpanded(false)
    } finally {
      if (sourceCheckRequestRef.current === requestId) {
        setIsCheckingSource(false)
      }
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

  return {
    sourceType,
    setSourceType,
    url,
    setUrl,
    file,
    setFile,
    language,
    setLanguage,
    model,
    setModel,
    device,
    setDevice,
    nproc,
    setNproc,
    outputFormat,
    setOutputFormat,
    saveAudio,
    setSaveAudio,
    useTimestamps,
    setUseTimestamps,
    isSubmitting,
    error,
    errorMessage,
    setError,
    job,
    options,
    sourceCheck,
    isCheckingSource,
    isSourceCheckExpanded,
    setIsSourceCheckExpanded,
    deferredLogText,
    deferredTranscript,
    deferredPreview,
    normalizedUrl,
    effectiveTimestamps,
    maxCpuThreads,
    effectiveCpuThreads,
    handleSubmit,
    handleCheckSource,
    handleCopy,
    handleDownload,
  }
}

export type TranscriberController = ReturnType<typeof useTranscriberController>
