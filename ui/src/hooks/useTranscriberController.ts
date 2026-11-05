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
    restorePunctuation: boolean
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
  effective_device?: string | null
  effectiveDevice?: string | null
  backend?: string | null
  effective_backend?: string | null
  effectiveBackend?: string | null
}

export type OptionsResponse = {
  models: string[]
  devices: string[]
  deviceDetails: Array<{ id: string; available: boolean; name: string; reason: string | null }>
  defaultDevice: string
  vulkanModels: string[]
  punctuationAvailable: boolean
  modelsDirectory: string | null
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

const FALLBACK_CPU_THREADS = 1
const DEFAULT_MODEL_OPTIONS = [
  'tiny',
  'base',
  'small',
  'medium',
  'large-v1',
  'large-v2',
  'large-v3',
  'large-v3-turbo',
  'large',
  'distil-large-v2',
  'distil-medium.en',
  'distil-small.en',
  'tiny.en',
  'base.en',
  'small.en',
  'medium.en',
]
const DEFAULT_LANGUAGE_OPTIONS = ['ru', 'auto', 'en', 'de', 'fr', 'es', 'it', 'pt', 'uk', 'ja', 'ko', 'zh']
const DEFAULT_OPTIONS: OptionsResponse = {
  models: DEFAULT_MODEL_OPTIONS,
  devices: ['auto', 'cpu', 'vulkan', 'cuda'],
  deviceDetails: [
    { id: 'auto', available: true, name: 'Automatic', reason: null },
    { id: 'cpu', available: true, name: 'CPU', reason: null },
    { id: 'vulkan', available: false, name: 'Vulkan', reason: null },
    { id: 'cuda', available: false, name: 'CUDA', reason: null },
  ],
  defaultDevice: 'auto',
  vulkanModels: [],
  punctuationAvailable: false,
  modelsDirectory: null,
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
  const deviceDetails = payload?.deviceDetails?.length ? payload.deviceDetails : DEFAULT_OPTIONS.deviceDetails
  const availableDevices = deviceDetails.filter((item) => item.available).map((item) => item.id)
  const backendThreadLimit = payload?.maxCpuThreads
  return {
    models: payload?.models?.length ? payload.models : DEFAULT_OPTIONS.models,
    devices: mergeOrderedValues(availableDevices, payload?.devices).filter((id) =>
      deviceDetails.some((item) => item.id === id && item.available),
    ),
    deviceDetails,
    defaultDevice: payload?.defaultDevice ?? 'auto',
    vulkanModels: payload?.vulkanModels ?? [],
    punctuationAvailable: payload?.punctuationAvailable ?? false,
    modelsDirectory: payload?.modelsDirectory ?? null,
    outputFormats: mergeOrderedValues(DEFAULT_OPTIONS.outputFormats, payload?.outputFormats) as OutputFormat[],
    languages: mergeOrderedValues(DEFAULT_LANGUAGE_OPTIONS, payload?.languages),
    maxCpuThreads:
      typeof backendThreadLimit === 'number' && Number.isFinite(backendThreadLimit) && backendThreadLimit > 0
        ? Math.floor(backendThreadLimit)
        : FALLBACK_CPU_THREADS,
  }
}

export function useTranscriberController(shouldLoadOptions: boolean) {
  const [sourceType, setSourceType] = useState<SourceType>('url')
  const [url, setUrl] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [language, setLanguage] = useState('ru')
  const [model, setModel] = useState('small')
  const [device, setDevice] = useState('auto')
  const [nproc, setNproc] = useState(FALLBACK_CPU_THREADS)
  const [outputFormat, setOutputFormat] = useState<OutputFormat>('srt')
  const [saveAudio, setSaveAudio] = useState(false)
  const [useTimestamps, setUseTimestamps] = useState(true)
  const [restorePunctuation, setRestorePunctuation] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<UiError | null>(null)
  const [job, setJob] = useState<JobResponse | null>(null)
  const [options, setOptions] = useState<OptionsResponse>(DEFAULT_OPTIONS)
  const [sourceCheck, setSourceCheck] = useState<SourceCheckResponse | null>(null)
  const [isCheckingSource, setIsCheckingSource] = useState(false)
  const [isSourceCheckExpanded, setIsSourceCheckExpanded] = useState(false)
  const [copyStatus, setCopyStatus] = useState<'success' | 'error' | null>(null)
  const [optionsError, setOptionsError] = useState(false)
  const [optionsRetryCount, setOptionsRetryCount] = useState(0)
  const [optionsLoaded, setOptionsLoaded] = useState(false)
  const sourceCheckRequestRef = useRef(0)
  const optionsRequestRef = useRef<Promise<void> | null>(null)

  const deferredLogText = useDeferredValue(job?.logs.join('\n') ?? '')
  const deferredTranscript = useDeferredValue(job?.result_text ?? '')
  const deferredPreview = useDeferredValue(job?.rendered_result ?? '')
  const currentJobId = job?.id
  const currentJobStatus = job?.status
  const normalizedUrl = url.trim()

  useEffect(() => {
    if (!shouldLoadOptions || optionsLoaded || optionsRequestRef.current) {
      return
    }

    let active = true

    async function loadOptions() {
      try {
        const payload = await fetchJson<Partial<OptionsResponse>>('/api/options')
        if (!active) {
          return
        }

        startTransition(() => {
          const normalizedOptions = normalizeOptionsResponse(payload)
          setOptionsError(false)
          setOptionsLoaded(true)
          setOptions(normalizedOptions)
          setDevice((current) =>
            normalizedOptions.devices.includes(current)
              ? current
              : normalizedOptions.devices.includes(normalizedOptions.defaultDevice)
                ? normalizedOptions.defaultDevice
                : normalizedOptions.devices[0] ?? 'cpu',
          )
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
        if (active) {
          optionsRequestRef.current = null
          setOptionsError(true)
        }
      }
    }

    const request = loadOptions()
    optionsRequestRef.current = request
    return () => {
      active = false
      if (!optionsLoaded && optionsRequestRef.current === request) optionsRequestRef.current = null
    }
  }, [shouldLoadOptions, optionsLoaded, optionsRetryCount])

  useEffect(() => {
    if (!currentJobId || (currentJobStatus !== 'queued' && currentJobStatus !== 'running')) {
      return
    }

    let active = true
    let inFlight = false
    const timer = window.setInterval(async () => {
      if (inFlight) return
      inFlight = true
      try {
        const nextJob = await fetchJson<JobResponse>(`/api/jobs/${currentJobId}`)
        if (!active) return
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
        if (!active) return
        setError(
          fetchError instanceof Error
            ? { kind: 'message', message: fetchError.message }
            : { kind: 'translation', key: 'errors.unableRefreshJob' },
        )
      } finally {
        inFlight = false
      }
    }, 1000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [currentJobId, currentJobStatus])

  function handleSourceChange() {
    sourceCheckRequestRef.current += 1
    setSourceCheck(null)
    setIsCheckingSource(false)
    setIsSourceCheckExpanded(false)
  }

  function handleSetSourceType(value: SourceType) {
    handleSourceChange()
    setSourceType(value)
  }

  function handleSetUrl(value: string) {
    handleSourceChange()
    setUrl(value)
  }

  function handleSetLanguage(value: string) {
    setLanguage(value)
    if (value !== 'ru') setRestorePunctuation(false)
  }

  const effectiveTimestamps = outputFormat === 'srt' ? true : useTimestamps
  const maxCpuThreads = Math.max(options.maxCpuThreads || 1, 1)
  const effectiveCpuThreads = Math.min(Math.max(nproc, 1), maxCpuThreads)
  const availableModels = options.models.filter((item) =>
    language === 'en' || (!item.endsWith('.en') && !item.startsWith('distil-')),
  ).filter((item) => {
    const vulkanMode = device === 'vulkan' || (device === 'auto' && options.deviceDetails.some((item) => item.id === 'vulkan' && item.available))
    return !vulkanMode || options.vulkanModels.includes(item)
  })

  const selectedModel = availableModels.includes(model) ? model : availableModels[0] ?? model

  const errorMessage = useMemo(() => {
    if (!error) {
      return null
    }

    return error.kind === 'message' ? error.message : error.key
  }, [error])

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (isSubmitting || job?.status === 'queued' || job?.status === 'running') return
    setError(null)
    setCopyStatus(null)

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
    formData.append('model', selectedModel)
    formData.append('device', device)
    if (device !== 'cuda') {
      formData.append('nproc', String(effectiveCpuThreads))
    }
    formData.append('outputFormat', outputFormat)
    formData.append('saveAudio', String(saveAudio))
    formData.append('useTimestamps', String(effectiveTimestamps))
    formData.append('restorePunctuation', String(language === 'ru' ? restorePunctuation : false))

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
    if (!deferredTranscript || typeof navigator.clipboard?.writeText !== 'function') {
      setCopyStatus('error')
      return
    }

    try {
      await navigator.clipboard.writeText(deferredTranscript)
      setCopyStatus('success')
    } catch {
      setCopyStatus('error')
    }
  }

  function handleDownload() {
    if (!job?.downloadUrl) {
      return
    }

    window.open(`${API_BASE}${job.downloadUrl}`, '_blank', 'noopener,noreferrer')
  }

  return {
    sourceType,
    setSourceType: handleSetSourceType,
    url,
    setUrl: handleSetUrl,
    file,
    setFile,
    language,
    setLanguage: handleSetLanguage,
    selectedModel,
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
    restorePunctuation,
    setRestorePunctuation,
    isSubmitting,
    error,
    errorMessage,
    setError,
    job,
    options,
    optionsError,
    retryOptions: () => setOptionsRetryCount((count) => count + 1),
    availableModels,
    sourceCheck,
    isCheckingSource,
    isSourceCheckExpanded,
    setIsSourceCheckExpanded,
    deferredLogText,
    deferredTranscript,
    deferredPreview,
    copyStatus,
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
