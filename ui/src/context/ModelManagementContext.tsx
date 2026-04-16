/* eslint-disable react-refresh/only-export-components */
import {
  createContext,
  startTransition,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

import { fetchJson } from '../lib/api'

export type ModelPanelStatus = 'downloaded' | 'not_downloaded' | 'unknown'
export type ModelLanguageScope = 'multilingual' | 'english'
export type ModelFamily = 'standard' | 'distil'
export type OperationJobStatus = 'queued' | 'running' | 'success' | 'error'
export type OperationType = 'download' | 'delete'
export type NoticeKind = 'info' | 'success' | 'error'

export type ModelPanelItem = {
  id: string
  displayName: string
  backendValue: string
  hfRepoId: string
  languageScope: ModelLanguageScope
  family: ModelFamily
  enabled: boolean
  isDownloaded: boolean
  cacheLocation: string | null
  downloadedSizeBytes: number | null
  lastModified: string | null
  status: ModelPanelStatus
}

export type ModelPanelResponse = {
  catalog: ModelPanelItem[]
  summary: {
    downloadedCount: number
    availableCount: number
    totalDownloadedSizeBytes: number
  }
}

export type ModelOperationJobResponse = {
  jobId: string
  modelId: string
  status: OperationJobStatus
  progress: {
    percent: number
    label: string
  }
  message: string
  error: string | null
}

export type ActiveOperation = {
  operationType: OperationType
  jobId: string
  modelId: string
}

export type Notice = {
  kind: NoticeKind
  text: string
}

const EMPTY_SUMMARY: ModelPanelResponse['summary'] = {
  downloadedCount: 0,
  availableCount: 0,
  totalDownloadedSizeBytes: 0,
}

type LoadPanelOptions = {
  keepLoadingState?: boolean
}

type ModelManagementContextValue = {
  panelData: ModelPanelResponse | null
  catalog: ModelPanelItem[]
  summary: ModelPanelResponse['summary']
  isLoading: boolean
  error: string | null
  activeOperation: ActiveOperation | null
  downloadJobs: Record<string, ModelOperationJobResponse>
  deleteJobs: Record<string, ModelOperationJobResponse>
  modelNotices: Record<string, Notice>
  globalNotice: Notice | null
  loadPanel: (options?: LoadPanelOptions) => Promise<void>
  startOperation: (modelId: string, operationType: OperationType) => Promise<void>
}

const ModelManagementContext = createContext<ModelManagementContextValue | null>(null)

export function ModelManagementProvider(props: { children: ReactNode }) {
  const [panelData, setPanelData] = useState<ModelPanelResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeOperation, setActiveOperation] = useState<ActiveOperation | null>(null)
  const [downloadJobs, setDownloadJobs] = useState<Record<string, ModelOperationJobResponse>>({})
  const [deleteJobs, setDeleteJobs] = useState<Record<string, ModelOperationJobResponse>>({})
  const [modelNotices, setModelNotices] = useState<Record<string, Notice>>({})
  const [globalNotice, setGlobalNotice] = useState<Notice | null>(null)

  const loadPanel = useCallback(async (options?: LoadPanelOptions) => {
    if (!options?.keepLoadingState) {
      setIsLoading(true)
    }

    try {
      const payload = await fetchJson<ModelPanelResponse>('/api/models/panel')
      startTransition(() => {
        setPanelData(payload)
        setError(null)
      })
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Unable to load model catalog.')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadPanel()
  }, [loadPanel])

  useEffect(() => {
    if (!activeOperation) {
      return
    }

    let active = true
    const timer = window.setInterval(async () => {
      try {
        const endpoint =
          activeOperation.operationType === 'download'
            ? `/api/models/download/${activeOperation.jobId}`
            : `/api/models/delete/${activeOperation.jobId}`
        const job = await fetchJson<ModelOperationJobResponse>(endpoint)
        if (!active) {
          return
        }

        startTransition(() => {
          if (activeOperation.operationType === 'download') {
            setDownloadJobs((current) => ({
              ...current,
              [job.modelId]: job,
            }))
          } else {
            setDeleteJobs((current) => ({
              ...current,
              [job.modelId]: job,
            }))
          }
        })

        if (job.status === 'success') {
          const successMessage = job.message || 'Model operation completed successfully.'
          setActiveOperation(null)
          setModelNotices((current) => ({
            ...current,
            [job.modelId]: { kind: 'success', text: successMessage },
          }))
          setGlobalNotice({ kind: 'success', text: successMessage })
          await loadPanel({ keepLoadingState: true })
        } else if (job.status === 'error') {
          const failureMessage = job.error || job.message || 'Model operation failed.'
          setActiveOperation(null)
          setModelNotices((current) => ({
            ...current,
            [job.modelId]: { kind: 'error', text: failureMessage },
          }))
          setGlobalNotice({ kind: 'error', text: failureMessage })
        }
      } catch (pollError) {
        if (!active) {
          return
        }

        setActiveOperation(null)
        setGlobalNotice({
          kind: 'error',
          text: pollError instanceof Error ? pollError.message : 'Unable to refresh model operation status.',
        })
      }
    }, 1000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [activeOperation, loadPanel])

  const startOperation = useCallback(
    async (modelId: string, operationType: OperationType) => {
      setGlobalNotice(null)

      try {
        const job = await fetchJson<ModelOperationJobResponse>(`/api/models/${operationType}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ modelId }),
        })

        startTransition(() => {
          if (operationType === 'download') {
            setDownloadJobs((current) => ({
              ...current,
              [job.modelId]: job,
            }))
          } else {
            setDeleteJobs((current) => ({
              ...current,
              [job.modelId]: job,
            }))
          }
        })

        if (job.status === 'queued' || job.status === 'running') {
          const infoText = job.message || 'The backend accepted the operation.'
          setActiveOperation({
            operationType,
            jobId: job.jobId,
            modelId: job.modelId,
          })
          setModelNotices((current) => ({
            ...current,
            [job.modelId]: { kind: 'info', text: infoText },
          }))
          return
        }

        if (job.status === 'success') {
          const successMessage = job.message || 'Model operation completed successfully.'
          setModelNotices((current) => ({
            ...current,
            [job.modelId]: { kind: 'success', text: successMessage },
          }))
          setGlobalNotice({ kind: 'success', text: successMessage })
          await loadPanel({ keepLoadingState: true })
          return
        }

        const failureMessage = job.error || job.message || 'Model operation failed.'
        setModelNotices((current) => ({
          ...current,
          [job.modelId]: { kind: 'error', text: failureMessage },
        }))
        setGlobalNotice({ kind: 'error', text: failureMessage })
      } catch (operationError) {
        const message = operationError instanceof Error ? operationError.message : 'Model operation failed.'
        setGlobalNotice({ kind: 'error', text: message })
        setModelNotices((current) => ({
          ...current,
          [modelId]: { kind: 'error', text: message },
        }))
      }
    },
    [loadPanel],
  )

  const value = useMemo<ModelManagementContextValue>(
    () => ({
      panelData,
      catalog: panelData?.catalog ?? [],
      summary: panelData?.summary ?? EMPTY_SUMMARY,
      isLoading,
      error,
      activeOperation,
      downloadJobs,
      deleteJobs,
      modelNotices,
      globalNotice,
      loadPanel,
      startOperation,
    }),
    [
      activeOperation,
      deleteJobs,
      downloadJobs,
      error,
      globalNotice,
      isLoading,
      loadPanel,
      modelNotices,
      panelData,
      startOperation,
    ],
  )

  return <ModelManagementContext.Provider value={value}>{props.children}</ModelManagementContext.Provider>
}

export function useModelManagement() {
  const context = useContext(ModelManagementContext)

  if (!context) {
    throw new Error('useModelManagement must be used within ModelManagementProvider')
  }

  return context
}
