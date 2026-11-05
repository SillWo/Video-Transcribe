import { ArrowLeft } from 'lucide-react'
import { useEffect } from 'react'
import { Link } from 'react-router-dom'

import { ModelManagementDashboard } from '../components/ModelManagementDashboard'
import { useI18n } from '../i18n/useI18n'
import { useModelManagement } from '../context/ModelManagementContext'

const COPY = {
  en: {
    title: 'Local model management',
    description: 'Manage CPU and CUDA models. Vulkan models download automatically the first time you use them.',
    location: 'Project model folder',
    back: 'Back to transcriber',
  },
  ru: {
    title: 'Управление локальными моделями',
    description: 'Управляйте моделями CPU и CUDA. Модели Vulkan загружаются автоматически при первом использовании.',
    location: 'Папка моделей проекта',
    back: 'Вернуться к транскрибатору',
  },
} as const

export function ModelsPage() {
  const { locale } = useI18n()
  const { modelsDirectory } = useModelManagement()
  const copy = locale === 'ru' ? COPY.ru : COPY.en

  useEffect(() => {
    document.title = copy.title
  }, [copy.title])

  return (
    <>
      <section className="rounded-[28px] border border-stone-200 bg-white/90 p-5 shadow-[0_20px_60px_-30px_rgba(41,37,36,0.35)] backdrop-blur">
        <div className="grid gap-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="max-w-3xl">
              <div className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                {locale === 'ru' ? 'Библиотека моделей' : 'Model library'}
              </div>
              <h1 className="mt-2 break-words text-3xl font-semibold text-stone-900">{copy.title}</h1>
              <p className="mt-2 text-sm text-stone-500">{copy.description}</p>
              {modelsDirectory ? (
                <p className="mt-3 break-all rounded-xl bg-stone-50 px-3 py-2 text-xs text-stone-600">
                  {copy.location}: {modelsDirectory}
                </p>
              ) : null}
            </div>
            <Link
              to="/"
              className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl border border-stone-200 bg-white px-4 py-2 text-sm font-semibold text-stone-800 transition-[background-color] hover:bg-stone-50"
            >
              <ArrowLeft className="h-4 w-4" />
              {copy.back}
            </Link>
          </div>
        </div>
      </section>

      <ModelManagementDashboard />
    </>
  )
}
