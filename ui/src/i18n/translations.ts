export type UiLocale = 'ru' | 'en'

export type TranslationParams = Record<string, string | number>

type TranslationTree = {
  [key: string]: string | TranslationTree
}

const en = {
  languageSwitcher: {
    label: 'Interface language',
    ru: 'RU',
    en: 'EN',
  },
  summary: {
    eyebrow: 'Current Setup',
    title: 'Local video transcription',
    badge: 'Browser + FastAPI + Whisper',
    fields: {
      source: 'source',
      lang: 'lang',
      model: 'model',
      device: 'device',
      cores: 'cores',
      format: 'format',
      timestamps: 'timestamps',
      keepAudio: 'keep audio',
    },
    values: {
      sourceType: {
        url: 'video link',
        file: 'local file',
      },
      timestamps: {
        forced: 'forced',
        on: 'on',
        off: 'off',
      },
      toggles: {
        on: 'on',
        off: 'off',
      },
    },
  },
  controls: {
    title: 'Controls',
    description: 'Pick a source, set Whisper options, then start the local pipeline.',
    sourceLabel: 'Source',
    sourceOptions: {
      url: {
        title: 'Video link',
        description: 'YouTube, Rutube and other sources supported by yt-dlp',
      },
      file: {
        title: 'Local file',
        description: 'Upload a media file from this machine',
      },
    },
    fields: {
      url: 'URL',
      mediaFile: 'Media file',
      language: 'Language',
      model: 'Model',
      device: 'Device',
      cpuThreads: 'CPU cores',
      output: 'Output',
    },
    hints: {
      cpuThreadsRange: 'Choose from 1 to {count} logical cores.',
    },
    placeholders: {
      url: 'https://www.youtube.com/watch?v=...',
    },
    file: {
      selected: 'Selected: {filename}',
      empty: 'No file selected',
    },
    toggles: {
      saveAudio: {
        title: 'Save intermediate audio',
        description: 'Keep extracted MP3 in the output folder',
      },
      timestamps: {
        title: 'Use timestamps',
        srtLocked: 'SRT always includes timestamps.',
        enabled: 'Apply timestamps to txt/json preview and saved file.',
      },
    },
    buttons: {
      starting: 'Starting',
      start: 'Start transcription',
    },
  },
  panels: {
    log: {
      title: 'Execution Log',
      description: 'Live output from whisper-gpu.py',
      empty: 'Log output will appear here after you start a job.',
    },
    transcript: {
      title: 'Transcript',
      description: 'Final text and saved file preview',
      copy: 'Copy',
      download: 'Download',
      plainText: 'Plain text',
      plainTextEmpty: 'Completed transcription text will appear here.',
      savedPreview: 'Saved file preview',
      savedPreviewEmpty: 'The selected txt/srt/json output will be previewed here.',
    },
  },
  status: {
    idle: 'Idle',
    job: {
      queued: 'Queued',
      running: 'Running',
      completed: 'Completed',
      failed: 'Failed',
    },
  },
  stages: {
    source: {
      label: 'Source',
      description: 'URL or local file',
    },
    audio: {
      label: 'Audio',
      description: 'Download or extract track',
    },
    recognition: {
      label: 'Recognition',
      description: 'Whisper pipeline',
    },
    result: {
      label: 'Result',
      description: 'Preview and download',
    },
  },
  errors: {
    enterVideoUrl: 'Enter a video URL.',
    chooseLocalFile: 'Choose a local media file.',
    jobFailed: 'Job failed',
    unableRefreshJob: 'Unable to refresh job status',
    unableStart: 'Unable to start transcription',
  },
  languageNames: {
    ru: 'Russian',
    en: 'English',
    de: 'German',
    fr: 'French',
    es: 'Spanish',
    it: 'Italian',
    pt: 'Portuguese',
    uk: 'Ukrainian',
    ja: 'Japanese',
    ko: 'Korean',
    zh: 'Chinese',
  },
} as const

type WidenTranslationTree<T> = {
  [K in keyof T]: T[K] extends string
    ? string
    : T[K] extends Record<string, unknown>
      ? WidenTranslationTree<T[K]>
      : never
}

type TranslationSchema = WidenTranslationTree<typeof en>

const ru: TranslationSchema = {
  languageSwitcher: {
    label: 'Язык интерфейса',
    ru: 'RU',
    en: 'EN',
  },
  summary: {
    eyebrow: 'Текущая конфигурация',
    title: 'Локальная расшифровка видео',
    badge: 'Браузер + FastAPI + Whisper',
    fields: {
      source: 'источник',
      lang: 'язык',
      model: 'модель',
      device: 'устройство',
      cores: 'ядра',
      format: 'формат',
      timestamps: 'таймкоды',
      keepAudio: 'аудио',
    },
    values: {
      sourceType: {
        url: 'ссылка',
        file: 'файл',
      },
      timestamps: {
        forced: 'всегда',
        on: 'вкл',
        off: 'выкл',
      },
      toggles: {
        on: 'вкл',
        off: 'выкл',
      },
    },
  },
  controls: {
    title: 'Управление',
    description: 'Выберите источник, настройте параметры Whisper и запустите локальный пайплайн.',
    sourceLabel: 'Источник',
    sourceOptions: {
      url: {
        title: 'Ссылка на видео',
        description: 'YouTube, Rutube и другие источники, поддерживаемые yt-dlp',
      },
      file: {
        title: 'Локальный файл',
        description: 'Загрузите медиафайл с этого компьютера',
      },
    },
    fields: {
      url: 'URL',
      mediaFile: 'Медиафайл',
      language: 'Язык',
      model: 'Модель',
      device: 'Устройство',
      cpuThreads: 'Кол-во ядер процессора',
      output: 'Вывод',
    },
    hints: {
      cpuThreadsRange: 'Введите значение от 1 до {count} логических ядер.',
    },
    placeholders: {
      url: 'https://www.youtube.com/watch?v=...',
    },
    file: {
      selected: 'Выбран файл: {filename}',
      empty: 'Файл не выбран',
    },
    toggles: {
      saveAudio: {
        title: 'Сохранять промежуточное аудио',
        description: 'Оставлять извлеченный MP3 в папке результата',
      },
      timestamps: {
        title: 'Использовать таймкоды',
        srtLocked: 'Формат SRT всегда содержит таймкоды.',
        enabled: 'Применять таймкоды к превью и сохраненному файлу txt/json.',
      },
    },
    buttons: {
      starting: 'Запуск',
      start: 'Начать расшифровку',
    },
  },
  panels: {
    log: {
      title: 'Журнал выполнения',
      description: 'Живой вывод из whisper-gpu.py',
      empty: 'Логи появятся здесь после запуска задачи.',
    },
    transcript: {
      title: 'Расшифровка',
      description: 'Итоговый текст и превью сохраненного файла',
      copy: 'Копировать',
      download: 'Скачать',
      plainText: 'Текст',
      plainTextEmpty: 'Готовый текст расшифровки появится здесь.',
      savedPreview: 'Превью файла',
      savedPreviewEmpty: 'Здесь будет показано превью выбранного результата txt/srt/json.',
    },
  },
  status: {
    idle: 'Ожидание',
    job: {
      queued: 'В очереди',
      running: 'В работе',
      completed: 'Готово',
      failed: 'Ошибка',
    },
  },
  stages: {
    source: {
      label: 'Источник',
      description: 'Ссылка или локальный файл',
    },
    audio: {
      label: 'Аудио',
      description: 'Загрузка или извлечение дорожки',
    },
    recognition: {
      label: 'Распознавание',
      description: 'Пайплайн Whisper',
    },
    result: {
      label: 'Результат',
      description: 'Превью и скачивание',
    },
  },
  errors: {
    enterVideoUrl: 'Введите ссылку на видео.',
    chooseLocalFile: 'Выберите локальный медиафайл.',
    jobFailed: 'Задача завершилась с ошибкой',
    unableRefreshJob: 'Не удалось обновить статус задачи',
    unableStart: 'Не удалось запустить расшифровку',
  },
  languageNames: {
    ru: 'Русский',
    en: 'Английский',
    de: 'Немецкий',
    fr: 'Французский',
    es: 'Испанский',
    it: 'Итальянский',
    pt: 'Португальский',
    uk: 'Украинский',
    ja: 'Японский',
    ko: 'Корейский',
    zh: 'Китайский',
  },
}

export const UI_LOCALE_STORAGE_KEY = 'video-transcribe.ui.locale'
export const DEFAULT_UI_LOCALE: UiLocale = 'ru'
export const supportedUiLocales: UiLocale[] = ['ru', 'en']

export const translations: Record<UiLocale, TranslationSchema> = {
  ru,
  en,
}

function getNestedValue(tree: TranslationTree, key: string): string | undefined {
  return key.split('.').reduce<string | TranslationTree | undefined>((current, part) => {
    if (typeof current !== 'object' || current === null) {
      return undefined
    }

    return current[part]
  }, tree) as string | undefined
}

function interpolate(template: string, params?: TranslationParams) {
  if (!params) {
    return template
  }

  return template.replace(/\{(\w+)\}/g, (_, token: string) => String(params[token] ?? `{${token}}`))
}

export function translate(
  locale: UiLocale,
  key: string,
  params?: TranslationParams,
  fallback?: string,
) {
  const template =
    getNestedValue(translations[locale], key) ??
    getNestedValue(translations[DEFAULT_UI_LOCALE], key) ??
    fallback ??
    key

  return interpolate(template, params)
}

export function isUiLocale(value: string): value is UiLocale {
  return supportedUiLocales.includes(value as UiLocale)
}

export function getLanguageName(locale: UiLocale, code: string) {
  return translate(locale, `languageNames.${code}`, undefined, code.toUpperCase())
}
