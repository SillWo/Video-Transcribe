type SettingsSummaryProps = {
  sourceType: 'url' | 'file'
  language: string
  model: string
  device: string
  outputFormat: 'txt' | 'srt' | 'json'
  saveAudio: boolean
  useTimestamps: boolean
}

const itemClass =
  'rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-xs font-medium uppercase tracking-[0.18em] text-stone-600'

export function SettingsSummary(props: SettingsSummaryProps) {
  return (
    <div className="rounded-[28px] border border-stone-200 bg-white/85 p-4 shadow-[0_12px_40px_-24px_rgba(41,37,36,0.35)] backdrop-blur">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-orange-600">
            Current Setup
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-stone-900">
            Local video transcription
          </h1>
        </div>
        <div className="rounded-full border border-orange-200 bg-orange-50 px-3 py-1 text-xs font-semibold text-orange-700">
          Browser + FastAPI + Whisper
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <span className={itemClass}>source: {props.sourceType}</span>
        <span className={itemClass}>lang: {props.language}</span>
        <span className={itemClass}>model: {props.model}</span>
        <span className={itemClass}>device: {props.device}</span>
        <span className={itemClass}>format: {props.outputFormat}</span>
        <span className={itemClass}>
          timestamps: {props.outputFormat === 'srt' ? 'forced' : props.useTimestamps ? 'on' : 'off'}
        </span>
        <span className={itemClass}>keep audio: {props.saveAudio ? 'on' : 'off'}</span>
      </div>
    </div>
  )
}
