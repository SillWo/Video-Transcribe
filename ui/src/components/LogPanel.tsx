import { ScrollText } from 'lucide-react'

import { SectionCard } from './SectionCard'

type LogPanelProps = {
  logText: string
}

export function LogPanel({ logText }: LogPanelProps) {
  return (
    <SectionCard
      title="Execution Log"
      description="Live output from whisper-gpu.py"
      action={<ScrollText className="h-5 w-5 text-stone-400" />}
      className="h-full"
    >
      <div className="min-h-[280px] rounded-3xl bg-stone-950 p-4 text-sm text-stone-100">
        <pre className="max-h-[340px] overflow-auto whitespace-pre-wrap break-words font-mono leading-6">
          {logText || 'Log output will appear here after you start a job.'}
        </pre>
      </div>
    </SectionCard>
  )
}
