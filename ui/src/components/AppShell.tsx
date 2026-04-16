import type { ReactNode } from 'react'

export function AppShell(props: { children: ReactNode }) {
  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(251,146,60,0.20),_transparent_30%),radial-gradient(circle_at_top_right,_rgba(56,189,248,0.15),_transparent_28%),linear-gradient(180deg,_#fffaf5_0%,_#f5f5f4_48%,_#fafaf9_100%)] px-4 py-6 text-stone-800 md:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">{props.children}</div>
    </main>
  )
}
