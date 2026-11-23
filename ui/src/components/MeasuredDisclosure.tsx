import { useLayoutEffect, useRef, type ReactNode } from 'react'

type Props = { open: boolean; className?: string; children: ReactNode }

/** Measures content height for short, interruptible disclosure transitions. */
export default function MeasuredDisclosure({ open, className = '', children }: Props) {
  const panel = useRef<HTMLDivElement>(null)
  const content = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const node = panel.current
    if (!node) return
    const currentHeight = node.getBoundingClientRect().height
    node.style.height = `${currentHeight}px`
    void node.offsetHeight
    const targetHeight = open ? content.current?.scrollHeight ?? 0 : 0
    const frame = requestAnimationFrame(() => { if (panel.current === node) node.style.height = `${targetHeight}px` })
    return () => cancelAnimationFrame(frame)
  }, [open])
  useLayoutEffect(() => {
    if (!open || !panel.current || !content.current) return
    const observer = new ResizeObserver(() => {
      if (panel.current && content.current) panel.current.style.height = `${content.current.scrollHeight}px`
    })
    observer.observe(content.current)
    return () => observer.disconnect()
  }, [open])
  return <div ref={panel} className={`measured-disclosure ${className}`} aria-hidden={!open} inert={!open}><div ref={content} className="measured-disclosure-inner">{children}</div></div>
}
