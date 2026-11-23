import type { ReactNode } from 'react'

type Props = { show: boolean; className: string; children: ReactNode; role?: 'status' | 'alert' }

/** CSS keeps transient notices mounted so enter and exit transitions can retarget cleanly. */
export default function ToastNotice({ show, className, children, role = 'status' }: Props) {
  return <div className={`${className} toast-motion ${show ? 'toast-visible' : 'toast-exiting'}`} role={role} aria-live={role === 'alert' ? 'assertive' : 'polite'} aria-hidden={!show} inert={!show}>
    {children}
  </div>
}
