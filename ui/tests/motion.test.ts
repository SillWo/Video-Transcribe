import assert from 'node:assert/strict'
import test from 'node:test'
import { readFile } from 'node:fs/promises'

const read = (path: string) => readFile(new URL(path, import.meta.url), 'utf8')

test('motion tokens and reduced-motion treatment stay shared and movement-light', async () => {
  const css = await read('../src/redesign.css')
  assert.match(css, /--ease-out:\s*cubic-bezier\(\.23, 1, \.32, 1\)/)
  assert.match(css, /--ease-in-out:\s*cubic-bezier\(\.77, 0, \.175, 1\)/)
  assert.match(css, /@media \(prefers-reduced-motion: reduce\)/)
  assert.match(css, /\.queue-popover, \.toast-motion, \.measured-disclosure, dialog \{ transform: none; \}/)
  assert.doesNotMatch(css, /transition:\s*all\b/)
})

test('queue opens instantly for keyboard, exits on pointer with interruptible timing, and restores trigger focus on Escape', async () => {
  const component = await read('../src/components/GlobalStatusBar.tsx')
  const css = await read('../src/redesign.css')
  assert.match(component, /event\.detail === 0/)
  assert.match(component, /setOpenSource\(keyboard \? 'keyboard' : 'pointer'\)/)
  assert.match(component, /clearTimeout\(closePopoverTimer\.current\)/)
  assert.match(component, /queueTriggerRef\.current\?\.focus\(\)/)
  assert.match(css, /\.queue-popover\.instant \{ transition: none; \}/)
  assert.match(css, /\.queue-popover \{[^}]*180ms/)
})

test('transient notices exit symmetrically without becoming focus traps', async () => {
  const toast = await read('../src/components/ToastNotice.tsx')
  const css = await read('../src/redesign.css')
  assert.match(toast, /toast-visible/)
  assert.match(toast, /inert=\{!show\}/)
  assert.match(toast, /toast-exiting/)
  assert.doesNotMatch(toast, /aria-modal|showModal/)
  assert.match(css, /\.toast-motion\.toast-visible \{[^}]*opacity: 1;[^}]*transform: translateY\(0\)/)
  assert.match(css, /\.toast-motion\.toast-exiting \{[^}]*opacity: 0;[^}]*transform: translateY\(8%\)/)
})

test('disclosure height is measured and interrupted transitions are cleaned up', async () => {
  const disclosure = await read('../src/components/MeasuredDisclosure.tsx')
  assert.match(disclosure, /content\.current\?\.scrollHeight/)
  assert.match(disclosure, /cancelAnimationFrame\(frame\)/)
  assert.match(disclosure, /getBoundingClientRect\(\)\.height/)
  assert.match(disclosure, /ResizeObserver/)
  assert.match(disclosure, /void node\.offsetHeight/)
  assert.match(disclosure, /inert=\{!open\}/)
})

test('confirmation dialogs use native modal focus and Escape cancellation', async () => {
  const app = await read('../src/App.tsx')
  const archive = await read('../src/pages/ArchivePage.tsx')
  const status = await read('../src/components/GlobalStatusBar.tsx')
  assert.match(app, /dialog\.showModal\(\)/)
  assert.match(app, /onCancel=.*onClose\(\)/)
  assert.match(archive, /dialog\.showModal\(\)/)
  assert.match(archive, /onCancel=.*setConfirmDelete\(false\)/)
  assert.match(status, /dialog\.showModal\(\)/)
  assert.match(status, /onCancel=.*setConfirmCancel\(false\)/)
})

test('compact archive layouts keep meeting evidence reachable from an accessible disclosure', async () => {
  const archive = await read('../src/pages/ArchivePage.tsx')
  const css = await read('../src/redesign.css')
  assert.match(archive, /aria-expanded=\{compactEvidenceOpen\}/)
  assert.match(archive, /className="archive-compact-evidence" aria-label="Видео и кадры встречи"/)
  assert.match(css, /@media \(max-width: 1160px\)[\s\S]*\.archive-evidence-toggle \{ display: inline-flex; \}/)
})
