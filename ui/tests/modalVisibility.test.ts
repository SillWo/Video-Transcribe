import assert from 'node:assert/strict'
import test from 'node:test'
import { readFile } from 'node:fs/promises'
import { shouldShowPreservedModal } from '../src/lib/modalVisibility.ts'

test('preserved-screen modals only enter the top layer while their owning route is visible', () => {
  assert.equal(shouldShowPreservedModal(true, true), true)
  assert.equal(shouldShowPreservedModal(false, true), false)
  assert.equal(shouldShowPreservedModal(true, false), false)
})

test('MeetingWorkspace closes and clears modal state when its preserved route hides', async () => {
  const component = await readFile(new URL('../src/components/MeetingWorkspace.tsx', import.meta.url), 'utf8')
  assert.match(component, /shouldShowPreservedModal\(visible, Boolean\(errorDialog\)\)/)
  assert.match(component, /if \(!visible\) \{[\s\S]*if \(dialog\.open\) dialog\.close\(\)[\s\S]*setErrorDialogState\(null\)/)
  assert.match(component, /shouldShowPreservedModal\(visible, Boolean\(backupConfirmation\)\)/)
})
