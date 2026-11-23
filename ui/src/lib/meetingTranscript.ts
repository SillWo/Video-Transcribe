export function isMeetingLinkedTranscript(meetingId?: string | null) {
  return Boolean(meetingId?.trim())
}

export function isEditableTranscription(meetingId: string | null | undefined, status: string) {
  return status === 'completed' && !isMeetingLinkedTranscript(meetingId)
}

function normalizeRoot(value?: string | null) {
  const normalized = (value ?? '').trim().replace(/[\\/]+/g, '\\').toLocaleLowerCase('en-US')
  return /^[a-z]:\\$/.test(normalized) ? normalized : normalized.replace(/\\+$/, '')
}

export function canOpenLinkedMeeting(meetingId: string | null | undefined, meetingRoot: string | null | undefined, currentRoot: string | null | undefined) {
  const expectedRoot = normalizeRoot(meetingRoot)
  const selectedRoot = normalizeRoot(currentRoot)
  return Boolean(meetingId?.trim() && expectedRoot && selectedRoot && expectedRoot === selectedRoot)
}

export function isSameMeetingJobScope(
  expected: { id: string; meetingId?: string | null; meetingRoot?: string | null },
  actual: { id: string; meetingId?: string | null; meetingRoot?: string | null },
) {
  const expectedRoot = normalizeRoot(expected.meetingRoot)
  return Boolean(
    expected.id === actual.id &&
    expected.meetingId && expected.meetingId === actual.meetingId &&
    expectedRoot && expectedRoot === normalizeRoot(actual.meetingRoot),
  )
}

export function linkedMeetingArchiveTarget(meetingId: string) {
  const params = new URLSearchParams({ kind: 'meeting', id: meetingId })
  return `/archive?${params.toString()}`
}
