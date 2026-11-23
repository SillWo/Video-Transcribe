import assert from 'node:assert/strict'
import test from 'node:test'
import { canOpenLinkedMeeting, isEditableTranscription, isMeetingLinkedTranscript, isSameMeetingJobScope, linkedMeetingArchiveTarget } from '../src/lib/meetingTranscript.ts'

test('meeting-linked tasks are identified without changing ordinary audio jobs', () => {
  assert.equal(isMeetingLinkedTranscript('meeting-123'), true)
  assert.equal(isMeetingLinkedTranscript('  '), false)
  assert.equal(isMeetingLinkedTranscript(undefined), false)
  assert.equal(isEditableTranscription('meeting-123', 'completed'), false)
  assert.equal(isEditableTranscription(undefined, 'completed'), true)
  assert.equal(isEditableTranscription(undefined, 'running'), false)
})

test('opening a linked meeting requires the same normalized archive root', () => {
  assert.equal(canOpenLinkedMeeting('meeting-123', 'C:/Archive/', 'c:\\archive'), true)
  assert.equal(canOpenLinkedMeeting('meeting-123', 'D:/Archive', 'C:/Archive'), false)
  assert.equal(canOpenLinkedMeeting('meeting-123', 'C:/Archive', undefined), false)
  assert.equal(canOpenLinkedMeeting('', 'C:/Archive', 'C:/Archive'), false)
})

test('a delayed meeting preview response only applies to the same task, meeting, and archive root', () => {
  const expected = { id: 'job-1', meetingId: 'meeting-1', meetingRoot: 'C:/Archive/' }
  assert.equal(isSameMeetingJobScope(expected, { ...expected, meetingRoot: 'c:\\archive' }), true)
  assert.equal(isSameMeetingJobScope(expected, { ...expected, id: 'job-2' }), false)
  assert.equal(isSameMeetingJobScope(expected, { ...expected, meetingId: 'meeting-2' }), false)
  assert.equal(isSameMeetingJobScope(expected, { ...expected, meetingRoot: 'D:/Archive' }), false)
})

test('linked task navigation targets the meeting in the archive by its exact id', () => {
  assert.equal(linkedMeetingArchiveTarget('meeting id/1'), '/archive?kind=meeting&id=meeting+id%2F1')
})
