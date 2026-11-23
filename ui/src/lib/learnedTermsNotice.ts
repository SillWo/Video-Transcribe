export type LearnedTermUndo = { term: string; undo_id: string }

export function notifyLearnedTerms(terms: Array<{ term?: string; undo_id?: string }> | undefined) {
  const undoable = (terms ?? []).filter((item): item is { term: string; undo_id: string } => Boolean(item.term && item.undo_id))
  if (undoable.length) window.dispatchEvent(new CustomEvent<LearnedTermUndo[]>('learned-terms-added', { detail: undoable }))
}
