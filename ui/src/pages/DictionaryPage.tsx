import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { AlertCircle, ChevronDown, Pencil, Plus, Search, Trash2, Undo2, X } from 'lucide-react'
import { dictionaryApi, type DictionaryEntry, upsertDictionaryEntry } from '../lib/dictionaryApi'
import { getDictionarySettingsDraft, loadDictionarySettingsDraft, setDictionaryAutoLearn, subscribeDictionarySettingsDraft } from '../lib/dictionarySettingsDraft'
import ToastNotice from '../components/ToastNotice'
import MeasuredDisclosure from '../components/MeasuredDisclosure'

export function AutoLearnSetting({ disabled = false }: { disabled?: boolean }) {
  const state = useSyncExternalStore(subscribeDictionarySettingsDraft, getDictionarySettingsDraft, getDictionarySettingsDraft)
  useEffect(() => { void loadDictionarySettingsDraft() }, [])
  return <section className="settings-group dictionary-learning-setting"><h2>Словарь</h2><label className="dictionary-auto-toggle"><span><strong>Запоминать исправления автоматически</strong><small>Новые термины из правок расшифровок попадут в личный словарь.</small>{!state.loaded && <small>Загружаем настройку…</small>}{state.value !== state.savedValue && <small>Изменение сохранится вместе с настройками</small>}{state.error && <small role="alert">{state.error}</small>}</span><input type="checkbox" checked={state.value} disabled={state.busy || !state.loaded || disabled} onChange={(event) => setDictionaryAutoLearn(event.target.checked)} /></label></section>
}

export default function DictionaryPage({ visible = true }: { visible?: boolean }) {
  const [items, setItems] = useState<DictionaryEntry[]>([])
  const [query, setQuery] = useState('')
  const [term, setTerm] = useState('')
  const [definition, setDefinition] = useState('')
  const [editing, setEditing] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)
  const [provenance, setProvenance] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [undoId, setUndoId] = useState('')
  const [notice, setNotice] = useState('')
  const requestGeneration = useRef(0)
  const mounted = useRef(false)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; requestGeneration.current += 1 } }, [])
  const refreshList = useCallback(async () => {
    const generation = ++requestGeneration.current
    try {
      const result = await dictionaryApi.list()
      if (mounted.current && generation === requestGeneration.current) { setItems(result.items); setError('') }
    } catch (reason) {
      if (mounted.current && generation === requestGeneration.current) setError(reason instanceof Error ? reason.message : 'Не удалось загрузить словарь')
    }
  }, [])
  useEffect(() => { if (visible) void Promise.resolve().then(refreshList) }, [visible, refreshList])
  useEffect(() => {
    const refreshWhenVisible = () => { if (visible) void refreshList() }
    window.addEventListener('dictionary-changed', refreshWhenVisible)
    window.addEventListener('learned-terms-added', refreshWhenVisible)
    return () => {
      window.removeEventListener('dictionary-changed', refreshWhenVisible)
      window.removeEventListener('learned-terms-added', refreshWhenVisible)
    }
  }, [visible, refreshList])
  const visibleItems = useMemo(() => { const needle = query.trim().toLocaleLowerCase('ru'); return items.filter((item) => !needle || `${item.term} ${item.definition ?? ''}`.toLocaleLowerCase('ru').includes(needle)) }, [items, query])
  const resetForm = () => { setTerm(''); setDefinition(''); setEditing(null); setExpanded(false) }
  async function submit(event: React.FormEvent) {
    event.preventDefault(); if (!term.trim()) return
    setBusy(true); setError('')
    try { const result = editing ? await dictionaryApi.update(editing, term.trim(), definition.trim()) : await dictionaryApi.add(term.trim(), definition.trim()); setItems((current) => upsertDictionaryEntry(current, result.item)); window.dispatchEvent(new Event('dictionary-changed')); resetForm() }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Не удалось сохранить термин') }
    finally { setBusy(false) }
  }
  async function remove(item: DictionaryEntry) {
    setBusy(true); setError('')
    try { const result = await dictionaryApi.remove(item.id); setItems((current) => current.filter((row) => row.id !== item.id)); window.dispatchEvent(new Event('dictionary-changed')); setUndoId(result.undo_id); setNotice(`«${item.term}» удалён`); window.setTimeout(() => setNotice(''), 7000) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Не удалось удалить термин') }
    finally { setBusy(false) }
  }
  async function undo() {
    if (!undoId) return
    setBusy(true)
    try { const result = await dictionaryApi.undo(undoId); setItems((current) => upsertDictionaryEntry(current, result.item)); window.dispatchEvent(new Event('dictionary-changed')); setUndoId(''); setNotice('Термин восстановлен'); window.setTimeout(() => setNotice(''), 4000) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Не удалось восстановить термин') }
    finally { setBusy(false) }
  }
  function edit(item: DictionaryEntry) { setEditing(item.id); setTerm(item.term); setDefinition(item.definition ?? ''); setExpanded(true); setProvenance(null); document.querySelector('.dictionary-add-form')?.scrollIntoView({ block: 'nearest' }) }
  return <main className="dictionary-page">
    <header className="page-heading"><div><h1>Словарь</h1><p>Термины, которые помогают точнее распознавать речь</p></div><span className="dictionary-count">{items.length}</span></header>
    <label className="dictionary-search"><Search size={17} /><span className="sr-only">Найти термин</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Найти термин" /></label>
    <div className="dictionary-add-area">
      <button type="button" className="dictionary-add-toggle" aria-expanded={expanded} onClick={() => { setExpanded((value) => !value); if (expanded) resetForm() }}>{expanded ? <X size={16} /> : <Plus size={16} />}{editing ? 'Изменить термин' : 'Добавить термин'}<ChevronDown size={15} /></button>
      <MeasuredDisclosure open={expanded} className="dictionary-add-disclosure"><form className="dictionary-add-form" onSubmit={(event) => void submit(event)}><label>Термин<input autoFocus={expanded} tabIndex={expanded ? 0 : -1} value={term} onChange={(event) => setTerm(event.target.value)} required maxLength={64} /></label><label>Определение <span>необязательно</span><input tabIndex={expanded ? 0 : -1} value={definition} onChange={(event) => setDefinition(event.target.value)} /></label><div><button className="primary-button" type="submit" disabled={busy || !term.trim()}>{busy ? 'Сохраняем…' : editing ? 'Сохранить' : 'Добавить'}</button>{editing && <button className="quiet-button" type="button" onClick={resetForm}>Отмена</button>}</div></form></MeasuredDisclosure>
    </div>
    {error && <div className="dictionary-error" role="alert"><AlertCircle size={16} />{error}</div>}
    <section className="dictionary-list" aria-label="Термины">
      {visibleItems.length === 0 ? <div className="dictionary-empty"><strong>{query ? 'Ничего не найдено' : 'Словарь пока пуст'}</strong><span>{query ? 'Попробуйте изменить поиск' : 'Добавленные термины будут доступны для расшифровок'}</span></div> : visibleItems.map((item) => <article className="dictionary-entry" key={item.id}><div className="dictionary-entry-copy"><strong>{item.term}</strong>{item.definition && <p>{item.definition}</p>}
        {!!item.provenance?.length && <div className="dictionary-provenance"><button type="button" className="text-action" aria-expanded={provenance === item.id} aria-controls={`dictionary-source-${item.id}`} onClick={() => setProvenance((current) => current === item.id ? null : item.id)}>{provenance === item.id ? 'Скрыть источник' : `Источник · ${item.provenance.length}`}</button><MeasuredDisclosure open={provenance === item.id} className="dictionary-source-disclosure"><ul id={`dictionary-source-${item.id}`}>{item.provenance.map((source, index) => <li key={`${source.meeting_id ?? source.source ?? 'source'}-${index}`}>{source.meeting_title || source.source || source.meeting_id || 'Расшифровка'}{source.created_at ? ` · ${new Intl.DateTimeFormat('ru-RU', { dateStyle: 'medium' }).format(new Date(source.created_at))}` : ''}</li>)}</ul></MeasuredDisclosure></div>}
      </div><div className="dictionary-entry-actions"><button type="button" className="icon-button" aria-label={`Изменить ${item.term}`} title="Изменить" onClick={() => edit(item)}><Pencil size={16} /></button><button type="button" className="icon-button" aria-label={`Удалить ${item.term}`} title="Удалить" disabled={busy} onClick={() => void remove(item)}><Trash2 size={16} /></button></div></article>)}
    </section>
    <ToastNotice show={visible && Boolean(notice)} className="dictionary-toast"><span>{notice}</span>{undoId && <button type="button" onClick={() => void undo()} disabled={busy}><Undo2 size={14} />Отменить</button>}<button type="button" aria-label="Закрыть сообщение" onClick={() => { setUndoId(''); setNotice('') }}><X size={14} /></button></ToastNotice>
  </main>
}
