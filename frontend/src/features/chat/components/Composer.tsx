import type { FormEvent, KeyboardEvent } from 'react'
import { useEffect, useState } from 'react'
import { Send } from 'lucide-react'

import { useDebounce } from '../../../hooks/useDebounce'
import { draftStorage } from '../../../lib/storage'

interface ComposerProps {
  disabled?: boolean
  onSubmit: (value: string) => Promise<void>
}

export function Composer({ disabled, onSubmit }: ComposerProps) {
  const [value, setValue] = useState(() => draftStorage.getDraft() ?? '')
  const debouncedValue = useDebounce(value, 500)

  useEffect(() => {
    draftStorage.saveDraft(debouncedValue)
  }, [debouncedValue])

  useEffect(() => {
    return () => {
      draftStorage.clearDraft()
    }
  }, [])

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    const trimmed = value.trim()

    if (!trimmed || disabled) {
      return
    }

    await onSubmit(trimmed)
    setValue('')
    draftStorage.clearDraft()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void handleSubmit(event)
    }
  }

  return (
    <>
      {/* Mobile: icon button beside textarea */}
      <form onSubmit={handleSubmit} className="rounded-2xl border border-slate-300 bg-white p-2 shadow-sm lg:hidden">
        <div className="flex items-end gap-2">
          <textarea
            id="chat-input"
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            rows={1}
            placeholder="Digite sua pergunta..."
            className="min-h-10 max-h-32 flex-1 resize-none rounded-xl border border-sky-400 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-cyan-500 focus:ring-2 focus:ring-cyan-100 disabled:cursor-not-allowed disabled:opacity-70"
          />
          <button
            type="submit"
            disabled={disabled || value.trim().length === 0}
            className="shrink-0 rounded-xl bg-cyan-600 p-2.5 text-white transition hover:bg-cyan-700 disabled:cursor-not-allowed disabled:bg-cyan-300"
            aria-label="Enviar"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </form>

      {/* Desktop: full layout with label and text button */}
      <form onSubmit={handleSubmit} className="hidden rounded-2xl border border-slate-300 bg-white p-3 shadow-sm lg:block">
        <label htmlFor="chat-input-desktop" className="mb-2 block text-xs font-semibold uppercase tracking-[0.08em] text-slate-500">
          Sua pergunta
        </label>
        <textarea
          id="chat-input-desktop"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder="Digite sua pergunta sobre diabetes ou insulinoterapia"
          className="min-h-28 w-full resize-none rounded-xl border border-sky-400 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-cyan-500 focus:ring-2 focus:ring-cyan-100 disabled:cursor-not-allowed disabled:opacity-70"
        />
        <div className="mt-3 flex items-center justify-between gap-3">
          <p className="text-xs text-slate-500">Enter envia. Shift + Enter quebra linha.</p>
          <button
            type="submit"
            disabled={disabled || value.trim().length === 0}
            className="rounded-xl bg-cyan-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-cyan-700 disabled:cursor-not-allowed disabled:bg-cyan-300"
          >
            {disabled ? 'Enviando...' : 'Enviar'}
          </button>
        </div>
      </form>
    </>
  )
}
