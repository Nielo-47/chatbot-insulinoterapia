import { useEffect, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'

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
    <form onSubmit={handleSubmit} className="rounded-2xl border border-slate-300 bg-white p-3 shadow-sm">
      <label htmlFor="chat-input" className="mb-2 block text-xs font-semibold uppercase tracking-[0.08em] text-slate-500">
        Sua pergunta
      </label>
      <textarea
        id="chat-input"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder="Digite sua pergunta sobre diabetes ou insulinoterapia"
        className="min-h-24 w-full resize-none rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-cyan-500 focus:ring-2 focus:ring-cyan-100 disabled:cursor-not-allowed disabled:opacity-70"
      />
      <div className="mt-3 flex items-center justify-between gap-3">
        <p className="text-xs text-slate-500">Enter envia. Shift + Enter quebra linha.</p>
        <button
          type="submit"
          disabled={disabled || value.trim().length === 0}
          className="rounded-xl bg-cyan-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-cyan-700 disabled:cursor-not-allowed disabled:bg-cyan-300"
        >
          {disabled ? 'Enviando...' : 'Enviar'}
        </button>
      </div>
    </form>
  )
}
