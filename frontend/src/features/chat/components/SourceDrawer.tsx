import type { ChatMessage } from '../../../types/chat'

interface SourceDrawerProps {
  message: ChatMessage | null
  onClose: () => void
}

export function SourceDrawer({ message, onClose }: SourceDrawerProps) {
  if (!message || !message.sources || message.sources.length === 0) {
    return (
      <aside className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <h3 className="text-sm font-semibold text-slate-900">Referencias</h3>
        <p className="mt-2 text-sm text-slate-500">Selecione uma resposta para ver as fontes utilizadas.</p>
      </aside>
    )
  }

  return (
    <aside className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-emerald-900">Referencias da resposta</h3>
        <button
          type="button"
          onClick={onClose}
          className="rounded-full border border-emerald-300 px-2 py-1 text-xs font-medium text-emerald-800 transition hover:bg-emerald-100"
        >
          Limpar
        </button>
      </div>
      <ol className="space-y-2 pl-4 text-sm text-emerald-900">
        {message.sources.map((source) => (
          <li key={source.id} className="list-decimal rounded-lg bg-white/80 px-2 py-1">
            {source.label}
          </li>
        ))}
      </ol>
    </aside>
  )
}
