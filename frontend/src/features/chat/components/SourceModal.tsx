import { X } from 'lucide-react'
import { useEffect } from 'react'

interface SourceModalItem {
  page?: number
  content?: string
}

interface SourceModalProps {
  title: string
  items: SourceModalItem[]
  onClose: () => void
}

export function SourceModal({ title, items, onClose }: SourceModalProps) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="flex max-h-[80dvh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl bg-white shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-5 py-4">
          <h3 className="break-words text-base font-semibold text-slate-900">{title}</h3>
          <button
            type="button"
            onClick={onClose}
            aria-label="Fechar"
            className="shrink-0 rounded-full border border-slate-300 p-1.5 text-slate-600 transition hover:bg-slate-100"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
          {items.map((item, index) => (
            <section key={index} className="rounded-xl border border-emerald-100 bg-emerald-50/50 p-4">
              {item.page != null && (
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-emerald-700">
                  Página {item.page}
                </p>
              )}
              {item.content ? (
                <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-slate-700">
                  {item.content}
                </p>
              ) : (
                <p className="text-sm italic text-slate-400">Conteúdo não disponível para esta fonte.</p>
              )}
            </section>
          ))}
        </div>
      </div>
    </div>
  )
}
