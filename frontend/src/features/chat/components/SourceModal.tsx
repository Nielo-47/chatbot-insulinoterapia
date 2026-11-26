import { X } from 'lucide-react'
import { useEffect } from 'react'

interface SourceModalItem {
  page?: number
  content?: string
}

interface PageGroup {
  page?: number
  items: SourceModalItem[]
}

interface SourceModalProps {
  title: string
  items: SourceModalItem[]
  onClose: () => void
}

function groupByPage(items: SourceModalItem[]): PageGroup[] {
  const groups = new Map<string, PageGroup>()
  for (const item of items) {
    const key = item.page != null ? `page-${item.page}` : 'no-page'
    const existing = groups.get(key)
    if (existing) {
      existing.items.push(item)
    } else {
      groups.set(key, { page: item.page, items: [item] })
    }
  }
  return [...groups.values()]
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
          {groupByPage(items).map((group) => (
            <section key={group.page ?? 'no-page'} className="rounded-xl border border-emerald-100 bg-emerald-50/50 p-4">
              {group.page != null && (
                <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-emerald-700">
                  Página {group.page}
                </p>
              )}
              <div className="divide-y divide-emerald-100">
                {group.items.map((item, index) => (
                  <div key={index} className={index === 0 ? 'pt-0' : 'pt-3'}>
                    {item.content ? (
                      <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-slate-700">
                        {item.content}
                      </p>
                    ) : (
                      <p className="text-sm italic text-slate-400">Conteúdo não disponível para esta fonte.</p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>
    </div>
  )
}
