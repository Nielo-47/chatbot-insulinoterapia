import { useMemo, useState } from 'react'

import { friendlyFileName, truncate } from '../../../lib/text'
import type { ChatMessage, ChatSource } from '../../../types/chat'
import { SourceModal } from './SourceModal'

interface SourceDrawerProps {
  message: ChatMessage | null
  onClose: () => void
}

interface SourceGroup {
  path: string
  sources: ChatSource[]
  pages: number[]
}

function groupByFile(sources: ChatSource[]): SourceGroup[] {
  const groups = new Map<string, SourceGroup>()
  for (const source of sources) {
    const existing = groups.get(source.path)
    if (existing) {
      existing.sources.push(source)
      if (source.page != null && !existing.pages.includes(source.page)) {
        existing.pages.push(source.page)
      }
    } else {
      groups.set(source.path, {
        path: source.path,
        sources: [source],
        pages: source.page != null ? [source.page] : [],
      })
    }
  }
  return [...groups.values()]
}

export function SourceDrawer({ message, onClose }: SourceDrawerProps) {
  const [selectedPath, setSelectedPath] = useState<string | null>(null)

  const groups = useMemo(() => groupByFile(message?.sources ?? []), [message])

  if (!message || !message.sources || message.sources.length === 0) {
    return (
      <aside className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <h3 className="text-sm font-semibold text-slate-900">Referências</h3>
        <p className="mt-2 text-sm text-slate-500">Selecione uma resposta para ver as fontes utilizadas.</p>
      </aside>
    )
  }

  const selectedGroup = groups.find((group) => group.path === selectedPath) ?? null

  return (
    <aside className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-emerald-900">Referências da resposta</h3>
        <button
          type="button"
          onClick={onClose}
          className="rounded-full border border-emerald-300 px-2 py-1 text-xs font-medium text-emerald-800 transition hover:bg-emerald-100"
        >
          Limpar
        </button>
      </div>
      <ol className="max-h-72 space-y-2 overflow-y-auto pr-1">
        {groups.map((group) => {
          const firstContent = group.sources.find((source) => source.content)?.content
          const pageLabel =
            group.pages.length === 1
              ? `Página ${group.pages[0]}`
              : group.pages.length > 1
                ? `Páginas ${group.pages.join(', ')}`
                : null
          return (
            <li key={group.path}>
              <button
                type="button"
                onClick={() => setSelectedPath(group.path)}
                className="w-full rounded-lg border border-emerald-200 bg-white/80 px-3 py-2 text-left transition hover:border-emerald-400 hover:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-300"
              >
                <span className="flex items-center justify-between gap-2">
                  <span className="break-words font-medium text-emerald-900">
                    {friendlyFileName(group.path)}
                  </span>
                  <span className="shrink-0 text-xs font-medium text-emerald-700">
                    {group.sources.length > 1 ? `${group.sources.length} trechos` : 'Abrir'}
                  </span>
                </span>
                {pageLabel && <span className="text-xs text-slate-500">{pageLabel}</span>}
                {firstContent && (
                  <span className="mt-1 block break-words text-sm leading-snug text-slate-600">
                    {truncate(firstContent, 160)}
                  </span>
                )}
              </button>
            </li>
          )
        })}
      </ol>
      {selectedGroup && (
        <SourceModal
          title={friendlyFileName(selectedGroup.path)}
          items={selectedGroup.sources.map((source) => ({ page: source.page, content: source.content }))}
          onClose={() => setSelectedPath(null)}
        />
      )}
    </aside>
  )
}
