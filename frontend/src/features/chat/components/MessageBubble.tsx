import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import type { ChatMessage } from '../../../types/chat'

interface MessageBubbleProps {
  message: ChatMessage
  onShowSources: (message: ChatMessage) => void
}

export function MessageBubble({ message, onShowSources }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const hasSources = !isUser && !!message.sources && message.sources.length > 0
  const previewSources = message.sources?.slice(0, 5) ?? []
  const remainingSources = (message.sources?.length ?? 0) - previewSources.length

  return (
    <article
      onClick={() => {
        if (hasSources) {
          onShowSources(message)
        }
      }}
      className={clsx(
        'group max-w-3xl rounded-2xl border p-4 shadow-sm',
        isUser ? 'ml-auto border-sky-400 bg-sky-50' : 'mr-auto border-slate-200 bg-white',
        hasSources ? 'cursor-pointer transition hover:border-emerald-300' : null,
      )}
    >
      <header className="mb-2 flex items-center justify-between text-xs font-medium uppercase tracking-[0.08em] text-slate-500">
        <span>{isUser ? 'Voce' : 'Assistente'}</span>
        <time>{new Date(message.createdAt).toLocaleTimeString()}</time>
      </header>

      <div className={clsx('markdown-content text-sm leading-6', message.isError ? 'text-rose-700' : 'text-slate-800')}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ node: _node, ...props }) => (
              <a {...props} target="_blank" rel="noreferrer" className="font-medium text-cyan-700 underline decoration-cyan-400 underline-offset-2 hover:text-cyan-800" />
            ),
            code: ({ className, ...props }) => (
              <code {...props} className={clsx('rounded bg-slate-200 px-1 py-0.5 font-mono text-[0.85em]', className)} />
            ),
            pre: ({ className, ...props }) => (
              <pre
                {...props}
                className={clsx('overflow-x-auto rounded-xl bg-slate-900/95 p-3 font-mono text-xs text-slate-100', className)}
              />
            ),
          }}
        >
          {message.content}
        </ReactMarkdown>
      </div>

      {!isUser && message.summarized && (
        <div className="mt-3 inline-flex rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700">
          Historico comprimido automaticamente
        </div>
      )}

      {hasSources && (
        <div className="group/references relative mt-3 inline-flex">
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              onShowSources(message)
            }}
            className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700 transition hover:bg-emerald-100 md:opacity-0 md:group-hover:opacity-100 md:focus-visible:opacity-100"
          >
            Ver referencias ({message.sourceCount ?? previewSources.length + remainingSources})
          </button>

          <div className="pointer-events-none absolute left-0 top-full z-20 mt-2 hidden w-80 max-w-[min(22rem,calc(100vw-2rem))] rounded-2xl border border-emerald-200 bg-white p-3 text-xs text-slate-700 shadow-lg shadow-slate-200/60 group-hover/references:block group-focus-within/references:block">
            <p className="mb-2 font-semibold uppercase tracking-[0.08em] text-emerald-700">Fontes consultadas</p>
            <ul className="space-y-2">
              {previewSources.map((source) => (
                <li key={source.id} className="break-words rounded-lg bg-emerald-50 px-2 py-1">
                  {source.label}
                </li>
              ))}
              {remainingSources > 0 && (
                <li className="text-slate-500">+ {remainingSources} outras referencias</li>
              )}
            </ul>
          </div>
        </div>
      )}
    </article>
  )
}
