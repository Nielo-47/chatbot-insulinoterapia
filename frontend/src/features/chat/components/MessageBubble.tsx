import clsx from 'clsx'

import type { ChatMessage } from '../../../types/chat'

interface MessageBubbleProps {
  message: ChatMessage
  onShowSources: (message: ChatMessage) => void
}

export function MessageBubble({ message, onShowSources }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <article className={clsx('max-w-3xl rounded-2xl border p-4 shadow-sm', isUser ? 'ml-auto border-sky-400 bg-sky-50' : 'mr-auto border-slate-200 bg-white')}>
      <header className="mb-2 flex items-center justify-between text-xs font-medium uppercase tracking-[0.08em] text-slate-500">
        <span>{isUser ? 'Voce' : 'Assistente'}</span>
        <time>{new Date(message.createdAt).toLocaleTimeString()}</time>
      </header>

      <p className={clsx('whitespace-pre-wrap text-sm leading-6', message.isError ? 'text-rose-700' : 'text-slate-800')}>{message.content}</p>

      {!isUser && message.sources && message.sources.length > 0 && (
        <button
          type="button"
          onClick={() => onShowSources(message)}
          className="mt-3 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700 transition hover:bg-emerald-100"
        >
          Ver referencias ({message.sourceCount ?? message.sources.length})
        </button>
      )}
    </article>
  )
}
