import { BotMessageSquare, BookOpenText, LogOut, RefreshCcw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

let messageIdCounter = 0

function createMessageId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  messageIdCounter += 1
  return `msg-${messageIdCounter}`
}

import { ApiError, clearConversation, getConversationHistory, sendQuery } from '../../lib/api'
import type { AuthStatus, BackendStatus } from '../../types/app'
import type { ChatMessage } from '../../types/chat'
import { Composer } from './components/Composer'
import { FollowUpSuggestions } from './components/FollowUpSuggestions'
import { MessageBubble } from './components/MessageBubble'
import { SourceDrawer } from './components/SourceDrawer'

const initialMessage: ChatMessage = {
  id: 'welcome',
  role: 'assistant',
  content:
    'Olá. Sou seu assistente de insulinoterapia. Faça perguntas sobre aplicação, rotina e cuidados com diabetes para receber orientações seguras.',
  createdAt: new Date().toISOString(),
}

const defaultSuggestions = [
  'Como aplicar a insulina com caneta?',
  'Onde devo guardar minha insulina?',
  'O que fazer em caso de hipoglicemia?',
]

function normalizeSources(sources: Array<{path: string, page?: number, content?: string}>) {
  return sources.map((source, index) => ({
    id: `${index}-${source.path.slice(0, 24)}-${source.page || '0'}`,
    path: source.path,
    page: source.page,
    content: source.content,
  }));
}

const SUGGESTIONS_STORAGE_PREFIX = 'chat-followup-suggestions'

function suggestionsStorageKey(username: string): string {
  return `${SUGGESTIONS_STORAGE_PREFIX}:${username}`
}

function readCachedSuggestions(username: string): string[] | null {
  try {
    const raw = window.localStorage.getItem(suggestionsStorageKey(username))
    if (!raw) {
      return null
    }
    const parsed: unknown = JSON.parse(raw)
    if (Array.isArray(parsed) && parsed.every((item) => typeof item === 'string')) {
      return parsed as string[]
    }
    return null
  } catch {
    return null
  }
}

function writeCachedSuggestions(username: string, questions: string[]): void {
  try {
    window.localStorage.setItem(suggestionsStorageKey(username), JSON.stringify(questions))
  } catch {
    // Storage unavailable (private mode / quota) — suggestions simply won't persist.
  }
}

function clearCachedSuggestions(username: string): void {
  try {
    window.localStorage.removeItem(suggestionsStorageKey(username))
  } catch {
    // Ignore storage errors when clearing.
  }
}

interface ChatPageProps {
  username: string
  backendStatus: BackendStatus
  authStatus: AuthStatus
  onLogout: (reason?: 'manual' | 'expired' | 'deleted') => Promise<void>
  onDeleteAccount: () => Promise<void>
}

export function ChatPage({ username, backendStatus, authStatus, onLogout, onDeleteAccount }: ChatPageProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([initialMessage])
  const [cachedSuggestions, setCachedSuggestions] = useState<string[] | null>(() => readCachedSuggestions(username))
  const [activeSourcesMessage, setActiveSourcesMessage] = useState<ChatMessage | null>(null)
  const [isSending, setIsSending] = useState(false)
  const [isLoggingOut, setIsLoggingOut] = useState(false)
  const [isDeletingAccount, setIsDeletingAccount] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)
  const isSendingRef = useRef(false)
  const onLogoutRef = useRef(onLogout)
  onLogoutRef.current = onLogout

  useEffect(() => {
    void (async () => {
      try {
        const history = await getConversationHistory()
        const loadedMessages: ChatMessage[] = history.map((msg, index) => ({
          id: `history-${index}`,
          role: msg.role as 'user' | 'assistant',
          content: msg.content,
          createdAt: new Date().toISOString(),
          sources: normalizeSources(msg.sources),
        }))
        // Seed history only once, and never clobber messages the user already
        // sent while the fetch was in flight (that replace could corrupt the
        // list and, together with unstable keys, produce duplicate bubbles).
        setMessages((current) => {
          if (current.length > 1) {
            return current
          }
          return [initialMessage, ...loadedMessages]
        })
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          await onLogoutRef.current('expired')
          return
        }

        setLocalError(error instanceof Error ? error.message : 'Não foi possível carregar o histórico.')
      }
    })()
    // Run once on mount: onLogout changes identity on every App re-render, and
    // re-running this effect would re-fetch and replace the message list.
  }, [])

  const sortedMessages = useMemo(
    () => [...messages].sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()),
    [messages],
  )

  // Show the latest in-session assistant answer's follow-up questions; fall
  // back to the last cached ones (survives page refresh), then to templates.
  const activeSuggestions = useMemo(() => {
    for (let i = sortedMessages.length - 1; i >= 0; i -= 1) {
      const message = sortedMessages[i]
      if (message.role === 'assistant' && !message.isError && message.followUpQuestions?.length) {
        return message.followUpQuestions
      }
    }
    return cachedSuggestions ?? defaultSuggestions
  }, [sortedMessages, cachedSuggestions])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [sortedMessages.length, isSending])

  const handleSend = async (value: string) => {
    if (isSendingRef.current) {
      // Guard against concurrent submissions (double Enter / double click /
      // suggestion clicks): a second send would duplicate the message pair.
      return
    }

    if (backendStatus === 'offline') {
      const offlineMessage: ChatMessage = {
        id: createMessageId(),
        role: 'assistant',
        content: 'Não foi possível enviar sua mensagem porque o backend está indisponível no momento.',
        createdAt: new Date().toISOString(),
        isError: true,
      }
      setLocalError('Backend indisponível. Verifique se a API está ativa e tente novamente.')
      setMessages((current) => [...current, offlineMessage])
      return
    }

    const userMessage: ChatMessage = {
      id: createMessageId(),
      role: 'user',
      content: value,
      createdAt: new Date().toISOString(),
    }

    setMessages((current) => [...current, userMessage])
    isSendingRef.current = true
    setIsSending(true)
    setLocalError(null)

    try {
      const result = await sendQuery({ query: value })
      const assistantMessage: ChatMessage = {
        id: createMessageId(),
        role: 'assistant',
        content: result.response,
        createdAt: new Date().toISOString(),
        sources: normalizeSources(result.sources),
        summarized: result.summarized,
        followUpQuestions: result.followUpQuestions,
      }

      setMessages((current) => [...current, assistantMessage])
      if (result.followUpQuestions?.length) {
        setCachedSuggestions(result.followUpQuestions)
        writeCachedSuggestions(username, result.followUpQuestions)
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        await onLogoutRef.current('expired')
        return
      }

      const errorMessage: ChatMessage = {
        id: createMessageId(),
        role: 'assistant',
        content: error instanceof Error ? error.message : 'Erro inesperado na consulta.',
        createdAt: new Date().toISOString(),
        isError: true,
      }
      setMessages((current) => [...current, errorMessage])
    } finally {
      isSendingRef.current = false
      setIsSending(false)
    }
  }

  const handleClearConversation = async () => {
    try {
      await clearConversation()
      setLocalError(null)
      setActiveSourcesMessage(null)
      setMessages([initialMessage])
      setCachedSuggestions(null)
      clearCachedSuggestions(username)
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        await onLogout('expired')
        return
      }

      setLocalError(error instanceof Error ? error.message : 'Não foi possível limpar a conversa.')
    }
  }

  const handleLogout = async () => {
    setIsLoggingOut(true)
    try {
      await onLogout('manual')
    } finally {
      setIsLoggingOut(false)
    }
  }

  const handleDeleteAccount = async () => {
    const shouldDelete = window.confirm('Tem certeza que deseja excluir sua conta? Esta ação não pode ser desfeita.')

    if (!shouldDelete) {
      return
    }

    setIsDeletingAccount(true)
    try {
      await onDeleteAccount()
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        await onLogout('expired')
        return
      }

      setLocalError(error instanceof Error ? error.message : 'Não foi possível excluir a conta.')
    } finally {
      setIsDeletingAccount(false)
    }
  }

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-[radial-gradient(circle_at_top,_rgba(220,252,231,0.8),_rgba(255,255,255,1)_45%)]">
      <div className="mx-auto flex w-full max-w-7xl flex-1 min-h-0 flex-col gap-5 px-4 py-5 lg:flex-row lg:px-8 lg:py-8">
        <main className="flex min-h-0 flex-1 flex-col rounded-3xl border border-slate-200 bg-white/90 p-4 shadow-xl shadow-slate-200/40 backdrop-blur lg:p-6">
          <header className="mb-3 flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
            <div>
              <h1 className="font-serif text-2xl font-semibold text-slate-900 lg:text-3xl">
                Chatbot de Insulinoterapia
                <span className="ml-2 inline-flex items-center rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-[10px] font-semibold uppercase leading-none tracking-wide text-amber-700">
                  Teste Fechado
                </span>
              </h1>
            </div>
          </header>

          {backendStatus === 'offline' && (
            <div className="mb-4 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
              Não foi possível validar o backend. Verifique se a API está ativa e tente novamente.
            </div>
          )}

          {localError && (
            <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
              {localError}
            </div>
          )}

          <section className="chat-scroll-area min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
            {sortedMessages.map((message) => (
              <MessageBubble key={message.id} message={message} onShowSources={setActiveSourcesMessage} />
            ))}

            {isSending && (
              <article className="mr-auto max-w-md rounded-2xl border border-cyan-200 bg-cyan-50 p-4 text-sm text-cyan-800">
                Processando resposta...
              </article>
            )}
            <div ref={messagesEndRef} aria-hidden="true" />
          </section>

          <div className="mt-4">
            <FollowUpSuggestions suggestions={activeSuggestions} disabled={isSending} onSelect={(question) => void handleSend(question)} />
            <Composer disabled={isSending} onSubmit={handleSend} />
          </div>
        </main>

        <aside className="max-h-[40dvh] w-full min-h-0 shrink-0 overflow-y-auto lg:max-h-none lg:w-80">
          <div className="space-y-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="mb-2 inline-flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.06em] text-slate-700">
              <BookOpenText className="h-4 w-4" />
              Sobre o assistente
            </h2>
            <p className="text-sm text-slate-600">Perguntas e respostas com suporte de base de conhecimento e referências.</p>
            <p className="mt-2 text-xs text-slate-500">As respostas não substituem avaliação médica presencial.</p>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="mb-2 inline-flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.06em] text-slate-700">
              <BotMessageSquare className="h-4 w-4" />
              Conta
            </h2>
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
              <span className="font-medium">{username}</span>
            </div>
            <div className="mt-3 space-y-2">
              <button
                type="button"
                onClick={() => void handleClearConversation()}
                disabled={backendStatus === 'offline'}
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-70"
              >
                <RefreshCcw className="h-4 w-4" />
                Limpar conversa
              </button>
              <button
                type="button"
                onClick={() => void handleLogout()}
                disabled={isLoggingOut}
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-70"
              >
                <LogOut className="h-4 w-4" />
                {isLoggingOut ? 'Saindo...' : 'Sair'}
              </button>
              <button
                type="button"
                onClick={() => void handleDeleteAccount()}
                disabled={isDeletingAccount}
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-rose-300 bg-white px-3 py-2 text-sm font-medium text-rose-700 transition hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {isDeletingAccount ? 'Excluindo...' : 'Excluir conta'}
              </button>
            </div>
            {backendStatus === 'offline' && (
              <p className="mt-3 rounded-xl border border-rose-200 bg-rose-50 p-2 text-xs text-rose-700">
                O backend está indisponível no momento.
              </p>
            )}
            {(authStatus === 'expired' || authStatus === 'unknown') && (
              <p className="mt-2 rounded-xl border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800">
                Sessão com status {authStatus === 'expired' ? 'expirado' : 'indefinido'}.
              </p>
            )}
          </div>

          <SourceDrawer message={activeSourcesMessage} onClose={() => setActiveSourcesMessage(null)} />
          </div>
        </aside>
      </div>
    </div>
  )
}
