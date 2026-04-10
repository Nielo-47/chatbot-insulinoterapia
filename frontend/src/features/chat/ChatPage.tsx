import { useEffect, useMemo, useState } from 'react'
import { BotMessageSquare, LogOut, RefreshCcw, ShieldCheck } from 'lucide-react'
import { v4 as uuidv4 } from 'uuid'

import { clearSession, getCurrentUser, sendQuery, getConversationHistory } from '../../lib/api'
import { sessionStorageService } from '../../lib/storage'
import type { ChatMessage } from '../../types/chat'
import { Composer } from './components/Composer'
import { MessageBubble } from './components/MessageBubble'
import { SourceDrawer } from './components/SourceDrawer'

const initialMessage: ChatMessage = {
  id: 'welcome',
  role: 'assistant',
  content:
    'Ola. Sou seu assistente de insulinoterapia. Faca perguntas sobre aplicacao, rotina e cuidados com diabetes para receber orientacoes seguras.',
  createdAt: new Date().toISOString(),
}

function normalizeSources(sources: string[]) {
  return sources.map((source, index) => ({ id: `${index}-${source.slice(0, 24)}`, label: source }))
}

interface ChatPageProps {
  username: string
  onLogout: () => Promise<void>
}

export function ChatPage({ username, onLogout }: ChatPageProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([initialMessage])
  const [activeSourcesMessage, setActiveSourcesMessage] = useState<ChatMessage | null>(null)
  const [sessionId, setSessionId] = useState<string>('')
  const [isSending, setIsSending] = useState(false)
  const [backendReady, setBackendReady] = useState<boolean | null>(null)
  const [isLoggingOut, setIsLoggingOut] = useState(false)

  useEffect(() => {
    const storedSessionId = sessionStorageService.getSessionId() || uuidv4()
    sessionStorageService.setSessionId(storedSessionId)
    setSessionId(storedSessionId)
  }, [])

  useEffect(() => {
    void (async () => {
      try {
        await getCurrentUser()
        setBackendReady(true)
        // Load conversation history after confirming user is authenticated
        const history = await getConversationHistory()
        if (history.length > 0) {
          const loadedMessages: ChatMessage[] = history.map((msg, index) => ({
            id: `history-${index}`,
            role: msg.role as 'user' | 'assistant',
            content: msg.content,
            createdAt: new Date().toISOString(),
          }))
          // Load conversation history after the welcome message
          setMessages([initialMessage, ...loadedMessages])
        }
      } catch {
        setBackendReady(false)
      }
    })()
  }, [])

  const sortedMessages = useMemo(
    () => [...messages].sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()),
    [messages],
  )

  const handleSend = async (value: string) => {
    if (!sessionId) {
      return
    }

    const userMessage: ChatMessage = {
      id: uuidv4(),
      role: 'user',
      content: value,
      createdAt: new Date().toISOString(),
    }

    setMessages((current) => [...current, userMessage])
    setIsSending(true)

    try {
      const result = await sendQuery({ query: value, session_id: sessionId })
      const assistantMessage: ChatMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: result.response,
        createdAt: new Date().toISOString(),
        sources: normalizeSources(result.sources),
        sourceCount: result.source_count,
        summarized: result.summarized,
      }

      setMessages((current) => [...current, assistantMessage])

      if (result.session_id !== sessionId) {
        setSessionId(result.session_id)
        sessionStorageService.setSessionId(result.session_id)
      }
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: error instanceof Error ? error.message : 'Erro inesperado na consulta.',
        createdAt: new Date().toISOString(),
        isError: true,
      }
      setMessages((current) => [...current, errorMessage])
    } finally {
      setIsSending(false)
    }
  }

  const handleNewSession = async () => {
    if (sessionId) {
      try {
        await clearSession(sessionId)
      } catch {
        // Keep UI responsive even if session cleanup fails.
      }
    }

    const nextSessionId = uuidv4()
    setSessionId(nextSessionId)
    sessionStorageService.setSessionId(nextSessionId)
    setActiveSourcesMessage(null)
    setMessages([initialMessage])
  }

  const handleLogout = async () => {
    setIsLoggingOut(true)
    try {
      await onLogout()
    } finally {
      setIsLoggingOut(false)
    }
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(220,252,231,0.8),_rgba(255,255,255,1)_45%)]">
      <div className="mx-auto grid min-h-screen max-w-7xl grid-cols-1 gap-5 px-4 py-5 lg:grid-cols-[2.2fr_1fr] lg:px-8 lg:py-8">
        <main className="flex flex-col rounded-3xl border border-slate-200 bg-white/90 p-4 shadow-xl shadow-slate-200/40 backdrop-blur lg:p-6">
          <header className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-4">
            <div>
              <p className="mb-1 inline-flex items-center gap-2 rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.08em] text-cyan-700">
                <ShieldCheck className="h-3.5 w-3.5" />
                Assistente Clinico
              </p>
              <h1 className="font-serif text-2xl font-semibold text-slate-900 lg:text-3xl">Chatbot de Insulinoterapia</h1>
              <p className="mt-1 text-sm text-slate-600">Perguntas e respostas com suporte de base de conhecimento e referencias.</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
                <span className="font-medium">{username}</span>
              </div>
              <button
                type="button"
                onClick={() => void handleNewSession()}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100"
              >
                <RefreshCcw className="h-4 w-4" />
                Nova conversa
              </button>
              <button
                type="button"
                onClick={() => void handleLogout()}
                disabled={isLoggingOut}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-70"
              >
                <LogOut className="h-4 w-4" />
                {isLoggingOut ? 'Saindo...' : 'Sair'}
              </button>
            </div>
          </header>

          {backendReady === false && (
            <div className="mb-4 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
              Nao foi possivel validar o backend. Verifique se a API esta ativa e tente novamente.
            </div>
          )}

          <section className="flex-1 space-y-3 overflow-y-auto pr-1">
            {sortedMessages.map((message) => (
              <MessageBubble key={message.id} message={message} onShowSources={setActiveSourcesMessage} />
            ))}

            {isSending && (
              <article className="mr-auto max-w-md rounded-2xl border border-cyan-200 bg-cyan-50 p-4 text-sm text-cyan-800">
                Processando resposta...
              </article>
            )}
          </section>

          <div className="mt-4">
            <Composer disabled={isSending || backendReady === false} onSubmit={handleSend} />
          </div>
        </main>

        <aside className="space-y-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="mb-2 inline-flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.06em] text-slate-700">
              <BotMessageSquare className="h-4 w-4" />
              Sessao
            </h2>
            <p className="break-all text-xs text-slate-500">ID: {sessionId || 'carregando...'}</p>
            <p className="mt-2 text-xs text-slate-500">As respostas nao substituem avaliacao medica presencial.</p>
          </div>

          <SourceDrawer message={activeSourcesMessage} onClose={() => setActiveSourcesMessage(null)} />
        </aside>
      </div>
    </div>
  )
}
