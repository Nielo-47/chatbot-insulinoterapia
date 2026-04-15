import { useEffect, useMemo, useState } from 'react'
import { BotMessageSquare, LogOut, RefreshCcw, ShieldCheck } from 'lucide-react'

import { ApiError, clearConversation, getConversationHistory, sendQuery } from '../../lib/api'
import type { AuthStatus, BackendStatus } from '../../types/app'
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
  backendStatus: BackendStatus
  authStatus: AuthStatus
  onLogout: (reason?: 'manual' | 'expired' | 'deleted') => Promise<void>
  onDeleteAccount: () => Promise<void>
}

export function ChatPage({ username, backendStatus, authStatus, onLogout, onDeleteAccount }: ChatPageProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([initialMessage])
  const [activeSourcesMessage, setActiveSourcesMessage] = useState<ChatMessage | null>(null)
  const [isSending, setIsSending] = useState(false)
  const [isLoggingOut, setIsLoggingOut] = useState(false)
  const [isDeletingAccount, setIsDeletingAccount] = useState(false)
  const [compressionNotice, setCompressionNotice] = useState<string | null>(null)
  const [localError, setLocalError] = useState<string | null>(null)

  useEffect(() => {
    void (async () => {
      try {
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
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          await onLogout('expired')
          return
        }

        setLocalError(error instanceof Error ? error.message : 'Nao foi possivel carregar o historico.')
      }
    })()
  }, [onLogout])

  const sortedMessages = useMemo(
    () => [...messages].sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()),
    [messages],
  )

  const handleSend = async (value: string) => {
    if (backendStatus === 'offline') {
      return
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: value,
      createdAt: new Date().toISOString(),
    }

    setMessages((current) => [...current, userMessage])
    setIsSending(true)
    setLocalError(null)
    setCompressionNotice(null)

    try {
      const result = await sendQuery({ query: value })
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: result.response,
        createdAt: new Date().toISOString(),
        sources: normalizeSources(result.sources),
        sourceCount: result.source_count,
        summarized: result.summarized,
      }

      setMessages((current) => [...current, assistantMessage])
      if (result.summarized) {
        setCompressionNotice('Historico comprimido automaticamente pelo backend.')
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        await onLogout('expired')
        return
      }

      const errorMessage: ChatMessage = {
        id: crypto.randomUUID(),
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

  const handleClearConversation = async () => {
    try {
      await clearConversation()
      setCompressionNotice(null)
      setLocalError(null)
      setActiveSourcesMessage(null)
      setMessages([initialMessage])
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        await onLogout('expired')
        return
      }

      setLocalError(error instanceof Error ? error.message : 'Nao foi possivel limpar a conversa.')
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
    const shouldDelete = window.confirm('Tem certeza que deseja excluir sua conta? Esta acao nao pode ser desfeita.')

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

      setLocalError(error instanceof Error ? error.message : 'Nao foi possivel excluir a conta.')
    } finally {
      setIsDeletingAccount(false)
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
              <div className={`rounded-xl border px-3 py-2 text-sm ${backendStatus === 'online' ? 'border-emerald-200 bg-emerald-50 text-emerald-800' : 'border-rose-200 bg-rose-50 text-rose-700'}`}>
                Backend: {backendStatus === 'online' ? 'online' : 'indisponivel'}
              </div>
              <div className="rounded-xl border border-cyan-200 bg-cyan-50 px-3 py-2 text-sm text-cyan-800">
                Auth: {authStatus === 'authenticated' ? 'autenticado' : authStatus === 'invalid' ? 'invalido' : authStatus === 'unknown' ? 'indefinido' : 'deslogado'}
              </div>
              <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
                <span className="font-medium">{username}</span>
              </div>
              <button
                type="button"
                onClick={() => void handleClearConversation()}
                disabled={backendStatus === 'offline'}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100"
              >
                <RefreshCcw className="h-4 w-4" />
                Limpar conversa
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
              <button
                type="button"
                onClick={() => void handleDeleteAccount()}
                disabled={isDeletingAccount}
                className="inline-flex items-center gap-2 rounded-xl border border-rose-300 bg-white px-3 py-2 text-sm font-medium text-rose-700 transition hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {isDeletingAccount ? 'Excluindo...' : 'Excluir conta'}
              </button>
            </div>
          </header>

          {backendStatus === 'offline' && (
            <div className="mb-4 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
              Nao foi possivel validar o backend. Verifique se a API esta ativa e tente novamente.
            </div>
          )}

          {compressionNotice && (
            <div className="mb-4 rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800">
              {compressionNotice}
            </div>
          )}

          {localError && (
            <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
              {localError}
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
            <Composer disabled={isSending || backendStatus === 'offline'} onSubmit={handleSend} />
          </div>
        </main>

        <aside className="space-y-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="mb-2 inline-flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.06em] text-slate-700">
              <BotMessageSquare className="h-4 w-4" />
              Conta e status
            </h2>
            <p className="text-xs text-slate-500">Usuario autenticado: {username}</p>
            <p className="mt-2 text-xs text-slate-500">As respostas nao substituem avaliacao medica presencial.</p>
            <p className="mt-2 text-xs text-slate-500">
              Compressao de historico ocorre automaticamente quando necessario e nao pode ser acionada manualmente.
            </p>
          </div>

          <SourceDrawer message={activeSourcesMessage} onClose={() => setActiveSourcesMessage(null)} />
        </aside>
      </div>
    </div>
  )
}
