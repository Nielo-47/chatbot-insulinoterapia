import { useState, type FormEvent } from 'react'
import { LogIn } from 'lucide-react'

import type { AuthStatus, BackendStatus } from '../../types/app'

interface LoginPageProps {
  onLogin: (username: string, password: string) => Promise<void>
  errorMessage?: string | null
  isSubmitting?: boolean
  backendStatus: BackendStatus
  authStatus: AuthStatus
}

export function LoginPage({ onLogin, errorMessage, isSubmitting, backendStatus, authStatus }: LoginPageProps) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    await onLogin(username.trim(), password)
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(209,250,229,0.75),_rgba(255,255,255,1)_48%)] px-4 py-6">
      <div className="mx-auto flex min-h-[calc(100vh-3rem)] max-w-6xl items-center justify-center">
        <div className="grid w-full overflow-hidden rounded-[2rem] border border-slate-200 bg-white/90 shadow-2xl shadow-slate-200/50 lg:grid-cols-[1.1fr_0.9fr]">
          <section className="flex flex-col justify-between gap-8 bg-slate-950 px-8 py-10 text-white sm:px-10 sm:py-12">
            <div>
              <h1 className="font-serif text-4xl font-semibold leading-tight text-white sm:text-5xl">
                Assistente de insulinoterapia
              </h1>
              <p className="mt-4 max-w-xl text-sm leading-6 text-slate-300 sm:text-base">
                Entre com sua conta para acessar o chat e manter o historico da conversa vinculado ao seu usuario.
              </p>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/5 p-5 text-sm text-slate-200 backdrop-blur">
              <p className="font-medium text-white">Conta de teste</p>
              <p className="mt-2 text-slate-300">Use as credenciais exibidas ao lado para entrar e validar o fluxo manualmente.</p>
            </div>
          </section>

          <section className="flex items-center justify-center px-6 py-10 sm:px-10 sm:py-12">
            <form onSubmit={handleSubmit} className="w-full max-w-md space-y-5">
              <div>
                <h2 className="font-serif text-3xl font-semibold text-slate-900">Entrar</h2>
                <p className="mt-2 text-sm text-slate-600">Entre com o usuario seedado para abrir o chat.</p>
              </div>

              {backendStatus === 'offline' && (
                <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                  O backend esta indisponivel no momento. Tente novamente quando o servico estiver online.
                </div>
              )}

              {authStatus === 'invalid' && (
                <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                  Sua sessao expirou ou ficou invalida. Entre novamente.
                </div>
              )}

              <label className="block space-y-2">
                <span className="text-sm font-medium text-slate-700">Usuario</span>
                <input
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  autoComplete="username"
                  className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-cyan-500 focus:ring-4 focus:ring-cyan-100"
                  placeholder="demo"
                />
              </label>

              <label className="block space-y-2">
                <span className="text-sm font-medium text-slate-700">Senha</span>
                <input
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  autoComplete="current-password"
                  className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-cyan-500 focus:ring-4 focus:ring-cyan-100"
                  placeholder="demo12345"
                />
              </label>

              {errorMessage && (
                <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{errorMessage}</div>
              )}

              <button
                type="submit"
                disabled={isSubmitting || backendStatus === 'offline'}
                className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-cyan-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-cyan-700 disabled:cursor-not-allowed disabled:bg-cyan-300"
              >
                <LogIn className="h-4 w-4" />
                {isSubmitting ? 'Entrando...' : backendStatus === 'offline' ? 'Backend indisponivel' : 'Entrar'}
              </button>

              <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-4 py-3 text-xs leading-5 text-slate-600">
                Credenciais seedadas: <span className="font-semibold">demo</span> / <span className="font-semibold">demo12345</span>
              </div>
            </form>
          </section>
        </div>
      </div>
    </div>
  )
}