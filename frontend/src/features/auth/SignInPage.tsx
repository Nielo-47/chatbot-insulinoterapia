import { LogIn } from 'lucide-react'

import { getAuthentikLoginUrl } from '../../lib/env'
import type { AuthStatus, BackendStatus } from '../../types/app'

interface SignInPageProps {
  backendStatus: BackendStatus
  authStatus: AuthStatus
}

export function SignInPage({ backendStatus, authStatus }: SignInPageProps) {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(209,250,229,0.75),_rgba(255,255,255,1)_48%)] px-4 py-6">
      <div className="mx-auto flex min-h-[calc(100vh-3rem)] max-w-md items-center justify-center">
        <div className="w-full overflow-hidden rounded-[2rem] border border-slate-200 bg-white/90 p-8 shadow-2xl shadow-slate-200/50 sm:p-10">
          <h1 className="font-serif text-3xl font-semibold text-slate-900 sm:text-4xl">
            Assistente de insulinoterapia
          </h1>
          <span className="mt-3 inline-flex items-center rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-amber-700">
            Teste Fechado
          </span>

          <p className="mt-4 text-sm leading-6 text-slate-600">
            O acesso e feito pelo provedor de identidade. Clique em Entrar para autenticar e continuar.
          </p>

          {backendStatus === 'offline' && (
            <div className="mt-5 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              O backend esta indisponivel no momento. Tente novamente quando o servico estiver online.
            </div>
          )}

          {authStatus === 'expired' && (
            <div className="mt-5 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              Sua sessao expirou. Entre novamente.
            </div>
          )}

          <button
            type="button"
            onClick={() => window.location.assign(getAuthentikLoginUrl())}
            disabled={backendStatus === 'offline'}
            className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-cyan-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-cyan-700 disabled:cursor-not-allowed disabled:bg-cyan-300"
          >
            <LogIn className="h-4 w-4" />
            Entrar com Authentik
          </button>
        </div>
      </div>
    </div>
  )
}
