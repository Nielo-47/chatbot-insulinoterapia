import { useState } from 'react'
import { LogIn, Mail, Lock, UserPlus } from 'lucide-react'

import { supabase } from '../../lib/supabase'
import { translateAuthError } from '../../lib/authErrors'
import { navigateTo } from '../../lib/router'
import type { AuthStatus, BackendStatus } from '../../types/app'
import { AuthShell, TextField } from './AuthShell'

interface SignInPageProps {
  backendStatus: BackendStatus
  authStatus: AuthStatus
}

export function SignInPage({ backendStatus, authStatus }: SignInPageProps) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      const { error: authError } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password,
      })
      if (authError) {
        setError(translateAuthError(authError))
      }
      // On success the onAuthStateChange handler in App.tsx picks up the new
      // session and renders the chat page.
    } catch {
      setError('Erro inesperado ao tentar entrar.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <AuthShell>
      <h2 className="font-serif text-2xl font-semibold text-slate-900 sm:text-3xl">
        Entrar
      </h2>
      <p className="mt-2 text-sm leading-6 text-slate-600">
        Entre com o e-mail e a senha fornecidos para participar do teste.
      </p>

      {backendStatus === 'offline' && (
        <div className="mt-5 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          O backend está indisponível no momento. Tente novamente quando o
          serviço estiver online.
        </div>
      )}

      {authStatus === 'expired' && (
        <div className="mt-5 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          Sua sessão expirou. Entre novamente.
        </div>
      )}

      <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
        <TextField
          id="email"
          label="E-mail"
          icon={Mail}
          type="email"
          autoComplete="email"
          placeholder="seu@email.com"
          value={email}
          onChange={setEmail}
          disabled={submitting || backendStatus === 'offline'}
        />
        <TextField
          id="password"
          label="Senha"
          icon={Lock}
          type="password"
          autoComplete="current-password"
          placeholder="Sua senha"
          value={password}
          onChange={setPassword}
          disabled={submitting || backendStatus === 'offline'}
        />

        {error && (
          <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={submitting || backendStatus === 'offline'}
          className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-cyan-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-cyan-600/25 transition hover:bg-cyan-700 disabled:cursor-not-allowed disabled:bg-cyan-300 disabled:shadow-none"
        >
          <LogIn className="h-4 w-4" />
          {submitting ? 'Entrando...' : 'Entrar'}
        </button>
      </form>

      <div className="mt-6 border-t border-slate-200 pt-5">
        <button
          type="button"
          onClick={() => navigateTo('/signin')}
          className="inline-flex w-full items-center justify-center gap-2 rounded-2xl border border-emerald-300 bg-emerald-50 px-4 py-3 text-sm font-semibold text-emerald-700 transition hover:bg-emerald-100"
        >
          <UserPlus className="h-4 w-4" />
          Criar conta
        </button>
        <p className="mt-2 text-center text-xs text-slate-500">
          Novo participante? Crie sua conta para entrar.
        </p>
      </div>
    </AuthShell>
  )
}
