import { useState } from 'react'
import { ArrowLeft, Mail, Lock, UserPlus } from 'lucide-react'

import { supabase } from '../../lib/supabase'
import { translateAuthError } from '../../lib/authErrors'
import { navigateTo } from '../../lib/router'
import type { AuthStatus, BackendStatus } from '../../types/app'
import { AuthShell, TextField } from './AuthShell'

interface SignUpPageProps {
  backendStatus: BackendStatus
  authStatus: AuthStatus
}

export function SignUpPage({ backendStatus, authStatus }: SignUpPageProps) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError(null)
    setInfo(null)

    if (password.length < 6) {
      setError('A senha deve ter pelo menos 6 caracteres.')
      return
    }
    if (password !== confirmPassword) {
      setError('As senhas não coincidem.')
      return
    }

    setSubmitting(true)
    try {
      const { data, error: signUpError } = await supabase.auth.signUp({
        email: email.trim(),
        password,
      })
      if (signUpError) {
        setError(translateAuthError(signUpError))
      } else if (data.session) {
        // Email confirmation is enabled in this Supabase project, so a new
        // signup only reaches this branch if confirmation was skipped. In that
        // case the onAuthStateChange handler in App.tsx renders the chat page.
      } else if (data.user) {
        setInfo('Conta criada. Confirme o e-mail enviado para ativar o acesso.')
      } else {
        setError('Não foi possível criar a conta. Tente novamente.')
      }
    } catch {
      setError('Erro inesperado ao tentar criar a conta.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <AuthShell>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => navigateTo('/')}
          className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-slate-300 bg-white text-slate-600 transition hover:bg-slate-100"
          aria-label="Voltar para a página de entrada"
        >
          <ArrowLeft className="h-4 w-4" />
        </button>
        <h2 className="font-serif text-2xl font-semibold text-slate-900 sm:text-3xl">
          Criar conta
        </h2>
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-600">
        Cadastre-se para participar do teste fechado e começar a usar o
        assistente.
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

      {info && (
        <div className="mt-5 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
          {info}
        </div>
      )}

      <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
        <TextField
          id="signup-email"
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
          id="signup-password"
          label="Senha"
          icon={Lock}
          type="password"
          autoComplete="new-password"
          placeholder="Pelo menos 6 caracteres"
          value={password}
          onChange={setPassword}
          disabled={submitting || backendStatus === 'offline'}
        />
        <TextField
          id="signup-confirm-password"
          label="Confirmar senha"
          icon={Lock}
          type="password"
          autoComplete="new-password"
          placeholder="Repita a senha"
          value={confirmPassword}
          onChange={setConfirmPassword}
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
          <UserPlus className="h-4 w-4" />
          {submitting ? 'Criando conta...' : 'Criar conta'}
        </button>
      </form>

      <p className="mt-6 text-center text-sm text-slate-500">
        Já tem uma conta?{' '}
        <button
          type="button"
          onClick={() => navigateTo('/')}
          className="font-semibold text-cyan-700 underline decoration-cyan-400 underline-offset-2 transition hover:text-cyan-800"
        >
          Entrar
        </button>
      </p>
    </AuthShell>
  )
}
