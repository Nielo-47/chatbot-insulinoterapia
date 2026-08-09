import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'
import { BotMessageSquare, BookOpenText, History } from 'lucide-react'

const FEATURES: Array<{ icon: LucideIcon; title: string; description: string }> = [
  {
    icon: BotMessageSquare,
    title: 'Assistente especializado',
    description: 'Orientações sobre aplicação, rotina e cuidados com diabetes.',
  },
  {
    icon: BookOpenText,
    title: 'Referências das fontes',
    description: 'Respostas fundamentadas em base de conhecimento com citações.',
  },
  {
    icon: History,
    title: 'Histórico persistente',
    description: 'Suas conversas ficam salvas e sincronizadas na sua conta.',
  },
]

interface AuthShellProps {
  children: ReactNode
}

export function AuthShell({ children }: AuthShellProps) {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(209,250,229,0.75),_rgba(255,255,255,1)_48%)] px-4 py-6">
      <div className="mx-auto flex min-h-[calc(100vh-3rem)] max-w-5xl items-center justify-center">
        <div className="w-full overflow-hidden rounded-[2rem] border border-slate-200 bg-white/90 shadow-2xl shadow-slate-200/50 backdrop-blur">
          <div className="grid lg:grid-cols-[1.1fr_1fr]">
            <div className="relative flex flex-col justify-between gap-8 bg-gradient-to-br from-emerald-50 via-teal-50/50 to-white p-8 sm:p-10 lg:border-r lg:border-slate-200">
              <div>
                <span className="inline-flex items-center rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-amber-700">
                  Teste Fechado
                </span>
                <h1 className="mt-4 font-serif text-3xl font-semibold leading-tight text-slate-900 sm:text-4xl">
                  Assistente de insulinoterapia
                </h1>
                <p className="mt-3 max-w-md text-sm leading-6 text-slate-600">
                  Perguntas e respostas sobre aplicação, rotina e cuidados com
                  diabetes, com suporte de base de conhecimento e referências.
                </p>
              </div>

              <ul className="space-y-4">
                {FEATURES.map((feature) => (
                  <li key={feature.title} className="flex items-start gap-3">
                    <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-cyan-200 bg-cyan-50 text-cyan-700">
                      <feature.icon className="h-5 w-5" />
                    </span>
                    <span>
                      <span className="block text-sm font-semibold text-slate-800">
                        {feature.title}
                      </span>
                      <span className="block text-xs leading-5 text-slate-500">
                        {feature.description}
                      </span>
                    </span>
                  </li>
                ))}
              </ul>

              <p className="max-w-md rounded-2xl border border-slate-200 bg-white/70 px-4 py-3 text-xs leading-5 text-slate-500">
                As respostas não substituem avaliação médica presencial.
              </p>
            </div>

            <div className="flex flex-col justify-center p-8 sm:p-10">{children}</div>
          </div>
        </div>
      </div>
    </div>
  )
}

interface TextFieldProps {
  id: string
  label: string
  icon: LucideIcon
  type?: 'text' | 'email' | 'password'
  autoComplete?: string
  placeholder?: string
  value: string
  onChange: (value: string) => void
  disabled?: boolean
}

export function TextField({
  id,
  label,
  icon: Icon,
  type = 'text',
  autoComplete,
  placeholder,
  value,
  onChange,
  disabled,
}: TextFieldProps) {
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-sm font-medium text-slate-700">
        {label}
      </label>
      <div className="relative">
        <Icon className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
        <input
          id={id}
          type={type}
          autoComplete={autoComplete}
          required
          placeholder={placeholder}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          disabled={disabled}
          className="w-full rounded-2xl border border-slate-300 bg-white py-3 pl-10 pr-4 text-sm text-slate-900 placeholder-slate-400 outline-none transition focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/30 disabled:cursor-not-allowed disabled:bg-slate-100"
        />
      </div>
    </div>
  )
}
