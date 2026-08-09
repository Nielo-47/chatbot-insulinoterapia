import clsx from 'clsx'
import { Lightbulb } from 'lucide-react'

interface FollowUpSuggestionsProps {
  suggestions: string[]
  disabled?: boolean
  onSelect: (question: string) => void
}

export function FollowUpSuggestions({ suggestions, disabled, onSelect }: FollowUpSuggestionsProps) {
  if (suggestions.length === 0) {
    return null
  }

  return (
    <div className="mb-3">
      <p className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-[0.08em] text-slate-500">
        <Lightbulb className="h-3.5 w-3.5" />
        Perguntas sugeridas
      </p>
      <div className="flex flex-wrap gap-2">
        {suggestions.slice(0, 3).map((question, index) => (
          <button
            key={`${index}-${question}`}
            type="button"
            onClick={() => onSelect(question)}
            disabled={disabled}
            className={clsx(
              'rounded-xl border border-cyan-200 bg-cyan-50 px-3 py-2 text-left text-xs font-medium text-cyan-800 transition',
              disabled ? 'cursor-not-allowed opacity-60' : 'hover:border-cyan-400 hover:bg-cyan-100 active:bg-cyan-200',
            )}
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  )
}
