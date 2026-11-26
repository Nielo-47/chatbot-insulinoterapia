import type { ClientResponseError } from 'pocketbase'

// PocketBase returns its error messages in English. Map the known response
// statuses/codes to Portuguese so the closed-test participants see the same
// language as the rest of the UI. Unknown errors fall back to a generic
// message instead of leaking an English string.
const PT_BR_MESSAGES: Record<number, string> = {
  400: 'Dados inválidos enviados. Verifique os campos e tente novamente.',
  401: 'E-mail ou senha incorretos.',
  403: 'Você não tem permissão para esta operação.',
  404: 'Não encontramos uma conta com este e-mail.',
  429: 'Muitas tentativas. Aguarde alguns minutos e tente novamente.',
}

const GENERIC_ERROR = 'Não foi possível concluir a operação. Tente novamente.'

function messageFor(error: ClientResponseError): string {
  // Network-level failures surface as status 0.
  if (error.status === 0) {
    return 'Falha de conexão com o servidor. Verifique sua internet.'
  }
  return PT_BR_MESSAGES[error.status] ?? GENERIC_ERROR
}

export function translateAuthError(error: unknown): string | null {
  if (!error) return null
  const name = (error as { name?: string })?.name
  if (name === 'ClientResponseError') {
    return messageFor(error as ClientResponseError)
  }
  return GENERIC_ERROR
}
