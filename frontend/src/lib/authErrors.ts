import type { AuthError } from '@supabase/supabase-js'

// Supabase Auth returns its error messages in English. Map the error codes to
// Portuguese so the closed-test participants see the same language as the rest
// of the UI. Unknown codes fall back to a generic message instead of leaking
// an English string.
const PT_BR_MESSAGES: Record<string, string> = {
  invalid_credentials: 'E-mail ou senha incorretos.',
  email_exists: 'Ja existe uma conta com este e-mail.',
  user_already_exists: 'Ja existe uma conta com este e-mail.',
  email_not_confirmed: 'Confirme seu e-mail antes de entrar.',
  phone_not_confirmed: 'Confirme seu telefone antes de entrar.',
  invalid_email: 'Digite um e-mail valido.',
  weak_password: 'A senha deve ter pelo menos 6 caracteres.',
  user_not_found: 'Nao encontramos uma conta com este e-mail.',
  user_banned: 'Esta conta foi desativada.',
  over_request_rate_limit: 'Muitas tentativas. Aguarde alguns minutos e tente novamente.',
  over_email_send_rate_limit: 'Muitas tentativas. Aguarde alguns minutos e tente novamente.',
  over_sms_send_rate_limit: 'Muitas tentativas. Aguarde alguns minutos e tente novamente.',
  session_expired: 'Sua sessao expirou. Entre novamente.',
  same_password: 'A nova senha deve ser diferente da anterior.',
  failed_to_fetch: 'Falha de conexao com o servidor. Verifique sua internet.',
  fetch_error: 'Falha de conexao com o servidor. Verifique sua internet.',
  timeout: 'A requisicao demorou demais. Tente novamente.',
  bad_json: 'Dados invalidos enviados. Tente novamente.',
  unexpected_failure: 'Erro inesperado. Tente novamente.',
}

const GENERIC_ERROR = 'Nao foi possivel concluir a operacao. Tente novamente.'

export function translateAuthError(error: AuthError | null): string | null {
  if (!error) return null
  return error.code ? (PT_BR_MESSAGES[error.code] ?? GENERIC_ERROR) : GENERIC_ERROR
}
