const defaultApiBase = () => {
  return '/api'
}

export const env = {
  apiBaseUrl: (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') || defaultApiBase(),
  requestTimeoutMs: Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS || '60000'),
  // Supabase project credentials. VITE_SUPABASE_URL must NOT have a trailing slash.
  supabaseUrl: (import.meta.env.VITE_SUPABASE_URL as string | undefined)?.replace(/\/$/, '') || '',
  supabaseAnonKey: (import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY as string | undefined) || '',
}
