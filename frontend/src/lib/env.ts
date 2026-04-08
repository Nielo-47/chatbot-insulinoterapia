const defaultApiBase = () => {
  return '/api'
}

export const env = {
  apiBaseUrl: (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') || defaultApiBase(),
  requestTimeoutMs: Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS || '60000'),
}
