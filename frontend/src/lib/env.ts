const defaultApiBase = () => {
  return '/api'
}

const defaultPocketbasePath = () => {
  return '/pb'
}

export const env = {
  apiBaseUrl: (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') || defaultApiBase(),
  requestTimeoutMs: Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS || '60000'),
  // Same-origin path of the ui nginx proxy that forwards to the PocketBase
  // container (no trailing slash).
  pocketbaseUrl: (import.meta.env.VITE_POCKETBASE_URL as string | undefined)?.replace(/\/$/, '') || defaultPocketbasePath(),
}
