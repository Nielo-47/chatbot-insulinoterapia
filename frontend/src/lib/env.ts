const defaultApiBase = () => {
  return '/api'
}

export const env = {
  apiBaseUrl: (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') || defaultApiBase(),
  requestTimeoutMs: Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS || '60000'),
  // Authentik is served on the same origin by the nginx edge (default '/'),
  // or at an absolute URL when VITE_AUTHENTIK_BASE_URL is set.
  authentikBaseUrl: ((import.meta.env.VITE_AUTHENTIK_BASE_URL as string | undefined) || '/').replace(/\/$/, ''),
  authentikLoginFlow: (import.meta.env.VITE_AUTHENTIK_LOGIN_FLOW as string | undefined) || 'default-authentication-flow',
  authentikLogoutFlow: (import.meta.env.VITE_AUTHENTIK_LOGOUT_FLOW as string | undefined) || 'default-invalidation-flow',
}

export function getAuthentikLoginUrl(): string {
  return `${env.authentikBaseUrl}/if/flow/${env.authentikLoginFlow}/`
}

export function getAuthentikLogoutUrl(): string {
  return `${env.authentikBaseUrl}/if/flow/${env.authentikLogoutFlow}/`
}
