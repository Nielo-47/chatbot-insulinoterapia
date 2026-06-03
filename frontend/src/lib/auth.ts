const TOKEN_KEY = 'diabetes-chatbot-access-token'
const GUEST_SESSION_KEY = 'diabetes-chatbot-guest-session-id'

function createGuestSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }

  return `guest-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export const authStorage = {
  getToken(): string | null {
    return window.localStorage.getItem(TOKEN_KEY)
  },
  setToken(token: string): void {
    window.localStorage.setItem(TOKEN_KEY, token)
  },
  clearToken(): void {
    window.localStorage.removeItem(TOKEN_KEY)
  },
  getGuestSessionId(): string {
    const existing = window.localStorage.getItem(GUEST_SESSION_KEY)
    if (existing) {
      return existing
    }

    const sessionId = createGuestSessionId()
    window.localStorage.setItem(GUEST_SESSION_KEY, sessionId)
    return sessionId
  },
}