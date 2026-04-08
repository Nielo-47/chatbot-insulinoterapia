const SESSION_KEY = 'diabetes-chatbot-session-id'

export const sessionStorageService = {
  getSessionId(): string | null {
    return window.localStorage.getItem(SESSION_KEY)
  },
  setSessionId(sessionId: string): void {
    window.localStorage.setItem(SESSION_KEY, sessionId)
  },
  clearSessionId(): void {
    window.localStorage.removeItem(SESSION_KEY)
  },
}
