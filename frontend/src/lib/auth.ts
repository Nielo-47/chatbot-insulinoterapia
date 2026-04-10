const TOKEN_KEY = 'diabetes-chatbot-access-token'

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
}