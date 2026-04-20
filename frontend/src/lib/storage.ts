const TYPED_MESSAGE_KEY = 'diabetes-chatbot-typed-message-v1'

export const draftStorage = {
  getDraft(): string | null {
    try {
      return window.localStorage.getItem(TYPED_MESSAGE_KEY) ?? null
    } catch {
      return null
    }
  },
  saveDraft(message: string): void {
    try {
      window.localStorage.setItem(TYPED_MESSAGE_KEY, message)
    } catch {
      // storage unavailable (private mode, quota exceeded, etc.)
    }
  },
  clearDraft(): void {
    try {
      window.localStorage.removeItem(TYPED_MESSAGE_KEY)
    } catch {
      // ignore
    }
  },
}
