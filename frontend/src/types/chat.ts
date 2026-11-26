export type MessageRole = 'user' | 'assistant' | 'system'

export interface ChatSource {
  id?: string
  path: string          // file path (relative)
  page?: number        // optional page number
  content?: string     // full chunk text
  label?: string       // computed friendly label (optional)
}

export interface ChatMessage {
  id?: string  // Backend-generated, not needed from frontend
  role: MessageRole
  content: string
  createdAt: string
  sources?: ChatSource[]
  summarized?: boolean
  followUpQuestions?: string[]
  isError?: boolean
}

export interface QueryPayload {
  query: string
}

export interface QueryResult {
  response: string
  sources: ChatSource[]
  followUpQuestions: string[]
  summarized: boolean
}

export interface ConversationHistoryMessage {
  role: MessageRole
  content: string
  sources: ChatSource[]
}
