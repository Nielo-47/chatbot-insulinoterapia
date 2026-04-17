export type MessageRole = 'user' | 'assistant' | 'system'

export interface ChatSource {
  id: string
  label: string
}

export interface ChatMessage {
  id?: string  // Backend-generated, not needed from frontend
  role: MessageRole
  content: string
  createdAt: string
  sources?: ChatSource[]
  sourceCount?: number
  summarized?: boolean
  isError?: boolean
}

export interface QueryPayload {
  query: string
}

export interface QueryResult {
  response: string
  sources: string[]
  source_count: number
  summarized: boolean
}

export interface ConversationHistoryMessage {
  role: MessageRole
  content: string
  sources: string[]
  source_count: number
}
