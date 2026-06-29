export type MessageRole = 'user' | 'assistant' | 'system'

export interface ChatSource {
  id?: string
  path: string          // file path (relative)
  page?: number        // optional page number
  excerpt?: string     // short text excerpt
  label?: string       // computed friendly label (optional)
}

export interface ChatMessage {
  id?: string  // Backend-generated, not needed from frontend
  role: MessageRole
  content: string
  createdAt: string
  sources?: ChatSource[]
  summarized?: boolean
  isError?: boolean
}

export interface QueryPayload {
  query: string
}

export interface QueryResult {
  response: string
  sources: ChatSource[]
  summarized: boolean
}

export interface ConversationHistoryMessage {
  role: MessageRole
  content: string
  sources: ChatSource[]
}

export type StreamStage =
  | 'retrieving'
  | 'generating'
  | 'critiquing'
  | 'refining'
  | 'persisting'
  | 'summarizing'
  | 'done'

export interface StreamStageEvent {
  stage: StreamStage
}

export interface StreamTokenEvent {
  token: string
}

export interface StreamDoneEvent {
  response: string
  sources: ChatSource[]
  summarized: boolean
  session_id: string
}

export type StreamEvent =
  | { type: 'stage'; data: StreamStageEvent }
  | { type: 'token'; data: StreamTokenEvent }
  | { type: 'done'; data: StreamDoneEvent }
