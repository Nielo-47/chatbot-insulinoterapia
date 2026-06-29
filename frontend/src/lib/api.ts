import { z } from 'zod'

import { env } from './env'
import { authStorage } from './auth'
import type { ConversationHistoryMessage, QueryPayload, QueryResult, StreamEvent } from '../types/chat'

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    Object.setPrototypeOf(this, ApiError.prototype)
  }
}

const sourceItemSchema = z.object({
  id: z.string().optional(),
  path: z.string(),
  page: z.number().optional(),
  excerpt: z.string().optional(),
  label: z.string().optional(),
})

const queryResultSchema = z.object({
  response: z.string(),
  sources: z.array(sourceItemSchema),
  summarized: z.boolean(),
})

const healthResultSchema = z.object({
  status: z.string(),
  message: z.string(),
})

const loginResultSchema = z.object({
  access_token: z.string(),
  token_type: z.literal('bearer'),
})

const currentUserSchema = z.object({
  id: z.number(),
  username: z.string(),
})

const conversationHistorySchema = z.object({
  messages: z.array(
    z.object({
      role: z.enum(['user', 'assistant', 'system']),
      content: z.string(),
      sources: z.array(sourceItemSchema).default([]),
    }),
  ),
})

async function request<T>(
  path: string,
  init: RequestInit,
  schema: z.ZodSchema<T>,
  options?: { skipAuth?: boolean },
): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), env.requestTimeoutMs)
  const token = authStorage.getToken()
  const headers = new Headers(init.headers)
  headers.set('Content-Type', 'application/json')

  if (!env.authEnabled) {
    headers.set('X-Guest-Session-Id', authStorage.getGuestSessionId())
  }

  if (env.authEnabled && !options?.skipAuth && token) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  try {
    const response = await fetch(`${env.apiBaseUrl}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
    })

    if (!response.ok) {
      let detail = `Request failed with status ${response.status}`
      try {
        const errorBody = (await response.json()) as { detail?: string }
        if (errorBody?.detail) {
          detail = errorBody.detail
        }
      } catch {
        // Keep the status-only message when the response is not JSON.
      }
      throw new ApiError(detail, response.status)
    }

    const json = await response.json()
    return schema.parse(json)
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('A requisicao demorou demais. Tente novamente.')
    }

    if (error instanceof z.ZodError) {
      throw new Error('Resposta inesperada do servidor.')
    }

    if (error instanceof Error) {
      throw error
    }

    throw new Error('Erro inesperado ao comunicar com o servidor.')
  } finally {
    window.clearTimeout(timeout)
  }
}

export async function checkHealth(): Promise<void> {
  await request('/health', { method: 'GET' }, healthResultSchema)
}

export async function login(username: string, password: string): Promise<{ accessToken: string; tokenType: 'bearer' }> {
  const result = await request(
    '/auth/login',
    {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    },
    loginResultSchema,
    { skipAuth: true },
  )

  authStorage.setToken(result.access_token)
  return {
    accessToken: result.access_token,
    tokenType: result.token_type,
  }
}

export async function getCurrentUser(): Promise<{ id: number; username: string }> {
  return request('/auth/me', { method: 'GET' }, currentUserSchema)
}

export async function deleteAccount(): Promise<void> {
  await request('/auth/me', { method: 'DELETE' }, z.object({ message: z.string() }))
}

export async function clearAuthSession(): Promise<void> {
  authStorage.clearToken()
}

export async function getConversationHistory(): Promise<ConversationHistoryMessage[]> {
  const result = await request('/user/conversations', { method: 'GET' }, conversationHistorySchema)
  return result.messages
}

export async function sendQuery(payload: QueryPayload): Promise<QueryResult> {
  return request('/query', { method: 'POST', body: JSON.stringify(payload) }, queryResultSchema)
}

function buildStreamHeaders(): Headers {
  const headers = new Headers()
  headers.set('Content-Type', 'application/json')
  const token = authStorage.getToken()

  if (!env.authEnabled) {
    headers.set('X-Guest-Session-Id', authStorage.getGuestSessionId())
  }

  if (env.authEnabled && token) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  return headers
}

async function* parseSSEStream(response: Response): AsyncGenerator<StreamEvent> {
  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('Response body is not readable')
  }

  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // Process complete SSE messages from buffer
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // Keep incomplete line in buffer

      let currentEvent = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          const dataStr = line.slice(6)
          try {
            const data = JSON.parse(dataStr)
            if (currentEvent === 'stage') {
              yield { type: 'stage', data }
            } else if (currentEvent === 'token') {
              yield { type: 'token', data }
            } else if (currentEvent === 'done') {
              yield { type: 'done', data }
            }
          } catch {
            // Skip malformed JSON
          }
          currentEvent = ''
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

export async function sendQueryStream(
  payload: QueryPayload,
  signal?: AbortSignal,
): Promise<AsyncGenerator<StreamEvent>> {
  const response = await fetch(`${env.apiBaseUrl}/query/stream`, {
    method: 'POST',
    headers: buildStreamHeaders(),
    body: JSON.stringify(payload),
    signal,
  })

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`
    try {
      const errorBody = (await response.json()) as { detail?: string }
      if (errorBody?.detail) {
        detail = errorBody.detail
      }
    } catch {
      // Keep the status-only message
    }
    throw new ApiError(detail, response.status)
  }

  return parseSSEStream(response)
}

export async function clearConversation(): Promise<void> {
  await request(`/user/conversations`, { method: 'DELETE' }, z.object({ message: z.string() }))
}

export const clearSession = clearConversation
