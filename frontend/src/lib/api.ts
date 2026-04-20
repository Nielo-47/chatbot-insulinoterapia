import { z } from 'zod'

import { env } from './env'
import { authStorage } from './auth'
import type { ConversationHistoryMessage, QueryPayload, QueryResult } from '../types/chat'

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

  if (!options?.skipAuth && token) {
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

export async function clearConversation(): Promise<void> {
  await request(`/user/conversations`, { method: 'DELETE' }, z.object({ message: z.string() }))
}

export const clearSession = clearConversation
