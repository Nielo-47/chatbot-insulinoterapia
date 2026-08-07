import { z } from 'zod'

import { env } from './env'
import { supabase } from './supabase'
import type { ConversationHistoryMessage, QueryPayload, QueryResult } from '../types/chat'

const MAX_ERROR_LENGTH = 200

function sanitizeError(message: string): string {
  // Strip non-printable characters: keep only printable ASCII, space, tab, newline, carriage return
  const cleaned = message
    .split('')
    .filter((ch) => {
      const code = ch.charCodeAt(0)
      return code === 0x09 || code === 0x0a || code === 0x0d || (code >= 0x20 && code <= 0x7e)
    })
    .join('')
  // Truncate to max length
  return cleaned.length > MAX_ERROR_LENGTH ? cleaned.slice(0, MAX_ERROR_LENGTH) + '…' : cleaned
}

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

const currentUserSchema = z.object({
  id: z.string(),
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

async function getAccessToken(): Promise<string | null> {
  const { data } = await supabase.auth.getSession()
  return data.session?.access_token ?? null
}

async function request<T>(path: string, init: RequestInit, schema: z.ZodSchema<T>): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), env.requestTimeoutMs)
  const headers = new Headers(init.headers)
  headers.set('Content-Type', 'application/json')

  const token = await getAccessToken()
  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  const doFetch = () =>
    fetch(`${env.apiBaseUrl}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
    })

  try {
    let response = await doFetch()

    if (response.status === 401) {
      // The access token may have just expired: silently refresh it once and
      // retry. If the refresh itself fails (e.g. revoked session) the retry
      // returns 401 again and we surface the unauthenticated state.
      const { data: refreshed } = await supabase.auth.refreshSession()
      if (refreshed.session?.access_token) {
        headers.set('Authorization', `Bearer ${refreshed.session.access_token}`)
        response = await doFetch()
      }
    }

    if (response.status === 401) {
      throw new ApiError('Nao autenticado', 401)
    }

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
      throw new ApiError(sanitizeError(detail), response.status)
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

export async function getCurrentUser(): Promise<{ id: string; username: string }> {
  return request('/auth/me', { method: 'GET' }, currentUserSchema)
}

export async function deleteAccount(): Promise<void> {
  // The Supabase access token is the proof of identity; the backend deletes
  // the Supabase Auth user (service role) and then purges local data. No
  // password re-entry exists.
  await request('/auth/me', { method: 'DELETE' }, z.object({ message: z.string() }))
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
