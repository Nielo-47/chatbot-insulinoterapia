import { z } from 'zod'

import { env } from './env'
import { authStorage } from './auth'
import type { QueryPayload, QueryResult } from '../types/chat'

const queryResultSchema = z.object({
  response: z.string(),
  sources: z.array(z.string()),
  source_count: z.number(),
  summarized: z.boolean(),
  session_id: z.string(),
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

async function request<T>(
  path: string,
  init: RequestInit,
  schema: z.ZodSchema<T>,
  options?: { skipAuth?: boolean },
): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), env.requestTimeoutMs)
  const token = authStorage.getToken()
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init.headers || {}),
  }

  if (!options?.skipAuth && token) {
    headers.Authorization = `Bearer ${token}`
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
      throw new Error(detail)
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

export async function clearAuthSession(): Promise<void> {
  authStorage.clearToken()
}

export async function sendQuery(payload: QueryPayload): Promise<QueryResult> {
  return request('/query', { method: 'POST', body: JSON.stringify(payload) }, queryResultSchema)
}

export async function clearSession(sessionId: string): Promise<void> {
  await request(`/session/${sessionId}`, { method: 'DELETE' }, z.object({ message: z.string() }))
}
