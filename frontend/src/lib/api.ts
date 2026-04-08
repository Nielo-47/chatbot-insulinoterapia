import { z } from 'zod'

import { env } from './env'
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

async function request<T>(path: string, init: RequestInit, schema: z.ZodSchema<T>): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), env.requestTimeoutMs)

  try {
    const response = await fetch(`${env.apiBaseUrl}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init.headers || {}),
      },
      signal: controller.signal,
    })

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`)
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

export async function sendQuery(payload: QueryPayload): Promise<QueryResult> {
  return request('/query', { method: 'POST', body: JSON.stringify(payload) }, queryResultSchema)
}

export async function clearSession(sessionId: string): Promise<void> {
  await request(`/session/${sessionId}`, { method: 'DELETE' }, z.object({ message: z.string() }))
}
