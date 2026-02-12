import axios from 'axios'
import { useStore } from './store'
import type {
  QueryResponse,
  DocumentInfo,
  UploadResponse,
  TenantConfig,
  TenantConfigUpdate,
  HealthResponse,
} from './types'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  timeout: 120000,
})

client.interceptors.request.use((config) => {
  const apiKey = useStore.getState().apiKey
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey
  }
  return config
})

client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      useStore.getState().logout()
    }
    return Promise.reject(err)
  }
)

export const api = {
  health: () => client.get<HealthResponse>('/api/v1/health'),

  validateKey: (key: string) =>
    client.get<DocumentInfo[]>('/api/v1/documents', {
      headers: { 'X-API-Key': key },
    }),

  documents: {
    list: () => client.get<DocumentInfo[]>('/api/v1/documents'),
    upload: (file: File, onProgress?: (pct: number) => void) => {
      const form = new FormData()
      form.append('file', file)
      return client.post<UploadResponse>('/api/v1/documents/upload', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 600000, // 10 minutes per file for large PDFs
        onUploadProgress: (e) => {
          if (onProgress && e.total) onProgress(Math.round((e.loaded * 100) / e.total))
        },
      })
    },
    delete: (id: string) => client.delete(`/api/v1/documents/${id}`),
  },

  query: (query: string, documentId?: string, topK?: number) =>
    client.post<QueryResponse>('/api/v1/query', {
      query,
      document_id: documentId || null,
      top_k: topK || 10,
    }),

  /**
   * Streaming query via Server-Sent Events.
   * Calls onSources when sources arrive, onToken for each text chunk,
   * and onDone when generation is complete.
   */
  queryStream: (
    query: string,
    callbacks: {
      onSources: (sources: import('./types').SourceInfo[]) => void
      onToken: (token: string) => void
      onDone: (latencyMs: number) => void
      onError: (msg: string) => void
    },
    documentId?: string,
    topK?: number,
  ): { abort: () => void } => {
    const controller = new AbortController()
    const baseURL = import.meta.env.VITE_API_URL || ''
    const apiKey = useStore.getState().apiKey

    fetch(`${baseURL}/api/v1/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey ? { 'X-API-Key': apiKey } : {}),
      },
      body: JSON.stringify({
        query,
        document_id: documentId || null,
        top_k: topK || 10,
      }),
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) {
          if (res.status === 401) {
            useStore.getState().logout()
            callbacks.onError('Session expired. Please log in again.')
          } else if (res.status === 429) {
            callbacks.onError('Rate limit exceeded. Please wait a moment and try again.')
          } else {
            callbacks.onError('Sorry, something went wrong. Please try again.')
          }
          return
        }

        const reader = res.body?.getReader()
        if (!reader) {
          callbacks.onError('Streaming not supported by the browser.')
          return
        }

        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })

          // Parse SSE events from buffer
          const lines = buffer.split('\n')
          buffer = lines.pop() || '' // Keep incomplete line in buffer

          let currentEvent = ''
          let currentData = ''

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              currentEvent = line.slice(7)
            } else if (line.startsWith('data: ')) {
              currentData = line.slice(6)
            } else if (line === '' && currentEvent && currentData) {
              // End of event
              try {
                const parsed = JSON.parse(currentData)
                if (currentEvent === 'sources') {
                  callbacks.onSources(parsed)
                } else if (currentEvent === 'token') {
                  callbacks.onToken(parsed)
                } else if (currentEvent === 'done') {
                  callbacks.onDone(parsed.latency_ms)
                } else if (currentEvent === 'error') {
                  callbacks.onError(parsed)
                }
              } catch {
                // Ignore parse errors from partial data
              }
              currentEvent = ''
              currentData = ''
            }
          }
        }
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          callbacks.onError('Connection failed. Please try again.')
        }
      })

    return { abort: () => controller.abort() }
  },

  config: {
    get: () => client.get<TenantConfig>('/api/v1/config'),
    update: (data: TenantConfigUpdate) => client.put<TenantConfig>('/api/v1/config', data),
  },
}
