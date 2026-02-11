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
  timeout: 60000,
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

  config: {
    get: () => client.get<TenantConfig>('/api/v1/config'),
    update: (data: TenantConfigUpdate) => client.put<TenantConfig>('/api/v1/config', data),
  },
}
