export interface SourceInfo {
  document_title: string
  section: string
  page_numbers: number[]
  hierarchy_path: string
  chunk_id: string
  document_id: string
  relevance_score: number
  short_citation: string
  long_citation: string
  content: string
  context_before: string
  context_after: string
  cylaw_url?: string | null
}

export interface QueryResponse {
  answer: string
  sources: SourceInfo[]
  latency_ms: number
}

export interface DocumentInfo {
  id: string
  title: string
  document_type: string
  jurisdiction: string | null
  page_count: number
  chunks: number | null
  created_at: string | null
  cylaw_url?: string | null
}

export interface UploadResponse {
  id: string
  title: string
  document_type: string
  jurisdiction: string | null
  page_count: number
  chunks: number
}

export interface TenantConfig {
  language: string
  embedding_model: string
  embedding_provider: string
  llm_model: string
  reranker_model: string
  fts_language: string
}

export interface TenantConfigUpdate {
  language?: string
  embedding_model?: string
  embedding_provider?: string
  llm_model?: string
  reranker_model?: string
}

export interface HealthResponse {
  status: string
  version: string
  database: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceInfo[]
  latency_ms?: number
  isError?: boolean
}

export interface ParsedPart {
  type: 'text' | 'citation'
  content: string
  citationNumber?: number
}

export function parseCitations(text: string): ParsedPart[] {
  const parts: ParsedPart[] = []
  // Match citations with numbers, commas, and ranges like [1], [1, 2], [10-11], [1, 3-5]
  const regex = /\[(\d+(?:\s*[-,]\s*\d+)*)\]/g
  let lastIndex = 0
  let match

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', content: text.slice(lastIndex, match.index) })
    }
    // Parse and expand ranges like "1, 3-5" into [1, 3, 4, 5]
    const nums: number[] = []
    const segments = match[1].split(/\s*,\s*/)
    for (const seg of segments) {
      if (seg.includes('-')) {
        const [start, end] = seg.split('-').map(n => parseInt(n.trim()))
        for (let i = start; i <= end; i++) nums.push(i)
      } else {
        nums.push(parseInt(seg.trim()))
      }
    }
    nums.forEach((num, i) => {
      parts.push({
        type: 'citation',
        content: `[${num}]`,
        citationNumber: num,
      })
      if (i < nums.length - 1) {
        parts.push({ type: 'text', content: ' ' })
      }
    })
    lastIndex = regex.lastIndex
  }

  if (lastIndex < text.length) {
    parts.push({ type: 'text', content: text.slice(lastIndex) })
  }

  return parts.length > 0 ? parts : [{ type: 'text', content: text }]
}
