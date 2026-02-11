import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Loader2, MessageSquare, Sparkles } from 'lucide-react'
import { api } from '../api'
import type { ChatMessage as ChatMessageType } from '../types'
import ChatMessage from './ChatMessage'
import { useStore } from '../store'

const SUGGESTED_QUESTIONS = [
  'What are the key terms and definitions?',
  'Summarize the main obligations of each party.',
  'Are there any termination clauses?',
  'What are the liability limitations?',
]

function SkeletonLoader() {
  return (
    <div style={skeletonStyles.row}>
      <div style={skeletonStyles.avatar} />
      <div style={skeletonStyles.bubble}>
        <div style={{ ...skeletonStyles.line, width: '90%' }} />
        <div style={{ ...skeletonStyles.line, width: '75%' }} />
        <div style={{ ...skeletonStyles.line, width: '60%' }} />
        <div style={skeletonStyles.metaRow}>
          <div style={{ ...skeletonStyles.line, width: 40, height: 10 }} />
          <div style={{ ...skeletonStyles.line, width: 60, height: 10 }} />
        </div>
      </div>
    </div>
  )
}

const skeletonStyles: Record<string, React.CSSProperties> = {
  row: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: 10,
    padding: '4px 24px',
  },
  avatar: {
    width: 32,
    height: 32,
    borderRadius: '50%',
    background: 'var(--bg-3)',
    flexShrink: 0,
    animation: 'shimmer 1.5s ease-in-out infinite',
  },
  bubble: {
    padding: '12px 16px',
    borderRadius: 'var(--radius-md, 8px)',
    background: 'var(--bg-2)',
    maxWidth: '80%',
    minWidth: 280,
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  line: {
    height: 14,
    borderRadius: 4,
    background: 'var(--bg-3)',
    animation: 'shimmer 1.5s ease-in-out infinite',
  },
  metaRow: {
    display: 'flex',
    gap: 8,
    marginTop: 4,
  },
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessageType[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const listRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const closeSourcePanel = useStore((s) => s.closeSourcePanel)
  const sourcePanelOpen = useStore((s) => s.sourcePanelOpen)

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
    }
  }, [messages])

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault()
    const query = input.trim()
    if (!query || loading) return

    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: query }])
    setLoading(true)

    try {
      const { data } = await api.query(query)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
          latency_ms: data.latency_ms,
        },
      ])
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, something went wrong. Please try again.' },
      ])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  // Global keyboard shortcuts
  const handleGlobalKey = useCallback(
    (e: KeyboardEvent) => {
      // Escape closes source panel
      if (e.key === 'Escape' && sourcePanelOpen) {
        closeSourcePanel()
        return
      }
      // / focuses chat input when not already typing
      if (e.key === '/' && document.activeElement?.tagName !== 'INPUT' && document.activeElement?.tagName !== 'TEXTAREA') {
        e.preventDefault()
        inputRef.current?.focus()
      }
    },
    [closeSourcePanel, sourcePanelOpen]
  )

  useEffect(() => {
    window.addEventListener('keydown', handleGlobalKey)
    return () => window.removeEventListener('keydown', handleGlobalKey)
  }, [handleGlobalKey])

  const handleSuggestionClick = (question: string) => {
    setInput(question)
    // Auto-submit
    setTimeout(() => {
      setInput('')
      setMessages((prev) => [...prev, { role: 'user', content: question }])
      setLoading(true)
      api.query(question).then(({ data }) => {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: data.answer,
            sources: data.sources,
            latency_ms: data.latency_ms,
          },
        ])
      }).catch(() => {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: 'Sorry, something went wrong. Please try again.' },
        ])
      }).finally(() => {
        setLoading(false)
        inputRef.current?.focus()
      })
    }, 50)
  }

  return (
    <div style={styles.container}>
      <div ref={listRef} style={styles.messageList}>
        {messages.length === 0 && (
          <div style={styles.empty}>
            <MessageSquare size={48} color="var(--bg-3)" />
            <h2 style={styles.emptyTitle}>Ask about your legal documents</h2>
            <p style={styles.emptyText}>
              Upload a document, then ask questions. Answers include precise citations you can click to view the source.
            </p>
            <div style={styles.suggestions}>
              <div style={styles.suggestionsLabel}>
                <Sparkles size={13} />
                <span>Try asking</span>
              </div>
              <div style={styles.suggestionsGrid}>
                {SUGGESTED_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    style={styles.suggestionBtn}
                    onClick={() => handleSuggestionClick(q)}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <ChatMessage key={i} message={msg} />
        ))}
        {loading && <SkeletonLoader />}
      </div>

      <form onSubmit={handleSubmit} style={styles.inputBar}>
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents...  (press / to focus)"
          rows={1}
          style={styles.textarea}
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          style={{
            ...styles.sendBtn,
            opacity: loading || !input.trim() ? 0.4 : 1,
          }}
        >
          {loading ? <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> : <Send size={18} />}
        </button>
      </form>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes shimmer {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 0.7; }
        }
      `}</style>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    background: 'var(--bg-0)',
  },
  messageList: {
    flex: 1,
    overflowY: 'auto',
    padding: '24px 0',
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  empty: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    padding: 40,
    textAlign: 'center',
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: 600,
    color: 'var(--text-2)',
  },
  emptyText: {
    fontSize: 14,
    color: 'var(--text-3)',
    maxWidth: 400,
    lineHeight: 1.6,
  },
  suggestions: {
    marginTop: 16,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 10,
    maxWidth: 500,
  },
  suggestionsLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    fontSize: 12,
    color: 'var(--text-3)',
    fontWeight: 500,
  },
  suggestionsGrid: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 8,
    justifyContent: 'center',
  },
  suggestionBtn: {
    padding: '8px 14px',
    fontSize: 13,
    color: 'var(--text-2)',
    background: 'var(--bg-2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md, 8px)',
    cursor: 'pointer',
    transition: 'all var(--transition)',
    textAlign: 'left' as const,
    lineHeight: 1.4,
  },
  inputBar: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: 10,
    padding: '16px 24px',
    borderTop: '1px solid var(--border)',
    background: 'var(--bg-1)',
    flexShrink: 0,
  },
  textarea: {
    flex: 1,
    resize: 'none',
    padding: '12px 16px',
    fontSize: 14,
    lineHeight: 1.5,
    background: 'var(--bg-2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    color: 'var(--text-1)',
    outline: 'none',
    maxHeight: 120,
    fontFamily: 'inherit',
  },
  sendBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 44,
    height: 44,
    borderRadius: 'var(--radius-md)',
    background: 'var(--accent)',
    color: '#000',
    border: 'none',
    cursor: 'pointer',
    flexShrink: 0,
    transition: 'opacity var(--transition)',
  },
}
