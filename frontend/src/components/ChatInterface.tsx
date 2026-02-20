import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Loader2, MessageSquare, Sparkles, Plus, SlidersHorizontal } from 'lucide-react'
import { api } from '../api'
import ChatMessage from './ChatMessage'
import SourceTogglePopover from './SourceTogglePopover'
import { useStore } from '../store'

const SUGGESTED_QUESTIONS = [
  'What are the key provisions of GDPR Article 17?',
  'Summarize ECHR case law on right to privacy.',
  'What are the liability limitations under Cyprus law?',
  'Explain the EU directive on consumer protection.',
]

let _msgCounter = 0
function nextId() { return `msg-${Date.now()}-${++_msgCounter}` }

export default function ChatInterface() {
  const messages = useStore((s) => s.messages)
  const setMessages = useStore((s) => s.setMessages)
  const selectedDocumentId = useStore((s) => s.selectedDocumentId)
  const sourceToggles = useStore((s) => s.sourceToggles)
  const activeConversationId = useStore((s) => s.activeConversationId)
  const setActiveConversationId = useStore((s) => s.setActiveConversationId)
  const conversations = useStore((s) => s.conversations)
  const setConversations = useStore((s) => s.setConversations)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sourcePopoverOpen, setSourcePopoverOpen] = useState(false)
  const [sessionFileName, setSessionFileName] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const closeSourcePanel = useStore((s) => s.closeSourcePanel)
  const sourcePanelOpen = useStore((s) => s.sourcePanelOpen)
  const anySourceOff = !sourceToggles.cylaw || !sourceToggles.hudoc || !sourceToggles.eurlex || sourceToggles.families.length > 0

  // Auto-scroll as new tokens stream in
  const lastContentLen = useRef(0)
  useEffect(() => {
    const lastMsg = messages[messages.length - 1]
    const contentLen = lastMsg?.content?.length || 0
    if (listRef.current && contentLen !== lastContentLen.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
      lastContentLen.current = contentLen
    }
  }, [messages])

  const abortRef = useRef<{ abort: () => void } | null>(null)

  const submitQuery = useCallback(async (query: string) => {
    if (!query || loading) return

    setInput('')
    const assistantId = nextId()
    setMessages((prev) => [...prev, { id: nextId(), role: 'user', content: query }])
    setMessages((prev) => [...prev, { id: assistantId, role: 'assistant', content: '' }])
    setLoading(true)

    let accumulatedContent = ''

    const handle = api.queryStream(
      query,
      {
        onSources: (sources) => {
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId ? { ...m, sources } : m)
          )
        },
        onToken: (token) => {
          accumulatedContent += token
          const content = accumulatedContent
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId ? { ...m, content } : m)
          )
        },
        onDone: (latencyMs, conversationId) => {
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId ? { ...m, latency_ms: latencyMs } : m)
          )
          setLoading(false)
          abortRef.current = null
          inputRef.current?.focus()

          // If server created/returned a conversation_id, track it
          if (conversationId && !activeConversationId) {
            setActiveConversationId(conversationId)
            // Refresh conversation list to pick up the new one
            api.conversations.list().then(({ data }) => setConversations(data)).catch(() => {})
          }
        },
        onError: (msg) => {
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId ? { ...m, content: msg, isError: true } : m)
          )
          setLoading(false)
          abortRef.current = null
          inputRef.current?.focus()
        },
        onConversationId: (id) => {
          if (!activeConversationId) {
            setActiveConversationId(id)
          }
        },
      },
      selectedDocumentId ?? undefined,
      undefined,
      sourceToggles,
      activeConversationId ?? undefined,
    )

    abortRef.current = handle
  }, [loading, selectedDocumentId, setMessages, sourceToggles, activeConversationId, setActiveConversationId, setConversations])

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault()
    submitQuery(input.trim())
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
      if (e.key === 'Escape' && sourcePanelOpen) {
        closeSourcePanel()
        return
      }
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
    submitQuery(question)
  }

  const handleChatUpload = async (file: File) => {
    // Chat-scoped upload: attach PDF to current conversation
    let convId = activeConversationId
    if (!convId) {
      // Create a conversation first
      try {
        const { data } = await api.conversations.create('New Chat')
        convId = data.id
        setActiveConversationId(convId)
        api.conversations.list().then(({ data: convos }) => setConversations(convos)).catch(() => {})
      } catch {
        console.error('Failed to create conversation for upload')
        return
      }
    }
    try {
      await api.documents.upload(file, undefined, undefined, convId, 'session')
      setSessionFileName(file.name)
      window.dispatchEvent(new Event('documents-changed'))
    } catch (err) {
      console.error('Chat upload failed:', err)
    }
  }

  return (
    <div style={styles.container}>
      <div ref={listRef} style={styles.messageList}>
        {messages.length === 0 && (
          <div style={styles.empty}>
            <MessageSquare size={48} color="var(--bg-3)" />
            <h2 style={styles.emptyTitle}>Ask about legal documents</h2>
            <p style={styles.emptyText}>
              Search across Cyprus Law, ECHR case law, and EU legislation. Answers include precise citations you can click to view the source.
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
          <ChatMessage
            key={msg.id}
            message={msg}
            isStreaming={loading && i === messages.length - 1 && msg.role === 'assistant'}
          />
        ))}
      </div>

      {/* Session file indicator */}
      {sessionFileName && (
        <div style={styles.sessionChip}>
          <span style={styles.sessionChipText}>{sessionFileName}</span>
          <button
            style={styles.sessionChipClose}
            onClick={() => setSessionFileName(null)}
            title="Dismiss"
          >
            &times;
          </button>
        </div>
      )}

      <form onSubmit={handleSubmit} style={styles.inputBar}>
        {/* Upload button */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          style={{ display: 'none' }}
          onChange={async (e) => {
            const file = e.target.files?.[0]
            if (!file) return
            await handleChatUpload(file)
            e.target.value = ''
          }}
        />
        <button
          type="button"
          style={styles.iconBtn}
          onClick={() => fileInputRef.current?.click()}
          title="Upload document to this chat"
          disabled={loading}
        >
          <Plus size={18} />
        </button>

        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => {
            setInput(e.target.value)
            const el = e.target
            el.style.height = 'auto'
            el.style.height = Math.min(el.scrollHeight, 120) + 'px'
          }}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about legal documents...  (press / to focus)"
          rows={1}
          style={styles.textarea}
          disabled={loading}
        />

        {/* Source settings button */}
        <div style={{ position: 'relative' as const }}>
          <button
            type="button"
            style={styles.iconBtn}
            onClick={() => setSourcePopoverOpen(!sourcePopoverOpen)}
            title="Search sources"
          >
            <SlidersHorizontal size={18} />
            {anySourceOff && <span style={styles.indicatorDot} />}
          </button>
          <SourceTogglePopover
            open={sourcePopoverOpen}
            onClose={() => setSourcePopoverOpen(false)}
          />
        </div>

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
      {input.length > 0 && (
        <div style={styles.charCount}>
          {input.length}/2000
        </div>
      )}
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
  sessionChip: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '4px 12px',
    margin: '0 24px',
    background: 'var(--accent-dim)',
    borderRadius: 'var(--radius-sm)',
    fontSize: 12,
    color: 'var(--accent)',
  },
  sessionChipText: {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    flex: 1,
  },
  sessionChipClose: {
    background: 'none',
    border: 'none',
    color: 'var(--accent)',
    fontSize: 16,
    cursor: 'pointer',
    padding: '0 2px',
    lineHeight: 1,
  },
  inputBar: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: 10,
    padding: '16px 24px',
    borderTop: '1px solid var(--glass-border)',
    background: 'var(--glass-bg)',
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
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
    transition: 'border-color 0.2s ease, box-shadow 0.3s ease',
  },
  iconBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 40,
    height: 40,
    borderRadius: 'var(--radius-md)',
    background: 'transparent',
    color: 'var(--text-2)',
    border: '1px solid var(--border)',
    cursor: 'pointer',
    flexShrink: 0,
    transition: 'all var(--transition)',
    position: 'relative' as const,
  },
  indicatorDot: {
    position: 'absolute' as const,
    top: 6,
    right: 6,
    width: 7,
    height: 7,
    borderRadius: '50%',
    background: '#f59e0b',
    border: '1.5px solid var(--bg-1)',
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
  charCount: {
    textAlign: 'right' as const,
    fontSize: 11,
    color: 'var(--text-3)',
    padding: '0 24px 4px',
    marginTop: -8,
  },
}
