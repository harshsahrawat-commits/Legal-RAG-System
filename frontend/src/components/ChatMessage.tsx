import { useMemo, useState, useRef, useEffect } from 'react'
import { User, Bot, Clock, Copy, Check, AlertCircle, FileText } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage as ChatMessageType, SourceInfo } from '../types'
import { parseCitations } from '../types'
import { useStore } from '../store'

interface Props {
  message: ChatMessageType
  isStreaming?: boolean
}

function CitationBadge({
  num,
  source,
  valid,
  onClick,
}: {
  num: number
  source?: SourceInfo
  valid: boolean
  onClick: () => void
}) {
  const [hover, setHover] = useState(false)
  const [tooltipPos, setTooltipPos] = useState<'above' | 'below'>('above')
  const ref = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    if (hover && ref.current) {
      const rect = ref.current.getBoundingClientRect()
      setTooltipPos(rect.top < 160 ? 'below' : 'above')
    }
  }, [hover])

  return (
    <span
      ref={ref}
      style={{
        ...badgeStyles.badge,
        background: valid ? 'var(--accent-dim, rgba(6,182,212,0.15))' : 'var(--bg-3)',
        color: valid ? 'var(--accent, #06b6d4)' : 'var(--text-3)',
        cursor: valid ? 'pointer' : 'not-allowed',
        position: 'relative',
      }}
      onClick={valid ? onClick : undefined}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if ((e.key === 'Enter' || e.key === ' ') && valid) { e.preventDefault(); onClick() } }}
      aria-label={valid && source ? `Citation ${num}: ${source.section || source.document_title}` : `Citation ${num}`}
    >
      [{num}]
      {hover && valid && source && (
        <span
          style={{
            ...badgeStyles.tooltip,
            ...(tooltipPos === 'below' ? { top: '100%', bottom: 'auto', marginTop: 6 } : {}),
          }}
        >
          <strong style={{ display: 'block', marginBottom: 4 }}>
            {source.section || source.document_title}
          </strong>
          <span
            style={{
              display: 'block',
              fontSize: 11,
              color: 'var(--text-2)',
              lineHeight: 1.4,
            }}
          >
            {source.content.slice(0, 120)}
            {source.content.length > 120 ? '...' : ''}
          </span>
        </span>
      )}
    </span>
  )
}

const badgeStyles: Record<string, React.CSSProperties> = {
  badge: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 22,
    height: 20,
    padding: '0 6px',
    fontSize: 11,
    fontWeight: 700,
    borderRadius: 10,
    transition: 'all var(--transition)',
    verticalAlign: 'middle',
    marginInline: 2,
    userSelect: 'none' as const,
    gap: 3,
  },
  tooltip: {
    position: 'absolute',
    bottom: '100%',
    left: '50%',
    transform: 'translateX(-50%)',
    marginBottom: 6,
    width: 260,
    padding: '10px 12px',
    background: 'var(--bg-1)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm, 6px)',
    boxShadow: 'var(--shadow-lg, 0 4px 12px rgba(0,0,0,0.3))',
    fontSize: 12,
    fontWeight: 400,
    color: 'var(--text-1)',
    zIndex: 1000,
    textAlign: 'left' as const,
    pointerEvents: 'none' as const,
    whiteSpace: 'normal' as const,
    lineHeight: 1.4,
  },
}

const THINKING_MESSAGES = [
  'Searching through legal documents\u2026',
  'Analyzing relevant sections\u2026',
  'Preparing your response\u2026',
]

function ThinkingIndicator({ hasSources, sourceCount }: { hasSources: boolean; sourceCount: number }) {
  const [statusIndex, setStatusIndex] = useState(0)
  const [textVisible, setTextVisible] = useState(true)

  useEffect(() => {
    if (hasSources) return
    const interval = setInterval(() => {
      setTextVisible(false)
      setTimeout(() => {
        setStatusIndex((prev) => (prev + 1) % THINKING_MESSAGES.length)
        setTextVisible(true)
      }, 500)
    }, 4000)
    return () => clearInterval(interval)
  }, [hasSources])

  return (
    <div style={{ minWidth: 260, animation: 'fadeIn 0.25s ease' }}>
      {hasSources ? (
        <>
          <div style={thinkingStyles.sourcesBadge}>
            <FileText size={12} />
            <span>{sourceCount} source{sourceCount !== 1 ? 's' : ''} found</span>
          </div>
          <div style={thinkingStyles.generatingRow}>
            <span style={thinkingStyles.pulseDot} />
            <span>Generating answer\u2026</span>
          </div>
        </>
      ) : (
        <>
          <div style={thinkingStyles.statusRow}>
            <span style={thinkingStyles.pulseDot} />
            <span style={{
              ...thinkingStyles.statusText,
              opacity: textVisible ? 1 : 0,
            }}>
              {THINKING_MESSAGES[statusIndex]}
            </span>
          </div>
          <div style={thinkingStyles.skeletonGroup}>
            {[85, 70, 55].map((w, i) => (
              <div
                key={i}
                style={{
                  ...thinkingStyles.skeletonLine,
                  width: `${w}%`,
                  animationDelay: `${i * 0.15}s`,
                }}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}

const thinkingStyles: Record<string, React.CSSProperties> = {
  statusRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  pulseDot: {
    display: 'inline-block',
    width: 6,
    height: 6,
    borderRadius: '50%',
    background: '#2dd4bf',
    animation: 'pulseDot 1.5s ease-in-out infinite',
    flexShrink: 0,
  },
  statusText: {
    fontSize: 13,
    color: 'rgba(255, 255, 255, 0.5)',
    transition: 'opacity 0.5s ease',
  },
  skeletonGroup: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 8,
  },
  skeletonLine: {
    height: 14,
    borderRadius: 4,
    background: 'linear-gradient(90deg, rgba(255,255,255,0.06) 25%, rgba(255,255,255,0.14) 50%, rgba(255,255,255,0.06) 75%)',
    backgroundSize: '200% 100%',
    animation: 'shimmerSlide 1.8s ease-in-out infinite',
  },
  sourcesBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    padding: '5px 10px',
    background: 'var(--accent-dim)',
    borderRadius: 'var(--radius-sm)',
    fontSize: 12,
    color: 'var(--accent)',
    fontWeight: 500,
    animation: 'slideUpFadeIn 0.3s ease-out forwards',
  },
  generatingRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    marginTop: 10,
    fontSize: 12,
    color: 'var(--text-3)',
    animation: 'fadeIn 0.3s ease',
  },
}

export default function ChatMessage({ message, isStreaming = false }: Props) {
  const openSourcePanel = useStore((s) => s.openSourcePanel)
  const isUser = message.role === 'user'
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  // Smooth transition: thinking → streaming content
  const [isFadingOut, setIsFadingOut] = useState(false)
  const wasThinkingRef = useRef(true)

  useEffect(() => {
    if (isStreaming && wasThinkingRef.current && message.content !== '') {
      // First token arrived — fade out thinking indicator
      setIsFadingOut(true)
      const timer = setTimeout(() => setIsFadingOut(false), 300)
      wasThinkingRef.current = false
      return () => clearTimeout(timer)
    }
    if (message.content === '') {
      wasThinkingRef.current = true
    }
  }, [isStreaming, message.content])

  const showThinkingUI = isStreaming && (message.content === '' || isFadingOut)

  const parts = useMemo(
    () => (isUser ? null : parseCitations(message.content)),
    [message.content, isUser]
  )

  const isValidCitation = (num: number) => {
    return message.sources && num >= 1 && num <= message.sources.length
  }

  const handleCitationClick = (num: number) => {
    if (isValidCitation(num)) {
      openSourcePanel(num - 1, message.sources!)
    }
  }

  const renderAssistantContent = () => {
    if (!parts) return null

    // Phase 1 & 2: Thinking indicator with smooth fade-out when content arrives
    if (showThinkingUI) {
      return (
        <div style={{
          opacity: isFadingOut ? 0 : 1,
          transition: 'opacity 0.3s ease-out',
        }}>
          <ThinkingIndicator
            hasSources={!!message.sources && message.sources.length > 0}
            sourceCount={message.sources?.length || 0}
          />
        </div>
      )
    }

    if (message.content === '') return null

    // Phase 3: Content fades in after thinking fades out
    const streamEnter = isStreaming ? { animation: 'fadeIn 0.2s ease-in' } : {}

    // Check if the content has any markdown formatting
    const hasMarkdown = /(\*\*.+?\*\*|__.+?__|#{1,3}\s|```.+?```|^\s*[-*]\s|\|.+\|)/ms.test(message.content)

    if (!hasMarkdown) {
      // Plain text with citations
      return (
        <div style={{ ...styles.text, ...streamEnter }}>
          {parts.map((part, i) =>
            part.type === 'text' ? (
              <span key={i}>{part.content}</span>
            ) : (
              <CitationBadge
                key={i}
                num={part.citationNumber!}
                source={message.sources?.[part.citationNumber! - 1]}
                valid={isValidCitation(part.citationNumber!)!}
                onClick={() => handleCitationClick(part.citationNumber!)}
              />
            )
          )}
        </div>
      )
    }

    // Markdown content: render text segments as markdown, keep citation badges inline
    return (
      <div style={{ ...styles.markdown, ...streamEnter }}>
        {parts.map((part, i) =>
          part.type === 'text' ? (
            <ReactMarkdown key={i} remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {part.content}
            </ReactMarkdown>
          ) : (
            <CitationBadge
              key={i}
              num={part.citationNumber!}
              source={message.sources?.[part.citationNumber! - 1]}
              valid={isValidCitation(part.citationNumber!)!}
              onClick={() => handleCitationClick(part.citationNumber!)}
            />
          )
        )}
      </div>
    )
  }

  return (
    <div style={{ ...styles.row, justifyContent: isUser ? 'flex-end' : 'flex-start' }}>
      {!isUser && (
        <div style={styles.avatar}>
          <Bot size={16} />
        </div>
      )}
      <div
        style={{
          ...styles.bubble,
          background: message.isError ? 'var(--danger-dim, rgba(239,68,68,0.12))' : isUser ? 'var(--bg-3)' : 'var(--bg-2)',
          borderBottomRightRadius: isUser ? 4 : 'var(--radius-md)',
          borderBottomLeftRadius: isUser ? 'var(--radius-md)' : 4,
          maxWidth: isUser ? '70%' : '80%',
        }}
      >
        {isUser ? (
          <p style={styles.text}>{message.content}</p>
        ) : message.isError ? (
          <div style={{ ...styles.text, display: 'flex', alignItems: 'flex-start', gap: 8 }}>
            <AlertCircle size={16} color="var(--danger)" style={{ flexShrink: 0, marginTop: 2 }} />
            <span>{message.content}</span>
          </div>
        ) : (
          renderAssistantContent()
        )}
        {!isUser && !showThinkingUI && (
          <div style={styles.meta}>
            {message.latency_ms != null && (
              <>
                <Clock size={12} />
                <span>{(message.latency_ms / 1000).toFixed(1)}s</span>
              </>
            )}
            {message.sources && message.sources.length > 0 && (
              <span>{message.latency_ms != null && <>&middot; </>}{message.sources.length} source{message.sources.length > 1 ? 's' : ''}</span>
            )}
            {!message.isError && (
              <button onClick={handleCopy} style={styles.copyBtn} title="Copy answer">
                {copied ? <Check size={12} color="var(--success)" /> : <Copy size={12} />}
              </button>
            )}
          </div>
        )}
      </div>
      {isUser && (
        <div style={{ ...styles.avatar, background: 'var(--bg-3)' }}>
          <User size={16} />
        </div>
      )}
    </div>
  )
}

const markdownComponents = {
  p: ({ children }: { children?: React.ReactNode }) => (
    <span style={{ display: 'inline' }}>{children}</span>
  ),
  strong: ({ children }: { children?: React.ReactNode }) => (
    <strong style={{ fontWeight: 600 }}>{children}</strong>
  ),
  em: ({ children }: { children?: React.ReactNode }) => (
    <em>{children}</em>
  ),
  ul: ({ children }: { children?: React.ReactNode }) => (
    <ul style={{ margin: '8px 0', paddingLeft: 20, listStyleType: 'disc' }}>{children}</ul>
  ),
  ol: ({ children }: { children?: React.ReactNode }) => (
    <ol style={{ margin: '8px 0', paddingLeft: 20 }}>{children}</ol>
  ),
  li: ({ children }: { children?: React.ReactNode }) => (
    <li style={{ marginBottom: 4 }}>{children}</li>
  ),
  h1: ({ children }: { children?: React.ReactNode }) => (
    <h3 style={{ fontSize: 16, fontWeight: 700, margin: '12px 0 4px' }}>{children}</h3>
  ),
  h2: ({ children }: { children?: React.ReactNode }) => (
    <h3 style={{ fontSize: 15, fontWeight: 600, margin: '10px 0 4px' }}>{children}</h3>
  ),
  h3: ({ children }: { children?: React.ReactNode }) => (
    <h4 style={{ fontSize: 14, fontWeight: 600, margin: '8px 0 4px' }}>{children}</h4>
  ),
  code: ({ children, className }: { children?: React.ReactNode; className?: string }) => {
    const isBlock = className?.startsWith('language-')
    if (isBlock) {
      return (
        <pre
          style={{
            background: 'var(--bg-3)',
            padding: 12,
            borderRadius: 'var(--radius-sm, 6px)',
            fontSize: 13,
            overflowX: 'auto',
            margin: '8px 0',
          }}
        >
          <code>{children}</code>
        </pre>
      )
    }
    return (
      <code
        style={{
          background: 'var(--bg-3)',
          padding: '1px 5px',
          borderRadius: 3,
          fontSize: 13,
        }}
      >
        {children}
      </code>
    )
  },
  blockquote: ({ children }: { children?: React.ReactNode }) => (
    <blockquote
      style={{
        borderLeft: '3px solid var(--accent)',
        paddingLeft: 12,
        margin: '8px 0',
        color: 'var(--text-2)',
      }}
    >
      {children}
    </blockquote>
  ),
  table: ({ children }: { children?: React.ReactNode }) => (
    <div style={{ overflowX: 'auto', margin: '8px 0' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 13 }}>{children}</table>
    </div>
  ),
  th: ({ children }: { children?: React.ReactNode }) => (
    <th
      style={{
        border: '1px solid var(--border)',
        padding: '6px 10px',
        textAlign: 'left',
        fontWeight: 600,
        background: 'var(--bg-3)',
      }}
    >
      {children}
    </th>
  ),
  td: ({ children }: { children?: React.ReactNode }) => (
    <td style={{ border: '1px solid var(--border)', padding: '6px 10px' }}>{children}</td>
  ),
}

const styles: Record<string, React.CSSProperties> = {
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
    background: 'var(--accent-dim)',
    color: 'var(--accent)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  bubble: {
    padding: '12px 16px',
    borderRadius: 'var(--radius-md)',
    lineHeight: 1.6,
  },
  text: {
    fontSize: 14,
    whiteSpace: 'pre-wrap' as const,
    wordBreak: 'break-word' as const,
    margin: 0,
  },
  markdown: {
    fontSize: 14,
    wordBreak: 'break-word' as const,
    margin: 0,
    lineHeight: 1.6,
  },
  meta: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    marginTop: 8,
    fontSize: 12,
    color: 'var(--text-3)',
  },
  copyBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 24,
    height: 24,
    borderRadius: 'var(--radius-sm, 4px)',
    color: 'var(--text-3)',
    marginLeft: 'auto',
    cursor: 'pointer',
  },
}
