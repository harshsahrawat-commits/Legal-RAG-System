import React, { useMemo, useState, useRef, useEffect, useCallback } from 'react'
import { User, Bot, Clock, Copy, Check, AlertCircle, FileText, ExternalLink, ChevronRight } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage as ChatMessageType, SourceInfo } from '../types'
import { parseCitations } from '../types'

interface Props {
  message: ChatMessageType
  isStreaming?: boolean
}

const SOURCE_COLORS: Record<string, { bg: string; fg: string; label: string }> = {
  cylaw: { bg: 'var(--badge-cylaw-bg, #EBF0F7)', fg: 'var(--badge-cylaw-fg, #3A5A8C)', label: 'CyLaw' },
  hudoc: { bg: 'var(--badge-echr-bg, #E6F5F2)', fg: 'var(--badge-echr-fg, #0A7B6E)', label: 'ECHR' },
  eurlex: { bg: 'var(--badge-eurlex-bg, #F5F0E6)', fg: 'var(--badge-eurlex-fg, #8C7A3A)', label: 'EU Law' },
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

  const origin = source?.source_origin || 'cylaw'
  const colors = SOURCE_COLORS[origin] || SOURCE_COLORS.cylaw

  return (
    <span
      ref={ref}
      style={{
        ...badgeStyles.badge,
        background: valid ? '#E6F5F2' : '#F0F0ED',
        color: valid ? '#0A9B8A' : 'var(--text-3)',
        border: valid ? '1px solid #C5E5DF' : '1px solid var(--border)',
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
          <span style={{ display: 'block', fontSize: 10, fontWeight: 600, color: colors.fg, marginBottom: 4, textTransform: 'uppercase' as const, letterSpacing: '0.05em', fontFamily: 'var(--font-mono)' }}>
            {colors.label}
          </span>
          <strong style={{ display: 'block', marginBottom: 4, color: 'var(--text-1)' }}>
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
    padding: '1px 5px',
    fontSize: 11,
    fontWeight: 500,
    fontFamily: 'var(--font-mono)',
    borderRadius: 4,
    transition: 'all var(--transition)',
    verticalAlign: 'super',
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
    background: '#FFFFFF',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm, 6px)',
    boxShadow: 'var(--shadow-lg)',
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

function OriginBadge({ origin }: { origin: string }) {
  const colors = SOURCE_COLORS[origin] || SOURCE_COLORS.cylaw
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      padding: '3px 8px',
      fontSize: 10,
      fontWeight: 500,
      fontFamily: 'var(--font-mono)',
      borderRadius: 4,
      background: colors.bg,
      color: colors.fg,
      letterSpacing: '0.5px',
      lineHeight: 1,
      whiteSpace: 'nowrap' as const,
      textTransform: 'uppercase' as const,
    }}>
      {colors.label}
    </span>
  )
}

function RelevanceBar({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const barColor = pct >= 80 ? '#0A9B8A' : pct >= 60 ? '#5AADA3' : '#A0CCC6'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 80 }}>
      <div style={{
        flex: 1,
        height: 3,
        borderRadius: 2,
        background: '#E8E8E5',
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          borderRadius: 2,
          background: barColor,
          transition: 'width 0.3s ease',
        }} />
      </div>
      <span style={{
        fontSize: 10.5,
        fontFamily: 'var(--font-mono)',
        color: 'var(--text-3)',
        minWidth: 28,
        textAlign: 'right' as const,
      }}>
        {pct}%
      </span>
    </div>
  )
}

function getSourceUrl(source: SourceInfo): string | null {
  if (source.external_url) return source.external_url
  if (source.cylaw_url) return source.cylaw_url
  return null
}

function SourceRow({
  source,
  index,
  isActive,
  messageId,
}: {
  source: SourceInfo
  index: number
  isActive: boolean
  messageId: string
}) {
  const [hover, setHover] = useState(false)
  const origin = source.source_origin || 'cylaw'
  const url = getSourceUrl(source)

  return (
    <div
      id={`source-${messageId}-${index + 1}`}
      style={{
        display: 'grid',
        gridTemplateColumns: '40px 1fr auto',
        gap: 8,
        alignItems: 'start',
        padding: '14px 16px',
        borderRadius: 6,
        background: isActive ? '#E6F5F2' : hover ? '#FFFFFF' : 'transparent',
        borderLeft: isActive ? '3px solid #0A9B8A' : '3px solid transparent',
        borderBottom: '1px solid #EAEAE7',
        transition: 'all 0.15s ease',
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      {/* Col 1: Citation number */}
      <span style={{
        fontSize: 12,
        fontFamily: 'var(--font-mono)',
        color: isActive ? '#0A9B8A' : 'var(--text-3)',
        paddingTop: 1,
      }}>
        [{index + 1}]
      </span>

      {/* Col 2: Title + origin badge + short citation */}
      <div style={{ display: 'flex', flexDirection: 'column' as const, gap: 4, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' as const }}>
          {url ? (
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                fontSize: 14,
                fontWeight: 600,
                color: 'var(--text-1)',
                textDecoration: 'none',
                lineHeight: 1.4,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap' as const,
                maxWidth: '100%',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--accent)'; e.currentTarget.style.textDecoration = 'underline' }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-1)'; e.currentTarget.style.textDecoration = 'none' }}
            >
              {source.section || source.document_title}
              <ExternalLink size={10} style={{ marginLeft: 4, verticalAlign: 'middle', opacity: 0.4 }} />
            </a>
          ) : (
            <span style={{
              fontSize: 14,
              fontWeight: 600,
              color: 'var(--text-1)',
              lineHeight: 1.4,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap' as const,
            }}>
              {source.section || source.document_title}
            </span>
          )}
          <OriginBadge origin={origin} />
        </div>
        {source.short_citation && (
          <span style={{
            fontSize: 12.5,
            color: 'var(--text-2)',
            lineHeight: 1.4,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap' as const,
          }}>
            {source.short_citation}
          </span>
        )}
      </div>

      {/* Col 3: Relevance bar + View button */}
      <div style={{ display: 'flex', flexDirection: 'column' as const, alignItems: 'flex-end', gap: 6, paddingTop: 1 }}>
        <RelevanceBar score={source.relevance_score} />
        {url && (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
              padding: '4px 10px',
              fontSize: 10.5,
              fontFamily: 'var(--font-mono)',
              fontWeight: 500,
              color: 'var(--accent)',
              background: '#FFFFFF',
              border: '1px solid var(--border)',
              borderRadius: 4,
              textDecoration: 'none',
              textTransform: 'uppercase' as const,
              letterSpacing: '0.5px',
              transition: 'all 0.15s ease',
              whiteSpace: 'nowrap' as const,
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#E6F5F2'
              e.currentTarget.style.borderColor = '#C5E5DF'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = '#FFFFFF'
              e.currentTarget.style.borderColor = 'var(--border)'
            }}
          >
            View
            <ExternalLink size={9} />
          </a>
        )}
      </div>
    </div>
  )
}

function InlineSources({
  sources,
  messageId,
  expanded,
  onToggle,
  activeCitation,
}: {
  sources: SourceInfo[]
  messageId: string
  expanded: boolean
  onToggle: () => void
  activeCitation: number | null
}) {
  return (
    <div style={{ marginTop: 8 }}>
      {/* Toggle header */}
      <button
        onClick={onToggle}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '6px 0',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: 'var(--accent)',
          fontSize: 11,
          fontWeight: 500,
          textTransform: 'uppercase' as const,
          letterSpacing: '2px',
          fontFamily: 'var(--font-mono)',
          transition: 'color 0.15s ease',
        }}
      >
        <ChevronRight
          size={13}
          style={{
            transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s ease',
          }}
        />
        Sources &middot; {sources.length}
      </button>

      {/* Expanded source list */}
      {expanded && (
        <div style={{
          display: 'flex',
          flexDirection: 'column' as const,
          gap: 0,
          paddingTop: 4,
          background: '#F7F7F5',
          borderRadius: 8,
          padding: 16,
          animation: 'fadeIn 0.2s ease',
        }}>
          {sources.map((source, i) => (
            <SourceRow
              key={source.chunk_id || i}
              source={source}
              index={i}
              isActive={activeCitation === i + 1}
              messageId={messageId}
            />
          ))}
        </div>
      )}
    </div>
  )
}

const THINKING_MESSAGES = [
  'Searching through legal documents\u2026',
  'Analyzing relevant sections\u2026',
  'Preparing your response\u2026',
]

function ThinkingIndicator({ hasSources, sourceCount }: { hasSources: boolean; sourceCount: number }) {
  const [statusIndex, setStatusIndex] = useState(0)
  const [textVisible, setTextVisible] = useState(true)
  const fadeTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    if (hasSources) return
    const interval = setInterval(() => {
      setTextVisible(false)
      fadeTimer.current = setTimeout(() => {
        setStatusIndex((prev) => (prev + 1) % THINKING_MESSAGES.length)
        setTextVisible(true)
      }, 500)
    }, 4000)
    return () => {
      clearInterval(interval)
      if (fadeTimer.current) clearTimeout(fadeTimer.current)
    }
  }, [hasSources])

  return (
    <div style={{ minWidth: 260, minHeight: 80, animation: 'fadeIn 0.25s ease' }}>
      {hasSources && (
        <div style={thinkingStyles.sourcesBadge}>
          <FileText size={12} />
          <span>{sourceCount} source{sourceCount !== 1 ? 's' : ''} found</span>
        </div>
      )}

      <div style={{ ...thinkingStyles.statusRow, marginTop: hasSources ? 10 : 0 }}>
        <span style={thinkingStyles.pulseDot} />
        <span style={{
          ...thinkingStyles.statusText,
          opacity: hasSources ? 1 : (textVisible ? 1 : 0),
        }}>
          {hasSources ? 'Generating answer\u2026' : THINKING_MESSAGES[statusIndex]}
        </span>
      </div>

      <div style={thinkingStyles.skeletonGroup}>
        {(hasSources ? [75, 60] : [85, 70, 55]).map((w, i) => (
          <div
            key={i}
            style={{
              ...thinkingStyles.skeletonLine,
              width: `${w}%`,
              animation: `shimmerSlide 1.8s ease-in-out ${i * 0.15}s infinite`,
            }}
          />
        ))}
      </div>
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
    background: '#0A9B8A',
    animation: 'pulseDot 1.5s ease-in-out infinite',
    flexShrink: 0,
  },
  statusText: {
    fontSize: 13,
    color: 'var(--text-3)',
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
    background: 'linear-gradient(90deg, #E8E8E5 25%, #D8D8D5 50%, #E8E8E5 75%)',
    backgroundSize: '200% 100%',
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
}

function ChatMessage({ message, isStreaming = false }: Props) {
  const isUser = message.role === 'user'
  const [copied, setCopied] = useState(false)
  const [sourcesExpanded, setSourcesExpanded] = useState(false)
  const [activeCitation, setActiveCitation] = useState<number | null>(null)

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  // Smooth transition: thinking -> streaming content
  const [isFadingOut, setIsFadingOut] = useState(false)
  const wasThinkingRef = useRef(true)
  const fadeTimerRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    if (isStreaming && wasThinkingRef.current && message.content !== '') {
      wasThinkingRef.current = false
      setIsFadingOut(true)
      fadeTimerRef.current = setTimeout(() => setIsFadingOut(false), 150)
    }
    if (message.content === '') {
      wasThinkingRef.current = true
    }
  }, [isStreaming, message.content])

  useEffect(() => {
    return () => {
      if (fadeTimerRef.current) clearTimeout(fadeTimerRef.current)
    }
  }, [])

  const showThinkingUI = isStreaming && (message.content === '' || isFadingOut)

  const parts = useMemo(
    () => (isUser ? null : parseCitations(message.content)),
    [message.content, isUser]
  )

  const isValidCitation = (num: number) => {
    return message.sources && num >= 1 && num <= message.sources.length
  }

  const handleCitationClick = useCallback((num: number) => {
    if (!message.sources || num < 1 || num > message.sources.length) return
    setActiveCitation(num)
    setSourcesExpanded(true)
    // Scroll to the source row after DOM update
    requestAnimationFrame(() => {
      const el = document.getElementById(`source-${message.id}-${num}`)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      }
    })
  }, [message.sources, message.id])

  const renderAssistantContent = () => {
    if (!parts) return null

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

    const hasMarkdown = /(\*\*.+?\*\*|__.+?__|#{1,3}\s|```.+?```|^\s*[-*]\s|\|.+\|)/ms.test(message.content)

    if (!hasMarkdown) {
      return (
        <div style={styles.text}>
          {parts.map((part, i) =>
            part.type === 'text' ? (
              <span key={i}>{part.content}</span>
            ) : (
              <CitationBadge
                key={i}
                num={part.citationNumber!}
                source={message.sources?.[part.citationNumber! - 1]}
                valid={isValidCitation(part.citationNumber!)}
                onClick={() => handleCitationClick(part.citationNumber!)}
              />
            )
          )}
        </div>
      )
    }

    return (
      <div style={styles.markdown}>
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
              valid={isValidCitation(part.citationNumber!)}
              onClick={() => handleCitationClick(part.citationNumber!)}
            />
          )
        )}
      </div>
    )
  }

  const hasSources = !isUser && !showThinkingUI && message.sources && message.sources.length > 0

  return (
    <div style={{ ...styles.row, justifyContent: isUser ? 'flex-end' : 'flex-start' }}>
      {!isUser && (
        <div style={styles.avatar}>
          <Bot size={16} />
        </div>
      )}
      <div style={{ maxWidth: isUser ? '70%' : '80%', display: 'flex', flexDirection: 'column' as const }}>
        <div
          style={{
            ...styles.bubble,
            ...(message.isError ? {
              background: 'var(--danger-dim)',
              border: '1px solid rgba(217,75,75,0.2)',
            } : isUser ? {
              background: 'var(--accent)',
              color: '#FFFFFF',
              borderRadius: '16px 16px 4px 16px',
            } : {
              background: '#FFFFFF',
              border: '1px solid var(--border)',
              borderLeft: '3px solid var(--accent)',
              boxShadow: '0 1px 4px rgba(0,0,0,0.03)',
              borderRadius: 12,
            }),
          }}
        >
          {isUser ? (
            <p style={{ ...styles.text, color: '#FFFFFF', fontWeight: 500, fontSize: 14.5 }}>{message.content}</p>
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

        {/* Inline collapsible sources */}
        {hasSources && (
          <div style={{ paddingLeft: 4 }}>
            <InlineSources
              sources={message.sources!}
              messageId={message.id}
              expanded={sourcesExpanded}
              onToggle={() => {
                setSourcesExpanded(!sourcesExpanded)
                if (sourcesExpanded) setActiveCitation(null)
              }}
              activeCitation={activeCitation}
            />
          </div>
        )}
      </div>
      {isUser && (
        <div style={{ ...styles.avatar, background: 'var(--accent-dim)', color: 'var(--accent)' }}>
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
            background: '#F3F3F0',
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
          background: '#F3F3F0',
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
        background: '#F0F0ED',
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
    maxWidth: 820,
    margin: '0 auto',
    width: '100%',
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
    padding: '28px 32px',
    borderRadius: 'var(--radius-md)',
    lineHeight: 1.78,
  },
  text: {
    fontSize: 15.5,
    whiteSpace: 'pre-wrap' as const,
    wordBreak: 'break-word' as const,
    margin: 0,
    lineHeight: 1.78,
    letterSpacing: '0.01em',
    color: 'var(--text-1)',
  },
  markdown: {
    fontSize: 15.5,
    wordBreak: 'break-word' as const,
    margin: 0,
    lineHeight: 1.78,
    letterSpacing: '0.01em',
    color: 'var(--text-1)',
  },
  meta: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    marginTop: 16,
    paddingTop: 12,
    borderTop: '1px solid #F0F0ED',
    fontFamily: 'var(--font-mono)',
    fontSize: 11.5,
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

export default React.memo(ChatMessage)
