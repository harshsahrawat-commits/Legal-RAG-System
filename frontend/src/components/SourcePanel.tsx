import { useEffect, useCallback, useState } from 'react'
import { X, ChevronLeft, ChevronRight, Copy, Check, ChevronDown, ChevronUp } from 'lucide-react'
import { useStore } from '../store'

function formatPageNumbers(pages: number[]): string {
  if (!pages || pages.length === 0) return 'N/A'
  if (pages.length <= 5) return pages.join(', ')
  // Compact range format for many pages
  const sorted = [...pages].sort((a, b) => a - b)
  return `pp. ${sorted[0]}-${sorted[sorted.length - 1]}`
}

function stripMarkdownArtifacts(text: string): string {
  return text
    .replace(/\*\*([^*]+)\*\*/g, '$1')  // **bold** → bold
    .replace(/\*([^*]+)\*/g, '$1')       // *italic* → italic
    .replace(/^Case \d+:\d+-cv-\d+-\w+ Document \d+ Filed .+$/gm, '')  // PACER stamps
    .trim()
}

export default function SourcePanel() {
  const { sourcePanelOpen, selectedSourceIndex, currentSources, closeSourcePanel, navigateSource } =
    useStore()
  const [copied, setCopied] = useState(false)
  const [contextExpanded, setContextExpanded] = useState(false)

  const source = currentSources[selectedSourceIndex] ?? null
  const total = currentSources.length
  const hasPrev = selectedSourceIndex > 0
  const hasNext = selectedSourceIndex < total - 1

  // Reset state when source changes
  useEffect(() => {
    setCopied(false)
    setContextExpanded(false)
  }, [selectedSourceIndex])

  const handleKey = useCallback(
    (e: KeyboardEvent) => {
      if (!sourcePanelOpen) return
      if (e.key === 'Escape') closeSourcePanel()
      if (e.key === 'ArrowLeft' && hasPrev) navigateSource('prev')
      if (e.key === 'ArrowRight' && hasNext) navigateSource('next')
    },
    [sourcePanelOpen, hasPrev, hasNext, closeSourcePanel, navigateSource]
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [handleKey])

  const handleCopy = async () => {
    if (!source) return
    try {
      await navigator.clipboard.writeText(source.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Fallback for non-HTTPS contexts
      const ta = document.createElement('textarea')
      ta.value = source.content
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  if (!sourcePanelOpen || !source) return null

  const relevancePct = Math.round(source.relevance_score * 100)
  const hasContext = !!(source.context_before || source.context_after)
  const cleanedContent = stripMarkdownArtifacts(source.content)
  const pageDisplay = formatPageNumbers(source.page_numbers)

  return (
    <>
      <div style={styles.overlay} onClick={closeSourcePanel} />
      <div style={styles.panel}>
        {/* Header */}
        <div style={styles.header}>
          <div style={styles.headerLeft}>
            <span style={styles.headerTitle}>Source [{selectedSourceIndex + 1}]</span>
            <span style={styles.headerCount}>{selectedSourceIndex + 1} of {total}</span>
          </div>
          <div style={styles.headerRight}>
            <button onClick={() => navigateSource('prev')} disabled={!hasPrev} style={{ ...styles.navBtn, opacity: hasPrev ? 1 : 0.3 }}>
              <ChevronLeft size={18} />
            </button>
            <button onClick={() => navigateSource('next')} disabled={!hasNext} style={{ ...styles.navBtn, opacity: hasNext ? 1 : 0.3 }}>
              <ChevronRight size={18} />
            </button>
            <button onClick={closeSourcePanel} style={styles.closeBtn}>
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div style={styles.body}>
          {/* Document identity */}
          <div style={styles.metaSection}>
            <span style={styles.docTitle}>{source.document_title}</span>
            <span style={styles.shortCitation}>{source.short_citation}</span>
            {pageDisplay !== 'N/A' && (
              <span style={styles.pageInfo}>Pages: {pageDisplay}</span>
            )}

            {/* Relevance bar */}
            <div style={styles.relevanceRow}>
              <div style={styles.barTrack}>
                <div
                  style={{
                    ...styles.barFill,
                    width: `${relevancePct}%`,
                    background: 'var(--accent, #06b6d4)',
                  }}
                />
              </div>
              <span style={styles.relevancePct}>{relevancePct}%</span>
            </div>
          </div>

          {/* Source text */}
          <div style={styles.sourceSection}>
            <div style={styles.sourceBlock}>
              <div style={styles.sourceLabelRow}>
                <span style={styles.sourceLabel}>Source content</span>
                <button onClick={handleCopy} style={styles.copyBtn} title="Copy to clipboard">
                  {copied ? <Check size={14} color="var(--success, #22c55e)" /> : <Copy size={14} />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <p style={styles.sourceText}>{cleanedContent}</p>
            </div>

            {hasContext && (
              <button
                onClick={() => setContextExpanded(!contextExpanded)}
                style={styles.contextToggle}
              >
                {contextExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                <span>{contextExpanded ? 'Hide' : 'Show'} surrounding context</span>
              </button>
            )}

            {contextExpanded && source.context_before && (
              <div style={styles.contextBlock}>
                <span style={styles.contextLabel}>Context before</span>
                <p style={styles.contextText}>{source.context_before}</p>
              </div>
            )}

            {contextExpanded && source.context_after && (
              <div style={styles.contextBlock}>
                <span style={styles.contextLabel}>Context after</span>
                <p style={styles.contextText}>{source.context_after}</p>
              </div>
            )}
          </div>

          {/* Citation */}
          <div style={styles.citationSection}>
            <span style={styles.contextLabel}>Citation</span>
            <p style={styles.citationText}>{source.long_citation}</p>
          </div>
        </div>
      </div>
    </>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0,0,0,0.5)',
    zIndex: 998,
  },
  panel: {
    position: 'fixed',
    right: 0,
    top: 0,
    height: '100vh',
    width: 480,
    maxWidth: '100vw',
    background: 'var(--bg-1)',
    borderLeft: '1px solid var(--border)',
    boxShadow: 'var(--shadow-lg)',
    zIndex: 999,
    display: 'flex',
    flexDirection: 'column',
    animation: 'slideIn 0.25s ease-out',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '14px 20px',
    borderBottom: '1px solid var(--border)',
    flexShrink: 0,
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  headerTitle: {
    fontSize: 15,
    fontWeight: 600,
  },
  headerCount: {
    fontSize: 12,
    color: 'var(--text-3)',
  },
  headerRight: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
  },
  navBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 32,
    height: 32,
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-2)',
    transition: 'all var(--transition)',
  },
  closeBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 32,
    height: 32,
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-2)',
    marginLeft: 8,
  },
  body: {
    flex: 1,
    overflowY: 'auto',
    padding: 20,
    display: 'flex',
    flexDirection: 'column',
    gap: 20,
  },
  metaSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    padding: 16,
    background: 'var(--bg-2)',
    borderRadius: 'var(--radius-md)',
    border: '1px solid var(--border)',
  },
  docTitle: {
    fontSize: 16,
    fontWeight: 700,
    color: 'var(--text-1)',
    lineHeight: 1.3,
  },
  shortCitation: {
    fontSize: 12,
    color: 'var(--text-3)',
    lineHeight: 1.4,
  },
  pageInfo: {
    fontSize: 12,
    color: 'var(--text-3)',
  },
  relevanceRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    marginTop: 4,
  },
  barTrack: {
    flex: 1,
    height: 6,
    borderRadius: 3,
    background: 'var(--bg-3)',
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: 3,
    transition: 'width 0.3s ease',
  },
  relevancePct: {
    fontSize: 13,
    fontWeight: 600,
    color: 'var(--text-1)',
    minWidth: 36,
    textAlign: 'right',
  },
  sourceSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  contextBlock: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  contextLabel: {
    fontSize: 11,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: 'var(--text-3)',
  },
  contextText: {
    fontSize: 13,
    lineHeight: 1.6,
    color: 'var(--text-3)',
    padding: 12,
    background: 'var(--bg-2)',
    borderRadius: 'var(--radius-sm)',
    borderLeft: '3px solid var(--border)',
    margin: 0,
  },
  contextToggle: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '6px 0',
    fontSize: 12,
    color: 'var(--text-3)',
    cursor: 'pointer',
    background: 'none',
    border: 'none',
    transition: 'color var(--transition)',
  },
  sourceBlock: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  sourceLabelRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  sourceLabel: {
    fontSize: 11,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: 'var(--accent)',
  },
  copyBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    padding: '4px 8px',
    fontSize: 11,
    color: 'var(--text-3)',
    background: 'var(--bg-2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm, 4px)',
    cursor: 'pointer',
    transition: 'all var(--transition)',
  },
  sourceText: {
    fontSize: 14,
    lineHeight: 1.7,
    color: 'var(--text-1)',
    padding: 16,
    background: 'var(--bg-2)',
    borderRadius: 'var(--radius-sm)',
    borderLeft: '3px solid var(--accent)',
    margin: 0,
  },
  citationSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    padding: 12,
    background: 'var(--bg-2)',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--border)',
  },
  citationText: {
    fontSize: 12,
    lineHeight: 1.5,
    color: 'var(--text-2)',
    margin: 0,
    fontFamily: 'monospace',
  },
}
