import { useRef, useEffect } from 'react'
import { useStore } from '../store'
import type { SourceToggles } from '../types'

const SOURCE_CONFIG: { key: keyof SourceToggles; label: string; sublabel: string; color: string }[] = [
  { key: 'cylaw', label: 'CyLaw Documents', sublabel: 'Uploaded PDFs', color: '#06b6d4' },
  { key: 'hudoc', label: 'HUDOC (ECHR)', sublabel: 'European Court of Human Rights', color: '#8b5cf6' },
  { key: 'eurlex', label: 'EUR-Lex (EU Law)', sublabel: 'EU legislation & regulations', color: '#f59e0b' },
]

interface Props {
  open: boolean
  onClose: () => void
}

export default function SourceTogglePopover({ open, onClose }: Props) {
  const sourceToggles = useStore((s) => s.sourceToggles)
  const setSourceToggle = useStore((s) => s.setSourceToggle)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose()
      }
    }
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('mousedown', handleClick)
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('mousedown', handleClick)
      document.removeEventListener('keydown', handleKey)
    }
  }, [open, onClose])

  if (!open) return null

  return (
    <div ref={ref} style={styles.popover}>
      <div style={styles.header}>Search Sources</div>
      {SOURCE_CONFIG.map(({ key, label, sublabel, color }) => (
        <div key={key} style={styles.row} onClick={() => setSourceToggle(key, !sourceToggles[key])}>
          <div style={styles.labelGroup}>
            <div style={styles.labelRow}>
              <span style={{ ...styles.dot, background: color }} />
              <span style={styles.label}>{label}</span>
            </div>
            <span style={styles.sublabel}>{sublabel}</span>
          </div>
          <button
            style={{
              ...styles.toggle,
              background: sourceToggles[key] ? color : 'var(--bg-3)',
            }}
            onClick={(e) => {
              e.stopPropagation()
              setSourceToggle(key, !sourceToggles[key])
            }}
            aria-label={`Toggle ${label}`}
            role="switch"
            aria-checked={sourceToggles[key]}
          >
            <span
              style={{
                ...styles.toggleDot,
                transform: sourceToggles[key] ? 'translateX(16px)' : 'translateX(0)',
              }}
            />
          </button>
        </div>
      ))}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  popover: {
    position: 'absolute',
    bottom: '100%',
    right: 0,
    marginBottom: 8,
    width: 280,
    background: 'var(--bg-1)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md, 8px)',
    boxShadow: 'var(--shadow-lg, 0 4px 12px rgba(0,0,0,0.3))',
    zIndex: 100,
    overflow: 'hidden',
  },
  header: {
    padding: '10px 14px',
    fontSize: 12,
    fontWeight: 600,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
    color: 'var(--text-3)',
    borderBottom: '1px solid var(--border)',
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '10px 14px',
    cursor: 'pointer',
    transition: 'background 0.15s',
  },
  labelGroup: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 2,
  },
  labelRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    flexShrink: 0,
  },
  label: {
    fontSize: 13,
    fontWeight: 500,
    color: 'var(--text-1)',
  },
  sublabel: {
    fontSize: 11,
    color: 'var(--text-3)',
    marginLeft: 16,
  },
  toggle: {
    width: 36,
    height: 20,
    borderRadius: 10,
    border: 'none',
    cursor: 'pointer',
    position: 'relative' as const,
    flexShrink: 0,
    transition: 'background 0.2s',
    padding: 0,
  },
  toggleDot: {
    display: 'block',
    width: 16,
    height: 16,
    borderRadius: '50%',
    background: '#fff',
    position: 'absolute' as const,
    top: 2,
    left: 2,
    transition: 'transform 0.2s',
    boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
  },
}
