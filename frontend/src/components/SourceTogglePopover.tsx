import { useRef, useEffect } from 'react'
import { useStore } from '../store'
import type { SourceToggles } from '../types'

const SOURCE_CONFIG: { key: keyof Omit<SourceToggles, 'families'>; label: string; sublabel: string; color: string }[] = [
  { key: 'cylaw', label: 'CyLaw Documents', sublabel: 'Cyprus Law', color: 'var(--badge-cylaw-fg, #3A5A8C)' },
  { key: 'hudoc', label: 'HUDOC (ECHR)', sublabel: 'European Court of Human Rights', color: 'var(--badge-echr-fg, #0A7B6E)' },
  { key: 'eurlex', label: 'EUR-Lex (EU Law)', sublabel: 'EU legislation & regulations', color: 'var(--badge-eurlex-fg, #8C7A3A)' },
]

interface Props {
  open: boolean
  onClose: () => void
}

export default function SourceTogglePopover({ open, onClose }: Props) {
  const sourceToggles = useStore((s) => s.sourceToggles)
  const setSourceToggle = useStore((s) => s.setSourceToggle)
  const families = useStore((s) => s.families)
  const setFamilyToggle = useStore((s) => s.setFamilyToggle)
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

  const activeFamilies = families.filter((f) => f.is_active)

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
              background: sourceToggles[key] ? 'var(--accent)' : '#D0D0CC',
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

      {activeFamilies.length > 0 && (
        <>
          <div style={styles.divider} />
          <div style={styles.header}>My Collections</div>
          {activeFamilies.map((fam) => {
            const isOn = sourceToggles.families.includes(fam.id)
            return (
              <div key={fam.id} style={styles.row} onClick={() => setFamilyToggle(fam.id, !isOn)}>
                <div style={styles.labelGroup}>
                  <div style={styles.labelRow}>
                    <span style={{ ...styles.dot, background: 'var(--accent)' }} />
                    <span style={styles.label}>{fam.name}</span>
                  </div>
                  <span style={styles.sublabel}>{fam.document_count} document{fam.document_count !== 1 ? 's' : ''}</span>
                </div>
                <button
                  style={{
                    ...styles.toggle,
                    background: isOn ? 'var(--accent)' : '#D0D0CC',
                  }}
                  onClick={(e) => {
                    e.stopPropagation()
                    setFamilyToggle(fam.id, !isOn)
                  }}
                  aria-label={`Toggle ${fam.name}`}
                  role="switch"
                  aria-checked={isOn}
                >
                  <span
                    style={{
                      ...styles.toggleDot,
                      transform: isOn ? 'translateX(16px)' : 'translateX(0)',
                    }}
                  />
                </button>
              </div>
            )
          })}
        </>
      )}
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
    background: '#FFFFFF',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md, 8px)',
    boxShadow: 'var(--shadow-lg)',
    zIndex: 100,
    overflow: 'hidden',
    animation: 'fadeIn 0.15s ease',
  },
  header: {
    padding: '10px 14px',
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    fontWeight: 500,
    textTransform: 'uppercase' as const,
    letterSpacing: '2px',
    color: 'var(--accent)',
    borderBottom: '1px solid var(--border)',
  },
  divider: {
    height: 1,
    background: 'var(--border)',
    margin: '4px 0',
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
    boxShadow: '0 1px 2px rgba(0,0,0,0.15)',
  },
}
