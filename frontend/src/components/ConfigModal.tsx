import { useState, useEffect } from 'react'
import { X, Loader2, Check } from 'lucide-react'
import { api } from '../api'
import type { TenantConfig } from '../types'

interface Props {
  open: boolean
  onClose: () => void
}

export default function ConfigModal({ open, onClose }: Props) {
  const [config, setConfig] = useState<TenantConfig | null>(null)
  const [language, setLanguage] = useState('en')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')
  const [loadError, setLoadError] = useState(false)

  useEffect(() => {
    if (!open) return
    setLoadError(false)
    setError('')
    api.config.get().then(({ data }) => {
      setConfig(data)
      setLanguage(data.language)
    }).catch(() => {
      setLoadError(true)
    })
  }, [open])

  const handleSave = async () => {
    setSaving(true)
    setSaved(false)
    setError('')
    try {
      const { data } = await api.config.update({ language })
      setConfig(data)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch {
      setError('Failed to save settings. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  if (!open) return null

  return (
    <>
      <div style={styles.overlay} onClick={onClose} />
      <div style={styles.modal}>
        <div style={styles.header}>
          <h2 style={styles.title}>Settings</h2>
          <button onClick={onClose} style={styles.closeBtn}>
            <X size={18} />
          </button>
        </div>

        {loadError ? (
          <div style={styles.loading}>
            <span style={{ color: 'var(--danger)', fontSize: 13 }}>Failed to load settings</span>
          </div>
        ) : config ? (
          <div style={styles.body}>
            <div style={styles.field}>
              <label style={styles.label}>Language</label>
              <select value={language} onChange={(e) => setLanguage(e.target.value)} style={styles.select}>
                <option value="en">English</option>
                <option value="el">Greek</option>
              </select>
            </div>

            <div style={styles.field}>
              <label style={styles.label}>Embedding Model</label>
              <input value={config.embedding_model} readOnly style={styles.readOnly} />
            </div>

            <div style={styles.field}>
              <label style={styles.label}>LLM Model</label>
              <input value={config.llm_model} readOnly style={styles.readOnly} />
            </div>

            <div style={styles.field}>
              <label style={styles.label}>Reranker</label>
              <input value={config.reranker_model} readOnly style={styles.readOnly} />
            </div>

            <div style={styles.field}>
              <label style={styles.label}>FTS Language</label>
              <input value={config.fts_language} readOnly style={styles.readOnly} />
            </div>

            {error && (
              <div style={{ color: 'var(--danger)', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>{error}</span>
              </div>
            )}

            <button onClick={handleSave} disabled={saving} style={styles.saveBtn}>
              {saving ? (
                <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
              ) : saved ? (
                <><Check size={16} /> Saved</>
              ) : (
                'Save Changes'
              )}
            </button>
          </div>
        ) : (
          <div style={styles.loading}>
            <Loader2 size={24} style={{ animation: 'spin 1s linear infinite' }} color="var(--text-3)" />
          </div>
        )}
      </div>

    </>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0,0,0,0.3)',
    zIndex: 1000,
  },
  modal: {
    position: 'fixed',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: 440,
    maxWidth: '90vw',
    background: 'var(--bg-1)',
    borderRadius: 'var(--radius-lg)',
    border: '1px solid var(--border)',
    boxShadow: 'var(--shadow-lg)',
    zIndex: 1001,
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '16px 20px',
    borderBottom: '1px solid var(--border)',
  },
  title: {
    fontSize: 16,
    fontWeight: 600,
  },
  closeBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 32,
    height: 32,
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-2)',
  },
  body: {
    padding: 20,
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  loading: {
    padding: 40,
    display: 'flex',
    justifyContent: 'center',
  },
  field: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  label: {
    fontSize: 12,
    fontWeight: 500,
    color: 'var(--text-3)',
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
  },
  select: {
    padding: '10px 14px',
    fontSize: 14,
    background: 'var(--bg-2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-1)',
    outline: 'none',
    cursor: 'pointer',
  },
  readOnly: {
    padding: '10px 14px',
    fontSize: 13,
    background: 'var(--bg-2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-3)',
    outline: 'none',
    fontFamily: 'monospace',
  },
  saveBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginTop: 8,
    padding: '10px 20px',
    background: 'var(--accent)',
    color: '#FFFFFF',
    fontWeight: 600,
    fontSize: 14,
    borderRadius: 'var(--radius-sm)',
    border: 'none',
    cursor: 'pointer',
  },
}
