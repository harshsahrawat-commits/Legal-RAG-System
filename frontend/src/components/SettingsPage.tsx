import { useState, useEffect, useCallback } from 'react'
import { ArrowLeft, Upload, Trash2, Loader2, AlertCircle, FileText } from 'lucide-react'
import { api } from '../api'
import { useStore } from '../store'
import type { DocumentInfo, SourceToggles } from '../types'

const SOURCE_LABELS: Record<keyof SourceToggles, { label: string; desc: string }> = {
  cylaw: { label: 'CyLaw (Cyprus Law)', desc: 'Cyprus legislation and legal texts' },
  hudoc: { label: 'HUDOC (ECHR)', desc: 'European Court of Human Rights case law' },
  eurlex: { label: 'EUR-Lex (EU Law)', desc: 'European Union legislation and regulations' },
}

export default function SettingsPage() {
  const user = useStore((s) => s.user)
  const logout = useStore((s) => s.logout)
  const setSettingsOpen = useStore((s) => s.setSettingsOpen)
  const sourceToggles = useStore((s) => s.sourceToggles)
  const setSourceToggle = useStore((s) => s.setSourceToggle)

  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [loadingDocs, setLoadingDocs] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState('')
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)

  const loadDocs = useCallback(async () => {
    try {
      const { data } = await api.documents.list()
      setDocuments(data)
    } catch {
      // ignore for API key users
    } finally {
      setLoadingDocs(false)
    }
  }, [])

  useEffect(() => {
    loadDocs()
  }, [loadDocs])

  const handleUpload = async (files: FileList | null) => {
    if (!files?.length) return
    const pdfFiles = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'))
    if (!pdfFiles.length) {
      setError('Only PDF files are supported')
      return
    }

    setError('')
    setUploading(true)
    setUploadProgress(0)

    for (const file of pdfFiles) {
      try {
        await api.documents.upload(file, setUploadProgress)
        loadDocs()
      } catch {
        setError(`Failed to upload ${file.name}`)
      }
    }

    setUploading(false)
    setUploadProgress(0)
  }

  const handleDelete = async (id: string) => {
    if (confirmDeleteId !== id) {
      setConfirmDeleteId(id)
      setTimeout(() => setConfirmDeleteId((cur) => (cur === id ? null : cur)), 3000)
      return
    }
    setConfirmDeleteId(null)
    try {
      await api.documents.delete(id)
      setDocuments((prev) => prev.filter((d) => d.id !== id))
    } catch {
      setError('Failed to delete document')
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.inner}>
        {/* Back button */}
        <button onClick={() => setSettingsOpen(false)} style={styles.backBtn}>
          <ArrowLeft size={18} />
          <span>Back to chat</span>
        </button>

        <h1 style={styles.pageTitle}>Settings</h1>

        {/* Account Section */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Account</h2>
          <div style={styles.card}>
            {user ? (
              <div style={styles.accountRow}>
                {user.avatar_url ? (
                  <img src={user.avatar_url} alt="" style={styles.avatar} />
                ) : (
                  <div style={styles.avatarPlaceholder}>
                    {(user.name || user.email || '?')[0].toUpperCase()}
                  </div>
                )}
                <div style={styles.accountInfo}>
                  {user.name && <div style={styles.accountName}>{user.name}</div>}
                  <div style={styles.accountEmail}>{user.email}</div>
                </div>
                <button onClick={logout} style={styles.signOutBtn}>Sign Out</button>
              </div>
            ) : (
              <div style={styles.accountRow}>
                <span style={styles.accountEmail}>Authenticated via API key</span>
                <button onClick={logout} style={styles.signOutBtn}>Sign Out</button>
              </div>
            )}
          </div>
        </section>

        {/* Search Sources Section */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Search Sources</h2>
          <div style={styles.card}>
            {(Object.keys(SOURCE_LABELS) as (keyof SourceToggles)[]).map((key) => (
              <label key={key} style={styles.toggleRow}>
                <div style={styles.toggleInfo}>
                  <span style={styles.toggleLabel}>{SOURCE_LABELS[key].label}</span>
                  <span style={styles.toggleDesc}>{SOURCE_LABELS[key].desc}</span>
                </div>
                <input
                  type="checkbox"
                  checked={sourceToggles[key]}
                  onChange={(e) => setSourceToggle(key, e.target.checked)}
                  style={styles.checkbox}
                />
              </label>
            ))}
          </div>
        </section>

        {/* Documents Section */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>My Documents</h2>
          <p style={styles.sectionDesc}>
            Files uploaded here are included in all your queries as persistent search sources.
          </p>

          <div
            style={styles.dropZone}
            onClick={() => document.getElementById('settings-file-input')?.click()}
          >
            {uploading ? (
              <>
                <Loader2 size={24} style={{ animation: 'spin 1s linear infinite' }} />
                <span style={styles.dropText}>Uploading... {uploadProgress}%</span>
              </>
            ) : (
              <>
                <Upload size={24} color="var(--text-3)" />
                <span style={styles.dropText}>Click to upload PDF</span>
              </>
            )}
            <input
              id="settings-file-input"
              type="file"
              accept=".pdf"
              multiple
              style={{ display: 'none' }}
              onChange={(e) => { handleUpload(e.target.files); e.target.value = '' }}
            />
          </div>

          {error && (
            <div style={styles.errorRow}>
              <AlertCircle size={14} />
              <span>{error}</span>
            </div>
          )}

          <div style={styles.docList}>
            {loadingDocs && <p style={styles.emptyText}>Loading documents...</p>}
            {!loadingDocs && documents.length === 0 && (
              <p style={styles.emptyText}>No documents uploaded yet</p>
            )}
            {documents.map((doc) => (
              <div key={doc.id} style={styles.docItem}>
                <FileText size={16} color="var(--text-2)" style={{ flexShrink: 0 }} />
                <span style={styles.docTitle} title={doc.title}>{doc.title}</span>
                <span style={styles.docMeta}>{doc.page_count} pages</span>
                <button
                  onClick={() => handleDelete(doc.id)}
                  style={{
                    ...styles.deleteBtn,
                    ...(confirmDeleteId === doc.id ? { color: 'var(--danger)', fontWeight: 600, fontSize: 11 } : {}),
                  }}
                  title={confirmDeleteId === doc.id ? 'Click again to confirm' : 'Delete'}
                >
                  {confirmDeleteId === doc.id ? 'Confirm?' : <Trash2 size={14} />}
                </button>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    flex: 1,
    overflowY: 'auto',
    background: 'var(--bg-0)',
  },
  inner: {
    maxWidth: 640,
    margin: '0 auto',
    padding: '32px 24px',
  },
  backBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    fontSize: 14,
    color: 'var(--text-2)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    padding: 0,
    marginBottom: 24,
  },
  pageTitle: {
    fontSize: 24,
    fontWeight: 700,
    marginBottom: 32,
  },
  section: {
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: 'var(--text-3)',
    marginBottom: 12,
  },
  sectionDesc: {
    fontSize: 13,
    color: 'var(--text-3)',
    marginBottom: 12,
  },
  card: {
    background: 'var(--bg-1)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: '16px',
  },
  accountRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: '50%',
    flexShrink: 0,
  },
  avatarPlaceholder: {
    width: 40,
    height: 40,
    borderRadius: '50%',
    background: 'var(--accent-dim, rgba(6,182,212,0.15))',
    color: 'var(--accent)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 16,
    fontWeight: 700,
    flexShrink: 0,
  },
  accountInfo: {
    flex: 1,
    minWidth: 0,
  },
  accountName: {
    fontSize: 15,
    fontWeight: 600,
  },
  accountEmail: {
    fontSize: 13,
    color: 'var(--text-2)',
  },
  signOutBtn: {
    padding: '8px 16px',
    fontSize: 13,
    fontWeight: 500,
    color: 'var(--danger)',
    background: 'none',
    border: '1px solid var(--danger)',
    borderRadius: 'var(--radius-sm)',
    cursor: 'pointer',
    flexShrink: 0,
  },
  toggleRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '10px 0',
    cursor: 'pointer',
    borderBottom: '1px solid var(--border)',
  },
  toggleInfo: {
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  },
  toggleLabel: {
    fontSize: 14,
    fontWeight: 500,
  },
  toggleDesc: {
    fontSize: 12,
    color: 'var(--text-3)',
  },
  checkbox: {
    width: 18,
    height: 18,
    accentColor: 'var(--accent)',
    cursor: 'pointer',
  },
  dropZone: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 24,
    border: '2px dashed var(--border)',
    borderRadius: 'var(--radius-md)',
    cursor: 'pointer',
    transition: 'all var(--transition)',
    marginBottom: 12,
  },
  dropText: {
    fontSize: 13,
    color: 'var(--text-3)',
  },
  errorRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    color: 'var(--danger)',
    fontSize: 12,
    marginBottom: 8,
  },
  docList: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  docItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '10px 12px',
    background: 'var(--bg-1)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
  },
  docTitle: {
    flex: 1,
    fontSize: 13,
    fontWeight: 500,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  docMeta: {
    fontSize: 12,
    color: 'var(--text-3)',
    flexShrink: 0,
  },
  deleteBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 28,
    height: 28,
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-3)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    flexShrink: 0,
  },
  emptyText: {
    color: 'var(--text-3)',
    fontSize: 13,
    textAlign: 'center',
    padding: 20,
  },
}
