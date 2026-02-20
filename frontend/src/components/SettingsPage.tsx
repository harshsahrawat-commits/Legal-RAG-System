import { useState, useEffect, useCallback } from 'react'
import { ArrowLeft, Upload, Trash2, Loader2, AlertCircle, FileText, Plus, Pencil, Check, X, FolderOpen } from 'lucide-react'
import { api } from '../api'
import { useStore } from '../store'
import type { DocumentInfo, DocumentFamily, SourceToggles } from '../types'

const SOURCE_LABELS: Record<keyof Omit<SourceToggles, 'families'>, { label: string; desc: string }> = {
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
  const families = useStore((s) => s.families)
  const setFamilies = useStore((s) => s.setFamilies)

  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [loadingDocs, setLoadingDocs] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadFamilyId, setUploadFamilyId] = useState<string>('')
  const [error, setError] = useState('')
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)

  // Family management state
  const [newFamilyName, setNewFamilyName] = useState('')
  const [creatingFamily, setCreatingFamily] = useState(false)
  const [renamingFamilyId, setRenamingFamilyId] = useState<string | null>(null)
  const [renameFamilyValue, setRenameFamilyValue] = useState('')
  const [confirmDeleteFamilyId, setConfirmDeleteFamilyId] = useState<string | null>(null)

  const loadDocs = useCallback(async () => {
    try {
      const { data } = await api.documents.list()
      setDocuments(data)
    } catch {
      // ignore errors
    } finally {
      setLoadingDocs(false)
    }
  }, [])

  const loadFamilies = useCallback(async () => {
    try {
      const { data } = await api.families.list()
      setFamilies(data)
    } catch {
      // ignore
    }
  }, [setFamilies])

  useEffect(() => {
    loadDocs()
    loadFamilies()
  }, [loadDocs, loadFamilies])

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
        await api.documents.upload(file, setUploadProgress, uploadFamilyId || undefined)
        loadDocs()
        loadFamilies()
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
      loadFamilies()
    } catch {
      setError('Failed to delete document')
    }
  }

  const handleMoveDoc = async (docId: string, familyId: string | null) => {
    try {
      await api.documents.moveToFamily(docId, familyId)
      setDocuments((prev) =>
        prev.map((d) => d.id === docId ? { ...d, family_id: familyId } : d)
      )
      loadFamilies()
    } catch {
      setError('Failed to move document')
    }
  }

  // Family CRUD
  const handleCreateFamily = async () => {
    if (!newFamilyName.trim()) return
    setCreatingFamily(true)
    try {
      await api.families.create(newFamilyName.trim())
      setNewFamilyName('')
      loadFamilies()
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to create family')
    }
    setCreatingFamily(false)
  }

  const handleRenameFamily = async () => {
    if (!renamingFamilyId || !renameFamilyValue.trim()) {
      setRenamingFamilyId(null)
      return
    }
    try {
      await api.families.rename(renamingFamilyId, renameFamilyValue.trim())
      loadFamilies()
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to rename family')
    }
    setRenamingFamilyId(null)
  }

  const handleDeleteFamily = async (id: string) => {
    if (confirmDeleteFamilyId !== id) {
      setConfirmDeleteFamilyId(id)
      setTimeout(() => setConfirmDeleteFamilyId((cur) => (cur === id ? null : cur)), 3000)
      return
    }
    setConfirmDeleteFamilyId(null)
    try {
      await api.families.delete(id)
      loadFamilies()
      loadDocs()
    } catch {
      setError('Failed to delete family')
    }
  }

  const handleToggleFamilyActive = async (fam: DocumentFamily) => {
    try {
      await api.families.setActive(fam.id, !fam.is_active)
      loadFamilies()
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to toggle family')
    }
  }

  // Group documents by family
  const unassignedDocs = documents.filter((d) => !d.family_id)
  const docsByFamily: Record<string, DocumentInfo[]> = {}
  for (const d of documents) {
    if (d.family_id) {
      if (!docsByFamily[d.family_id]) docsByFamily[d.family_id] = []
      docsByFamily[d.family_id].push(d)
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
            ) : null}
          </div>
        </section>

        {/* Search Sources Section */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Search Sources</h2>
          <div style={styles.card}>
            {(Object.keys(SOURCE_LABELS) as (keyof Omit<SourceToggles, 'families'>)[]).map((key) => (
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

        {/* Document Families Section */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Document Collections</h2>
          <p style={styles.sectionDesc}>
            Organize your documents into collections. Activate up to 3 collections to include them in search.
          </p>

          {/* Create family */}
          <div style={styles.createRow}>
            <input
              value={newFamilyName}
              onChange={(e) => setNewFamilyName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleCreateFamily()}
              placeholder="New collection name..."
              style={styles.createInput}
              disabled={creatingFamily}
            />
            <button
              onClick={handleCreateFamily}
              disabled={creatingFamily || !newFamilyName.trim()}
              style={{
                ...styles.createBtn,
                opacity: creatingFamily || !newFamilyName.trim() ? 0.4 : 1,
              }}
            >
              <Plus size={16} />
            </button>
          </div>

          {/* Family list */}
          <div style={styles.familyList}>
            {families.map((fam) => (
              <div key={fam.id} style={styles.familyItem}>
                {renamingFamilyId === fam.id ? (
                  <div style={styles.familyRenameRow}>
                    <input
                      value={renameFamilyValue}
                      onChange={(e) => setRenameFamilyValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleRenameFamily()
                        if (e.key === 'Escape') setRenamingFamilyId(null)
                      }}
                      style={styles.renameInput}
                      autoFocus
                    />
                    <button onClick={handleRenameFamily} style={styles.tinyBtn}><Check size={14} /></button>
                    <button onClick={() => setRenamingFamilyId(null)} style={styles.tinyBtn}><X size={14} /></button>
                  </div>
                ) : (
                  <>
                    <FolderOpen size={16} color="var(--text-2)" style={{ flexShrink: 0 }} />
                    <span style={styles.familyName}>{fam.name}</span>
                    <span style={styles.familyCount}>{fam.document_count}</span>

                    {/* Active toggle */}
                    <button
                      style={{
                        ...styles.activeToggle,
                        background: fam.is_active ? '#10b981' : 'var(--bg-3)',
                      }}
                      onClick={() => handleToggleFamilyActive(fam)}
                      title={fam.is_active ? 'Deactivate' : 'Activate'}
                    >
                      <span style={{
                        ...styles.activeToggleDot,
                        transform: fam.is_active ? 'translateX(14px)' : 'translateX(0)',
                      }} />
                    </button>

                    <button
                      onClick={() => { setRenamingFamilyId(fam.id); setRenameFamilyValue(fam.name) }}
                      style={styles.tinyBtn}
                      title="Rename"
                    >
                      <Pencil size={13} />
                    </button>
                    <button
                      onClick={() => handleDeleteFamily(fam.id)}
                      style={{
                        ...styles.tinyBtn,
                        ...(confirmDeleteFamilyId === fam.id ? { color: 'var(--danger)', fontWeight: 600, fontSize: 11 } : {}),
                      }}
                      title={confirmDeleteFamilyId === fam.id ? 'Click to confirm' : 'Delete'}
                    >
                      {confirmDeleteFamilyId === fam.id ? 'Confirm?' : <Trash2 size={13} />}
                    </button>
                  </>
                )}
              </div>
            ))}
            {families.length === 0 && (
              <p style={styles.emptyText}>No collections yet</p>
            )}
          </div>
        </section>

        {/* Documents Section */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>My Documents</h2>
          <p style={styles.sectionDesc}>
            Upload PDFs and optionally assign them to a collection.
          </p>

          {/* Upload target family selector */}
          <div style={styles.uploadRow}>
            <select
              value={uploadFamilyId}
              onChange={(e) => setUploadFamilyId(e.target.value)}
              style={styles.familySelect}
            >
              <option value="">No collection (unassigned)</option>
              {families.map((f) => (
                <option key={f.id} value={f.id}>{f.name}</option>
              ))}
            </select>
          </div>

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

          {/* Documents grouped by family */}
          <div style={styles.docList}>
            {loadingDocs && <p style={styles.emptyText}>Loading documents...</p>}
            {!loadingDocs && documents.length === 0 && (
              <p style={styles.emptyText}>No documents uploaded yet</p>
            )}

            {/* Family-grouped documents */}
            {families.map((fam) => {
              const fDocs = docsByFamily[fam.id]
              if (!fDocs?.length) return null
              return (
                <div key={fam.id}>
                  <div style={styles.docGroupLabel}>
                    <FolderOpen size={13} color="var(--text-3)" />
                    <span>{fam.name}</span>
                  </div>
                  {fDocs.map((doc) => renderDocItem(doc, fam.id))}
                </div>
              )
            })}

            {/* Unassigned documents */}
            {unassignedDocs.length > 0 && (
              <div>
                {families.length > 0 && (
                  <div style={styles.docGroupLabel}>
                    <span>Unassigned</span>
                  </div>
                )}
                {unassignedDocs.map((doc) => renderDocItem(doc, null))}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  )

  function renderDocItem(doc: DocumentInfo, _currentFamilyId: string | null) {
    return (
      <div key={doc.id} style={styles.docItem}>
        <FileText size={16} color="var(--text-2)" style={{ flexShrink: 0 }} />
        <span style={styles.docTitle} title={doc.title}>{doc.title}</span>
        <span style={styles.docMeta}>{doc.page_count}p</span>

        {/* Move to family dropdown */}
        <select
          value={doc.family_id || ''}
          onChange={(e) => handleMoveDoc(doc.id, e.target.value || null)}
          style={styles.moveSelect}
          title="Move to collection"
        >
          <option value="">—</option>
          {families.map((f) => (
            <option key={f.id} value={f.id}>{f.name}</option>
          ))}
        </select>

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
    )
  }
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
    background: 'var(--glass-bg)',
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
    border: '1px solid var(--glass-border)',
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
    background: 'var(--accent-dim)',
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

  // Family management
  createRow: {
    display: 'flex',
    gap: 8,
    marginBottom: 12,
  },
  createInput: {
    flex: 1,
    padding: '8px 12px',
    fontSize: 13,
    background: 'var(--bg-2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-1)',
    outline: 'none',
    fontFamily: 'inherit',
  },
  createBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 36,
    height: 36,
    borderRadius: 'var(--radius-sm)',
    background: 'var(--accent)',
    color: '#000',
    border: 'none',
    cursor: 'pointer',
    flexShrink: 0,
  },
  familyList: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
    marginBottom: 8,
  },
  familyItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '8px 12px',
    background: 'var(--glass-bg)',
    border: '1px solid var(--glass-border)',
    borderRadius: 'var(--radius-sm)',
  },
  familyRenameRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    flex: 1,
  },
  familyName: {
    flex: 1,
    fontSize: 13,
    fontWeight: 500,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  familyCount: {
    fontSize: 11,
    color: 'var(--text-3)',
    background: 'var(--bg-2)',
    padding: '2px 6px',
    borderRadius: 10,
    flexShrink: 0,
  },
  activeToggle: {
    width: 32,
    height: 18,
    borderRadius: 9,
    border: 'none',
    cursor: 'pointer',
    position: 'relative' as const,
    flexShrink: 0,
    transition: 'background 0.2s',
    padding: 0,
  },
  activeToggleDot: {
    display: 'block',
    width: 14,
    height: 14,
    borderRadius: '50%',
    background: '#fff',
    position: 'absolute' as const,
    top: 2,
    left: 2,
    transition: 'transform 0.2s',
    boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
  },
  renameInput: {
    flex: 1,
    padding: '4px 8px',
    fontSize: 13,
    background: 'var(--bg-2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-1)',
    outline: 'none',
    fontFamily: 'inherit',
  },
  tinyBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 24,
    height: 24,
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-3)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    flexShrink: 0,
  },

  // Upload
  uploadRow: {
    marginBottom: 8,
  },
  familySelect: {
    width: '100%',
    padding: '8px 12px',
    fontSize: 13,
    background: 'var(--bg-2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-1)',
    outline: 'none',
    fontFamily: 'inherit',
  },
  dropZone: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 24,
    border: '2px dashed var(--glass-border)',
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

  // Document list
  docList: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  docGroupLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    fontSize: 11,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: 'var(--text-3)',
    padding: '10px 4px 4px',
  },
  docItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '8px 12px',
    background: 'var(--glass-bg)',
    border: '1px solid var(--glass-border)',
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
  moveSelect: {
    padding: '4px 8px',
    fontSize: 11,
    background: 'var(--bg-2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-2)',
    outline: 'none',
    fontFamily: 'inherit',
    maxWidth: 100,
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
