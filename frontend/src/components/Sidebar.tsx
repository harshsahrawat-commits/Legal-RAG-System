import { useState, useEffect, useCallback, useMemo } from 'react'
import { Upload, FileText, Trash2, Loader2, AlertCircle, Search, ChevronLeft, ArrowUpDown } from 'lucide-react'
import { api } from '../api'
import type { DocumentInfo } from '../types'

const TYPE_COLORS: Record<string, string> = {
  contract: '#06b6d4',
  statute: '#8b5cf6',
  case_law: '#f59e0b',
  regulation: '#22c55e',
  unknown: '#6b7280',
}

type SortBy = 'name' | 'date' | 'type'

export default function Sidebar() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadStats, setUploadStats] = useState({ current: 0, total: 0, success: 0, failed: 0 })
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState<SortBy>('date')
  const [collapsed, setCollapsed] = useState(false)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [loadingDocs, setLoadingDocs] = useState(true)

  const loadDocs = useCallback(async () => {
    try {
      const { data } = await api.documents.list()
      setDocuments(data)
    } catch {
      setError('Failed to load documents')
    } finally {
      setLoadingDocs(false)
    }
  }, [])

  useEffect(() => {
    loadDocs()
  }, [loadDocs])

  const filteredAndSorted = useMemo(() => {
    let docs = documents
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      docs = docs.filter(
        (d) =>
          d.title.toLowerCase().includes(q) ||
          d.document_type.toLowerCase().includes(q)
      )
    }
    return [...docs].sort((a, b) => {
      if (sortBy === 'name') return a.title.localeCompare(b.title)
      if (sortBy === 'type') return a.document_type.localeCompare(b.document_type)
      // date — newest first
      return (b.created_at ?? '').localeCompare(a.created_at ?? '')
    })
  }, [documents, searchQuery, sortBy])

  const handleUpload = async (files: FileList | null) => {
    if (!files?.length) return
    const pdfFiles = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'))
    if (pdfFiles.length === 0) {
      setError('Only PDF files are supported')
      return
    }

    setError('')
    setUploading(true)
    setUploadProgress(0)
    setUploadStats({ current: 0, total: pdfFiles.length, success: 0, failed: 0 })

    let success = 0, failed = 0
    for (let i = 0; i < pdfFiles.length; i++) {
      const file = pdfFiles[i]
      setUploadStats(prev => ({ ...prev, current: i + 1 }))
      setUploadProgress(0)
      try {
        await api.documents.upload(file, setUploadProgress)
        success++
        setUploadStats(prev => ({ ...prev, success }))
        loadDocs()
      } catch (e) {
        failed++
        setUploadStats(prev => ({ ...prev, failed }))
        console.error(`Failed to upload ${file.name}:`, e)
      }
    }

    setUploading(false)
    setUploadProgress(0)
    if (failed > 0) {
      setError(`${failed} of ${pdfFiles.length} uploads failed`)
    }
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

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    handleUpload(e.dataTransfer.files)
  }

  const cycleSortBy = () => {
    setSortBy((prev) => {
      if (prev === 'date') return 'name'
      if (prev === 'name') return 'type'
      return 'date'
    })
  }

  if (collapsed) {
    return (
      <aside style={styles.collapsedSidebar}>
        <button onClick={() => setCollapsed(false)} style={styles.expandBtn} title="Expand sidebar">
          <FileText size={18} />
        </button>
        {documents.length > 0 && (
          <span style={styles.collapsedCount}>{documents.length}</span>
        )}
      </aside>
    )
  }

  return (
    <aside style={styles.sidebar}>
      <div style={styles.section}>
        <div style={styles.sectionHeader}>
          <h3 style={styles.sectionTitle}>Upload Document</h3>
          <button onClick={() => setCollapsed(true)} style={styles.collapseBtn} title="Collapse sidebar">
            <ChevronLeft size={16} />
          </button>
        </div>
        <div
          style={{ ...styles.dropZone, ...(dragOver ? styles.dropZoneActive : {}) }}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          {uploading ? (
            <>
              <Loader2 size={24} style={{ animation: 'spin 1s linear infinite' }} />
              <span style={styles.dropText}>
                Uploading {uploadStats.current}/{uploadStats.total}
              </span>
              <span style={styles.progressText}>
                {uploadStats.success > 0 && `${uploadStats.success} done`}
                {uploadStats.failed > 0 && ` / ${uploadStats.failed} failed`}
              </span>
              {uploadProgress > 0 && uploadProgress < 100 && (
                <span style={styles.progressText}>{uploadProgress}% uploading...</span>
              )}
            </>
          ) : (
            <>
              <Upload size={24} color="var(--text-3)" />
              <span style={styles.dropText}>Drop PDF here or click</span>
            </>
          )}
          <input
            id="file-input"
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
      </div>

      <div style={{ ...styles.section, flex: 1, overflow: 'hidden' }}>
        <div style={styles.sectionHeader}>
          <h3 style={styles.sectionTitle}>
            Documents
            <span style={styles.countBadge}>{documents.length}</span>
          </h3>
          <button onClick={cycleSortBy} style={styles.sortBtn} title={`Sort by ${sortBy}`}>
            <ArrowUpDown size={13} />
            <span>{sortBy}</span>
          </button>
        </div>

        {/* Search */}
        <div style={styles.searchWrap}>
          <Search size={14} color="var(--text-3)" style={{ flexShrink: 0 }} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter documents..."
            style={styles.searchInput}
          />
        </div>

        <div style={styles.docList}>
          {loadingDocs && (
            <p style={styles.emptyText}>Loading documents...</p>
          )}
          {!loadingDocs && filteredAndSorted.length === 0 && documents.length === 0 && (
            <p style={styles.emptyText}>No documents uploaded yet</p>
          )}
          {filteredAndSorted.length === 0 && documents.length > 0 && (
            <p style={styles.emptyText}>No documents match "{searchQuery}"</p>
          )}
          {filteredAndSorted.map((doc) => (
            <div key={doc.id} style={styles.docCard}>
              <div style={styles.docHeader}>
                <FileText size={16} color="var(--text-2)" style={{ flexShrink: 0 }} />
                <span style={styles.docTitle} title={doc.title}>{doc.title}</span>
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
              <div style={styles.docMeta}>
                <span
                  style={{
                    ...styles.badge,
                    color: TYPE_COLORS[doc.document_type] || TYPE_COLORS.unknown,
                    borderColor: TYPE_COLORS[doc.document_type] || TYPE_COLORS.unknown,
                  }}
                >
                  {doc.document_type}
                </span>
                <span style={styles.metaText}>{doc.page_count} pages</span>
              </div>
            </div>
          ))}
        </div>
      </div>

    </aside>
  )
}

const styles: Record<string, React.CSSProperties> = {
  sidebar: {
    width: 280,
    background: 'var(--bg-1)',
    borderRight: '1px solid var(--border)',
    display: 'flex',
    flexDirection: 'column',
    flexShrink: 0,
    overflow: 'hidden',
    transition: 'width 0.2s ease',
  },
  collapsedSidebar: {
    width: 48,
    background: 'var(--bg-1)',
    borderRight: '1px solid var(--border)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    paddingTop: 12,
    gap: 8,
    flexShrink: 0,
    transition: 'width 0.2s ease',
  },
  expandBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 32,
    height: 32,
    borderRadius: 'var(--radius-sm, 4px)',
    color: 'var(--text-2)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
  },
  collapsedCount: {
    fontSize: 11,
    fontWeight: 600,
    color: 'var(--text-3)',
  },
  section: {
    padding: '16px',
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  sectionHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  sectionTitle: {
    fontSize: 12,
    fontWeight: 600,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
    color: 'var(--text-3)',
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    margin: 0,
  },
  countBadge: {
    fontSize: 10,
    fontWeight: 700,
    background: 'var(--accent-dim, rgba(6,182,212,0.15))',
    color: 'var(--accent, #06b6d4)',
    padding: '1px 6px',
    borderRadius: 10,
  },
  collapseBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 24,
    height: 24,
    borderRadius: 'var(--radius-sm, 4px)',
    color: 'var(--text-3)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
  },
  sortBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    padding: '2px 6px',
    fontSize: 10,
    fontWeight: 500,
    color: 'var(--text-3)',
    background: 'none',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm, 4px)',
    cursor: 'pointer',
    textTransform: 'capitalize' as const,
  },
  searchWrap: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '6px 10px',
    background: 'var(--bg-2)',
    borderRadius: 'var(--radius-sm, 4px)',
    border: '1px solid var(--border)',
  },
  searchInput: {
    flex: 1,
    border: 'none',
    background: 'none',
    color: 'var(--text-1)',
    fontSize: 12,
    outline: 'none',
    fontFamily: 'inherit',
  },
  dropZone: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 20,
    border: '2px dashed var(--border)',
    borderRadius: 'var(--radius-md)',
    cursor: 'pointer',
    transition: 'all var(--transition)',
  },
  dropZoneActive: {
    borderColor: 'var(--accent)',
    background: 'var(--accent-dim)',
  },
  dropText: {
    fontSize: 13,
    color: 'var(--text-3)',
  },
  progressText: {
    fontSize: 11,
    color: 'var(--text-3)',
  },
  errorRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    color: 'var(--danger)',
    fontSize: 12,
  },
  docList: {
    flex: 1,
    overflowY: 'auto' as const,
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 8,
  },
  emptyText: {
    color: 'var(--text-3)',
    fontSize: 13,
    textAlign: 'center' as const,
    padding: 20,
  },
  docCard: {
    padding: 12,
    background: 'var(--bg-2)',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--border)',
  },
  docHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  docTitle: {
    flex: 1,
    fontSize: 13,
    fontWeight: 500,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
  },
  deleteBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 28,
    height: 28,
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-3)',
    flexShrink: 0,
    transition: 'color var(--transition)',
  },
  docMeta: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginTop: 8,
  },
  badge: {
    fontSize: 11,
    fontWeight: 500,
    padding: '2px 8px',
    borderRadius: 20,
    border: '1px solid',
  },
  metaText: {
    fontSize: 12,
    color: 'var(--text-3)',
  },
}
