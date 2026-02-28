import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import {
  Plus, MessageSquare, Trash2, Pencil, Check, X, Search,
  ChevronLeft, Settings, LogOut, MoreHorizontal, Scale, Download,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useStore } from '../store'
import { generateExportHTML } from '../utils/exportConversation'
import type { Conversation, ChatMessage } from '../types'

function timeGroup(dateStr: string): string {
  const d = new Date(dateStr)
  const now = new Date()
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const itemDay = new Date(d.getFullYear(), d.getMonth(), d.getDate())
  const diffDays = Math.floor((today.getTime() - itemDay.getTime()) / 86400000)

  if (diffDays === 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays <= 7) return 'Previous 7 days'
  return 'Older'
}

const GROUP_ORDER = ['Today', 'Yesterday', 'Previous 7 days', 'Older']

export default function Sidebar() {
  const navigate = useNavigate()
  const conversations = useStore((s) => s.conversations)
  const setConversations = useStore((s) => s.setConversations)
  const activeId = useStore((s) => s.activeConversationId)
  const setActiveId = useStore((s) => s.setActiveConversationId)
  const clearMessages = useStore((s) => s.clearMessages)
  const user = useStore((s) => s.user)
  const logout = useStore((s) => s.logout)

  const [collapsed, setCollapsed] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null)
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const contextMenuRef = useRef<HTMLDivElement>(null)

  const loadConversations = useCallback(async () => {
    try {
      const { data } = await api.conversations.list()
      setConversations(data)
    } catch {
      // Silently fail on conversation list errors
    } finally {
      setLoading(false)
    }
  }, [setConversations])

  useEffect(() => {
    loadConversations()
  }, [loadConversations])

  // Close context menu on outside click
  useEffect(() => {
    if (!menuOpenId) return
    const handleClickOutside = (e: MouseEvent) => {
      if (contextMenuRef.current && !contextMenuRef.current.contains(e.target as Node)) {
        setMenuOpenId(null)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [menuOpenId])

  const filteredConversations = useMemo(() => {
    if (!searchQuery.trim()) return conversations
    const q = searchQuery.toLowerCase()
    return conversations.filter((c) => c.title.toLowerCase().includes(q))
  }, [conversations, searchQuery])

  const grouped = useMemo(() => {
    const groups: Record<string, Conversation[]> = {}
    for (const c of filteredConversations) {
      const g = timeGroup(c.updated_at)
      if (!groups[g]) groups[g] = []
      groups[g].push(c)
    }
    return groups
  }, [filteredConversations])

  const handleNewChat = () => {
    setActiveId(null)
    clearMessages()
    navigate('/chat')
  }

  const handleSelectConversation = (id: string) => {
    if (id === activeId) return
    navigate(`/chat/${id}`)
  }

  const handleDelete = async (id: string) => {
    if (confirmDeleteId !== id) {
      setConfirmDeleteId(id)
      setMenuOpenId(null)
      setTimeout(() => setConfirmDeleteId((cur) => (cur === id ? null : cur)), 3000)
      return
    }
    setConfirmDeleteId(null)
    try {
      await api.conversations.delete(id)
      setConversations(conversations.filter((c) => c.id !== id))
      if (activeId === id) {
        setActiveId(null)
        clearMessages()
        navigate('/chat')
      }
    } catch {
      // ignore
    }
  }

  const startRename = (c: Conversation) => {
    setRenamingId(c.id)
    setRenameValue(c.title)
    setMenuOpenId(null)
  }

  const handleRename = async () => {
    if (!renamingId || !renameValue.trim()) {
      setRenamingId(null)
      return
    }
    try {
      await api.conversations.rename(renamingId, renameValue.trim())
      setConversations(
        conversations.map((c) =>
          c.id === renamingId ? { ...c, title: renameValue.trim() } : c
        )
      )
    } catch {
      // ignore
    }
    setRenamingId(null)
  }

  const handleExport = async (id: string, title: string) => {
    setMenuOpenId(null)
    try {
      let msgs: ChatMessage[]
      if (id === activeId) {
        msgs = useStore.getState().messages
      } else {
        const { data } = await api.conversations.messages(id)
        msgs = data.map((m) => ({
          id: m.id,
          role: m.role as 'user' | 'assistant',
          content: m.content,
          sources: m.sources ?? undefined,
          latency_ms: m.latency_ms ?? undefined,
        }))
      }
      if (!msgs.length) return
      const html = generateExportHTML(title, msgs)
      const blob = new Blob([html], { type: 'text/html' })
      const url = URL.createObjectURL(blob)
      window.open(url, '_blank')
      setTimeout(() => URL.revokeObjectURL(url), 60000)
    } catch {
      // Silently fail on export errors
    }
  }

  if (collapsed) {
    return (
      <aside style={styles.collapsedSidebar}>
        <button onClick={() => setCollapsed(false)} style={styles.expandBtn} title="Expand sidebar">
          <MessageSquare size={18} />
        </button>
        {conversations.length > 0 && (
          <span style={styles.collapsedCount}>{conversations.length}</span>
        )}
      </aside>
    )
  }

  return (
    <aside style={styles.sidebar}>
      {/* New Chat + Collapse */}
      <div style={styles.topBar}>
        <button onClick={handleNewChat} style={styles.newChatBtn}>
          <Plus size={16} />
          <span>New Chat</span>
        </button>
        <button onClick={() => { setCollapsed(true); setSearchQuery('') }} style={styles.collapseBtn} title="Collapse sidebar">
          <ChevronLeft size={16} />
        </button>
      </div>

      {/* Search bar */}
      <div style={styles.searchBar}>
        <Search size={14} color="var(--text-3)" style={{ flexShrink: 0 }} />
        <input
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search conversations..."
          style={styles.searchInput}
        />
        {searchQuery && (
          <button onClick={() => setSearchQuery('')} style={styles.searchClearBtn}>
            <X size={14} />
          </button>
        )}
      </div>

      {/* Conversation list */}
      <div style={styles.convList}>
        {loading && <p style={styles.emptyText}>Loading...</p>}
        {!loading && conversations.length === 0 && !searchQuery && (
          <p style={styles.emptyText}>No conversations yet</p>
        )}
        {!loading && searchQuery && filteredConversations.length === 0 && (
          <p style={styles.emptyText}>No conversations matching &lsquo;{searchQuery}&rsquo;</p>
        )}

        {GROUP_ORDER.map((group) => {
          const items = grouped[group]
          if (!items?.length) return null
          return (
            <div key={group}>
              <div style={styles.groupLabel}>{group}</div>
              {items.map((c) => (
                <div
                  key={c.id}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if ((e.key === 'Enter' || e.key === ' ') && renamingId !== c.id) {
                      e.preventDefault()
                      handleSelectConversation(c.id)
                    }
                  }}
                  style={{
                    ...styles.convItem,
                    ...(c.id === activeId ? styles.convItemActive : {}),
                    ...(confirmDeleteId === c.id ? styles.convItemDanger : {}),
                  }}
                  onClick={() => {
                    if (renamingId !== c.id) handleSelectConversation(c.id)
                  }}
                >
                  {renamingId === c.id ? (
                    <div style={styles.renameRow}>
                      <input
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleRename()
                          if (e.key === 'Escape') setRenamingId(null)
                        }}
                        style={styles.renameInput}
                        autoFocus
                        onClick={(e) => e.stopPropagation()}
                      />
                      <button onClick={(e) => { e.stopPropagation(); handleRename() }} style={styles.tinyBtn}>
                        <Check size={14} />
                      </button>
                      <button onClick={(e) => { e.stopPropagation(); setRenamingId(null) }} style={styles.tinyBtn}>
                        <X size={14} />
                      </button>
                    </div>
                  ) : (
                    <>
                      <span style={styles.convTitle} title={c.title}>{c.title}</span>
                      {confirmDeleteId === c.id ? (
                        <button
                          onClick={(e) => { e.stopPropagation(); handleDelete(c.id) }}
                          style={styles.confirmDeleteBtn}
                        >
                          Confirm?
                        </button>
                      ) : (
                        <div ref={menuOpenId === c.id ? contextMenuRef : undefined} style={{ position: 'relative' as const }}>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              setMenuOpenId(menuOpenId === c.id ? null : c.id)
                            }}
                            style={styles.moreBtn}
                          >
                            <MoreHorizontal size={14} />
                          </button>
                          {menuOpenId === c.id && (
                            <div style={styles.contextMenu}>
                              <button
                                style={styles.menuItem}
                                onClick={(e) => { e.stopPropagation(); startRename(c) }}
                              >
                                <Pencil size={13} /> Rename
                              </button>
                              <button
                                style={styles.menuItem}
                                onClick={(e) => { e.stopPropagation(); handleExport(c.id, c.title) }}
                              >
                                <Download size={13} /> Export
                              </button>
                              <button
                                style={{ ...styles.menuItem, color: 'var(--danger)' }}
                                onClick={(e) => { e.stopPropagation(); handleDelete(c.id) }}
                              >
                                <Trash2 size={13} /> Delete
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>
              ))}
            </div>
          )
        })}
      </div>

      {/* User footer */}
      <div style={styles.footer}>
        {user && (
          <div style={styles.userRow}>
            {user.avatar_url ? (
              <img src={user.avatar_url} alt="" style={styles.avatar} />
            ) : (
              <div style={styles.avatarPlaceholder}>
                {(user.name || user.email || '?')[0].toUpperCase()}
              </div>
            )}
            <span style={styles.userName} title={user.email}>
              {user.name || user.email}
            </span>
          </div>
        )}
        <div style={styles.footerActions}>
          <button onClick={() => navigate('/legal/terms')} style={styles.footerBtn} title="Legal">
            <Scale size={16} />
          </button>
          <button onClick={() => navigate('/settings')} style={styles.footerBtn} title="Settings">
            <Settings size={16} />
          </button>
          <button onClick={() => { logout(); navigate('/') }} style={styles.footerBtn} title="Sign out">
            <LogOut size={16} />
          </button>
        </div>
      </div>
    </aside>
  )
}

const styles: Record<string, React.CSSProperties> = {
  sidebar: {
    width: 280,
    background: '#F0F0ED',
    borderRight: '1px solid var(--border)',
    display: 'flex',
    flexDirection: 'column',
    flexShrink: 0,
    overflow: 'hidden',
  },
  collapsedSidebar: {
    width: 48,
    background: '#F0F0ED',
    borderRight: '1px solid var(--border)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    paddingTop: 12,
    gap: 8,
    flexShrink: 0,
  },
  expandBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 32,
    height: 32,
    borderRadius: 'var(--radius-sm)',
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
  topBar: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '12px 12px 8px',
  },
  newChatBtn: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '10px 14px',
    fontSize: 13.5,
    fontWeight: 500,
    color: '#FFFFFF',
    background: 'var(--accent)',
    border: 'none',
    borderRadius: 'var(--radius-sm)',
    cursor: 'pointer',
    transition: 'all var(--transition)',
  },
  collapseBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 32,
    height: 32,
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-3)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
  },
  searchBar: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    margin: '0 12px 4px',
    padding: '6px 10px',
    background: '#FFFFFF',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
  },
  searchInput: {
    flex: 1,
    border: 'none',
    outline: 'none',
    fontSize: 13,
    color: 'var(--text-1)',
    background: 'transparent',
    fontFamily: 'inherit',
    padding: 0,
  },
  searchClearBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 20,
    height: 20,
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-3)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    flexShrink: 0,
  },
  convList: {
    flex: 1,
    overflowY: 'auto',
    padding: '4px 8px',
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  },
  groupLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    fontWeight: 500,
    textTransform: 'uppercase',
    letterSpacing: '1.5px',
    color: 'var(--text-3)',
    padding: '12px 8px 4px',
  },
  convItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    padding: '10px 14px',
    borderRadius: 6,
    cursor: 'pointer',
    transition: 'background 0.2s, border-color 0.2s, box-shadow 0.2s',
    border: '1px solid transparent',
  },
  convItemActive: {
    background: '#FFFFFF',
    border: '1px solid var(--border)',
    boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
  },
  convItemDanger: {
    background: 'rgba(217,75,75,0.06)',
  },
  convTitle: {
    flex: 1,
    fontSize: 13.5,
    fontWeight: 500,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    color: 'var(--text-1)',
  },
  moreBtn: {
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
    opacity: 0.6,
    flexShrink: 0,
  },
  contextMenu: {
    position: 'absolute',
    top: '100%',
    right: 0,
    zIndex: 50,
    minWidth: 140,
    background: '#FFFFFF',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    boxShadow: 'var(--shadow-lg)',
    padding: 4,
    animation: 'fadeIn 0.15s ease',
  },
  menuItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    width: '100%',
    padding: '8px 12px',
    fontSize: 13,
    color: 'var(--text-1)',
    background: 'none',
    border: 'none',
    borderRadius: 'var(--radius-sm)',
    cursor: 'pointer',
    textAlign: 'left',
  },
  confirmDeleteBtn: {
    fontSize: 11,
    fontWeight: 600,
    color: 'var(--danger)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    flexShrink: 0,
    padding: '2px 6px',
  },
  renameRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    flex: 1,
  },
  renameInput: {
    flex: 1,
    padding: '4px 8px',
    fontSize: 13,
    background: '#FFFFFF',
    border: '1px solid var(--border-hover)',
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
    color: 'var(--text-2)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
  },
  footer: {
    padding: '12px',
    borderTop: '1px solid var(--border)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  userRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    flex: 1,
    minWidth: 0,
  },
  avatar: {
    width: 28,
    height: 28,
    borderRadius: '50%',
    flexShrink: 0,
  },
  avatarPlaceholder: {
    width: 28,
    height: 28,
    borderRadius: '50%',
    background: 'var(--accent-dim)',
    color: 'var(--accent)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 13,
    fontWeight: 700,
    flexShrink: 0,
  },
  userName: {
    fontSize: 13,
    color: 'var(--text-2)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  footerActions: {
    display: 'flex',
    gap: 2,
    flexShrink: 0,
  },
  footerBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 32,
    height: 32,
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-3)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
  },
  emptyText: {
    color: 'var(--text-3)',
    fontSize: 13,
    textAlign: 'center',
    padding: 24,
  },
}
