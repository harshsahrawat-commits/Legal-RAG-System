import { Scale, Settings, LogOut, Trash2 } from 'lucide-react'
import { useStore } from '../store'

interface HeaderProps {
  onSettingsClick: () => void
}

export default function Header({ onSettingsClick }: HeaderProps) {
  const logout = useStore((s) => s.logout)
  const clearMessages = useStore((s) => s.clearMessages)
  const hasMessages = useStore((s) => s.messages.length > 0)

  return (
    <header style={styles.header}>
      <div style={styles.left}>
        <Scale size={20} color="var(--accent)" />
        <span style={styles.brand}>Legal RAG</span>
      </div>
      <div style={styles.right}>
        {hasMessages && (
          <button onClick={clearMessages} style={styles.iconBtn} title="Clear chat" aria-label="Clear chat">
            <Trash2 size={18} />
          </button>
        )}
        <button onClick={onSettingsClick} style={styles.iconBtn} title="Settings" aria-label="Settings">
          <Settings size={18} />
        </button>
        <button onClick={logout} style={styles.iconBtn} title="Sign out" aria-label="Sign out">
          <LogOut size={18} />
        </button>
      </div>
    </header>
  )
}

const styles: Record<string, React.CSSProperties> = {
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 24px',
    borderBottom: '1px solid var(--border)',
    background: 'var(--bg-1)',
    flexShrink: 0,
  },
  left: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  brand: {
    fontSize: 16,
    fontWeight: 700,
    letterSpacing: '-0.01em',
  },
  right: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
  },
  iconBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 36,
    height: 36,
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-2)',
    transition: 'all var(--transition)',
  },
}
