import { Scale, Settings, LogOut } from 'lucide-react'
import { useStore } from '../store'

interface HeaderProps {
  onSettingsClick: () => void
}

export default function Header({ onSettingsClick }: HeaderProps) {
  const logout = useStore((s) => s.logout)

  return (
    <header style={styles.header}>
      <div style={styles.left}>
        <Scale size={20} color="var(--accent)" />
        <span style={styles.brand}>Legal RAG</span>
      </div>
      <div style={styles.right}>
        <button onClick={onSettingsClick} style={styles.iconBtn} title="Settings">
          <Settings size={18} />
        </button>
        <button onClick={logout} style={styles.iconBtn} title="Sign out">
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
