import { Scale } from 'lucide-react'

export default function Header() {
  return (
    <header style={styles.header}>
      <div style={styles.left}>
        <Scale size={20} color="var(--accent)" style={{ filter: 'drop-shadow(0 0 8px var(--accent-glow))' }} />
        <span style={styles.brand}>Legal RAG</span>
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
    borderBottom: '1px solid var(--glass-border)',
    background: 'var(--glass-bg)',
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
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
}
