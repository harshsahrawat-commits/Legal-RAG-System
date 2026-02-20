import { Scale } from 'lucide-react'

export default function Header() {
  return (
    <header style={styles.header}>
      <div style={styles.left}>
        <Scale size={20} color="var(--accent)" />
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
    height: 56,
    borderBottom: '1px solid var(--border)',
    background: '#FFFFFF',
    boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
    flexShrink: 0,
  },
  left: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  brand: {
    fontSize: 18,
    fontWeight: 700,
    letterSpacing: '-0.01em',
    color: 'var(--text-1)',
  },
}
