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
}
