import { useState } from 'react'
import { Scale, ArrowRight, Loader2 } from 'lucide-react'
import { api } from '../api'
import { useStore } from '../store'

export default function LoginScreen() {
  const [key, setKey] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const setApiKey = useStore((s) => s.setApiKey)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!key.trim()) return

    setLoading(true)
    setError('')

    try {
      await api.validateKey(key.trim())
      setApiKey(key.trim())
    } catch {
      setError('Invalid API key. Please check and try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <div style={styles.logo}>
          <Scale size={32} color="var(--accent)" />
          <h1 style={styles.title}>Legal RAG</h1>
        </div>
        <p style={styles.subtitle}>AI-powered legal document analysis with precise citations</p>

        <form onSubmit={handleSubmit} style={styles.form}>
          <label style={styles.label}>API Key</label>
          <input
            type="password"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder="lrag_xxxxxxxxxxxxxxxxxx"
            style={styles.input}
            autoFocus
          />

          {error && <p style={styles.error}>{error}</p>}

          <button type="submit" disabled={loading || !key.trim()} style={styles.button}>
            {loading ? (
              <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
            ) : (
              <>
                Sign In <ArrowRight size={16} />
              </>
            )}
          </button>
        </form>

        <p style={styles.footer}>Multi-tenant &middot; Multilingual &middot; Cited Answers</p>
      </div>

    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'var(--bg-0)',
  },
  card: {
    width: '100%',
    maxWidth: 400,
    padding: 40,
    background: 'var(--bg-1)',
    borderRadius: 'var(--radius-lg)',
    border: '1px solid var(--border)',
    boxShadow: 'var(--shadow-lg)',
    textAlign: 'center' as const,
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    marginBottom: 8,
  },
  title: {
    fontSize: 28,
    fontWeight: 700,
    letterSpacing: '-0.02em',
  },
  subtitle: {
    color: 'var(--text-2)',
    fontSize: 14,
    marginBottom: 32,
  },
  form: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 12,
    textAlign: 'left' as const,
  },
  label: {
    fontSize: 13,
    fontWeight: 500,
    color: 'var(--text-2)',
  },
  input: {
    width: '100%',
    padding: '12px 16px',
    fontSize: 15,
    background: 'var(--bg-2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-1)',
  },
  error: {
    color: 'var(--danger)',
    fontSize: 13,
    margin: 0,
  },
  button: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginTop: 8,
    padding: '12px 24px',
    background: 'var(--accent)',
    color: '#000',
    fontWeight: 600,
    fontSize: 15,
    borderRadius: 'var(--radius-sm)',
    border: 'none',
    cursor: 'pointer',
    transition: 'background var(--transition)',
  },
  footer: {
    marginTop: 24,
    fontSize: 12,
    color: 'var(--text-3)',
  },
}
